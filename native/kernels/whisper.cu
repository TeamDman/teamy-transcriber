// Whisper's numerical primitives. Model topology and cache ownership live in Rust.
// Activations and weights are row-major FP32; all work uses one owned stream.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <new>

struct Session { cudaStream_t stream{}; cublasHandle_t blas{}; };
static thread_local char last_error[512];
static int error(const char* op, int status) {
    snprintf(last_error, sizeof(last_error), "%s failed (%d)", op, status);
    return status ? status : -1;
}
#define CUDA(call) do { auto e = (call); if(e != cudaSuccess) return error(#call, int(e)); } while(0)
#define BLAS(call) do { auto e = (call); if(e != CUBLAS_STATUS_SUCCESS) return error(#call, int(e)); } while(0)
extern "C" const char* tw_error() { return last_error; }
extern "C" int tw_create(Session** out, int device, int tf32) {
    CUDA(cudaSetDevice(device));
    auto* s = new(std::nothrow) Session;
    if (!s) return error("session allocation", -1);
    auto ce = cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking);
    if (ce != cudaSuccess) { delete s; return error("stream creation", int(ce)); }
    auto be = cublasCreate(&s->blas);
    if (be != CUBLAS_STATUS_SUCCESS) { cudaStreamDestroy(s->stream); delete s; return error("cuBLAS creation", int(be)); }
    be = cublasSetStream(s->blas, s->stream);
    if (be == CUBLAS_STATUS_SUCCESS) be = cublasSetMathMode(s->blas, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
    if (be != CUBLAS_STATUS_SUCCESS) { cublasDestroy(s->blas); cudaStreamDestroy(s->stream); delete s; return error("cuBLAS configuration", int(be)); }
    *out = s; return 0;
}
extern "C" void tw_destroy(Session* s) { if (s) { cudaStreamSynchronize(s->stream); cublasDestroy(s->blas); cudaStreamDestroy(s->stream); delete s; } }
extern "C" int tw_alloc(Session*, float** p, size_t n) { CUDA(cudaMalloc(p, n*sizeof(float))); return 0; }
extern "C" void tw_free(Session* s, float* p) { cudaStreamSynchronize(s->stream); cudaFree(p); }
extern "C" int tw_upload(Session* s, float* dst, const float* src, size_t n) {
    CUDA(cudaMemcpyAsync(dst, src, n*sizeof(float), cudaMemcpyHostToDevice, s->stream));
    // The caller may release its host slice immediately after return.
    CUDA(cudaStreamSynchronize(s->stream)); return 0;
}
extern "C" int tw_download(Session* s, float* dst, const float* src, size_t n) {
    CUDA(cudaMemcpyAsync(dst, src, n*sizeof(float), cudaMemcpyDeviceToHost, s->stream));
    CUDA(cudaStreamSynchronize(s->stream)); return 0;
}
extern "C" int tw_copy(Session* s, float* dst, const float* src, size_t n) {
    CUDA(cudaMemcpyAsync(dst, src, n*sizeof(float), cudaMemcpyDeviceToDevice, s->stream)); return 0;
}
extern "C" int tw_sync(Session* s) { CUDA(cudaStreamSynchronize(s->stream)); return 0; }

__global__ void affine(float* x, const float* bias, const float* residual, int width, size_t n, int gelu) {
    size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i] + (bias ? bias[i % width] : 0.f);
        if (gelu) v = .5f*v*(1.f + erff(v * .7071067811865475f));
        if (residual) v += residual[i];
        x[i] = v;
    }
}
extern "C" int tw_linear(Session* s, const float* x, const float* w, const float* b, const float* r, float* y, int rows, int input, int output, int gelu) {
    const float one=1.f, zero=0.f;
    BLAS(cublasSgemm(s->blas,CUBLAS_OP_T,CUBLAS_OP_N,output,rows,input,&one,w,input,x,input,&zero,y,output));
    size_t n = size_t(rows)*output;
    if (b || r || gelu) affine<<<unsigned((n+255)/256),256,0,s->stream>>>(y,b,r,output,n,gelu);
    CUDA(cudaGetLastError()); return 0;
}
__global__ void norm(const float* x,const float* w,const float* b,float* y,int width) {
    __shared__ float sums[256];
    int tid=threadIdx.x; size_t base=size_t(blockIdx.x)*width;
    float sum=0; for(int i=tid;i<width;i+=256) sum += x[base+i];
    sums[tid]=sum; __syncthreads();
    for(int d=128;d;d>>=1) { if(tid<d) sums[tid]+=sums[tid+d]; __syncthreads(); }
    float mean=sums[0]/width; __syncthreads(); sum=0;
    for(int i=tid;i<width;i+=256) { float v=x[base+i]-mean; sum+=v*v; }
    sums[tid]=sum; __syncthreads();
    for(int d=128;d;d>>=1) { if(tid<d) sums[tid]+=sums[tid+d]; __syncthreads(); }
    float inv=rsqrtf(sums[0]/width+1.e-5f);
    for(int i=tid;i<width;i+=256) y[base+i]=(x[base+i]-mean)*inv*w[i]+b[i];
}
extern "C" int tw_norm(Session* s,const float* x,const float* w,const float* b,float* y,int rows,int width) {
    norm<<<rows,256,0,s->stream>>>(x,w,b,y,width); CUDA(cudaGetLastError()); return 0;
}
__global__ void columns(const float* x,float* col,int time,int channels,int out_time,int stride,int planar) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x, n=size_t(out_time)*channels*3;
    if(i<n) {
        int tap=i%3, c=(i/3)%channels, t=int(i/(3*channels))*stride+tap-1;
        col[i]=(t<0||t>=time)?0.f:x[planar?size_t(c)*time+t:size_t(t)*channels+c];
    }
}
extern "C" int tw_conv(Session* s,const float* x,const float* w,const float* b,float* col,float* y,int time,int input,int output,int stride,int planar) {
    int out_time=(time+stride-1)/stride;
    size_t n=size_t(out_time)*input*3;
    columns<<<unsigned((n+255)/256),256,0,s->stream>>>(x,col,time,input,out_time,stride,planar);
    CUDA(cudaGetLastError());
    return tw_linear(s,col,w,b,nullptr,y,out_time,input*3,output,1);
}
__global__ void add_position(float* x,const float* pos,int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=pos[i]; }
extern "C" int tw_position(Session* s,float* x,const float* pos,int n) {
    add_position<<<(n+255)/256,256,0,s->stream>>>(x,pos,n); CUDA(cudaGetLastError()); return 0;
}
__global__ void embed(const float* w,const float* pos,float* y,int token,int position,int width) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<width) y[i]=w[size_t(token)*width+i]+pos[size_t(position)*width+i];
}
extern "C" int tw_embed(Session* s,const float* w,const float* pos,float* y,int token,int position,int width) {
    embed<<<(width+255)/256,256,0,s->stream>>>(w,pos,y,token,position,width); CUDA(cudaGetLastError()); return 0;
}
__global__ void softmax(float* scores,int queries,int keys,int causal,int offset) {
    __shared__ float sums[256];
    int tid=threadIdx.x, row=blockIdx.x, query=row%queries;
    float* x=scores+size_t(row)*keys;
    float mx=-INFINITY;
    for(int j=tid;j<keys;j+=256) { float v=(!causal||j<=offset+query)?x[j]:-INFINITY; x[j]=v; mx=fmaxf(mx,v); }
    sums[tid]=mx; __syncthreads();
    for(int d=128;d;d>>=1) { if(tid<d) sums[tid]=fmaxf(sums[tid],sums[tid+d]); __syncthreads(); }
    mx=sums[0]; __syncthreads(); float sum=0;
    for(int j=tid;j<keys;j+=256) { float v=expf(x[j]-mx); x[j]=v; sum+=v; }
    sums[tid]=sum; __syncthreads();
    for(int d=128;d;d>>=1) { if(tid<d) sums[tid]+=sums[tid+d]; __syncthreads(); }
    sum=sums[0]; for(int j=tid;j<keys;j+=256) x[j]/=sum;
}
extern "C" int tw_attention(Session* s,const float* q,const float* k,const float* v,float* scores,float* out,int nq,int nk,int width,int heads,int causal,int offset) {
    int d=width/heads; float scale=1.f/sqrtf(float(d)),one=1.f,zero=0.f;
    long long score_stride=static_cast<long long>(nq)*nk;
    BLAS(cublasSgemmStridedBatched(s->blas,CUBLAS_OP_T,CUBLAS_OP_N,nk,nq,d,&scale,k,width,d,q,width,d,&zero,scores,nk,score_stride,heads));
    softmax<<<heads*nq,256,0,s->stream>>>(scores,nq,nk,causal,offset); CUDA(cudaGetLastError());
    BLAS(cublasSgemmStridedBatched(s->blas,CUBLAS_OP_N,CUBLAS_OP_N,d,nq,nk,&one,v,width,d,scores,nk,score_stride,&zero,out,width,d,heads));
    return 0;
}
// One block is sufficient for this reduction: copy one token rather than 50K logits.
__global__ void argmax(const float* x,const float* allowed,float* result,int n) {
    __shared__ float values[256]; __shared__ int ids[256];
    int tid=threadIdx.x, best=-1; float value=-INFINITY;
    for(int i=tid;i<n;i+=256) if(allowed[i] && isfinite(x[i]) && (x[i]>value || (x[i]==value && i>best))) {value=x[i];best=i;}
    values[tid]=value; ids[tid]=best; __syncthreads();
    for(int d=128;d;d>>=1) {
        if(tid<d && (values[tid+d]>values[tid] || (values[tid+d]==values[tid] && ids[tid+d]>ids[tid]))) {values[tid]=values[tid+d];ids[tid]=ids[tid+d];}
        __syncthreads();
    }
    if(tid==0) result[0]=float(ids[0]);
}
extern "C" int tw_argmax(Session* s,const float* x,const float* allowed,float* result,int n) {
    argmax<<<1,256,0,s->stream>>>(x,allowed,result,n); CUDA(cudaGetLastError()); return 0;
}
