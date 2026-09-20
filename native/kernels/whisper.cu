// Whisper's numerical primitives. Model topology and cache ownership live in Rust.
// Activations and weights are row-major FP32; all work uses one owned stream.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
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
// Incremental decoding has one input row. Keep its dot product, bias, GELU
// and residual in one launch instead of a general GEMM plus a second kernel.
// One warp owns each output; all products and reductions remain FP32.
template<bool Vectorized>
__global__ void matvec(const float* x, const float* w, const float* b,
                      const float* residual, float* y, int input, int output, int gelu) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= output) return;
    float sum = 0.f;
    if (Vectorized) {
        const float4* xv = reinterpret_cast<const float4*>(x);
        const float4* wv = reinterpret_cast<const float4*>(w + size_t(row) * input);
        float a = 0.f, c = 0.f, d = 0.f, e = 0.f;
        for (int j = lane; j < input / 4; j += 32) {
            float4 vx = xv[j], vw = wv[j];
            a = fmaf(vx.x, vw.x, a); c = fmaf(vx.y, vw.y, c);
            d = fmaf(vx.z, vw.z, d); e = fmaf(vx.w, vw.w, e);
        }
        sum = (a + c) + (d + e);
    } else {
        for (int j = lane; j < input; j += 32)
            sum = fmaf(x[j], w[size_t(row) * input + j], sum);
    }
    for (int step = 16; step; step >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, step);
    if (!lane) {
        float value = sum + (b ? b[row] : 0.f);
        if (gelu) value = .5f * value * (1.f + erff(value * .7071067811865475f));
        if (residual) value += residual[row];
        y[row] = value;
    }
}
extern "C" int tw_linear(Session* s, const float* x, const float* w, const float* b, const float* r, float* y, int rows, int input, int output, int gelu) {
    if (rows == 1) {
        bool aligned = input % 4 == 0 && ((reinterpret_cast<std::uintptr_t>(x) | reinterpret_cast<std::uintptr_t>(w)) & 15) == 0;
        if (aligned) matvec<true><<<(output+3)/4,128,0,s->stream>>>(x,w,b,r,y,input,output,gelu);
        else matvec<false><<<(output+3)/4,128,0,s->stream>>>(x,w,b,r,y,input,output,gelu);
        CUDA(cudaGetLastError()); return 0;
    }
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
// Independent single-query sequences: each block owns one sequence/head.
// Cache rows are [sequence, capacity, width]; no sequence reads another's KV.
// FP32 score, softmax and value reduction share one launch and no global scores.
__global__ void decode_attention(const float* q,const float* k,const float* v,
                                 float* out,int keys,int capacity,int width,int heads) {
    __shared__ float scores[1500];
    __shared__ float reduction[256];
    int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    int head=blockIdx.x, sequence=blockIdx.y;
    size_t qb=size_t(sequence)*width+head*64;
    size_t kb=size_t(sequence)*capacity*width+head*64;
    float q0=q[qb+lane],q1=q[qb+lane+32];
    for(int key=warp;key<keys;key+=8) {
        size_t base=kb+size_t(key)*width;
        float sum=fmaf(q0,k[base+lane],q1*k[base+lane+32]);
        for(int delta=16;delta;delta>>=1) sum+=__shfl_down_sync(0xffffffffu,sum,delta);
        if(!lane) scores[key]=sum*.125f;
    }
    __syncthreads();
    float maximum=-INFINITY;
    for(int key=tid;key<keys;key+=256) maximum=fmaxf(maximum,scores[key]);
    reduction[tid]=maximum; __syncthreads();
    for(int delta=128;delta;delta>>=1) {if(tid<delta) reduction[tid]=fmaxf(reduction[tid],reduction[tid+delta]); __syncthreads();}
    maximum=reduction[0]; __syncthreads();
    float sum=0.f;
    for(int key=tid;key<keys;key+=256) {float p=expf(scores[key]-maximum);scores[key]=p;sum+=p;}
    reduction[tid]=sum; __syncthreads();
    for(int delta=128;delta;delta>>=1) {if(tid<delta) reduction[tid]+=reduction[tid+delta]; __syncthreads();}
    float inverse=1.f/reduction[0]; __syncthreads();
    int channel=tid%64, group=tid/64;
    float value=0.f;
    for(int key=group;key<keys;key+=4)
        value=fmaf(scores[key]*inverse,v[kb+size_t(key)*width+channel],value);
    reduction[tid]=value; __syncthreads();
    if(tid<64) out[qb+tid]=(reduction[tid]+reduction[tid+64])+(reduction[tid+128]+reduction[tid+192]);
}
extern "C" int tw_decode_attention(Session* s,const float* q,const float* k,const float* v,
                                    float* out,int batch,int keys,int capacity,int width,int heads) {
    decode_attention<<<dim3(heads,batch),256,0,s->stream>>>(q,k,v,out,keys,capacity,width,heads);
    CUDA(cudaGetLastError()); return 0;
}
__global__ void cache_token(const float* x,float* cache,int position,int capacity,int width,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) cache[size_t(i/width)*capacity*width+size_t(position)*width+i%width]=x[i];
}
extern "C" int tw_cache_token(Session* s,const float* x,float* cache,int batch,int position,int capacity,int width) {
    int count=batch*width;
    cache_token<<<(count+255)/256,256,0,s->stream>>>(x,cache,position,capacity,width,count);
    CUDA(cudaGetLastError()); return 0;
}
__global__ void embed_batch(const float* w,const float* pos,const float* tokens,float* y,int position,int width,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) y[i]=w[size_t(int(tokens[i/width]))*width+i%width]+pos[size_t(position)*width+i%width];
}
extern "C" int tw_embed_batch(Session* s,const float* w,const float* pos,const float* tokens,float* y,int batch,int position,int width) {
    int count=batch*width;
    embed_batch<<<(count+255)/256,256,0,s->stream>>>(w,pos,tokens,y,position,width,count);
    CUDA(cudaGetLastError()); return 0;
}
// One block is sufficient for this reduction: copy one token rather than 50K logits.
__global__ void argmax(const float* x,const float* allowed,float* result,int n) {
    __shared__ float values[256]; __shared__ int ids[256];
    x += size_t(blockIdx.x)*n;
    int tid=threadIdx.x, best=-1; float value=-INFINITY;
    for(int i=tid;i<n;i+=256) if(allowed[i] && isfinite(x[i]) && (x[i]>value || (x[i]==value && i>best))) {value=x[i];best=i;}
    values[tid]=value; ids[tid]=best; __syncthreads();
    for(int d=128;d;d>>=1) {
        if(tid<d && (values[tid+d]>values[tid] || (values[tid+d]==values[tid] && ids[tid+d]>ids[tid]))) {values[tid]=values[tid+d];ids[tid]=ids[tid+d];}
        __syncthreads();
    }
    if(tid==0) result[blockIdx.x]=float(ids[0]);
}
extern "C" int tw_argmax(Session* s,const float* x,const float* allowed,float* result,int n) {
    argmax<<<1,256,0,s->stream>>>(x,allowed,result,n); CUDA(cudaGetLastError()); return 0;
}
extern "C" int tw_argmax_batch(Session* s,const float* x,const float* allowed,float* result,int batch,int n) {
    argmax<<<batch,256,0,s->stream>>>(x,allowed,result,n); CUDA(cudaGetLastError()); return 0;
}
