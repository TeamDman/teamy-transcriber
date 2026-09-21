// Phone encoder additions; included after the common CUDA session primitives.
__global__ void px_columns(const float* x,float* col,int time,int input,int ci,int kernel,int stride,int pad,int rows,int group) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(rows)*ci*kernel) {
        int k=i%kernel, c=(i/kernel)%ci, t=int(i/(ci*kernel))*stride+k-pad;
        col[i]=(t<0||t>=time)?0.f:x[size_t(t)*input+group*ci+c];
    }
}
__global__ void px_depthwise(const float* x,const float* w,const float* b,float* y,int time,int width,int kernel,int pad) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<time*width) { int t=i/width,c=i%width; float v=b[c];
        for(int k=0;k<kernel;k++) { int s=t+k-pad; if(s>=0&&s<time) v=fmaf(x[s*width+c],w[c*kernel+k],v); }
        y[i]=v;
    }
}
extern "C" int px_conv(Session* s,const float* x,const float* w,const float* b,float* col,float* y,int time,int input,int output,int kernel,int stride,int pad,int groups,int rows) {
    if(groups==input && input==output && stride==1 && rows==time) {
        px_depthwise<<<(time*input+255)/256,256,0,s->stream>>>(x,w,b,y,time,input,kernel,pad);
    } else {
        int ci=input/groups, co=output/groups;
        const float one=1.f,zero=0.f;
        for(int g=0;g<groups;g++) {
            size_t n=size_t(rows)*ci*kernel;
            px_columns<<<unsigned((n+255)/256),256,0,s->stream>>>(x,col,time,input,ci,kernel,stride,pad,rows,g);
            BLAS(cublasSgemm(s->blas,CUBLAS_OP_T,CUBLAS_OP_N,co,rows,ci*kernel,&one,w+size_t(g)*co*ci*kernel,ci*kernel,col,ci*kernel,&zero,y+g*co,output));
        }
        affine<<<(size_t(rows)*output+255)/256,256,0,s->stream>>>(y,b,nullptr,output,size_t(rows)*output,0);
    }
    CUDA(cudaGetLastError()); return 0;
}
// 0 swish, 1 GELU, 2 scaled residual, 3 multiplication.
__global__ void px_point(const float* x,const float* b,float* y,int n,int op,float scale) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) {float v=x[i];
        if(op==0) v=v/(1.f+expf(-v));
        if(op==1) v=.5f*v*(1.f+erff(v*.7071067811865475f));
        if(op==2) v=b[i]+scale*v;
        if(op==3) v*=b[i];
        y[i]=v;
    }
}
extern "C" int px_pointwise(Session* s,const float* x,const float* b,float* y,int n,int op,float scale) {
    px_point<<<(n+255)/256,256,0,s->stream>>>(x,b,y,n,op,scale); CUDA(cudaGetLastError());return 0;
}
__global__ void px_copy_rows(const float* x,float* y,int rows,int width,int xs,int ys,int xo,int yo) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<rows*width) y[(i/width)*ys+yo+i%width]=x[(i/width)*xs+xo+i%width];
}
extern "C" int px_rows(Session* s,const float* x,float* y,int rows,int width,int xs,int ys,int xo,int yo) {
    px_copy_rows<<<(rows*width+255)/256,256,0,s->stream>>>(x,y,rows,width,xs,ys,xo,yo); CUDA(cudaGetLastError());return 0;
}
extern "C" int px_norm(Session* s,const float* x,const float* w,const float* b,float* y,int rows,int width,float eps) {
    norm<<<rows,256,0,s->stream>>>(x,w,b,y,width,eps); CUDA(cudaGetLastError());return 0;
}
__global__ void px_softmax_kernel(const float* x,float* y,int width) {
    // CTC vocabulary is small (428); one block owns each row.
    __shared__ float v[512]; int t=threadIdx.x; int base=blockIdx.x*width;
    v[t]=t<width?x[base+t]:-INFINITY; __syncthreads();
    for(int d=256;d;d>>=1){if(t<d)v[t]=fmaxf(v[t],v[t+d]);__syncthreads();}
    float a=t<width?expf(x[base+t]-v[0]):0.f; __syncthreads();v[t]=a;__syncthreads();
    for(int d=256;d;d>>=1){if(t<d)v[t]+=v[t+d];__syncthreads();}
    if(t<width)y[base+t]=a/v[0];
}
extern "C" int px_softmax(Session* s,const float* x,float* y,int rows,int width) {
    px_softmax_kernel<<<rows,512,0,s->stream>>>(x,y,width);CUDA(cudaGetLastError());return 0;
}
