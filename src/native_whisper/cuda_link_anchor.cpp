// Keep the CUDA LibTorch import library reachable in Windows packaged builds.
namespace at::cuda {
void CachingHostAllocator_emptyCache();
}

extern "C" void teamy_transcriber_force_torch_cuda() {
    at::cuda::CachingHostAllocator_emptyCache();
}
