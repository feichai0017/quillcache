// FP16 -> FP8 (E4M3) quantize-on-offload kernel.
//
// Halves the host/DRAM footprint of a KV block as it cools out of HBM. Compiled
// with nvcc on a GPU box and loaded by the cudarc device tier. This is the
// reference kernel; a production build uses __nv_fp8_e4m3 for the encode and a
// per-tensor or per-block scale chosen from the KV magnitude distribution.

#include <cuda_fp16.h>

extern "C" __global__ void quantize_fp16_to_fp8(const __half *__restrict__ in,
                                                unsigned char *__restrict__ out,
                                                int n, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float v = __half2float(in[i]) * scale;
  // Clamp to the E4M3 representable magnitude (~448).
  if (v > 448.0f) {
    v = 448.0f;
  }
  if (v < -448.0f) {
    v = -448.0f;
  }
  // Reference encode: a real build emits __nv_fp8_e4m3(v).
  out[i] = (unsigned char)(__float2int_rn(v) & 0xff);
}
