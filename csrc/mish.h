#ifdef __CUDACC__
#include <cuda_runtime.h>
#define GLOBAL_INLINE __forceinline__ __host__ __device__
#else
#include <cmath>
#define GLOBAL_INLINE __inline__
#endif

template <typename scalar_t>
GLOBAL_INLINE
void mish_fwd_func(scalar_t &out, const scalar_t &inp) {
  out = inp * tanh(log(exp(inp) + scalar_t(1.0)));
}

template <typename scalar_t>
GLOBAL_INLINE
void mish_bwd_func(scalar_t &grad_inp, const scalar_t &inp, const scalar_t &grad_out) {
  scalar_t w, d;
  // TODO: Test performance using exp(x*inp) -> exp(inp)**x
  w = 4*(inp+1) + 4*exp(2*inp) + exp(3*inp) + exp(inp) * (4*inp+6);
  d = 2*exp(inp) + exp(2*inp) + 2;
  grad_inp = grad_out * exp(inp) * w / pow(d, 2);
};