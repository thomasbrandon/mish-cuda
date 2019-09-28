#include <torch/types.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <c10/util/Half.h>
#define GLOBAL_INLINE __forceinline__ __host__ __device__
#else
#include <cmath>
#define GLOBAL_INLINE __inline__
#endif

#define THRESHOLD 20

// TODO: Try and convert these to lambda functions
template <typename scalar_t>
GLOBAL_INLINE
void mish_fwd_func(scalar_t &out, const scalar_t &inp) {
  out = inp * tanh(
    inp < scalar_t(THRESHOLD) ? log1p(exp(inp)) : inp
  );
};

template <typename scalar_t>
GLOBAL_INLINE
void mish_bwd_func(scalar_t &grad_inp, const scalar_t &inp, const scalar_t &grad_out) {
  const scalar_t sp = inp < scalar_t(THRESHOLD) ? log1p(exp(inp)) : inp;
  const scalar_t grad_sp = 1 - exp(-sp);
  const scalar_t tsp = tanh(sp);
  const scalar_t grad_tsp = (1 - tsp*tsp) * grad_sp;
  const scalar_t grad = inp * grad_tsp + tsp;
  grad_inp = grad_out * grad;
};

// Specialisations for Half to calculate as float
// Increases precision and also lacking certain instrinsics for Half
template <>
GLOBAL_INLINE
void mish_fwd_func(c10::Half &out, const c10::Half &inp) {
  float res;
  mish_fwd_func<float>(res, (float)inp);
  out = res;
};

template <>
GLOBAL_INLINE
void mish_bwd_func(c10::Half &grad_inp, const c10::Half &inp, const c10::Half &grad_out) {
  float res;
  mish_bwd_func<float>(res, (float)inp, (float)grad_out);
  grad_inp = res;
};