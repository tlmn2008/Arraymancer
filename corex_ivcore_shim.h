/* Iluvatar CoreX (ivcore11) migration shim -- repo-local, force-included via
 * nim.cfg ONLY in the `-d:corex` build (see the @if corex branch in nim.cfg).
 * It is never referenced on the default NVIDIA/nvcc path.
 *
 * The CoreX CUDA headers (/usr/local/corex/include/IX/crt/host_defines.h) define
 *   #define __noinline__ __attribute__((noinline))
 * and clang implicitly force-includes them in `-x ivcore` (CUDA) mode.
 *
 * Nim's nimbase.h emits its N_NOINLINE macro as:
 *   rettype __attribute__((__noinline__)) name
 * With host_defines.h's macro active, the inner `__noinline__` token re-expands to
 *   __attribute__((__attribute__((noinline))))
 * which is invalid ("use of undeclared identifier 'noinline'").
 *
 * `__attribute__((__noinline__))` is a perfectly valid attribute spelling on its own, so we
 * simply drop the CUDA object-like macro that shadows this token. `__forceinline__` is left
 * intact because the Arraymancer device kernels legitimately use `__device__ __forceinline__`
 * (and Nim's nimbase.h never emits a bare `__forceinline__`). No SDK file is touched.
 */
#ifdef __noinline__
#undef __noinline__
#endif
