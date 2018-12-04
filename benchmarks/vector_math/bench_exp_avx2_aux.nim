import ../../laser/simd
import ospaths, strutils
from os import DirSep

const cSourcesPath = currentSourcePath.rsplit(DirSep, 1)[0] & '/'
{.passC: "-I" & cSourcesPath .}

func round_down_power_of_2(x: Natural, step: static Natural): int {.inline.} =
  static: assert (step and (step - 1)) == 0, "Step must be a power of 2"
  result = x and not(step - 1)

template vectorize(
      wrapped_func,
      funcname,
      simd_load,
      simd_store: untyped,
      unroll_factor: int) =
  proc funcname*(dst, src: ptr UncheckedArray[float32], len: Natural) =
    let unroll_stop = len.round_down_power_of_2(unroll_factor)

    for i in countup(0, unroll_stop - 1, unroll_factor):
      dst[i].addr.simd_store src[i].addr.simd_load.wrapped_func
    for i in unroll_stop ..< len:
      dst[i] = src[i]

{.passC: "-D__AVX2__" .}
proc avx2_fmath_exp_ps256*(x: m256): m256 {.
    importcpp: "fmath_avx2::exp_ps256(@)",
    header: cSourcesPath & "lib_fmath_avx2.cpp"
  .}

vectorize(avx2_fmath_exp_ps256, avx2_fmath_exp_ps256, mm256_load_ps, mm256_store_ps, 8)


