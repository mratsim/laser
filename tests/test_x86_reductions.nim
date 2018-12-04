# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  unittest, random, sequtils, strformat,
  ../laser/private/[error_functions, lift],
  ../laser/primitives/simd_math/reductions_sse3

# To be tested with and without OpenMP

randomize(0xDEADBEEF) # Random seed for reproducibility

liftReduction(sum, `+`, 0)
liftReduction(min, min, Inf)
liftReduction(max, max, -Inf)

template reduction_test(size: int, refcall, optcall: untyped) =
  test refcall.astToStr & ", Length {size}":
    let a = newSeqWith(size, float32 rand(1.0))

    let expected = refcall(a)
    let optimized = optcall(a[0].unsafeAddr, a.len)

    check:
      relative_error(optimized, expected) < 1e-5'f32
      absolute_error(optimized, expected) < 1e-5'f32

suite "[x86 SSE3] Reductions":
  reduction_test 100, sum, sum_sse3
  reduction_test 100, min, min_sse3
  reduction_test 100, max, max_sse3
