# Profile of my reference benchmark system

i9-9980XE, Skylake-X, 18 cores with AVX512 support
  - All clock turbo: 4.1 Ghz
  - All AVX clock turbo: 4.0 Ghz
  - All AVX512 clock turbo:  3.5 Ghz
  - 2x FMA units per core

# Theoretical performance

  - Theoretical peak GFlop/s
      - Cores: 18
      - CpuGhz: 3.5
      - VectorWidth: 16
      - Instr/cycle: 2x FMA
      - Flop/Instr: 2 (FMA = 1 add + 1 mul)

      ==> 4032 GFlop/s

# Performance report
Measured by https://github.com/Mysticial/Flops/

```
Running Skylake Purley tuned binary with 1 thread...

Single-Precision - 128-bit AVX - Add/Sub
    GFlops = 32.704
    Result = 4.19594e+06

Double-Precision - 128-bit AVX - Add/Sub
    GFlops = 16.4
    Result = 2.06506e+06

Single-Precision - 128-bit AVX - Multiply
    GFlops = 32.736
    Result = 4.17422e+06

Double-Precision - 128-bit AVX - Multiply
    GFlops = 16.368
    Result = 2.07052e+06

Single-Precision - 128-bit AVX - Multiply + Add
    GFlops = 31.536
    Result = 3.36271e+06

Double-Precision - 128-bit AVX - Multiply + Add
    GFlops = 15.768
    Result = 1.67479e+06

Single-Precision - 128-bit FMA3 - Fused Multiply Add
    GFlops = 65.664
    Result = 4.17071e+06

Double-Precision - 128-bit FMA3 - Fused Multiply Add
    GFlops = 32.832
    Result = 2.08244e+06

Single-Precision - 256-bit AVX - Add/Sub
    GFlops = 64
    Result = 8.26566e+06

Double-Precision - 256-bit AVX - Add/Sub
    GFlops = 32
    Result = 4.08321e+06

Single-Precision - 256-bit AVX - Multiply
    GFlops = 64.032
    Result = 8.16782e+06

Double-Precision - 256-bit AVX - Multiply
    GFlops = 32.016
    Result = 4.06897e+06

Single-Precision - 256-bit AVX - Multiply + Add
    GFlops = 61.536
    Result = 6.57213e+06

Double-Precision - 256-bit AVX - Multiply + Add
    GFlops = 30.768
    Result = 3.27099e+06

Single-Precision - 256-bit FMA3 - Fused Multiply Add
    GFlops = 128.064
    Result = 8.18976e+06

Double-Precision - 256-bit FMA3 - Fused Multiply Add
    GFlops = 64.032
    Result = 4.09577e+06

Single-Precision - 512-bit AVX512 - Add/Sub
    GFlops = 112.128
    Result = 1.42363e+07

Double-Precision - 512-bit AVX512 - Add/Sub
    GFlops = 56.064
    Result = 7.07992e+06

Single-Precision - 512-bit AVX512 - Multiply
    GFlops = 112.128
    Result = 1.42943e+07

Double-Precision - 512-bit AVX512 - Multiply
    GFlops = 56.064
    Result = 7.14002e+06

Single-Precision - 512-bit AVX512 - Multiply + Add
    GFlops = 112.128
    Result = 1.19709e+07

Double-Precision - 512-bit AVX512 - Multiply + Add
    GFlops = 56.064
    Result = 5.97725e+06

Single-Precision - 512-bit AVX512 - Fused Multiply Add
    GFlops = 224.256
    Result = 1.42891e+07

Double-Precision - 512-bit AVX512 - Fused Multiply Add
    GFlops = 112.128
    Result = 7.14155e+06


Running Skylake Purley tuned binary with 36 thread(s)...

Single-Precision - 128-bit AVX - Add/Sub
    GFlops = 588.8
    Result = 7.49435e+07

Double-Precision - 128-bit AVX - Add/Sub
    GFlops = 294.992
    Result = 3.76555e+07

Single-Precision - 128-bit AVX - Multiply
    GFlops = 590.4
    Result = 7.53444e+07

Double-Precision - 128-bit AVX - Multiply
    GFlops = 295.296
    Result = 3.76554e+07

Single-Precision - 128-bit AVX - Multiply + Add
    GFlops = 590.4
    Result = 6.2638e+07

Double-Precision - 128-bit AVX - Multiply + Add
    GFlops = 295.248
    Result = 3.13618e+07

Single-Precision - 128-bit FMA3 - Fused Multiply Add
    GFlops = 1181.09
    Result = 7.52832e+07

Double-Precision - 128-bit FMA3 - Fused Multiply Add
    GFlops = 590.592
    Result = 3.76499e+07

Single-Precision - 256-bit AVX - Add/Sub
    GFlops = 1151.42
    Result = 1.46472e+08

Double-Precision - 256-bit AVX - Add/Sub
    GFlops = 575.52
    Result = 7.3292e+07

Single-Precision - 256-bit AVX - Multiply
    GFlops = 1151.9
    Result = 1.46865e+08

Double-Precision - 256-bit AVX - Multiply
    GFlops = 575.76
    Result = 7.33665e+07

Single-Precision - 256-bit AVX - Multiply + Add
    GFlops = 1151.71
    Result = 1.22439e+08

Double-Precision - 256-bit AVX - Multiply + Add
    GFlops = 575.808
    Result = 6.12064e+07

Single-Precision - 256-bit FMA3 - Fused Multiply Add
    GFlops = 2302.66
    Result = 1.46864e+08

Double-Precision - 256-bit FMA3 - Fused Multiply Add
    GFlops = 1151.42
    Result = 7.34076e+07

Single-Precision - 512-bit AVX512 - Add/Sub
    GFlops = 2017.54
    Result = 2.57166e+08

Double-Precision - 512-bit AVX512 - Add/Sub
    GFlops = 1008.38
    Result = 1.28578e+08

Single-Precision - 512-bit AVX512 - Multiply
    GFlops = 2017.92
    Result = 2.57183e+08

Double-Precision - 512-bit AVX512 - Multiply
    GFlops = 1009.54
    Result = 1.28591e+08

Single-Precision - 512-bit AVX512 - Multiply + Add
    GFlops = 2019.46
    Result = 2.14496e+08

Double-Precision - 512-bit AVX512 - Multiply + Add
    GFlops = 1009.54
    Result = 1.07391e+08

Single-Precision - 512-bit AVX512 - Fused Multiply Add
    GFlops = 4036.61
    Result = 2.57604e+08

Double-Precision - 512-bit AVX512 - Fused Multiply Add
    GFlops = 2018.3
    Result = 1.2861e+08
```