2x Xeon Gold 6154, Skylake-SP, 2 sockets 36 cores with AVX512 support.

  - 2x FMA units per core

# Turbo table:

| Threads | Non AVX | AVX | AVX2 | AVX-512 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.7 | 3.7 | 3.6 | 3.5 |
| 72 | 3.7 | 3.7 | 3.3 | 2.7 |

# Theoretical perfortmance

  - Theoretical peak GFlop/s
      - Cores: 36
      - CpuGhz: 2.7
      - VectorWidth: 16
      - Instr/cycle: 2x FMA
      - Flop/Instr: 2 (FMA = 1 add + 1 mul)

      ==> 6221 GFlop/s


# Performance report
Measured by https://github.com/Mysticial/Flops/

Compile flags:
`flags="-O3 -pthread -std=c++11"`

```
Running Skylake Purley tuned binary with 1 thread...

Single-Precision - 128-bit AVX - Add/Sub
    GFlops = 29.6
    Result = 3.77062e+06

Double-Precision - 128-bit AVX - Add/Sub
    GFlops = 14.8
    Result = 1.89082e+06

Single-Precision - 128-bit AVX - Multiply
    GFlops = 29.616
    Result = 3.76264e+06

Double-Precision - 128-bit AVX - Multiply
    GFlops = 14.808
    Result = 1.88361e+06

Single-Precision - 128-bit AVX - Multiply + Add
    GFlops = 28.464
    Result = 3.01437e+06

Double-Precision - 128-bit AVX - Multiply + Add
    GFlops = 14.232
    Result = 1.49273e+06

Single-Precision - 128-bit FMA3 - Fused Multiply Add
    GFlops = 59.232
    Result = 3.75951e+06

Double-Precision - 128-bit FMA3 - Fused Multiply Add
    GFlops = 29.616
    Result = 1.88064e+06

Single-Precision - 256-bit AVX - Add/Sub
    GFlops = 56.448
    Result = 7.13658e+06

Double-Precision - 256-bit AVX - Add/Sub
    GFlops = 28.224
    Result = 3.60599e+06

Single-Precision - 256-bit AVX - Multiply
    GFlops = 56.544
    Result = 7.1869e+06

Double-Precision - 256-bit AVX - Multiply
    GFlops = 28.224
    Result = 3.57487e+06

Single-Precision - 256-bit AVX - Multiply + Add
    GFlops = 54.24
    Result = 5.74009e+06

Double-Precision - 256-bit AVX - Multiply + Add
    GFlops = 27.12
    Result = 2.85433e+06

Single-Precision - 256-bit FMA3 - Fused Multiply Add
    GFlops = 112.896
    Result = 7.17821e+06

Double-Precision - 256-bit FMA3 - Fused Multiply Add
    GFlops = 56.448
    Result = 3.58505e+06

Single-Precision - 512-bit AVX512 - Add/Sub
    GFlops = 109.568
    Result = 1.38866e+07

Double-Precision - 512-bit AVX512 - Add/Sub
    GFlops = 54.912
    Result = 6.97365e+06

Single-Precision - 512-bit AVX512 - Multiply
    GFlops = 109.824
    Result = 1.39655e+07

Double-Precision - 512-bit AVX512 - Multiply
    GFlops = 54.912
    Result = 6.95621e+06

Single-Precision - 512-bit AVX512 - Multiply + Add
    GFlops = 109.824
    Result = 1.16442e+07

Double-Precision - 512-bit AVX512 - Multiply + Add
    GFlops = 54.912
    Result = 5.8363e+06

Single-Precision - 512-bit AVX512 - Fused Multiply Add
    GFlops = 219.648
    Result = 1.39502e+07

Double-Precision - 512-bit AVX512 - Fused Multiply Add
    GFlops = 109.44
    Result = 6.91304e+06


Running Skylake Purley tuned binary with 72 thread(s)...

Single-Precision - 128-bit AVX - Add/Sub
    GFlops = 1061.66
    Result = 1.34525e+08

Double-Precision - 128-bit AVX - Add/Sub
    GFlops = 529.344
    Result = 6.71256e+07

Single-Precision - 128-bit AVX - Multiply
    GFlops = 1059.94
    Result = 1.34484e+08

Double-Precision - 128-bit AVX - Multiply
    GFlops = 530.76
    Result = 6.73558e+07

Single-Precision - 128-bit AVX - Multiply + Add
    GFlops = 1060.75
    Result = 1.12212e+08

Double-Precision - 128-bit AVX - Multiply + Add
    GFlops = 530.016
    Result = 5.6014e+07

Single-Precision - 128-bit FMA3 - Fused Multiply Add
    GFlops = 2124.67
    Result = 1.3479e+08

Double-Precision - 128-bit FMA3 - Fused Multiply Add
    GFlops = 1060.51
    Result = 6.73101e+07

Single-Precision - 256-bit AVX - Add/Sub
    GFlops = 1896.51
    Result = 2.404e+08

Double-Precision - 256-bit AVX - Add/Sub
    GFlops = 945.344
    Result = 1.19742e+08

Single-Precision - 256-bit AVX - Multiply
    GFlops = 1893.7
    Result = 2.40188e+08

Double-Precision - 256-bit AVX - Multiply
    GFlops = 949.776
    Result = 1.20462e+08

Single-Precision - 256-bit AVX - Multiply + Add
    GFlops = 1895.42
    Result = 2.00145e+08

Double-Precision - 256-bit AVX - Multiply + Add
    GFlops = 946.608
    Result = 1.0001e+08

Single-Precision - 256-bit FMA3 - Fused Multiply Add
    GFlops = 3793.15
    Result = 2.40581e+08

Double-Precision - 256-bit FMA3 - Fused Multiply Add
    GFlops = 1897.92
    Result = 1.20378e+08

Single-Precision - 512-bit AVX512 - Add/Sub
    GFlops = 3106.56
    Result = 3.94345e+08

Double-Precision - 512-bit AVX512 - Add/Sub
    GFlops = 1549.82
    Result = 1.96823e+08

Single-Precision - 512-bit AVX512 - Multiply
    GFlops = 3101.95
    Result = 3.94153e+08

Double-Precision - 512-bit AVX512 - Multiply
    GFlops = 1557.89
    Result = 1.97929e+08

Single-Precision - 512-bit AVX512 - Multiply + Add
    GFlops = 3114.62
    Result = 3.29293e+08

Double-Precision - 512-bit AVX512 - Multiply + Add
    GFlops = 1555.2
    Result = 1.64751e+08

Single-Precision - 512-bit AVX512 - Fused Multiply Add
    GFlops = 6241.54
    Result = 3.96636e+08

Double-Precision - 512-bit AVX512 - Fused Multiply Add
    GFlops = 3111.17
    Result = 1.97388e+08
```
