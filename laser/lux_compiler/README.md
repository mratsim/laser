# Lux Embedded DSL for High-Performance Computing


## Description

Lux will be an embedded language to describe tensor transformations.

Those tensor transformations will be described in a minimum of 2 axis:
  - The algorithm part for example for a matrix multiplication
    it would be C[i, j] = A[i, k] * B[k, j]
  - The schedule part for example parallelizing a loop, tiling a loop or using
    SIMD vectorization

Potential future axis would be multi-device support like
managing CPU sockets or MPI or multi-GPU.

## Implementation plan

At first Lux will work at compile-time and be implemented using Nim metaprogramming. It will mainly target CPU for a start.

In the future a compile-time CUDA output is planned and a JIT with LLVM backend as well that can target all LLVM supported platforms (Cuda, OpenCL, Vulkan, and maybe graphic shaders like DX12 or OpenGL).

Depending on Nim development, the compile-time version of Lux might be re-implemented later as DLL/.so compiler plugin. This would probably significantly improve compilation speed for large project and avoids compile-time limitations.

## References

https://github.com/mratsim/Arraymancer/issues/347
https://github.com/mratsim/compute-graph-optim
