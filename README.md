# Laser - Primitives for high performance computing

Carefully-tuned primitives for running tensor and image-processing code
on CPU, GPUs and accelerators.

The library is in heavy development. For now the CPU backend is being optimised.

## Library content

<!-- TOC -->

- [Laser - Primitives for high performance computing](#laser---primitives-for-high-performance-computing)
  - [Library content](#library-content)
    - [SIMD intrinsics for x86 and x86-64](#simd-intrinsics-for-x86-and-x86-64)
    - [OpenMP templates](#openmp-templates)
    - [`cpuinfo` for runtime CPU feature detection for x86, x86-64 and ARM](#cpuinfo-for-runtime-cpu-feature-detection-for-x86-x86-64-and-arm)
    - [JIT Assembler](#jit-assembler)
    - [Loop-fusion and strided iterators for matrix and tensors](#loop-fusion-and-strided-iterators-for-matrix-and-tensors)
    - [Raw tensor type](#raw-tensor-type)
    - [Optimised floating point parallel reduction for sum, min and max](#optimised-floating-point-parallel-reduction-for-sum-min-and-max)
    - [Optimised logarithmic, exponential, tanh, sigmoid, softmax ...](#optimised-logarithmic-exponential-tanh-sigmoid-softmax)
    - [Optimised transpose, batched transpose and NCHW <=> NHWC format conversion](#optimised-transpose-batched-transpose-and-nchw--nhwc-format-conversion)
    - [Optimised strided Matrix-Multiplication for integers and floats](#optimised-strided-matrix-multiplication-for-integers-and-floats)
      - [In the future](#in-the-future)
        - [Operation fusion](#operation-fusion)
        - [Pre-packing](#pre-packing)
        - [Batched matrix multiplication](#batched-matrix-multiplication)
        - [Small matrix multiplication](#small-matrix-multiplication)
    - [Optimised convolutions](#optimised-convolutions)
    - [State-of-the art random distributions and weighted random sampling](#state-of-the-art-random-distributions-and-weighted-random-sampling)
  - [Usage & Installation](#usage--installation)
  - [License](#license)

<!-- /TOC -->

### SIMD intrinsics for x86 and x86-64
```Nim
import laser/simd
```

Laser includes a wrapper for x86 and x86-64 to operate on 128-bit (SSE) and 256-bit (AVX) vectors of floats and integers. SIMD are added on a as-needed basis for Laser optimisation needs.

### OpenMP templates
```Nim
import laser/openmp
```

Laser includes several OpenMP templates to easu data-parallel programming in Nim:
  - The simple omp parallel for loops
  - Splitting into chunks and having a per-thread ptr+len pair to paralley algorithm that takes a ptr+len
  - `omp parallel`, `omp critical`, `omp master`, `omp barrier` and `omp flush` for fine-grained control over parallelism
  - `attachGC` and `detachGC` if you need to use Nim GC-ed types in a non-master thread.

Examples:
  - [ex02_omp_parallel_for.nim](./examples/ex02_omp_parallel_for.nim)
  - [ex03_omp_parallel_chunks](./examples/ex03_omp_parallel_chunks.nim)

### `cpuinfo` for runtime CPU feature detection for x86, x86-64 and ARM

```Nim
import laser/cpuinfo
```

Laser includes a wrapper for [`cpuinfo`](https://github.com/pytorch/cpuinfo) by Facebook's PyTorch team.
This allows to query runtime information about CPU SIMD capabilities and various L1, L2, L3, L4 CPU cache sizes
to optimize your compute-bound algorithms.

Example: [ex01_cpuinfo.nim](./examples/ex01_cpuinfo.nim)

### JIT Assembler

```Nim
import laser/photon_jit
```

Laser offers its own JIT assembler with features being added on a as needed basis.
It is very lightweight and easy to extend. Currently it only supports x86-64 with [the following
opcodes](./laser/photon_jit/x86_64/x86_64_ops.nim).

Examples:
  - [ex06_jit_hello_world.nim](./examples/ex06_jit_hello_world.nim)
  - [ex07_jit_brainfuck_vm.nim](./examples/ex07_jit_brainfuck_vm.nim)

### Loop-fusion and strided iterators for matrix and tensors

```Nim
import laser/strided_iteration/foreach
import laser/strided_iteration/foreach_staged
```

Usage - forEach:

```Nim
forEach x in a, y in b, z in c:
  x += y * z
```

Laser includes optimised macros to iterate on contiguous and strided tensors.
The iterators work with normal Nim syntax, are parallelized via OpenMP when it makes sense.

Any tensor type works as long as it exposes the following interface:
  - rank: the number of dimensions
  - size: the number of elements in the tensor
  - shape, strides: a container that supports `[]` indexing
  - unsafe_raw_data: a routine that returns
    a `ptr UncheckedArray[T]` or
    any type with `[]` indexing implemented, including mutable indexing.

A advanced iterator `forEach_staged` provides a lot of flexibility to deal with advanced need, for example for parallel reduction:

```Nim
proc reduction_localsum_critical[T](x, y: Tensor[T]): T =
  forEachStaged xi in x, yi in y:
    openmp_config:
      use_openmp: true
      use_simd: false
      nowait: true
      omp_grain_size: OMP_MEMORY_BOUND_GRAIN_SIZE
    iteration_kind:
      {contiguous, strided} # Default, "contiguous", "strided" are also possible
    before_loop:
      var local_sum = 0.T
    in_loop:
      local_sum += xi + yi
    after_loop:
      omp_critical:
        result += local_sum
```

Examples:
  - ex04 - TODO
  - [ex05_tensor_parallel_reduction](./examples/ex05_tensor_parallel_reduction.nim)

Benchmarks:
  - [iter_bench.nim](./benchmarks/loop_iteration/iter_bench.nim)
  - [iter_bench_prod.nim](./benchmarks/loop_iteration/iter_bench_prod.nim)

### Raw tensor type

```Nim
import laser/tensor/[datatypes, allocator, initialization] # WIP
```

Laser includes a low-level tensor type with only the low-level allocation and initialization needed:
  - Aligned allocator
  - Parallel zero-ing and copy (deep copy, copy from a seq)
  - Metadata initialisation
  - Tensor raw data access via pointers is using Nim compiler for safeguard.
    Immutable objects return a `RawImmutablePtr`
    and mutable objects return a `RawMutablePtr`
    to prevent you from accidentally modifying an immutable object when accessing raw memory.

An example of how to use that to build higher-level `newTensor` or `randomTensor`, `transpose` and `[]` is give in the `iter_bench` in the previous section.

### Optimised floating point parallel reduction for sum, min and max

```Nim
import laser/primitives/reductions
```

Floating-point reductions are not optimised by compilers by default because they can't assume that
`result = (a+b) + c` is equivalent to `result = a + (b + c)` due to how floating-point rounding work.
This forces serial evaluation of reductions unless `-ffast-math` flag is passed to the compiler.

The primitives work around that by keeping several accumulators in parallel to avoid waiting for a previous serial evaluation. This allows those kernels to maximise memory-bandwith of your computer.

Benchmarks:
  - [reduction_packed_sse](./benchmarks/fp_reduction_latency/reduction_packed_sse.nim)

### Optimised logarithmic, exponential, tanh, sigmoid, softmax ...

In heavy development.

Unfortunately the default logarithm and exponential functions included in C and C++ standard \<math.h\> library are extremely slow.

Benchmarks shows that a 10x speed improvement is possible while keeping excellent accuracy.

Benchmarks:
  - [bench_exp](./benchmarks/vector_math/bench_exp.nim)
  - [bench_exp_avx2](./benchmarks/vector_math/bench_exp_avx2.nim)

### Optimised transpose, batched transpose and NCHW <=> NHWC format conversion

```Nim
import laser/primitives/swapaxes
```

While logical transpose (just swapping the `shape` and `strides` metadata of the tensor/matrix) is often enough, we sometimes might need to transpose data physically in-memory.

Laser provides Optimised routines for physical transpose, batched transpose (N matrices) and also transposition of images from and to NCHW and NHWC i.e. [Image id, Color, Height, Width] and [Image id, Height, Width, Color].

90% of ML libraries including Nvidia's CuDNN prefer to work in NCHW while often images are decoded in HWC.

Benchmarks:
  - [transpose_bench](./benchmarks/transpose/transpose_bench.nim)

### Optimised strided Matrix-Multiplication for integers and floats

```Nim
import laser/primitives/matrix_multiplication/gemm
```

Matrix multiplication is the at the base of Machine Learning and numerical computing.

The Dense/Linear/Affine layer of neural network is just a matrix-multiplication and often convolutions are reframed into matrix multiplication to use the 20 years of optimisation research gone into [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) libraries.

Laser implements its own multithreaded BLAS with the following details:

  - It reaches 98% of OpenBLAS speed on float64 when multithreaded and 102% when single-threaded
  - It reaches 97% of OpenBLAS speed on float32 when multithreaded and 99% when single-threaded
  - It support strided matrices, for example resulting from slicing every 2 rows
    or every 2 columns: `myTensor[0::2, :]`.
    This is very useful when doing cross-validation as you don't need an extra copy before matrix-multiplication.
  - Contrary to 99% of the BLAS out there, it supports integers: `int32` and `int64` using SSE2 or AVX2 instructions
  - Extending support to new SIMD including ARM Neon and AVX512 is very easy, including software fallback is easy as well. For example this is how to [add AVX2 int32](./laser/primitives/matrix_multiplication/gemm_ukernel_avx2.nim) support with fused multiply-add fallback:
    ```Nim
    template int32x8_muladd_unfused_avx2(a, b, c: m256i): m256i =
    mm256_add_epi32(mm256_mullo_epi32(a, b), c)

    ukernel_generator(
          x86_AVX2,
          typ = int32,
          vectype = m256i,
          nb_scalars = 8,
          simd_setZero = mm256_setzero_si256,
          simd_broadcast_value = mm256_set1_epi32,
          simd_load_aligned = mm256_load_si256,
          simd_load_unaligned = mm256_loadu_si256,
          simd_store_unaligned = mm256_storeu_si256,
          simd_mul = mm256_mullo_epi32,
          simd_add = mm256_add_epi32,
          simd_fma = int32x8_muladd_unfused_avx2
        )
    ```

#### In the future

##### Operation fusion

The BLAS will allow easily fusing unary operations (like `max/relu`, `tanh` or `sigmoid`) and binary operations (like adding a bias) at the end of the matrix multiplication kernels.

As those operations are memory-bound and not compute-bound, and for matrix multiplication we already have all the data in memory (in the unary case) or half the data (in the binary case), we basically save lots by not looping once again on the matrix to apply them.

Similarly, you will be able to fuse operations before the matrix multiplication kernel, during the prepacking when data is being re-ordered for high performance processing. This will be useful
for backward propagation when before each matrix multiplication we must apply the derivatives of `relu`, `tanh` and `sigmoid`.

##### Pre-packing

Also pre-packing matrices and working on pre-packed matrices is being added. This is useful for matrices that are being used repeatedly, for example for batched matrix multiplication.

`im2col` prepacker that fuses the `convolution->matrix multiplication` (im2col) step with the matrix multiplication packing is also planned to get very efficient convolutions.

##### Batched matrix multiplication

We often have to bached matrix multiplication for examples N tensors A multiplied by a tensor B, or N tensors A multiplied by N tensors B, this is planned.

##### Small matrix multiplication

In many cases we don't deal with 1000x1000 matrices. For example the traditional image size is 224x224 and the overhead to re-pack matrices in an efficient format is not justified.

When reframing convolutions in terms of matrix multiplication this is even worse as the main convolution kernels are 1x1, 3x3, 5x5.

Optimised small matrix-multiplication is planned.

### Optimised convolutions

In heavy development.

Benchmarks:
  - [conv2D_bench](./benchmarks/convolution/conv2d_bench.nim)

### State-of-the art random distributions and weighted random sampling

In heavy development

Benchmarks of multinomial sampling for Natural Language Processing and Reinforcement Learning:
  -[bench_multinomial_samplers](./benchmarks/random_sampling/bench_multinomial_sampler)

## Usage & Installation

The library is split in relatively independant modules that can be used without the others.

For example to just use the SIMD and cpu-detection portion, just do:

```Nim
import laser/simd
import laser/cpuinfo
```

To just use OpenMP

```Nim
import laser/openmp
```

The library is unstable and will be published on nimble when more mature.
Basically it will be published when it's ready to be the CPU backend of [Arraymancer](https://github.com/mratsim/Arraymancer),
it will automatically profit from the dozens of tests and edge cases handled in Arraymancer test suite.

## License

* Laser is licensed under the Apache License version 2
* Facebook's cpuinfo is licensed under Simplified BSD (BSD 2 clauses)
