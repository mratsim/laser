TODO: very WIP

## Challenges

1. There is no batched GEMM in OpenBLAS
2. BLIS and cublas strided GEMM supports arbitrary strides which avoids a copy to a contiguous tensor.
   but neither MKL or OpenBLAS supports it. Arbitrary strides are resolved during packing.
3. In deep learning, GEMMs and convolutions (which often use GEMM) are always followed by a non-linear activation which is memory-bound. Allowing non-linearity + GEMM fusion would probably increase throughput tremendously.

## GEMM

[1] Anatomy of High-Performance Matrix Multiplication (Revised)
    Kazushige Goto, Robert A. Van de Geijn
  - http://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf

[2] Anatomy of High-Performance Many-Threaded Matrix Multiplication
    Smith et al
  - http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf

[3] Automating the Last-Mile for High Performance Dense Linear Algebra
    Veras et al
  - https://arxiv.org/pdf/1611.08035.pdf

[4] GEMM: From Pure C to SSE Optimized Micro Kernels
    Michael Lehn
  - http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/gemm/index.html


## Small GEMMs

- https://github.com/hfp/libxsmm
- High-Performance Matrix-Matrix Multiplications of Very Small Matrices
    - https://hal.archives-ouvertes.fr/hal-01409286/document
- Algorithms and Optimization Techniques for High-Performance Matrix-Matrix Multiplications of Very Small Matrices
    - https://www.icl.utk.edu/files/publications/2018/icl-utk-1075-2018.pdf
      includes Magma, MKL and libxsmm comparison
- BLASFEO: basic linear algebra subroutines for embedded optimization
    - https://arxiv.org/pdf/1704.02457.pdf
    - https://github.com/giaf/blasfeo

## Links

- https://www.codeproject.com/Articles/1169319/Reducing-Packing-Overhead-in-Matrix-Matrix-Multipl
- https://www.icl.utk.edu/files/publications/2017/icl-utk-1032-2017.pdf
- https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/
- Automatic Generation of Fast BLAS3-GEMM:
A Portable Compiler Approach,
    - http://www.cse.unsw.edu.au/~jingling/papers/cgo17.pdf
- http://www.cs.utexas.edu/users/flame/pubs/bamarker_thesis.pdf

## Implementations
- Matmul with OpenMP tasks
  https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf p78
- Optimizing Matrix Multiplication on Xeon Phi
  https://www.springer.com/cda/content/document/cda_downloaddocument/9783319064857-c9.pdf?SGWID=0-0-45-1471414-p176715535