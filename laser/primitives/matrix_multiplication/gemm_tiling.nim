# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#               Cache and register optimizations
#
# ############################################################

# Papers:
#   [1] Anatomy of High-Performance Matrix Multiplication (Revised)
#       Kazushige Goto, Robert A. Van de Geijn
#     - http://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf
#
#   [2] Anatomy of High-Performance Many-Threaded Matrix Multiplication
#       Smith et al
#     - http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf
#
#   [3] Automating the Last-Mile for High Performance Dense Linear Algebra
#       Veras et al
#     - https://arxiv.org/pdf/1611.08035.pdf
#
#   [4] GEMM: From Pure C to SSE Optimized Micro Kernels
#       Michael Lehn
#     - http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/gemm/index.html
#
#   Laser wiki - GEMM optimization resources
#     - https://github.com/numforge/laser/wiki/GEMM-optimization-resources

import
  ../../cpuinfo, ../../compiler_optim_hints,
  ../../private/memory,
  typetraits, macros,
  ./gemm_utils

# ############################################################
#
#                    Microkernel (µkernel)
#
# ############################################################

# We have to take into account vectorisation
# so that the microkernel can be processed with vectorized intrinsics.
#
# Caux [mr, nr] must stay in register.
#   - mr ~= nr is optimal to amortize register load cost
#   - some registers must be left to prefetch Ã and ~B (PackedA and PackedB)
#   - nr >= (flops/cycle) / (bytes/cycle) * sizeof(element)
#
# For example Haswell is capable of
#   - 32 single-precision FLOPs/cycle
#   - 32 bytes/cycle store and 64 bytes/cycle load (store C, load A and B)
#
#   so nr >= 32/32 * 4
#   For that number of FLOP it must schedule
#   2xFMA so consume 16 single-precision float
#   so mr*nr >= 16

type
  MicroKernel* = object
    mr*, nr*: int
    vec_size*: int
    cpu_simd*: CPUFeatureX86
    c_unit_stride*: bool # We can use SIMD for the epilogue of C has a unit_stride

    # TODO: ARM support
    #   - https://github.com/nim-lang/Nim/issues/9679
    #   - https://github.com/nim-lang/Nim/issues/9678

  CPUFeatureX86* = enum
    x86_Generic,
    # x86_MMX,
    x86_SSE,
    x86_SSE2,
    x86_AVX,
    x86_AVX2,
    x86_AVX512

  X86_FeatureMap = array[CPUFeatureX86, int]

const X86_vecsize_float: X86_FeatureMap = [
  x86_Generic:         1,
  x86_SSE:     128 div 8,
  x86_SSE2:    128 div 8,
  x86_AVX:     256 div 8,
  x86_AVX2:    256 div 8,
  x86_AVX512:  512 div 8
]

const X86_vecsize_int: X86_FeatureMap = [
  x86_Generic:         1,
  x86_SSE:             1,
  x86_SSE2:    128 div 8, # SSE2 generalises 128-bit reg to int
  x86_AVX:     128 div 8,
  x86_AVX2:    256 div 8, # AVX2 generalises 256-bit reg to int
  x86_AVX512:  512 div 8
]

# mr * nr < number of registers - 4
# 4 registers are needed to hold Ã and ~B and loop index
const X86_regs: X86_FeatureMap = [
  x86_Generic: 2,
  x86_SSE:     2, # 8 XMM regs in 32-bit, 16 in 64-bit (we assume 32-bit mode)
  x86_SSE2:    6,
  x86_AVX:     6, # 16 YMM registers
  x86_AVX2:    6,
  x86_AVX512:  6  # 32 ZMM registers
]

func x86_ukernel*(cpu: CPUFeatureX86, T: typedesc, c_unit_stride: bool): MicroKernel =
  result.c_unit_stride = c_unit_stride
  when T is SomeFloat:
    result.vecsize =  X86_vecsize_float[cpu]
  else:
    result.vecsize =  X86_vecsize_int[cpu]
  # TODO: Complex support

  # The inner microkernel loop does:
  #   AB[m][n] = A[m] * B[n]
  # So n should be the vector size
  # if most matrices are row-Major.
  # This avoids dealing with transpose
  # in the inner loop and untranspose in the epilogue

  result.mr = X86_regs[cpu] # Base nb of registers
  result.nr = max(1, result.vecsize div sizeof(T))

  # 64-bit - use 8/12 out of the 16 XMM/YMM registers
  result.nr *= 2            # base regs * 2 - TODO: 32-bit run out of regs.
  result.cpu_simd = cpu

  # TODO: For AVX-512, we assume that CPU have 2 AVX512 units
  #       This is true on Skylake-X and Xeon-W and Xeon-SP Gold 6XXX
  #       bot not on Xeon-SP Bronze, Silver and Gold 5XXX

  #### Comparison with others
  # Example: for AVX2 Haswell CPU
  # This returns 6x16 ukernel for float and 4x8 for double.
  #
  # BLIS paper [3] recommends 4x12 and tested 8x4 for float,
  # the difference being between the broadcast and shuffle
  # operations. In implementation they use 6x16 ukernel.
  #
  # OpenBLAS uses 12x4 for float and 4x8 for double
  # MKLDNN also uses 16x6 for float and 8x6 for double (?)
  #   see unroll_factor: https://github.com/intel/mkl-dnn/blob/21fb5f2af1dd14e132af4f1b79160977ee487818/src/cpu/gemm/gemm_utils.hpp#L30-L48

#############################################
# Workaround "undeclared identifier mr or nr"
# for some reason the compiler cannot access fields in
# the static MicroKernel.

macro extract_mr*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.mr
macro extract_nr*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.nr
macro extract_vecsize*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.vecsize
macro extract_cpu_simd*(ukernel: static MicroKernel): untyped =
  let simd = ukernel.cpu_simd
  result = quote do: CPUFeatureX86(`simd`)
macro extract_c_unit_stride*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.c_unit_stride

# ############################################################
#
#                    Loop tiling
#
# ############################################################

type Tiles*[T] = ref object
  a*: ptr UncheckedArray[T]
  b*: ptr UncheckedArray[T]
  mc*, nc*, kc*: int
  # Nim doesn't support arbitrary increment with OpenMP
  # So we store indexing/edge case data in tiles for the parallelized loop
  jr_num_nr_tiles*: int
  a_alloc_mem: pointer # TODO Save on cache line, use an aligned allocator to not track this
  b_alloc_mem: pointer
  # The Tiles data structure takes 64-byte = 1 cache-line
  # TODO: packed and aligned

proc deallocTiles[T](tiles: Tiles[T]) =
  if not tiles.a_alloc_mem.isNil:
    deallocShared tiles.a_alloc_mem
  if not tiles.b_alloc_mem.isNil:
    deallocShared tiles.b_alloc_mem

proc newTiles*(
        ukernel: static MicroKernel,
        T: typedesc,
        M, N, K: Natural,
        ): Tiles[T] =
  # BLIS paper [2] section II Figure 2:
  #   - kc * nr in L1 cache µkernel
  #   - mc * kc in L2 cache Ã
  #   - kc * nc in L3 cache ~B (no L3 in Xeon Phi ¯\_(ツ)_/¯)
  new result, deallocTiles[T]
  const
    nr = ukernel.nr
    mr = ukernel.mr

  result.nc = N # We don't partition over N

  # ## Panel sizes
  # - TLB constraint
  #   TA ̃ + 2(TBj + TCj)≤T
  #   Note: optimizing the similar problem mckc/(2mc+2kc)
  #         under the constraint that mckc ≤ K is the problem
  #         of maximizing the area of a rectangle
  #         while minimizing the perimeter,
  #
  # Goto paper [1] section 6.3: choosing kc
  #   - kc should be as large as possible to amortize the mr*nr updates of Cj
  #   - Elements from Bj [kc, nr] must remain in L1 cache.
  #   - kc * nr should occupy less than half the L1 cache
  #     so that Ã and Caux do not evict element of Bj
  #   - Ã [kc, mc] should occupy
  #     a considerable fraction of the L2 cache
  #   In our experience optimal choice is so that "kc" float64 occupy half a page
  #     -> a page is 4096 bytes = 512 float64 -> half a page = 256

  # Goto paper [1] section 6.3: choosing mc
  #   - mc*kc should fill a considerable part of (1) the memory addressable
  #     by the TLB and (2) the L2 cache
  #     In practice mc is chosen so that A occupies about half the smaller of (1) and (2)

  # TODO: heuristics to compute the size
  result.mc = min(120, M)
  result.kc = min(360, K)

  # During packing the max size is unroll_stop*kc+kc*LR, LR = MR or NR
  let bufA_size = T.sizeof * result.kc*(result.mc+mr) # Size + Padding when packing
  let bufB_size = T.sizeof * result.kc*(result.nc+nr) # Size + Padding when packing

  # Note, if we parallelize on ic loop
  #   Each thread will access it's own part of A
  #   and so the buffer needs to be multiplied by the number of threads.
  result.a_alloc_mem = allocShared(bufA_size + 63)
  result.b_alloc_mem = allocShared(bufB_size + 63)
  result.a = assume_aligned align_raw_data(T, result.a_alloc_mem)
  result.b = assume_aligned align_raw_data(T, result.b_alloc_mem)

  # jr loop is parallel
  # Workaround for Nim OpenMP for loop not supporting non-unit increment
  result.jr_num_nr_tiles = (result.nc+nr-1) div nr
