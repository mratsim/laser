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
    cpu_simd*: CPUFeatureX86
    nb_scalars*: int # Ideally MicroKernel should be generic over T
    nb_vecs_nr*: int
    c_unit_stride*: bool # We can use SIMD for the epilogue of C has a unit_stride
    pt*: int # Parallelization threshold

    # TODO: ARM support
    #   - https://github.com/nim-lang/Nim/issues/9679
    #   - https://github.com/nim-lang/Nim/issues/9678

  CPUFeatureX86* = enum
    x86_Generic,
    x86_SSE,
    x86_SSE2,
    x86_SSE4_1,
    x86_AVX,
    x86_AVX_FMA,
    x86_AVX2,
    x86_AVX512
    #   Note that Skylake SP, Xeon Bronze Silver and Gold 5XXX
    #   only have a single AVX512 port and AVX2 can be faster
    #   due to AVX512 downclocking

  X86_FeatureMap = array[CPUFeatureX86, int]

const X86_vecwidth_float: X86_FeatureMap = [
  x86_Generic:         1,
  x86_SSE:     128 div 8,
  x86_SSE2:    128 div 8,
  x86_SSE4_1:  128 div 8,
  x86_AVX:     256 div 8,
  x86_AVX_FMA: 256 div 8,
  x86_AVX2:    256 div 8,
  x86_AVX512:  512 div 8
]

const X86_vecwidth_int: X86_FeatureMap = [
  x86_Generic:         1,
  x86_SSE:             1,
  x86_SSE2:    128 div 8,
  x86_SSE4_1:  128 div 8,
  x86_AVX:     128 div 8,  # Not even addition with integer AVX
  x86_AVX_FMA: 128 div 8,
  x86_AVX2:    256 div 8,
  x86_AVX512:  512 div 8
]

# 4 registers are needed for loop indexes
# To issue 2xFMAs in parallel we need to use 2x general purpose registers
# We want to hold C of size MR * NR completely in SIMD registers as well
# On x86-64 with 16 general purpose registers (GPR) and 32 SIMD registers (SIMD):
#    - 2xMR + 4 <= 16 GPR
#    - MR * NR <= 32
when defined(amd64): # 64-bit
  # MR configuration - registers for the rows of Ã
  # 16 General purpose registers
  const X86_regs: X86_FeatureMap = [
    x86_Generic: 2,
    x86_SSE:     6,
    x86_SSE2:    6,
    x86_SSE4_1:  6,
    x86_AVX:     6,
    x86_AVX_FMA: 6,
    x86_AVX2:    6,
    x86_AVX512:  6
  ]

  # NR configuration - Nb of ~B SIMD vectors
  const NbVecs: X86_FeatureMap = [
      x86_Generic: 1,
      x86_SSE:     2, # 16 XMM registers - 16/6 = 2
      x86_SSE2:    2,
      x86_SSE4_1:  2,
      x86_AVX:     2, # 16 YMM registers - 16/6 = 2
      x86_AVX_FMA: 2,
      x86_AVX2:    2,
      x86_AVX512:  4  # 32 ZMM registers - 32/6 = 5 - bench: 4 is 8% faster
    ]
else: # 32-bit
  # MR configuration - registers for the rows of Ã
  # 8 General purpose registers
  const X86_regs: X86_FeatureMap = [
    x86_Generic: 2,
    x86_SSE:     2,
    x86_SSE2:    2,
    x86_SSE4_1:  2,
    x86_AVX:     2,
    x86_AVX_FMA: 2,
    x86_AVX2:    2,
    x86_AVX512:  2 
  ]

  # NR configuration - Nb of ~B SIMD vectors
  const NbVecs: X86_FeatureMap = [
      x86_Generic: 1,
      x86_SSE:     2, # 8 XMM registers
      x86_SSE2:    2,
      x86_SSE4_1:  2,
      x86_AVX:     2, # 8 YMM registers
      x86_AVX_FMA: 2,
      x86_AVX2:    2,
      x86_AVX512:  2  # 8 ZMM registers
    ]

func x86_ukernel*(cpu: CPUFeatureX86, T: typedesc, c_unit_stride: bool): MicroKernel =
  result.cpu_simd = cpu
  result.c_unit_stride = c_unit_stride
  result.pt = 128
  when T is SomeFloat:
    result.nb_scalars = max(1, X86_vecwidth_float[cpu] div T.sizeof)
  elif T is SomeInteger: # Integers
    result.nb_scalars = max(1, X86_vecwidth_int[cpu] div T.sizeof)
  else:
    {.error: "Unsupported type: " & T.type.name.}

  # The inner microkernel loop does:
  #   AB[m][n] = A[m] * B[n]
  # So n should be the vector size
  # if most matrices are row-Major.
  # This avoids dealing with transpose
  # in the inner loop and untranspose in the epilogue

  result.mr = X86_regs[cpu]                 # 2~6 registers for the rows of Ã
  result.nb_vecs_nr = NbVecs[cpu]           # x2 for 2 SIMD vectors of B
  result.nr = result.nb_vecs_nr * result.nb_scalars

#############################################
# Workaround "undeclared identifier mr or nr"
# for some reason the compiler cannot access fields in
# the static MicroKernel.

macro extract_mr*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.mr
macro extract_nr*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.nr
macro extract_cpu_simd*(ukernel: static MicroKernel): untyped =
  let simd = ukernel.cpu_simd
  result = quote do: CPUFeatureX86(`simd`)
macro extract_nb_scalars*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.nb_scalars
macro extract_nb_vecs_nr*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.nb_vecs_nr
macro extract_c_unit_stride*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.c_unit_stride
macro extract_pt*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.pt


# ############################################################
#
#                    Loop tiling
#
# ############################################################

# multithreading info in [2] and https://github.com/flame/blis/blob/master/docs/Multithreading.md

type Tiles*[T] = ref object
  a*: ptr UncheckedArray[T]
  b*: ptr UncheckedArray[T]
  mc*, nc*, kc*: int

  # Multithreading
    # Nim doesn't support arbitrary increment with OpenMP
    # So we store indexing/edge case data in tiles for the parallelized loop
  ic_num_mc_tiles*: int   # For private L1-L2 and shared L3
  jr_num_nr_tiles*: int   # For private L1 and shared L2 cache

  # Allocation data
    # TODO Save on cache line, use an aligned allocator to not track this
  a_alloc_mem: pointer
  b_alloc_mem: pointer
  # The Tiles data structure takes 64-byte = 1 cache-line


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
  result.mc = min( 768 div T.sizeof, M)
  result.kc = min(2048 div T.sizeof, K)

  # Parallel config
  # Ic loop parallel means that each thread will share a panel B and pack a different A
  result.ic_num_mc_tiles = (M+result.mc-1) div result.mc
  result.jr_num_nr_tiles = (result.nc+nr-1) div nr

  # Packing
  # During packing the max size is unroll_stop*kc+kc*LR, LR = MR or NR
  let bufA_size = T.sizeof * result.kc*(result.mc+mr) * result.ic_num_mc_tiles
  let bufB_size = T.sizeof * result.kc*(result.nc+nr)

  result.a_alloc_mem = allocShared(bufA_size + 63)
  result.b_alloc_mem = allocShared(bufB_size + 63)
  result.a = assume_aligned align_raw_data(T, result.a_alloc_mem)
  result.b = assume_aligned align_raw_data(T, result.b_alloc_mem)
