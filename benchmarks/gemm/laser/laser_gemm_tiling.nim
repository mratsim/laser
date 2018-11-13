# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

####################################
#
# Runtime tile dimensioning for GEMM
#
####################################

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
#   [5] On Composing Matrix Multiplication from Kernels
#       Bryan marker
#     - http://www.cs.utexas.edu/users/flame/pubs/bamarker_thesis.pdf

# We assume that the CPU has at least 2 levels of cache

import
  ../../../laser/[cpuinfo, compiler_optim_hints],
  typetraits,
  ./laser_gemm_utils

# ##########################################################################################

# #####################
#
# Microkernel (µkernel)
#
# #####################

# We have to take into account vectorisation
# so that the microkernel can be processed with vectorized intrinsics.
#
# Caux [mr, nr] must stay in register.
#   - mr ~= nr is optimal to amortize register load cost
#   - some registers must be left to prefetch bufA and bufB
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

    # TODO: ARM support with a when arm ...
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
# 4 registers are needed to hold bufA and bufB
const X86_regs: X86_FeatureMap = [
  x86_Generic: 1,
  x86_SSE:     4, # 8 XMM regs in 32-bit, 16 in 64-bit (we assume 32-bit mode)
  x86_SSE2:    6,
  x86_AVX:     6, # 16 YMM registers
  x86_AVX2:    6,
  x86_AVX512:  6  # 32 ZMM registers
]

const PageSize* = 512

func x86_ukernel*(cpu: CPUFeatureX86, T: typedesc): MicroKernel =
  # Cannot reproduce a small case :/
  var ukernel: MicroKernel # triggers "Object Constructor needs an object type)"

  when T is SomeFloat:
    ukernel.vecsize =  X86_vecsize_float[cpu]
  else:
    ukernel.vecsize =  X86_vecsize_int[cpu]
  # TODO: Complex support

  # The inner microkernel loop does:
  #   AB[m][n] = A[m] * B[n]
  # So n should be the vector size
  # if most compute are row-Major.
  # This avoids dealing with transpose
  # in the inner loop and untranspose in the epilogue

  ukernel.mr = X86_regs[cpu]
  ukernel.nr = max(1, ukernel.vecsize div sizeof(T))

  when sizeof(int) == 8: # 64-bit - use 8/12 out of the 16 XMM/YMM registers
    ukernel.nr *= 2
  ukernel.cpu_simd = cpu

  return ukernel

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
  # OpenBLAS uses 16x4 for float and 4x8 for double
  # MKLDNN also uses 16x6 for float and 8x6 for double (?)
  #   see unroll_factor: https://github.com/intel/mkl-dnn/blob/21fb5f2af1dd14e132af4f1b79160977ee487818/src/cpu/gemm/gemm_utils.hpp#L30-L48

#############################################
# Workaround "undeclared identifier mr or nr"
# for some reason the compiler cannot access fields in
# the static MicroKernel.
import macros
macro extract_mr*(ukernel: static MicroKernel): untyped =
  let mr = ukernel.mr
  result = newLit mr
macro extract_nr*(ukernel: static MicroKernel): untyped =
  let nr = ukernel.nr
  result = newLit nr

# ##########################################################################################

type Tile*[T] = object
  a*: ptr UncheckedArray[T]
  b*: ptr UncheckedArray[T]
  mc*, nc*, kc*: int
  # Nim doesn't support arbitrary increment with OpenMP
  # So we store indexing/edge case data in tiles
  jr_num_nr_tiles*: int
  allocated_mem: pointer # TODO Save on cache line, use an aligned allocator to not track this

proc `=destroy`[T](tile: var Tile[T]) =
  if not tile.allocated_mem.isNil:
    deallocShared tile.allocated_mem

func align_raw_data(T: typedesc, p: pointer): ptr UncheckedArray[T] =
  static: assert T.supportsCopyMem

  withCompilerOptimHints()
  let address = cast[ByteAddress](p)
  let aligned_ptr{.restrict.} = block: # We cannot directly apply restrict to the default "result"
    let remainder = address and (LASER_MEM_ALIGN - 1) # mod power of 2
    if remainder == 0:
      assume_aligned cast[ptr UncheckedArray[T]](p)
    else:
      let offset = LASER_MEM_ALIGN - remainder
      assume_aligned cast[ptr UncheckedArray[T]](address +% offset)
  return aligned_ptr

func partition(dim, max_chunk_size, tile_dim: Natural): Natural =
  # Partition a dimension into mc, nc or kc chunks.
  # Chunks are chosen according to max_chunk_size which depends on the L1/L2 cache
  # and according to the microkernel tile dimension
  #
  # Return a chunk size

  if max_chunk_size >= dim:
    return dim

  let nb_chunks = dim div max_chunk_size + int(dim mod max_chunk_size != 0)
  result = dim div nb_chunks + int(dim mod nb_chunks != 0)

  let remainder = result mod tile_dim
  if remainder != 0:
    let candidate = result + tile_dim - remainder
    if candidate <= max_chunk_size:
      return candidate

proc newTiles*(
        ukernel: static MicroKernel,
        T: typedesc,
        M, N, K: Natural,
        ): Tile[T] =
  # Goto paper [1] section 6.3: choosing kc
  #   - kc should be as large as possible to amortize the mr*nr updates of Cj
  #   - Elements from Bj [kc, nr] must remain in L1 cache.
  #   - kc * nr should occupy less than half the L1 cache
  #     so that bufA and Caux do not evict element of Bj
  #   - bufA [kc, mc] should occupy
  #     a considerable fraction of the L2 cache
  #   In our experience optimal choice is so that "kc" float64 occupy half a page
  #     -> a page is 4096 bytes = 512 float64 -> half a page = 256

  # Goto paper [1] section 6.3: choosing mc
  #   - mc*kc should fill a considerable part of (1) the memory addressable
  #     by the TLB and (2) the L2 cache
  #     In practice mc is chosen so that A occupies about half the smaller of (1) and (2)

  # BLIS paper [2] section II Figure 2:
  #   - kc * nr in L1 cache µkernel
  #   - mc * kc in L2 cache Ã
  #   - kc * nc in L3 cache ~B (no L3 in Xeon Phi ¯\_(ツ)_/¯)
  const
    nr = ukernel.nr
    mr = ukernel.mr
  let
    l1 = cpuinfo_get_l1d_caches().size.int              # L1 cache
    l2h = cpuinfo_get_l2_caches().size.int * 3 div 5    # More than half L2 cache
    line_size = cpuinfo_get_l1d_caches().line_size.int

  const Kc_Threshold = 240
  # We use Mir-GLAS approach here

  # First determine a candidate kc depending on l2 cache constraints
  result.kc = (l2h - M * nr * T.sizeof) div ((M+nr)*T.sizeof)
  result.mc = M
  result.nc = N # We don't partition over N

  # TLB constraint
  # TA ̃ + 2(TBj + TCj)≤T
  # Note: optimizing the similar problem mckc/(2mc+2kc)
  #       under the constraint that mckc ≤ K is the problem
  #       of maximizing the area of a rectangle
  #       while minimizing the perimeter,
  let TLB = PageSize + 2*(nr*line_size + nr*mr * T.sizeof)

  let halfPerimeter = T.sizeof * (mr + nr)

  # Second, check if we have a small gemm
  #         or if we don't fit kc*nr in TLB or L1 cache
  if result.kc < Kc_Threshold or
      l1 < TLB + result.kc * halfPerimeter:

    result.kc = (l1 - TLB) div halfPerimeter
    result.kc = partition(K, result.kc, mr)

    # Now compute mc*kc so that it takes half the L2 cache
    # Loading it in L1 must not evict kc*nr in L1
    result.mc = (l2h - result.kc * nr * T.sizeof)
    result.mc = result.mc div (T.sizeof * (result.kc + nr))
    result.mc = partition(M, result.mc, nr)
  else:
    result.kc = partition(K, result.kc, mr)

  # Example:
  #   - AVX2 Broadwell
  #   - 32768 l1 and 262144 l2 cache
  #   - float32
  #   - mr*nr microkernel = 12*6
  #
  #   -> kc = 304, mc = 120

  let bufA_size = T.sizeof * result.kc * result.mc
  let bufB_size = T.sizeof * result.kc * nr

  # Note, if we parallelize on ic loop
  # Each thread will access it's own part of A
  # and so the buffer needs to be multiplied by the number of threads.
  result.allocated_mem = allocShared0(bufA_size + bufB_size + PageSize)
    # PageSize has enough space for 64 bytes alignement
    # This help for prefetching next page in the TLB
  result.a = assume_aligned align_raw_data(T, result.allocated_mem)
  result.b = result.a + bufA_size # TODO: align as well?

  # Workaround for Nim OpenMP for loop not supporting non-unit increment
  # we don't partition N so N = NC
  result.jr_num_nr_tiles = (N+nr-1) div nr
