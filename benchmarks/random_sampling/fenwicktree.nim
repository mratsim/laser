# ############################################################
#
#                      Some optimisations
#
# ############################################################

type
  PrefetchRW* {.size: cint.sizeof.} = enum
    Read = 0
    Write = 1
  PrefetchLocality* {.size: cint.sizeof.} = enum
    NoTemporalLocality = 0 # Data can be discarded from CPU cache after access
    LowTemporalLocality = 1
    ModerateTemporalLocality = 2
    HighTemporalLocality = 3 # Data should be left in all levels of cache possible
    # Translation
    # 0 - use no cache eviction level
    # 1 - L1 cache eviction level
    # 2 - L2 cache eviction level
    # 3 - L1 and L2 cache eviction level

proc builtin_prefetch(data: pointer, rw: PrefetchRW, locality: PrefetchLocality) {.importc: "__builtin_prefetch", noDecl.}

# ############################################################
#
#                      Implementation
#
# ############################################################

# Based on paper - A New Data Structure for Cumulative Frequency Tables
#       http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.8917&rep=rep1&type=pdf
# And - A Scalable Asynchronous Distributed Algorithm for Topic Modeling
#       https://www.cs.utexas.edu/~rofuyu/papers/nomad-lda-www.pdf

import std/bitops, random

type
  Sampler*[T] = distinct seq[T]
    ## Sampler implemented as a F+Tree
    # ⚠️ : We use 0-based indexing contrary to litterature
    #     as Fenwick-Tree use coding based on the number of bits
    #     the bitwise operations will be different
    #                  root at 0    root at 1
    # Left child        ix*2 + 1     ix*2
    # Right child       ix*2 + 2     ix*2 + 1
    # Parent            (ix-1)/2     ix/2

template len[T](s: Sampler[T]): int = (seq[T])(s).len
template `[]=`[T](s: var Sampler[T], idx: int, val: T) = (seq[T])(s)[idx] = val
template `[]`[T](s: Sampler[T], idx: int): T = (seq[T])(s)[idx]
template `$`[T](s: Sampler[T]): string = $(seq[T])(s)
  # No borrows for generics in Nim :/ so we do it manually

func `[]=`[T](s: var Sampler[T], slice: Slice[int], oa: openarray[T]) =
  assert slice.a < s.len
  assert slice.b < s.len
  assert slice.b - slice.a + 1 == oa.len

  for i in 0 ..< oa.len:
    s[slice.a + i] = oa[i]

func nextPowerOf2*(x: int): int =
  ## Returns x if x is a power of 2
  ## or the next biggest power of 2
  1 shl (fastLog2(x-1) + 1)

func newSampler[T](weights: openarray[T]): Sampler[T] =
  ## Initialise a sampler from weights.
  ##
  ## Weights can be a probability distribution (if they sum to 1)
  ## but alternatively the weight of each index (if they don't sum to 1)
  ##
  ## i.e. normalising inputs is not necessary.
  ##
  ## For example:
  ##
  ## [0.1, 0.4, 0.2, 0.3]
  ##   - will sample 0 with a probability of 10%,
  ##   -             1 with a probability of 40%
  ##   - ...
  ##
  ## Alternatively
  ##
  ## [3, 3, 1, 5]
  ##   - will sample 0 with a probability of 3/(3 + 3 + 1 + 5) = 3/12 = 25%
  ##   - ...

  # A complete tree (all leaves used) with n leaves
  # has n-1 internal nodes, for a total of 2N-1 nodes
  # so we allocate enough to hold a complete tree
  let n = weights.len.nextPowerOf2()
  let leaves_offset = n-1
  let size = leaves_offset + n
  result = (Sampler[T])(newSeq[T](size))

  # The internal nodes contain the cumulative probability distribution
  # The leaves contain the probability distribution from the input weights
  result[leaves_offset ..< leaves_offset + weights.len] = weights

  # Now we build the internal nodes as the cumulative probability of their children
  # For a node at position i, its children are at position 2*i+1 and 2*i+2
  #
  #   Perf note
  #     Sometimes people are worried that it's slower than 1-based indexing
  #     where we can just use 2*i and 2*i+1 to find children
  #     and i/2 to find parent instead of (i-1)/2.
  #       1. It doesn't make a difference because CPUs support Scale*Index+Offset
  #          addressing mode if scale is a power of 2.
  #          (Search for SIB - Scaled Index Byte)
  #          The only difference is only 1 more byte in the instruction.
  #          Execution time is the same, it only affects the instruction cache.
  #       2. The performance is dominated by fetching data from memory. During
  #          The previous memory fetch, the CPU can precompute the next location to fetch.
  #          as there is no dependencies.
  #    Furthermore if you are sampling N items where N is a power of 2
  #    you waste N-1 space with 1-based indexing.
  for i in countdown(leaves_offset-1, 0):
    result[i] = result[2*i+1] + result[2*i+2]

template sampleImpl[T](result: var int, s: Sampler[T], u: T) =

  {.pragma: restrict, codegenDecl: "$# __restrict $#".}
  let sp{.restrict.} = cast[ptr UncheckedArray[T]](s[0].unsafeAddr)

  # leaves_offset are at n-1, but we don't have n anymore
  # However size = 2n-1 = 2(n-1+1) - 1 = 2(n-1)+2 - 1 = 2(n-1) + 1
  # So a logical right shift will divide by 2 and truncate to n-1.
  let leaves_offset = s.len shr 1
  var i = 0

  # A F+tree guarantees the following:
  #   if u >= left CDF (lCDF) => result ∈ right branch
  #   and left branch otherwise
  while i < leaves_offset:
    let left = 2*i+1
    builtin_prefetch sp[2*left+1].unsafeAddr, Read, NoTemporalLocality
    let pLeft = sp[left]
    if u >= pleft:
      # We choose the right child and substract the left CDF
      # to maintain u ∈ [0, rCDF]
      u -= pLeft
      i = left + 1
    else:
      i = left
  result = i - leaves_offset


proc sample(s: Sampler): int =
  var u = rand(s[0])
  sampleImpl(result, s, u)

proc sample(s: Sampler, rng: Rand): int =
  var u = rng.rand(s[0])
  sampleImpl(result, s, u)

when isMainModule:
  block:
    let p = [0.3, 1.5, 0.4, 0.3]

    let sampler = newSampler(p)
    echo sampler
    echo sampler.sample
  block:
    let p = [0.3, 1.5, 0.4, 0.3, 0.3]

    let sampler = newSampler(p)
    echo sampler
    echo sampler.sample
  block:
    let p = [0.3, 1.5, 0.4, 0.3, 0.3, 1.5, 0.4, 0.3]

    let sampler = newSampler(p)
    echo sampler
    echo sampler.sample

  import tables
  block:
    let p = [0.1, 0.4, 0.3, 0.2]
    let sampler = newSampler(p)

    var c = initCountTable[int]()

    for _ in 0 ..< 1_000_000:
      c.inc(sampler.sample)

    echo c
