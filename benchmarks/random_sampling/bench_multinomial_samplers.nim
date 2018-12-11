# ##########################################
# Benchmarking tools
import random, times, stats, strformat, math, sequtils

proc warmup() =
  # Warmup - make sure cpu is on max perf
  let start = epochTime()
  var foo = 123
  for i in 0 ..< 300_000_000:
    foo += i*i mod 456
    foo = foo mod 789

  # Compiler shouldn't optimize away the results as cpuTime rely on sideeffects
  let stop = epochTime()
  echo &"Warmup: {stop - start:>4.4f} s, result {foo} (displayed to avoid compiler optimizing warmup away)"

template printStats(name: string, output: typed) {.dirty.} =
  echo "\n" & name
  echo &"Collected {stats.n} test samples in {global_stop - global_start:>4.3f} seconds"
  echo &"Average time: {stats.mean * 1000 :>4.3f} ms"
  echo &"Stddev  time: {stats.standardDeviationS * 1000 :>4.3f} ms"
  echo &"Min     time: {stats.min * 1000 :>4.3f} ms"
  echo &"Max     time: {stats.max * 1000 :>4.3f} ms"
  echo &"Perf:         {req_ops.float / stats.mean / float(10^3):>4.3f} millions samplings/s (VocabSize = {$VocabSize})"
  echo "\nDisplay output[0] to make sure it's not optimized away"
  echo output[0] # Prevents compiler from optimizing stuff away

template bench(name: string, initialisation, body: untyped) {.dirty.}=
  block: # Actual bench
    var stats: RunningStat
    let global_start = epochTime()
    for _ in 0 ..< nb_bench_runs:
      initialisation
      let start = epochTime()
      body
      let stop = epochTime()
      stats.push stop - start
    let global_stop = epochTime()
    printStats(name, output)

# ############################################################
#
#               Baseline multinomial sampling
#
# ############################################################

import algorithm, math, sequtils

func cumsum[T](x: openarray[T]): seq[T] =
  result = newSeq[T](x.len)
  result[0] = x[0]
  for i in 1 ..< x.len:
    result[i] = x[i] + result[i-1]

func searchsorted[T](a, v: openarray[T], leftSide: static bool = true): seq[T] =
  result = newSeqUninitialized[T](v.len)
  for i in 0 ..< v.len:
    when leftSide:
      result[i] = a.lowerBound(v[i])
    else:
      result[i] = a.upperBound(v[i])

proc sanityChecks() =
  block: # Numpy
    let p = [1, 2, 3, 4, 5]

    doAssert searchsorted(p, [3]) == @[2]
    doAssert searchsorted(p, [3], leftSide = false) == @[3]
    doassert searchsorted(p, [-10, 10, 2, 3]) == @[0, 5, 1, 2]

  block: # Tensorflow
    let row1 = [0, 3, 9, 9, 10]
    let row2 = [1, 2, 3, 4, 5]

    let val1 = [2, 4, 9]
    let val2 = [0, 2, 6]

    doAssert searchsorted(row1, val1) == @[1, 2, 2]
    doAssert searchsorted(row2, val2) == @[0, 1, 5]

    doAssert searchsorted(row1, val1, leftSide = false) == @[1, 2, 4]
    doAssert searchsorted(row2, val2, leftSide = false) == @[0, 2, 5]

# ############################################################
#
#                       Params
#
# ############################################################

const
  BatchSize = 128
  VocabSize = 50000
  NbSamples = 1 # we sample 1 word at a time. (we could sample 3 topics or or other ...)
  NbSamplesWithoutReplacement = 10

let req_ops = BatchSize
# let req_bytes = sizeof(float32) * BatchSize * VocabSize

# ############################################################
#
#                     Tensor sampling
#
# ############################################################

# We simulate generating a word from a vocabulary of 50000 potential words
# Our "Tensor" type will be a seq[array[50000, float32]] for simplicity each elements containing a probability.

# The softmax implementation (``exp(xi) / ∑i exp(xi)``) is a one-ass streaming softmax
# that doesn't need to pass a second time over the data to divide by the sum of exponentials
# nor a third time to be numerically stable (to find the max and substract it from all exponentials)

# Note that exponential is very slow and can be improved by a facter 10, cf vector_math benchs

import random, times

type Tensor[N: static int; T] = seq[array[N, T]]

func streaming_max_sumexp[N, T](t: array[N, T]): tuple[max, sumexp: T] =
  result.max = T(-Inf)
  result.sumexp = 0.t

  for x in t:
    if x <= result.max:
      result.sumexp += exp(x - result.max)
    else:
      result.sumexp = result.sumexp * exp(result.max - x) + 1
      result.max = x

func stable_softmax[T](x, max, sumexp: T): T {.inline.}=
  exp(x - max) / sumexp

proc softmax[N, T](t: Tensor[N, T]): Tensor[N, T] {.noInit.} =
  ## Exponential normalisation of each row
  ## Shape of input and output [batch_size, nb_classes/categories/words]

  result = newSeq[array[N, T]](t.len)

  for i in 0 || t.high:
    let (max, sumexp) = t[i].streaming_max_sumexp()
    for j in 0 ..< N:
      result[i][j] = stable_softmax(t[i][j], max, sumexp)

proc cumsum[N, T](t: Tensor[N, T]): Tensor[N, T] =
  result = newSeq[array[N, T]](t.len)
  for i in 0 || t.high:
    result[i][0] = t[i][0]
    for j in 1 ..< N:
      result[i][j] = t[i][j] + result[i][j-1]

## Only useful with replacement
# proc searchsorted[M, N, T](input: Tensor[N, T], values: Tensor[M, T], leftSide: static bool = true): Tensor[M, int] =
#   doAssert input.len == values.len
#   result = newSeq[array[M, int]](values.len)

#   for i in 0 || values.high:
#     for j in 0 ..< M:
#       when leftSide:
#         result[i][j] = input[i].lowerBound(values[i][j])
#       else:
#         result[i][j] = input[i].upperBound(values[i][j])

proc searchsorted[N, T](input: Tensor[N, T], values: seq[T], leftSide: static bool = true): seq[int] =
  doAssert input.len == values.len
  result = newSeq[int](values.len)

  for i in 0 || values.high:
    when leftSide:
      result[i] = input[i].lowerBound(values[i])
    else:
      result[i] = input[i].upperBound(values[i])

proc sample[N, T](probs: Tensor[N, T], nb_samples: static int): auto =
  ## Samples without replacement

  var rng {.global.} = initRand(0xDEADBEEF)

  when nb_samples == 1:
    # Can we use a single uniform sample
    # or do we need an array of probability?
    let uniform_samples = newSeqWith(probs.len, T(rng.rand(1.0)))

    # let start = epochTime()
    block:
      let cdf = probs.cumsum()
      result = cdf.searchsorted(uniform_samples, leftSide = false)
    # let stop = epochTime()

    # echo &"Sampling time: {stop - start:>4.4f} s"
  else:
    result = newSeq[array[nb_samples, int]](probs.len)

    ## Each time we sample we must zero the corresponding probability and recompute the CDF
    var samples_count = 0
    var p = probs

    while samples_count < min(N, nb_samples):
      # We can't sample more than once from our population
      # so we 0 the probability after sampling
      # Problem: assume we have [0.2, 0.3, 0.4, 0.1]
      # after one round         [0.2, 0.0, 0.4, 0.1] <--- everything must be rescaled
      var cdf = p.cumsum()
      block: # rescale
        for row in 0 || cdf.high:
          for col in cdf[row].mitems:
            col /= cdf[row][^1] # THe last column contains the sum of all probabilities
      let uniform_samples = newSeqWith(cdf.len, T(rng.rand(1.0)))
      let new_sample = cdf.searchsorted(uniform_samples, leftSide = false)
      block:
        for i in 0 || result.high:
          # store result
          result[i][samples_count] = new_sample[i]
          # zero the probability
          p[i][new_sample[i]] = 0.T
      inc samples_count

# ############################################################
#
#                       Bench
#
# ############################################################

proc benchCDF_sampling[N, T](a: Tensor[N, T], nb_bench_runs: int) =
  var output: seq[int]
  bench("Classic sampling via inverse CDF"):
    discard
  do:
    # Main work
    output = sample(a, NbSamples)

proc benchCDF_multisampling[N, T](a: Tensor[N, T], nb_bench_runs: int) =
  var output: Tensor[NbSamplesWithoutReplacement, int]
  bench(&"{NbSamplesWithoutReplacement} samplings without replacement"):
    discard
  do:
    # Main work
    output = sample(a, NbSamplesWithoutReplacement)

import ./fenwicktree

proc benchFtree_prefetch[N, T](a: Tensor[N, T], nb_bench_runs: int) =
  var output = newSeq[int](a.len)
  bench("Sampling via F+tree with prefetch"):
    discard
  do:
    # Main work
    for i in 0 ..< a.len:
      let sampler = newSampler[T](a[i])
      output[i] = sampler.sample(prefetch = true)

proc benchFtree_multisampling_prefetch[N, T](a: Tensor[N, T], nb_bench_runs: int) =
  var output = newSeq[array[NbSamplesWithoutReplacement, int]](a.len)
  bench(&"{NbSamplesWithoutReplacement} samplings without replacement"):
    discard
  do:
    # Main work
    for i in 0 ..< a.len:
      var sampler = newSampler(a[i])
      for j in 0 ..< NbSamplesWithoutReplacement:
        output[i][j] = sampler.sampleAndRemove(prefetch = true)


proc benchFtree[N, T](a: Tensor[N, T], nb_bench_runs: int) =
  var output = newSeq[int](a.len)
  bench("Sampling via F+tree without prefetch"):
    discard
  do:
    # Main work
    for i in 0 ..< a.len:
      let sampler = newSampler(a[i])
      output[i] = sampler.sample(prefetch = false)

proc benchFtree_multisampling[N, T](a: Tensor[N, T], nb_bench_runs: int) =
  var output = newSeq[array[NbSamplesWithoutReplacement, int]](a.len)
  bench(&"{NbSamplesWithoutReplacement} samplings without replacement without prefetch"):
    discard
  do:
    # Main work
    for i in 0 ..< a.len:
      var sampler = newSampler(a[i])
      for j in 0 ..< NbSamplesWithoutReplacement:
        output[i][j] = sampler.sampleAndRemove(prefetch = false)

# ############################################################
#
#                       Run everything
#
# ############################################################

proc randomTensor[T: SomeFloat](batchSize: int, N: static int, typ: type T): Tensor[N, T] =

  result = newSeq[array[N, T]](batchSize)
  for i in 0 ..< batchSize:
    for j in 0 ..< N:
      result[i][j] = T(rand(1.0))

when isMainModule:
  randomize(42) # For reproducibility
  warmup()

  block:
    let a = randomTensor(BatchSize, VocabSize, float32)
    benchCDF_sampling(a, 1000)
    benchCDF_multisampling(a, 1000)
    benchFtree_prefetch(a, 1000)
    benchFtree_multisampling_prefetch(a, 1000)
    benchFtree(a, 1000)
    benchFtree_multisampling(a, 1000)

# Warmup: 1.1930 s, result 224 (displayed to avoid compiler optimizing warmup away)

# Classic sampling via inverse CDF
# Collected 1000 test samples in 9.044 seconds
# Average time: 9.044 ms
# Stddev  time: 1.162 ms
# Min     time: 8.198 ms
# Max     time: 35.947 ms
# Perf:         14.154 millions samplings/s (VocabSize = 50000)

# Display output[0] to make sure it's not optimized away
# 2

# 10 samplings without replacement
# Collected 1000 test samples in 156.663 seconds
# Average time: 156.663 ms
# Stddev  time: 26.336 ms
# Min     time: 147.814 ms
# Max     time: 672.546 ms
# Perf:         0.817 millions samplings/s (VocabSize = 50000)

# Display output[0] to make sure it's not optimized away
# [12913, 8020, 41233, 19747, 46496, 18194, 40084, 34881, 42615, 9536]

# Sampling via F+tree with prefetch
# Collected 1000 test samples in 13.507 seconds
# Average time: 13.507 ms
# Stddev  time: 0.616 ms
# Min     time: 12.810 ms
# Max     time: 21.890 ms
# Perf:         9.476 millions samplings/s (VocabSize = 50000)

# Display output[0] to make sure it's not optimized away
# 29338

# 10 samplings without replacement
# Collected 1000 test samples in 13.658 seconds
# Average time: 13.658 ms
# Stddev  time: 0.576 ms
# Min     time: 12.945 ms
# Max     time: 22.702 ms
# Perf:         9.372 millions samplings/s (VocabSize = 50000)

# Display output[0] to make sure it's not optimized away
# [30153, 46437, 38451, 11943, 20761, 23742, 2702, 27470, 1530, 13165]

# Sampling via F+tree without prefetch
# Collected 1000 test samples in 13.549 seconds
# Average time: 13.548 ms
# Stddev  time: 0.742 ms
# Min     time: 12.792 ms
# Max     time: 25.680 ms
# Perf:         9.448 millions samplings/s (VocabSize = 50000)

# Display output[0] to make sure it's not optimized away
# 39064

# 10 samplings without replacement without prefetch
# Collected 1000 test samples in 13.680 seconds
# Average time: 13.680 ms
# Stddev  time: 0.583 ms
# Min     time: 13.003 ms
# Max     time: 22.149 ms
# Perf:         9.357 millions samplings/s (VocabSize = 50000)

# Display output[0] to make sure it's not optimized away
# [47509, 3677, 45871, 43620, 47576, 46527, 9768, 19118, 25929, 40719]
