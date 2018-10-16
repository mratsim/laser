# MIT License
# Copyright (c) 2018 Mamy André-Ratsimbazafy

import
  ./tensor,
  ./iter01_global,
  ./iter02_pertensor

import math, random, times, stats, strformat

proc warmup() =
  # Warmup - make sure cpu is on max perf
  let start = cpuTime()
  var foo = 123
  for i in 0 ..< 300_000_000:
    foo += i*i mod 456
    foo = foo mod 789

  # Compiler shouldn't optimize away the results as cpuTime rely on sideeffects
  let stop = cpuTime()
  echo &"Warmup: {stop - start:>4.4f} s, result {foo} (displayed to avoid compiler optimizing warmup away)"

template printStats() {.dirty.} =
  echo &"\nTensors of Float64 bench"
  echo &"Collected {stats.n} samples"
  echo &"Average broadcast time: {stats.mean * 1000 :>4.3f}ms"
  echo &"Stddev  broadcast time: {stats.standardDeviationS * 1000 :>4.3f}ms"
  echo &"Min     broadcast time: {stats.min * 1000 :>4.3f}ms"
  echo &"Max     broadcast time: {stats.max * 1000 :>4.3f}ms"
  echo "\nDisplay output[[0,0]] to make sure it's not optimized away"
  echo output[[0, 0]] # Prevents compiler from optimizing stuff away

proc mainBench1(a, b, c: Tensor, nb_samples: int) =
  ## Bench with standard lib
  var output = newTensor[float64](a.shape)

  block: # Actual bench
    var stats: RunningStat
    for _ in 0 ..< nb_samples:
      let start = cpuTime()
      materialize(output, a, b, c):
        a + b - sin c
      let stop = cpuTime()
      stats.push stop - start
    printStats()

proc mainBench2(a, b, c: Tensor, nb_samples: int) =
  ## Bench with standard lib
  var output = newTensor[float64](a.shape)

  block: # Actual bench
    var stats: RunningStat
    for _ in 0 ..< nb_samples:
      let start = cpuTime()
      forEach o in output, x in a, y in b, z in c:
        o = x + y - sin z
      let stop = cpuTime()
      stats.push stop - start
    printStats()

when isMainModule:
  warmup()
  block: # All contiguous
    let
      a = randomTensor([1000, 1000], 1.0)
      b = randomTensor([1000, 1000], 1.0)
      c = randomTensor([1000, 1000], 1.0)
    mainBench1(a, b, c, 1000)
    mainBench2(a, b, c, 1000)

    # Warmup: 1.2838 s, result 224 (displayed to avoid compiler optimizing warmup away)

    #######################################################################################

    # Tensors of Float64 bench
    # Collected 1000 samples
    # Average broadcast time: 22.840ms
    # Stddev  broadcast time: 1.955ms
    # Min     broadcast time: 21.410ms
    # Max     broadcast time: 40.491ms

    # Display output[[0,0]] to make sure it's not optimized away
    # 0.06512909995725152

    ######################################################################################

    # Tensors of Float64 bench
    # Collected 1000 samples
    # Average broadcast time: 9.403ms
    # Stddev  broadcast time: 0.538ms
    # Min     broadcast time: 8.538ms
    # Max     broadcast time: 15.061ms

    # Display output[[0,0]] to make sure it's not optimized away
    # 0.06512909995725152

  block: # Non C contiguous (but no Fortran contiguous fast-path)
    let
      a = randomTensor([100, 10000], 1.0)
      b = randomTensor([10000, 100], 1.0).transpose
      c = randomTensor([10000, 100], 1.0).transpose
    mainBench1(a, b, c, 1000)
    mainBench2(a, b, c, 1000)

    # Tensors of Float64 bench
    # Collected 1000 samples
    # Average broadcast time: 55.418ms
    # Stddev  broadcast time: 7.909ms
    # Min     broadcast time: 49.098ms
    # Max     broadcast time: 122.145ms

    # Display output[[0,0]] to make sure it's not optimized away
    # 0.3163590358464783

    # Tensors of Float64 bench
    # Collected 1000 samples
    # Average broadcast time: 40.173ms
    # Stddev  broadcast time: 3.903ms
    # Min     broadcast time: 36.701ms
    # Max     broadcast time: 76.235ms

    # Display output[[0,0]] to make sure it's not optimized away
    # 0.3163590358464783
