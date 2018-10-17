# MIT License
# Copyright (c) 2018 Mamy André-Ratsimbazafy

import
  ./tensor,
  ./iter01_global,
  ./iter02_pertensor,
  ./iter03_global_triot,
  ./iter05_fusedpertensor,
  ./metadata

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

template printStats(name: string) {.dirty.} =
  echo "\n" & name & " - float64"
  echo &"Collected {stats.n} samples in {global_stop - global_start:>4.3f} seconds"
  echo &"Average time: {stats.mean * 1000 :>4.3f}ms"
  echo &"Stddev  time: {stats.standardDeviationS * 1000 :>4.3f}ms"
  echo &"Min     time: {stats.min * 1000 :>4.3f}ms"
  echo &"Max     time: {stats.max * 1000 :>4.3f}ms"
  echo "\nDisplay output[[0,0]] to make sure it's not optimized away"
  echo output[[0, 0]] # Prevents compiler from optimizing stuff away

template bench(name: string, body: untyped) {.dirty.}=
  var output = newTensor[float64](a.shape)

  block: # Actual bench
    var stats: RunningStat
    let global_start = cpuTime()
    for _ in 0 ..< nb_samples:
      let start = cpuTime()
      body
      let stop = cpuTime()
      stats.push stop - start
    let global_stop = cpuTime()
    printStats(name)

proc mainBench_global(a, b, c: Tensor, nb_samples: int) =
  bench("Global reference iteration"):
    materialize(output, a, b, c):
      a + b - sin c

proc mainBench_perTensor(a, b, c: Tensor, nb_samples: int) =
  bench("Per tensor reference iteration"):
    forEach o in output, x in a, y in b, z in c:
      o = x + y - sin z

proc mainBench_global_triot(a, b, c: Tensor, nb_samples: int) =
  bench("Global TRIOT iteration"):
    triotForEach o in output, x in a, y in b, z in c:
      o = x + y - sin z

proc mainBench_fusedperTensor(a, b, c: Tensor, nb_samples: int) =
  bench("Fused per tensor reference iteration"):
    fusedForEach o in output, x in a, y in b, z in c:
      o = x + y - sin z

when isMainModule:
  randomize(42) # For reproducibility
  warmup()
  block: # All contiguous
    let
      a = randomTensor([1000, 1000], 1.0)
      b = randomTensor([1000, 1000], 1.0)
      c = randomTensor([1000, 1000], 1.0)
    mainBench_global(a, b, c, 1000)
    mainBench_perTensor(a, b, c, 1000)
    mainBench_global_triot(a, b, c, 1000)
    mainBench_fusedperTensor(a, b, c, 1000)

  block: # Non C contiguous (but no Fortran contiguous fast-path)
    let
      a = randomTensor([100, 10000], 1.0)
      b = randomTensor([10000, 100], 1.0).transpose
      c = randomTensor([10000, 100], 1.0).transpose
    mainBench_global(a, b, c, 1000)
    mainBench_perTensor(a, b, c, 1000)
    mainBench_global_triot(a, b, c, 1000)
    mainBench_fusedperTensor(a, b, c, 1000)


# Warmup: 1.1933 s, result 224 (displayed to avoid compiler optimizing warmup away)

############################################

# Global reference iteration - float64
# Collected 1000 samples in 21.296 seconds
# Average time: 21.292ms
# Stddev  time: 0.426ms
# Min     time: 21.139ms
# Max     time: 28.544ms

# Display output[[0,0]] to make sure it's not optimized away
# -0.41973403633413

# Per tensor reference iteration - float64
# Collected 1000 samples in 8.646 seconds
# Average time: 8.642ms
# Stddev  time: 0.195ms
# Min     time: 8.543ms
# Max     time: 11.056ms

# Display output[[0,0]] to make sure it's not optimized away
# -0.41973403633413

# Global TRIOT iteration - float64
# Collected 1000 samples in 19.514 seconds
# Average time: 19.510ms
# Stddev  time: 0.407ms
# Min     time: 19.349ms
# Max     time: 25.644ms

# Display output[[0,0]] to make sure it's not optimized away
# -0.41973403633413

# Fused per tensor reference iteration - float64
# Collected 1000 samples in 8.645 seconds
# Average time: 8.641ms
# Stddev  time: 0.253ms
# Min     time: 8.531ms
# Max     time: 13.235ms

# Display output[[0,0]] to make sure it's not optimized away
# -0.41973403633413

############################################

# Global reference iteration - float64
# Collected 1000 samples in 49.648 seconds
# Average time: 49.644ms
# Stddev  time: 2.316ms
# Min     time: 47.169ms
# Max     time: 78.987ms

# Display output[[0,0]] to make sure it's not optimized away
# 1.143903810108473

# Per tensor reference iteration - float64
# Collected 1000 samples in 36.795 seconds
# Average time: 36.790ms
# Stddev  time: 1.175ms
# Min     time: 34.855ms
# Max     time: 49.315ms

# Display output[[0,0]] to make sure it's not optimized away
# 1.143903810108473

# Global TRIOT iteration - float64
# Collected 1000 samples in 47.085 seconds
# Average time: 47.080ms
# Stddev  time: 1.337ms
# Min     time: 45.313ms
# Max     time: 68.331ms

# Display output[[0,0]] to make sure it's not optimized away
# 1.143903810108473

# Fused per tensor reference iteration - float64
# Collected 1000 samples in 30.588 seconds
# Average time: 30.583ms
# Stddev  time: 1.169ms
# Min     time: 28.384ms
# Max     time: 41.547ms

# Display output[[0,0]] to make sure it's not optimized away
# 1.143903810108473
