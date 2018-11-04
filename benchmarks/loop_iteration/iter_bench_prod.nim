# MIT License
# Copyright (c) 2018 Mamy André-Ratsimbazafy

# Iteration bench with the production implementation within Laser
import
  ../../laser/strided_iteration/foreach,
  ../../laser/tensor/[allocator, datatypes, initialization],
  ../../laser/[compiler_optim_hints, dynamic_stack_arrays]

withCompilerOptimHints()

proc newTensor*[T](shape: varargs[int]): Tensor[T] =
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)
  setZero(result, check_contiguous = false)

proc newTensor*[T](shape: Metadata): Tensor[T] =
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)
  setZero(result, check_contiguous = false)

proc randomTensor*[T](shape: openarray[int], max: T): Tensor[T] =
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)
  forEachContiguousSerial val in result:
    val = T(rand(max))

func transpose*(t: Tensor): Tensor =
  t.shape.reversed(result.shape)
  t.strides.reversed(result.strides)
  result.offset = t.offset
  result.storage = t.storage

func getIndex[T](t: Tensor[T], idx: varargs[int]): int =
  ## Convert [i, j, k, l ...] to the memory location referred by the index
  result = t.offset
  for i in 0 ..< t.shape.len:
    result += t.strides[i] * idx[i]

func `[]`*[T](t: Tensor[T], idx: varargs[int]): T {.inline.}=
  ## Index tensor
  t.storage.raw_data[t.getIndex(idx)]

################################################################

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
    let global_start = epochTime() # Due to multithreading we must use epoch time instead of CPU time
    for _ in 0 ..< nb_samples:     # or divide by the number of threads
      let start = epochTime()
      body
      let stop = epochTime()
      stats.push stop - start
    let global_stop = epochTime()
    printStats(name)

proc mainBench_libImpl(a, b, c: Tensor, nb_samples: int) =
  bench("Production implementation for tensor iteration"):
    forEach o in output, x in a, y in b, z in c:
      o = x + y - sin z

when isMainModule:
  randomize(42) # For reproducibility
  warmup()
  block: # All contiguous
    let
      a = randomTensor([1000, 1000], 1.0)
      b = randomTensor([1000, 1000], 1.0)
      c = randomTensor([1000, 1000], 1.0)
    mainBench_libImpl(a, b, c, 1000)

  block: # Non C contiguous (but no Fortran contiguous fast-path)
    let
      a = randomTensor([100, 10000], 1.0)
      b = randomTensor([10000, 100], 1.0).transpose
      c = randomTensor([10000, 100], 1.0).transpose
    mainBench_libImpl(a, b, c, 1000)
