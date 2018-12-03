
{.emit:"typedef float Float32x8 __attribute__ ((vector_size (32)));".}

type Float32x8 {.importc, bycopy.} = object
  raw: array[8, float32]

import math, times, random

func `+`(a, b: Float32x8): Float32x8 =
  {.emit: "`result` = `a` + `b`;".}

func double(v: Float32x8): Float32x8 =
  result = v + v

proc main() =
  var b = [0'f32, 0, 0, 0, 0, 0, 0, 0]
  var start = cpuTime()
  for i in 0..<1000000:
    let a = cast[Float32x8]([float32 rand(1.0), 2, rand(1.0), 4, rand(1.0), 6, rand(1.0), 8])
    b = cast[array[8, float32]](double(double(a)))
  var stop = cpuTime()
  echo b
  echo stop - start

when isMainModule:
  main()



