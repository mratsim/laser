# Apache v2
# Copyright (c) 2017-2018 Mamy André-Ratsimbazafy and the Arraymancer contributors

# Current iteration scheme in Arraymancer. Each tensor manages it's own loop
import ./tensor, ./mem_optim_hints, ./metadata

type
  IterKind = enum
    Values, Iter_Values, Offset_Values

template initStridedIteration(coord, backstrides, iter_pos: untyped, t: Tensor): untyped =
  ## Iterator init
  var iter_pos = 0
  withMemoryOptimHints()
  var coord {.align64, noInit.}: array[MAXRANK, int]
  var backstrides {.align64, noInit.}: array[MAXRANK, int]
  for i in 0..<t.rank:
    backstrides[i] = t.strides[i]*(t.shape[i]-1)
    coord[i] = 0

template advanceStridedIteration(coord, backstrides, iter_pos: untyped, t: Tensor): untyped =
  ## Computing the next position
  for k in countdown(t.rank - 1,0):
    if coord[k] < t.shape[k]-1:
      coord[k] += 1
      iter_pos += t.strides[k]
      break
    else:
      coord[k] = 0
      iter_pos -= backstrides[k]

template stridedIterationYield*(strider: IterKind, data, i, iter_pos: typed) =
  ## Iterator the return value
  when strider == IterKind.Values: yield data[iter_pos]
  elif strider == IterKind.Iter_Values: yield (i, data[iter_pos])
  elif strider == IterKind.Offset_Values: yield (iter_pos, data[iter_pos])

template stridedIteration*(strider: IterKind, t: Tensor): untyped =
  ## Iterate over a Tensor, displaying data as in C order, whatever the strides.

  # Get tensor data address with offset builtin
  withMemoryOptimHints()
  # TODO, allocate so that we can assume_align
  let data{.restrict.} = t.dataPtr # Warning ⚠: data pointed may be mutated

  # Optimize for loops in contiguous cases
  if t.is_C_Contiguous:
    for i in 0 ..< t.size:
      stridedIterationYield(strider, data, i, i)
  else:
    initStridedIteration(coord, backstrides, iter_pos, t)
    for i in 0 ..< t.size:
      stridedIterationYield(strider, data, i, iter_pos)
      advanceStridedIteration(coord, backstrides, iter_pos, t)

iterator items*[T](t: Tensor[T]): T {.noSideEffect.} =
  stridedIteration(IterKind.Values, t)

#########################################################

proc sanityChecks() =
  # Sanity checks

  let x = randomTensor([1, 2, 3], 10)
  let y = randomTensor([5, 2], 10)

  echo x # (shape: [1, 2, 3], strides: [6, 3, 1], offset: 0, storage: (data: @[1, 10, 5, 5, 7, 3]))
  echo y # (shape: [5, 2], strides: [2, 1], offset: 0, storage: (data: @[8, 3, 7, 9, 3, 8, 5, 3, 7, 1]))

  block:
    for val in x:
      echo val

when isMainModule:
  sanityChecks()
