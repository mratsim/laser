# MIT License
# Copyright (c) 2017-2018 Mamy André-Ratsimbazafy

import ./tensor

type
  IterKind* = enum
    Values, Enumerate

template strided_iteration[T](t: Tensor[T], strider: IterKind): untyped =
  ## Iterate over a Tensor, displaying data as in C order, whatever the strides.

  ## Iterator init
  var coord: array[MAXRANK, int] # Coordinates in the n-dimentional space
  var backstrides: array[MAXRANK, int] # Offset between end of dimension and beginning
  for i in 0..<t.rank:
    backstrides[i] = t.strides[i]*(t.shape[i]-1)
    coord[i] = 0

  var iter_pos = t.offset

  ## Iterator loop
  for i in 0 ..< t.shape.product:

    ## Templating the return value
    when strider == IterKind.Values: yield t.storage.data[iter_pos]
    elif strider == IterKind.Enumerate: yield (i, t.storage.data[iter_pos])
    # elif strider == IterKind.Coord_Values: yield (coord, t.storage.data[iter_pos])
    # elif strider == IterKind.MemOffset: yield iter_pos
    # elif strider == IterKind.MemOffset_Values: yield (iter_pos, t.storage.data[iter_pos])

    ## Computing the next position
    for k in countdown(t.rank - 1,0):
      if coord[k] < t.shape[k]-1:
        coord[k] += 1
        iter_pos += t.strides[k]
        break
      else:
        coord[k] = 0
        iter_pos -= backstrides[k]

iterator items*[T](t: Tensor[T]): T =
  strided_iteration(t, Values)

iterator enumerate*[T](t: Tensor[T]): (int, T) =
  strided_iteration(t, Enumerate)
