# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

template liftUnary*(func_name: untyped): untyped =
  ## Lift an unary function to make it work elementwise on a sequence
  proc func_name[T](x: seq[T]): seq[T] =
    result = newSeq[T](x.len)
    for i in 0 .. x.len:
      result[i] = func_name(x[i])

template liftReduction*(func_name, op, initial_value: untyped): untyped =
  ## Lift a reduction function to make it work elementwise on a sequence
  proc func_name[T](x: seq[T]): T =
    result = T(initial_value)
    for val in x:
      result = op(result, val)
