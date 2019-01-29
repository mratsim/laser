# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

proc relative_error*[T: SomeFloat](y, y_true: T): T {.inline.} =
  ## Relative error, |y_true - y|/max(|y_true|, |y|)
  ## Normally the relative error is defined as |y_true - y| / |y_true|,
  ## but here max is used to make it symmetric and to prevent dividing by zero,
  ## guaranteed to return zero in the case when both values are zero.
  let denom = max(abs(y_true), abs(y))
  if denom == 0.T:
    return 0.T
  result = abs(y_true - y) / denom

proc absolute_error*[T: SomeFloat](y, y_true: T): T {.inline.} =
  ## Absolute error for a single value, |y_true - y|
  result = abs(y_true - y)

proc mean_relative_error*[T: SomeFloat](y, y_true: seq[T]): T {.inline.} =
  doAssert y.len == y_true.len

  result = 0.T
  for i in 0 ..< y.len:
    result += relative_error(y[i], y_true[i])
  result = result / y.len.T

proc mean_absolute_error*[T: SomeFloat](y, y_true: seq[T]): T {.inline.} =
  doAssert y.len == y_true.len

  result = 0.T
  for i in 0 ..< y.len:
    result += absolute_error(y[i], y_true[i])
  result = result / y.len.T
