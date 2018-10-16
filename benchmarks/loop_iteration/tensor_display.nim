# Copyright 2017 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sugar, sequtils, typetraits,
  ./tensor, ./tensor_reference_iteration, utils


func bounds_display(t: Tensor,
          idx_data: tuple[val: string, idx: int]
          ): string =
  let (val,idx) = idx_data
  let s = t.shape.reversed

  if val == "|":
    return " | "

  for i,j in s[0 .. s.len-2]: # We don't take the last element (the row in C convention)
    if idx mod j == 0:
      return "\t" & $val & "|\n"
    if idx mod j == 1:
      return "|" & $val
  return "\t" & $val

proc disp2d(t: Tensor): string {.noSideEffect.} =
  ## Display a 2D-tensor
  var indexed_data: seq[(string,int)] = @[]
  for i, value in t.enumerate:
    indexed_data.add(($value,i+1))

  proc curry_bounds(tup: (string,int)): string {.noSideEffect.}= t.bounds_display(tup)
  return indexed_data.concatMap(curry_bounds)

proc `$`*[T](t: Tensor[T]): string =
  ## Pretty-print a tensor (when using ``echo`` for example)
  let desc = "Tensor of shape " & $t.shape & " of type \"" & T.name & '"'
  if t.rank <= 2:
    return desc & "\n" & t.disp2d
  else:
    return desc & "\n" & " -- NotImplemented: Display not implemented for tensors of rank > 4"
