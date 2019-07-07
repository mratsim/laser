# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  unittest,
  ../laser/tensor/[datatypes, initialization]

suite "\"Let\" tensors cannot be mutated through their raw view pointers":
  test "Check that the raw view API preserves immutability":
    let a = newTensor[int](2, 3)
    let p_a = a.unsafe_raw_data
    check:
      not compiles(p_a[0] = 100)
      p_a[0] == 0
      p_a[1] == 0

    var b = newTensor[int](2, 3)
    let p_b = b.unsafe_raw_data
    p_b[0] = 100
    check:
      p_b[0] == 100
      p_b[1] == 0
