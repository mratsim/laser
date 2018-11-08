# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

template `+=`*[T](p: var ptr (T or UncheckedArray[T]), offset: int) =
  ## Pointer arithmetic - in-place addition

  # Does: {.emit: "`p` += `offset`;".}
  const psym = p.astToStr()
  const osym = offset.astToStr()
  {.emit: psym & " += " & osym & ";".}
