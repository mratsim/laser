# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

template doWhile*(condition: bool, body: untyped): untyped =
  ## doWhile makes handling matrix edges easier than for loop
  ## A for loop would break too early and would required
  ## handling at the "N-2" iteration. This prevents incrementing
  ## by a non-unit stride as N-2 becomes a range.
  while true:
    `body`
    if not `condition`:
      break

template `+=`*(p: pointer, offset: int) =
  ## Pointer arithmetic - in-place addition
  ## the pointer is moved but we can't use a var as var
  ## uses a hidden pointer and the wrong one will be used in C.

  # Does: {.emit: "`p` += `offset`;".}
  const psym = p.astToStr()
  const osym = offset.astToStr()
  {.emit: psym & " += " & osym & ";".}

func `+`*[T](p: pointer or ptr, offset: int): type(p) {.inline.}=
  # Pointer arithmetic
  {.emit: "`result` = `p` + `offset`;".}
