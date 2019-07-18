# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  macros,
  # Internal
  ../core/[lux_types, lux_core_helpers]

export lux_types, lux_core_helpers

# ###########################################
#
#       Helpers for the base DSL routines
#
# ###########################################

proc symFnAndIndices*(
        stmts: var NimNode,
        fn: NimNode, indices: NimNode
      ): NimNode =
  ## Add statement that fill the .symbol field
  ## of functions and indices with their Nim ident
  ## Return a NimNode suitable for varargs call
  ## workaround https://github.com/nim-lang/Nim/issues/11769

  stmts.add quote do:
    if `fn`.symbol == "":
      `fn`.symbol = astToStr(`fn`)

  result = nnkArgList.newTree()

  for index in indices:
    stmts.add quote do:
      if `index`.isNil:
        new `index`
        `index`.symbol = astToStr(`index`)
    result.add index
