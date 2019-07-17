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

# proc hash(node: LuxNode): Hash =
#   # Use of hash is restricted to the OrderedSet used in this module.
#   # In other modules like codegen, we want to index via node ID.
#   # as 2 ASTs can have the same node ID, if one is the expression
#   # and the other is the variable it has been assigned to.
#   node.id

# proc searchDomains*(node: LuxNode, domainsFound: var OrderedSet[LuxNode]) =
#   case node.kind
#   of LValTensor, IntMut, FloatMut, IntLVal, FloatLVal:
#     node.prev_version.searchDomains(domainsFound)
#   of BinOp:
#     node.lhs.searchDomains(domainsFound)
#     node.rhs.searchDomains(domainsFound)
#   of Access, MutAccess:
#     for idx in node.indices:
#       # We need to handle A[i+j, 0] access
#       idx.searchDomains(domainsFound)
#   of Domain:
#     domainsFound.incl node
#   else:
#     discard

#   # Assign - we reach another assignment, i.e. managed in another loop
#   # AffineFor - We are already looping, we don't want double loop
#   # AffineIf - If loop index constraint, only make sense when accompanied by affine for

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
