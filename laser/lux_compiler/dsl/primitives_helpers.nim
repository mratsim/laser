# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  random,
  # Internal
  ../core/[lux_types, lux_core_helpers]

export lux_types, lux_core_helpers

# ###########################################
#
#       Helpers for the base DSL routines
#
# ###########################################

proc assign*(lhs, rhs: LuxNode): LuxNode =
  # Generate the Assign node
  # This also scans the domain /loop nest needed
  # to generate the assignment
  LuxNode(
    id: genId(),
    kind: Assign,
    lval: lhs,
    rval: rhs
    # domains: TODO
  )

proc newSym*(symbol: string, rhs: LuxNode): LuxNode =
  # Declare and allocate a new AST symbol
  # This also scans the domain/loop nest needed
  # to generate the assignment
  assign(
    lhs = LuxNode(
        # 1. Declare unallocated lval
        id: genId(),
        kind: LValTensor,
        symLVal: symbol,
        version: 0,
        prev_version: nil
      ),
    rhs = rhs
  )

proc lvalify*(node: var LuxNode) =
  ## Allocate an expression result
  ## to a mutable memory location.
  ##
  ## Solve the case where we have:
  ##
  ## .. code::nim
  ##   var B = A
  ##   B += C
  ##
  ## B must be attributed a memory location (l-value)
  ## to be mutated
  ##
  ## Also in the case
  ## .. code::nim
  ##   var B = A
  ##   var B2 = A
  ##   B += C
  ##   B2 *= C
  ##
  ## we want to reuse the same computation A
  ## but B and B2 should have unique ID in the AST tree
  ##
  ## # Summary
  ##
  ## Unique:
  ##   - lval symbol
  ##
  ## Reused:
  ##   - operation/expression ID

  let lval_id = genId()
  let lval_symbol = "lval_" & $lval_id
  node = LuxNode(
    id: lval_id,
    kind: LValTensor, # TODO accept scalars
    symLVal: lval_symbol,
    version: 1,
    prev_version: newSym(lval_symbol, node)
  )
