# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  random, sets, hashes, sequtils,
  # Internal
  ../core/[lux_types, lux_core_helpers]

export lux_types, lux_core_helpers

# ###########################################
#
#       Helpers for the base DSL routines
#
# ###########################################

proc hash(node: LuxNode): Hash =
  # Use of hash is restricted to the OrderedSet used in this module.
  # In other modules like codegen, we want to index via node ID.
  # as 2 ASTs can have the same node ID, if one is the expression
  # and the other is the variable it has been assigned to.
  node.id

proc searchDomains*(node: LuxNode, domainsFound: var OrderedSet[LuxNode]) =
  case node.kind
  of LValTensor, IntMut, FloatMut, IntLVal, FloatLVal:
    node.prev_version.searchDomains(domainsFound)
  of BinOp:
    node.lhs.searchDomains(domainsFound)
    node.rhs.searchDomains(domainsFound)
  of Access, MutAccess:
    for idx in node.indices:
      # We need to handle A[i+j, 0] access
      idx.searchDomains(domainsFound)
  of Domain:
    domainsFound.incl node
  else:
    discard

  # Assign - we reach another assignment, i.e. managed in another loop
  # AffineFor - We are already looping, we don't want double loop
  # AffineIf - If loop index constraint, only make sense when accompanied by affine for

proc assign*(lhs, rhs: LuxNode): LuxNode =
  ## Generate the Assign node
  ## This also scans the domain / loop nest needed
  ## to generate the assignment.
  ##
  ## Iteration domains are ordered from outermost to innermost
  ## for the LHS.
  ## And approximatively this way as well for the RHS.

  result = LuxNode(
    id: genId(),
    kind: Assign,
    lval: lhs,
    rval: rhs,
  )

  var domains = initOrderedSet[LuxNode](initialSize = 8)

  # Inner dimension of the lhs is always last
  # as prefetching for write operations is more expensive.
  rhs.searchDomains(domains)
  lhs.searchDomains(domains)

  result.domains = toSeq(domains)

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
