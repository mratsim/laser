# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard Library
  sets, tables, hashes, sequtils,
  # Internal
  ../../core/lux_types,
  ../../core/lux_core_helpers,
  ../../utils/iter_utils,
  # Debug
  ../../core/lux_print

# ###########################################
#
#            Pass: Build Loop
#
# ###########################################

# Lower the high-level loop-nest representation
# to a low-level one.
#
# In the future this will take scheduling information
# like parallel, unroll or vectorize

proc searchDomains(node: LuxNode,
                    domainsFound: var OrderedTable[Id, LuxNode],
                    boundsStmts: var LuxNode) =

  case node.kind
  of Domain:
    domainsFound[node.id] = node
  of BinOp:
    for idx in 1 ..< node.len:
      # We need to handle A[i+j, 0] access
      node[idx].searchDomains(domainsFound, boundsStmts)
  of Access:
    for idx in 1 ..< node.len:
      # We can't do bound checks and bounds definition
      # in symFnAndIndices due to Nim bug
      #
      # Furthermore, the single reference to a domain instance is lost
      # so we need to track domain Id in a field
      #
      # Important, domainsFound is indexed by domId not by LuxNode.id
      if node[idx].kind == Domain:
        if node[idx].iter.domId in domainsFound:
          boundsStmts.add Check.newTree(
            BinOp.newTree(
              newLux(Eq),
              domainsFound[node[idx].iter.domId].iter.stop,
              DimSize.newTree(
                node[0],
                newLux(idx-1)
              )
            )
          )
          # Canonicalize domain
          node[idx] = domainsFound[node[idx].iter.domId]
        else:
          assert node[idx].iter.stop.isNil
          assert node[idx].iter.domId != 0
          assert node[0].kind == Func
          node[idx].iter.start = newLux 0
          node[idx].iter.stop = DimSize.newTree(
            node[0],
            newLux(idx-1)
          )
          domainsFound[node[idx].iter.domId] = node[idx]

      else:
        # We need to handle A[i+j, 0] access
        # TODO bounds checking
        node[idx].searchDomains(domainsFound, boundsStmts)
  of DimSize..AffineIf:
    raise newException(ValueError, "Invalid AST at this phase [Build loop - loop domains]")
  else:
    discard

#   # Assign - we reach another assignment, i.e. managed in another loop
#   # AffineFor - We are already looping, we don't want double loop
#   # AffineIf - If loop index constraint, only make sense when accompanied by affine for

proc buildLoopsImpl(node: LuxNode, visited: var HashSet[Id], stmts: var LuxNode) =
  # We only need check lvalues for assign statements
  #
  # TODO: While assignment A[i, j] = expression
  #       and in-place op like sum += A[_]
  #       will always have a Func or Assign in the AST,
  #       there is the degenerate case:
  #         result = A[i, j]
  #       that will not be wrapped in a (useless) loop.
  if node.isNil:
    return

  case node.kind
  of Func:
    # We don't tag stages as visited
    # What if a stage is not ultimately used?
    # For now we assume no co-recursion
    if node.id in visited:
      return
    visited.incl node.id

    for stage in node.fn.stages:
      # build the prerequisite for this stage
      var stagePrelude = newLuxStmtList()
      buildLoopsImpl(stage.definition, visited, stagePrelude)
      stmts.add stagePrelude

      # infer the loop bounds and add bounds-checking
      var domains: OrderedTable[Id, LuxNode]
      var boundsStmts = newLuxStmtList()
      searchDomains(stage.definition, domains, boundsStmts)
      stmts.add boundsStmts

      # Build the nested loops.
      # TODO: recurrence and condition
      # Progressively wrap the assign statement from inner to outer loop
      # TODO: directly use implicit result - https://github.com/nim-lang/Nim/issues/11637

      # merge preserving order
      #   We want the inner-most loop to be unit-stride write as
      #   write prefetching is costlier than read prefetching.
      #   but we want to somewhat preserve loop order of contraction
      #   like A[i, j] += B[i,j,k] and A[i, j] += B[i,k,j]
      #
      #   So we should merge the lhs and rhs domain sets as
      #   {rhs} + {lhs}

      var lhs = Access.newTree(node)
      for param in stage.params:
        if param.kind == Domain:
          discard domains.hasKeyOrPut(param.iter.domId, param)
        else:
          assert param.kind in {IntLit, IntParam}
        lhs.add param

      let domainList = toSeq(domains.values())
      var loopStmt = Assign.newTree(
        lhs,
        stage.definition
      )

      for domain in domainList.reverse():
        loopStmt = AffineFor.newTree(
          domain,
          loopStmt
        )
      stmts.add loopStmt

  of IntLit..BinOpKind:
    return
  of BinOp..Access:
    if node.id in visited:
      return
    assert node.len == 3
    visited.incl node.id

    var lhsPrelude = newLuxStmtList()
    var rhsPrelude = newLuxStmtList()
    buildLoopsImpl(node[1], visited, lhsPrelude)
    buildLoopsImpl(node[2], visited, rhsPrelude)

    stmts.add lhsPrelude
    stmts.add rhsPrelude
  else:
    raise newException(ValueError, "Invalid AST at this phase [Build loop - loop generation]")

proc passBuildLoops*(asts: varargs[Fn]): seq[LuxNode] =
  ## Scan for Functions
  ## and wrap them in the required for loops

  var visited: HashSet[Id]

  # We keep 1 AST per input/output function
  # If a function has multiple output and only one is needed
  # This will allows dead-code elimination
  result.setLen(asts.len)
  for i, ast in asts:
    visited.clear()
    result[i] = newLuxStmtList()
    buildLoopsImpl(newLux(ast), visited, result[i])
