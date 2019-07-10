# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard Library
  tables,
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

proc buildLoopsImpl(node: LuxNode, visited: var Table[Id, LuxNode]): LuxNode =
  # We only need check lvalues for assign statements
  #
  # TODO: While assignment A[i, j] = expression
  #       and in-place op like sum += A[_]
  #       will always have a Mut or LVal or Assign in the AST,
  #       there is the degenerate case:
  #         result = A[i, j]
  #       that will not be wrapped in a (useless) loop.
  if node.isNil:
    return

  case node.kind
  of MutTensor, LValTensor, IntMut, FloatMut, IntLVal, FloatLVal:
    if node.prev_version.isNil:
      assert node.version == 0
      return node
    if node.id in visited:
      return visited[node.id]

    var new_node = new LuxNode
    new_node.id = node.id
    new_node.kind = node.kind
    new_node.symLVal = node.symLval
    new_node.version = node.version
    new_node.prev_version = buildLoopsImpl(
      node.prev_version,
      visited
    )

    visited[node.id] = new_node
    return new_node
  of Access, MutAccess:
    var new_node = new LuxNode
    new_node.id = node.id
    new_node.kind = node.kind
    new_node.tensorView = buildLoopsImpl(node.tensorView, visited)
    new_node.indices = node.indices
    return new_node

  of Assign:
    if node.id in visited:
      return visited[node.id]

    # Scan for previous assignments
    let lval = buildLoopsImpl(node.lval, visited)

    # reconstruct the inner AST
    var innerStmt = LuxNode(
      id: genId(),
      kind: Assign,
      lval: lval,
      rval: node.rval,
      # domains - now unneeded
    )

    let lvalConcrete = case lval.kind:
      of LValTensor: lval
      of Access, MutAccess: lval.tensorView
      else:
        raise newException(
          ValueError,
          "Found node \"" & $lval.kind & "\" when looking for a l-value"
        )

    # Progressively wrap the assign statement from inner to outer loop
    # TODO: directly use implicit result - https://github.com/nim-lang/Nim/issues/11637
    for domain in node.domains.reverse():
      innerStmt = LuxNode(
        id: genId(),
        kind: AffineFor,
        domain: domain,
        affineForBody: innerStmt,
        nestedLVal: lvalConcrete
      )

    visited[node.id] = innerStmt
    return innerStmt
  else:
    return node

proc passBuildLoops*(asts: varargs[LuxNode]): seq[LuxNode] =
  ## Scan for Assign statements
  ## and wrap them in the required for loops

  # TODO: should we mutate the AST in-place?
  var visited = initTable[Id, LuxNode](initialSize = 8)

  for ast in asts:
    result.add buildLoopsImpl(ast, visited)
