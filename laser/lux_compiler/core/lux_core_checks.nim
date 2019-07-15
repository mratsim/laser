# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Internal
  ./lux_types, ./lux_core_helpers

# ###########################################
#
#                AST checks
#
# ###########################################

# For composability, similar to Nim macros
# and Lisp, every LuxNodes except literals are
# seq[LuxNode]. That requires checking that the AST
# is well formed.

proc parseCheckBinOp*(node: LuxNode) =
  assert node.kind == BinOp
  assert node.len == 3
  assert node[0].kind == BinOpKind
  assert node[1].kind in LuxExpr
  assert node[2].kind in LuxExpr

proc parseAssign*(node: LuxNode) =
  assert node.kind == Assign
  assert node.len == 2
  assert node[0].kind == Func
  assert node[1].kind in LuxExpr

proc parseAccess*(node: LuxNode) =
  assert node.kind == Access
  assert node[0].kind == Func

  for i in 1 ..< node.children.len:
    assert node[i] in {
      IntImm, IntParam, Domain,
      BinOp
    }

proc parseShape*(node: LuxNode) =
  assert node.kind == Shape
  assert node.len == 2
  assert node[0].kind == Func
  assert node[1].kind == IntImm

proc parseAffineFor*(node: LuxNode) =
  assert node.kind == AffineFor
  assert node.len == 2
  assert node[0].kind == Domain
  assert node[1].kind == Statement

proc parseAffineIf*(node: LuxNode) =
  assert node.kind == AffineIf
  assert node.len == 2
  assert node[0].kind in LuxExpr
  assert node[1].kind == Statement
