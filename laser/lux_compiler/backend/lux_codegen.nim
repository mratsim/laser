# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  macros, tables,
  # Internal
  ../core/lux_types,
  ../platforms,
  # Debug
  ../core/lux_print

# ###########################################
#
#            Code generator
#
# ###########################################

# Generates low-level Nim code from Lux AST

proc codegen*(
    ast: LuxNode,
    arch: SimdArch,
    T: NimNode,
    params: seq[NimNode],
    visited: var Table[Id, NimNode],
    stmts: var NimNode): NimNode =
  ## Recursively walk the AST
  ## Append the corresponding Nim AST for generic instructions
  ## and returns a LValTensor, MutTensor or expression
  case ast.kind:
    of InTensor:
      return params[ast.symId]
    of IntImm:
      return newCall(SimdMap(arch, T, simdBroadcast), newLit(ast.intVal))
    of FloatImm:
      return newCall(SimdMap(arch, T, simdBroadcast), newLit(ast.floatVal))
    of MutTensor, LValTensor:
      let sym = ident(ast.symLVal)
      if ast.id in visited:
        return sym
      elif ast.prev_version.isNil:

        # TODO - need to add to the list of
        # buffer to initialize
        # And we need the size as well

        visited[ast.id] = sym
        return sym
      else:
        visited[ast.id] = sym
        var blck = newStmtList()
        let expression = codegen(ast.prev_version, arch, T, params, visited, blck)

        stmts.add blck
        if  not(expression.kind == nnkIdent and eqIdent(sym, expression)):
          stmts.add newAssignment(
            ident(ast.symLVal),
            expression
          )
        return ident(ast.symLVal)

    of AffineFor:
      if ast.id in visited:
        return visited[ast.id]

      var
        forLoop = nnkForStmt.newTree()
        loopPrologue = newStmtList()
        loopBody = newStmtList()

      let forLoopRange = nnkInfix.newTree(
        # TODO - hack - need a proper domain to loop range
        ident"..",
        newLit(ast.domain.start.intVal),
        nnkBracketExpr.newTree(
          nnkDotExpr.newTree(
            codegen(ast.domain.stop.tensor, arch, T, params, visited, loopPrologue),
            ident"shape"
          ),
          newLit(ast.domain.stop.axis)
        )
      )
      stmts.add loopPrologue

      discard codegen(ast.affineForBody, arch, T, params, visited, loopBody)

      forLoop.add ident(ast.domain.symDomain)
      forLoop.add forLoopRange
      forLoop.add loopBody

      stmts.add forLoop

      let nestedLVal = ident(ast.nestedLVal.symLVal)
      visited[ast.id] = nestedLVal
      return nestedLVal

    of Access, MutAccess:
      if ast.id in visited:
        return visited[ast.id]

      var preAccessStmt = newStmtList()
      let tensor = codegen(ast.tensorView, arch, T, params, visited, preAccessStmt)
      stmts.add preAccessStmt

      var bracketExpr = nnkBracketExpr.newTree()
      bracketExpr.add tensor
      for index in ast.indices:
        bracketExpr.add ident(index.symDomain)

      visited[ast.id] = bracketExpr
      return bracketExpr

    of Assign:
      if ast.id in visited:
        return visited[ast.id]

      var rvalStmt = newStmtList()
      let rval = codegen(ast.rval, arch, T, params, visited, rvalStmt)
      stmts.add rvalStmt

      var lvalStmt = newStmtList()
      let lval = codegen(ast.lval, arch, T, params, visited, lvalStmt)
      # "visited[ast.id] = lhs" is stored
      stmts.add lvalStmt

      lval.expectKind({nnkIdent, nnkBracketExpr})
      stmts.add newAssignment(lval, rval)
      # visited[ast.id] = lhs # Already done
      return lval

    of BinOp:
      if ast.id in visited:
        return visited[ast.id]

      var callStmt = nnkCall.newTree()
      var lhsStmt = newStmtList()
      var rhsStmt = newStmtList()

      let lhs = codegen(ast.lhs, arch, T, params, visited, lhsStmt)
      let rhs = codegen(ast.rhs, arch, T, params, visited, rhsStmt)

      stmts.add lhsStmt
      stmts.add rhsStmt

      case ast.binOpKind
      of Add: callStmt.add SimdMap(arch, T, simdAdd)
      of Mul: callStmt.add SimdMap(arch, T, simdMul)

      callStmt.add lhs
      callStmt.add rhs

      let op = genSym(nskLet, "op_")
      stmts.add newLetStmt(op, callStmt)
      visited[ast.id] = op
      return op

    else:
      raise newException(ValueError, "Unsupported code generation")

proc genKernel*(
      arch: SimdArch,
      io_ast: varargs[LuxNode],
      ids: seq[NimNode],
      ids_baseType: seq[NimNode],
      resultType: NimNode,
    ): NimNode =
  # Does topological ordering and dead-code elimination
  result = newStmtList()
  var visitedNodes = initTable[Id, NimNode]()

  for i, inOutVar in io_ast:
    if inOutVar.kind != InTensor:
      if inOutVar.kind in {MutTensor, LValTensor}:
        let sym = codegen(inOutVar, arch, ids_baseType[i], ids, visitedNodes, result)
        sym.expectKind nnkIdent
        if resultType.kind == nnkTupleTy:
          result.add newAssignment(
            nnkDotExpr.newTree(
              ident"result",
              ids[i]
            ),
            sym
          )
        else:
          result.add newAssignment(
            ident"result",
            sym
          )
      else:
        let expression = codegen(inOutVar, arch, ids_baseType[i], ids, visitedNodes, result)
        if resultType.kind == nnkTupleTy:
          result.add newAssignment(
            nnkDotExpr.newTree(
              ident"result",
              ids[i]
            ),
            expression
          )
        else:
          result.add newAssignment(
            ident"result",
            expression
          )
  # TODO: support var Tensor.
