# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  macros, tables,
  # Internal
  ./ast_definition,
  ../platforms

proc codegen*(
    ast: LuxNode,
    arch: SimdArch,
    params: seq[NimNode],
    visited: var Table[Id, NimNode],
    stmts: var NimNode): NimNode =
  ## Recursively walk the AST
  ## Append the corresponding Nim AST for generic instructions
  ## and returns a LVal, Output or expression
  case ast.kind:
    of Input:
      return params[ast.symId]
    of IntImm:
      return newCall(SimdTable[arch][simdBroadcast], newLit(ast.intVal))
    of FloatImm:
      return newCall(SimdTable[arch][simdBroadcast], newLit(ast.intVal))
    of Output, LVal:
      let sym = newIdentNode(ast.symLVal)
      if ast.id in visited:
        return sym
      elif ast.prev_version.isNil:
        visited[ast.id] = sym
        return sym
      else:
        visited[ast.id] = sym
        var blck = newStmtList()
        let expression = codegen(ast.prev_version, arch, params, visited, blck)
        stmts.add blck
        if not(expression.kind == nnkIdent and eqIdent(sym, expression)):
          stmts.add newAssignment(
            newIdentNode(ast.symLVal),
            expression
          )
        return newIdentNode(ast.symLVal)
    of Assign:
      if ast.id in visited:
        return visited[ast.id]

      var varAssign = false

      if ast.lhs.id notin visited and
            ast.lhs.kind == LVal and
            ast.lhs.prev_version.isNil and
            ast.rhs.id notin visited:
          varAssign = true

      var rhsStmt = newStmtList()
      let rhs = codegen(ast.rhs, arch, params, visited, rhsStmt)
      stmts.add rhsStmt

      var lhsStmt = newStmtList()
      let lhs = codegen(ast.lhs, arch, params, visited, lhsStmt)
      stmts.add lhsStmt

      lhs.expectKind(nnkIdent)
      if varAssign:
        stmts.add newVarStmt(lhs, rhs)
      else:
        stmts.add newAssignment(lhs, rhs)
      # visited[ast] = lhs # Already done
      return lhs

    of Add, Mul:
      if ast.id in visited:
        return visited[ast.id]

      var callStmt = nnkCall.newTree()
      var lhsStmt = newStmtList()
      var rhsStmt = newStmtList()

      let lhs = codegen(ast.lhs, arch, params, visited, lhsStmt)
      let rhs = codegen(ast.rhs, arch, params, visited, rhsStmt)

      stmts.add lhsStmt
      stmts.add rhsStmt

      case ast.kind
      of Add: callStmt.add SimdTable[arch][simdAdd]
      of Mul: callStmt.add SimdTable[arch][simdMul]
      else: raise newException(ValueError, "Unreachable code")

      callStmt.add lhs
      callStmt.add rhs

      let memloc = genSym(nskLet, "memloc_")
      stmts.add newLetStmt(memloc, callStmt)
      visited[ast.id] = memloc
      return memloc

proc bodyGen*(
    genSimd: bool, arch: SimdArch,
    io: varargs[LuxNode],
    ids: seq[NimNode],
    resultType: NimNode,
    ): NimNode =
  # Does topological ordering and dead-code elimination
  result = newStmtList()
  var visitedNodes = initTable[Id, NimNode]()

  for i, inOutVar in io:
    if inOutVar.kind != Input:
      if inOutVar.kind in {Output, LVal}:
        let sym = codegen(inOutVar, arch, ids, visitedNodes, result)
        sym.expectKind nnkIdent
        if resultType.kind == nnkTupleTy:
          result.add newAssignment(
            nnkDotExpr.newTree(
              newIdentNode"result",
              ids[i]
            ),
            sym
          )
        else:
          result.add newAssignment(
            newIdentNode"result",
            sym
          )
      else:
        let expression = codegen(inOutVar, arch, ids, visitedNodes, result)
        if resultType.kind == nnkTupleTy:
          result.add newAssignment(
            nnkDotExpr.newTree(
              newIdentNode"result",
              ids[i]
            ),
            expression
          )
        else:
          result.add newAssignment(
            newIdentNode"result",
            expression
          )
