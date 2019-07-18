# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  macros, tables,
  # Internal
  ../core/[lux_types, lux_core_helpers],
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
    # domainStack: var seq[Iter],
    visited: var Table[Id, NimNode],
    stmts: var NimNode): NimNode =
  ## Recursively walk the AST
  ## Append the corresponding Nim AST for generic instructions
  case ast.kind:
    of Func:
      return ident(ast.fn.symbol)
    of IntLit:
      return newLit ast.intVal
    of FloatLit:
      return newLit ast.floatVal
    of IntParam..BoolParam:
      return ident(ast.symParam)
    of Domain:
      # Append domId or use domainStack
      # or does Nim automatically scope/shadow nested for loop
      # with same ident?
      return ident(ast.iter.symbol)
    of BinOp:
      result = nnkInfix.newTree()
      case ast[0].bopKind
      of Add: result.add SimdMap(arch, T, simdAdd)
      of Mul: result.add SimdMap(arch, T, simdMul)
      of Eq: result.add bindSym"==" # Simd not applicable?

      result.add codegen(ast[1], arch, T, visited, stmts)
      result.add codegen(ast[2], arch, T, visited, stmts)
    of Access:
      result = nnkBracketExpr.newTree()
      result.add ident(ast[0].fn.symbol)
      for i in 1 ..< ast.len:
        result.add codegen(ast[i], ArchGeneric, getType(int), visited, stmts)
    of DimSize:
      result = nnkBracketExpr.newTree(
        nnkDotExpr.newTree(
          ident(ast[0].fn.symbol),
          ident"shape"
        ),
        newLit(ast[1].intVal)
      )
    # ------ Statements ----------------
    of StatementList:
      # Contrary to Nim, Lux StatementList are never expressions
      # So we don't need to return anything
      for subast in ast:
        discard codegen(subast, arch, T, visited, stmts)
    of Assign:
      stmts.add nnkAsgn.newTree(
        codegen(ast[0], arch, T, visited, stmts),
        codegen(ast[1], arch, T, visited, stmts)
      )
    of Check: # Boun checking
      stmts.add newCall(
        ident"doAssert",
        codegen(ast[0], ArchGeneric, getType(int), visited, stmts)
      )
    of AffineFor:
      var forStmt = nnkForStmt.newTree()
      var innerStmt = newStmtList()
      forStmt.add ident(ast[0].iter.symbol)
      forStmt.add nnkInfix.newTree(
        ident"..<",
        codegen(ast[0].iter.start, ArchGeneric, getType(int), visited, stmts),
        codegen(ast[0].iter.stop, ArchGeneric, getType(int), visited, stmts)
      )
      discard codegen(ast[1], arch, T, visited, innerStmt)
      forStmt.add innerStmt
      stmts.add forStmt
    else:
      raise newException(ValueError, "Unsupported code generation")

proc genKernel*(
      arch: SimdArch,
      kernel_ast: varargs[LuxNode],
      fns: varargs[Fn],
      ids: seq[NimNode], # Unused
      ids_baseType: seq[NimNode],
      resultType: NimNode,
    ): NimNode =
  # Does topological ordering and dead-code elimination
  result = newStmtList()
  var visitedNodes = initTable[Id, NimNode]()

  for i, fn in fns:
    if fn.stages.len == 0:
      # Input functions, no code to generate
      discard
    else:
      discard codegen(kernel_ast[i], arch, ids_baseType[i], visitedNodes, result)
      let symbol = ident(fn.symbol)
      if resultType.kind == nnkTupleTy:
        result.add newAssignment(
          nnkDotExpr.newTree(
            ident"result",
            ids[i]
          ),
          symbol
        )
      else:
        result.add newAssignment(
          ident"result",
          symbol
        )
