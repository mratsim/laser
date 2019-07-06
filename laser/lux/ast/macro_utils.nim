# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  macros

proc replaceType*(ast: NimNode, to_replace: NimNode, replacements: NimNode): NimNode =
  # Args:
  #   - The full syntax tree
  #   - replacement type
  #   - type to replace
  proc inspect(node: NimNode): NimNode =
    case node.kind:
    of {nnkIdent, nnkSym}: return node
    of nnkEmpty: return node
    of nnkLiterals: return node
    of nnkIdentDefs:
      let i = node.len - 2 # Type position
      if node[i] == to_replace:
        result = node.copyNimTree()
        result[i] = replacements
        return
      else:
        return node
    else:
      var rTree = node.kind.newTree()
      for child in node:
        rTree.add inspect(child)
      return rTree
  result = inspect(ast)

proc ct*(ident: NimNode): NimNode =
  nnkPragmaExpr.newTree(
    ident,
    nnkPragma.newTree(
      ident"compileTime"
    )
  )
