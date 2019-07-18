# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  macros

{.experimental: "dynamicBindSym".}
from ../../tensor/datatypes import Tensor

proc isType*(x: NimNode, t: string): bool =
  ## Compile-time type checking

  # We cannot instantiate a fully typed container
  # https://github.com/nim-lang/Nim/issues/6785
  # and https://github.com/nim-lang/RFCs/issues/44

  if x.kind == nnkBracketExpr:
    return sameType(bindSym(x[0]), bindSym(t))
  else:
    return sameType(bindSym(x), bindSym(t))

proc ct*(ident: NimNode): NimNode =
  nnkPragmaExpr.newTree(
    ident,
    nnkPragma.newTree(
      ident"compileTime"
    )
  )

proc liftTypes*(
        ast: NimNode,
        containerIdent: string,
        remapping = proc(x: NimNode): NimNode = x): NimNode =
  # InTensor:
  #   - A type signature
  #   - A container ident, for example "seq" or "Tensor"
  #     non-container will stay as-is, to allow multiplication of a container by a constant for example
  #   - An optional remapping function, by default identity
  #     we can use a SIMD map: float32 -> m128
  proc inspect(node: NimNode): NimNode =
    case node.kind:
    of nnkIdent, nnkSym: return node
    of nnkEmpty: return node
    of nnkLiterals: return node
    of nnkIdentDefs:
      if node[^2].isType(containerIdent):
        result = node.copyNimTree()
        result[^2] = remapping(node[^2][1])
        return
      else:
        return node
    else:
      var rTree = node.kind.newTree()
      for child in node:
        rTree.add inspect(child)
      return rTree
  result = inspect(ast)
