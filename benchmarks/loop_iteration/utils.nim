# MIT License
# Copyright (c) 2018 Mamy André-Ratsimbazafy

import macros, sequtils

func product*(x: varargs[int]): int {.inline.}=
  result = 1
  for val in x: result *= val

proc replaceNodes*(ast: NimNode, replacements: NimNode, to_replace: NimNode): NimNode =
  # Args:
  #   - The full syntax tree
  #   - an array of replacement value
  #   - an array of identifiers to replace
  proc inspect(node: NimNode): NimNode =
    case node.kind:
    of {nnkIdent, nnkSym}:
      for i, c in to_replace:
        if node.eqIdent($c):
          return replacements[i]
      return node
    of nnkEmpty: return node
    of nnkLiterals: return node
    else:
      var rTree = node.kind.newTree()
      for child in node:
        rTree.add inspect(child)
      return rTree
  result = inspect(ast)

proc pop*(tree: var NimNode): NimNode =
  ## varargs[untyped] consumes all arguments so the actual value should be popped
  ## https://github.com/nim-lang/Nim/issues/5855
  result = tree[tree.len-1]
  tree.del(tree.len-1)

func concatMap*[T](s: seq[T], f: proc(ss: T):string): string =
  ## Map a function to a sequence of T and concatenate the result as string
  return s.foldl(a & f(b), "")
