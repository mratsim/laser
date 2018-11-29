# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import macros, strutils

# ################################################################
#
#     Parser Generator for declarative x86_64 instruction format
#
# ################################################################

# The macro in this file will generate procs for the x86_64 assembler
# from a declarative format.
# The generated procs should include doc comments for nimsuggest
# and documentation generation.
#
# For example
# op MOV:
#   [dst32, src32]: [0x89, modrm(Direct, reg = src, rm = dst)]
#
# will generate
#   proc mov(a: Assembler[X86_64], dst32: static RegX86_32, src32: static RegX86_32) =
#     ## Copy 32-bit register content to another register
#     when dst32 in eax .. edi and src32 in eax .. edi:
#       a.add [0x89, modrm(Direct, reg = src, rm = dst)]
#     else:
#       a.add [
#          rex_prefix(
#            r = int(src32 in r8d .. r15d), # r extends "reg"
#            b = int(dst32 in r8d .. r15d)  # b extends "rm"
#          ),
#          0x89, modrm(Direct, reg = src, rm = dst)
#       ]

template assembler(): NimNode =
  newIdentDefs(
    ident"a", nnkVarTy.newTree(
      nnkBracketExpr.newTree(
        ident"Assembler",
        ident"X86_64"
      )
    )
  )
template reg(id, T: string): NimNode =
  newIdentDefs(
    ident id, nnkStaticTy.newTree(
      ident T
    )
  )
template imm(id, T1, T2: string): NimNode =
  newIdentDefs(
    ident id, nnkInfix.newTree(
      ident"or", ident T1, ident T2
    )
  )
template imm(id, T: string): NimNode =
  newIdentDefs(
    ident id, ident T
  )
template adr(id: string): NimNode =
  ## TODO support SIB/displacement indexing
  newIdentDefs(
    ident id, nnkStaticTy.newTree(
      nnkBracketExpr.newTree(
        ident"array",
        newLit 1,
        ident"RegX86_64"
      )
    )
  )
template label(id: string): NimNode =
  newIdentDefs(ident id, ident"Label")

macro op_generator*(instructions: untyped): untyped =
  # echo instructions.treeRepr
  result = newStmtList()

  # a: Assembler[X86_64] is the first param
  # of all procs
  let assemblerDef = assembler()

  for op in instructions:
    # Sanity checks
    op.expectKind nnkCommand
    assert op[0].eqIdent "op"
    op[1].expectKind nnkIdent # for example ident "MOV"
    op[2].expectKind nnkStmtList
    assert op.len == 3

    let procName = newIdentNode toLower($op[1])
    var procComment: NimNode

    ## Now iterate other all input params overload
    for overload in op[2]:
      if overload.kind == nnkCommentStmt:
        procComment = overload.copy
      else:
        # Sanity checks
        overload.expectKind nnkCall
        overload[0].expectKind nnkBracket    # for example [dst32, src32]
        overload[1].expectKind nnkStmtList
        overload[1][0].expectKind nnkBracket # for example [0x89, modrm(Direct, reg = src, rm = dst)]

        # Aliases
        let params = overload[0]
        let bytecode = overload[1][0]

        # Track if we use labels
        var useLabels = false

        # Let's create the proc parameters
        var procParams = @[newEmptyNode()] # void return type
        procParams.add assemblerDef        # a: Assembler[X86_64]

        # Workaround typechecking imm64: uint64 or pointer
        # we generate a pointer immediate proc separate from
        # the int procs
        var genPointerVersion = false

        # Iterate on the raw param
        for arg in params:
          var dst, src: NimNode

          let mnemo = $arg
          let pre = mnemo[0..2]  # mnemo prefix

          if pre == "dst":
            let suf = mnemo[3..^1]
            dst = arg
            procParams.add mnemo.reg("RegX86_" & suf)
          elif pre == "src":
            let suf = mnemo[3..^1]
            src = arg
            procParams.add mnemo.reg("RegX86_" & suf)
          elif pre == "imm":
            let suf = mnemo[3..^1]
            procParams.add mnemo.imm("uint" & suf, "int" & suf)
            if suf == "64": genPointerVersion = true
          elif pre == "adr":
            procParams.add mnemo.adr()
          elif pre == "lab":
            procParams.add mnemo.label()
          else:
            error "Wrong mnemonic \"" & mnemo & "\"." &
                  " Only dst, src, imm (+size suffix)," &
                  " adr and label are allowed.", arg
