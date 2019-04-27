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
#          rex(
#            r = int(src32 in r8d .. r15d), # r extends "reg"
#            b = int(dst32 in r8d .. r15d)  # b extends "rm"
#          ),
#          0x89, modrm(Direct, reg = src, rm = dst)
#       ]

proc assembler(a: NimNode): NimNode =
  ## Type section - "a: var Assembler[X86_64]"
  newIdentDefs(
    a, nnkVarTy.newTree(
      nnkBracketExpr.newTree(
        ident"Assembler",
        ident"X86_64"
      )
    )
  )
proc reg(id: NimNode, T: string): NimNode =
  ## Type section - "dst: static RegX86_64"
  newIdentDefs(
    id, nnkStaticTy.newTree(
      ident T
    )
  )
proc `or`(T1, T2: string): NimNode =
  ## Type section - "uint8 or int8"
  nnkInfix.newTree(
      bindSym"or", ident T1, ident T2
    )
proc imm(id: NimNode, T1, T2: string): NimNode =
  ## Type section - "T: uint8 or int8"
  newIdentDefs(
    id, T1 or T2
  )
proc imm(id: NimNode, T: string): NimNode =
  ## Type section - "T: uint8"
  newIdentDefs(
    id, ident T
  )
proc adr(id: NimNode): NimNode =
  ## Type section - "adr: static array[1, RegX86_64]"
  # TODO support SIB/displacement indexing
  newIdentDefs(
    id, nnkStaticTy.newTree(
      nnkBracketExpr.newTree(
        bindSym"array",
        newLit 1,
        ident"RegX86_64"
      )
    )
  )
proc label(id: NimNode): NimNode =
  ## Type section - "label: Label"
  newIdentDefs(id, ident"Label")

proc types(ids: NimNode): NimNode =
  ## Type section - "T: (uint16 or int16) or (uint32 or int32) or (uint§4 or int64)"
  assert ids[0].eqIdent "type"
  assert ids.len == 2

  # 1. build (uint64 or int64)
  result = (ident "uint" & $ids[1].intVal) or (ident "int" & $ids[1].intVal)

  # 2. Transform into "T: type(uint64 or int64)"
  result = newIdentDefs(
    ident"T", nnkCall.newTree(
      bindSym"type", result
    )
  )

proc searchRegRM(bracket: NimNode): tuple[reg, rm: NimNode] =
  ## Search in
  ##   - [rex(w=1), 0x89, modrm(Direct, reg = src64, rm = dst64)]
  ##   - [    0x66, 0xFF, modrm(Direct, opcode_ext = 0, rm = reg)]
  ## for reg = src64 and rm = dst64 and return the reg and rm nodes.

  bracket.expectKind nnkBracket
  # modrm if it exists is always the last one
  if bracket[^1].kind == nnkCall:
    let modrm = bracket[^1]
    assert modrm[0].eqIdent"modrm"
    let reg_or_ext = modrm[2]
    let rm = modrm[3]

    reg_or_ext.expectKind nnkExprEqExpr
    rm.expectKind nnkExprEqExpr
    if reg_or_ext[0].eqIdent"reg":
      result.reg = reg_or_ext[1]
    result.rm = rm[1]

proc regUpperExpr(param: string, reg: NimNode): NimNode =
  ## Proc input - "param = int(dst64 in (type dst64)(0b1000)..(type dst64)(0b1111))"
  ## to construct: rex(w = 1, b = static(int(dst64 in r8 .. r15)))
  nnkExprEqExpr.newTree(
    newIdentNode param,
    newCall(bindSym"static",
      newCall(bindSym"int",
        nnkInfix.newTree(
          bindSym"in", reg,
          nnkInfix.newTree(
            bindSym"..",
            newCall(
              newCall(bindSym"type", reg),
              newLit 0b1000
            ),
            newCall(
              newCall(bindSym"type", reg),
              newLit 0b1111
            )
          )
        )
      )
    )
  )

proc canonicalBody(bodySpecs: NimNode): NimNode =
  ## Convert [0x89, modrm(Direct, reg = src64, rm = dst64)]
  ## to [byte 0x89, byte modrm(Direct, reg = src64, rm = dst64)]
  result = nnkBracket.newTree()
  for node in bodySpecs:
    result.add newCall(bindSym"byte", node)

proc rexifyBody(bodySpecs, dst_or_adr0: NimNode): NimNode =
  ## Transform [rex(w=1), 0x89, modrm(Direct, reg = src64, rm = dst64)]
  ## into [rex(w=1, b=1, r=1) ... as needed
  result = nnkBracket.newTree()

  # we need to first scan `modrm` to see which register is used for reg and which for rm
  let (reg, rm) = block:
    let regrm = bodySpecs.searchRegRM() # reg/rm are dst, src or adr[0]
    if regrm.rm.kind == nnkNilLit:
      # There is no modrm, so implicitly rm = dst or rm = adr[0]
      (regrm.reg, dst_or_adr0)
    else: regrm

  # Then we known that b/r = dst/src in r8 .. r15
  if reg.kind != nnkNilLit:
    for i, node in bodySpecs:
      if i == 0:
        let rExpr = regUpperExpr("r", reg) # REX.R extends modrm.reg
        let bExpr = regUpperExpr("b", rm)  # REX.B extends modrm.rm
        if node.kind == nnkCall:
          # original is [rex(w=1), opcode, ...]
          # new is [rex(w=1, b=.., r=..), opcode, ...]
          assert node[0].eqIdent"rex", $node.treerepr  # TODO support VEX for SSE/AVX
          var newRex = node.copy()
          # Monkey-patch rex(w=1)
          newRex.add rExpr
          newRex.add bExpr
          # into rex(w=1, b=..., r=...)
          result.add newRex
        elif node.kind == nnkIntLit and node.intVal == 0x66: # 16-bit case
          # original is [0x66, opcode, ...]
          # new is [0x66, REX, opcode, ...]
          result.add node
          result.add newCall(ident"rex", rExpr, bExpr)
        else:
          # original is [opcode, ...]
          # new is [REX, opcode, ...]
          result.add newCall(ident"rex", rExpr, bExpr)
          result.add node
      else:
        result.add node
  else:
    # modrm(Direct, opcode_ext = 0, rm = dst) case
    for i, node in bodySpecs:
      if i == 0:
        let bExpr = regUpperExpr("b", rm)  # REX.B extends modrm.rm
        if node.kind == nnkCall:
          assert node[0].eqIdent"rex", $node.treerepr # TODO support VEX for SSE/AVX
          var newRex = node.copy()
          # Monkey-patch rex(w=1)
          newRex.add bExpr
          # into rex(w=1, r=...)
          result.add newRex
        elif node.kind == nnkIntLit and node.intVal == 0x66: # 16-bit case
          # order is [0x66, REX, opcode, ...]
          result.add node
          result.add newCall(ident"rex", bExpr)
        else:
          result.add newCall(ident"rex", bExpr)
          result.add node
      else:
        result.add node

proc processSpecs(procBody: var NimNode, assemblerSym, dst, src, adr, body: Nimnode) =
  # case [rex(w=1), 0x89, modrm(Direct, reg = src64, rm = dst64)]
  # If both registers are in 0b0_000 to 0b0_111 range keep the body as is
  # Otherwise we need to add a rex_prefix if there wasn't
  # or adapt b and r params if there was
  body.expectKind nnkBracket

  # Convert to [byte 0x89, byte modrm(Direct, reg = src64, rm = dst64)]
  let canonicalBody = body.canonicalBody()

  if dst.kind != nnkNilLit and src.kind != nnkNilLit:
    # Rex case [rex(b=1, r=1), 0x89, modrm(Direct, reg = src64, rm = dst64)]
    let rexBody = body.rexifyBody(dst)
    procBody.add quote do:
      when (`dst` in (type `dst`)(0b0_000) .. (type `dst`)(0b0_111)) and
              (`src` in (type `src`)(0b0_000) .. (type `src`)(0b0_111)):
        `assemblerSym`.code.add `canonicalBody`
      else:
        `assemblerSym`.code.add `rexBody`
  elif dst.kind != nnkNilLit:
    let rexBody = body.rexifyBody(dst)
    procBody.add quote do:
      when (`dst` in (type `dst`)(0b0_000) .. (type `dst`)(0b0_111)):
        `assemblerSym`.code.add `canonicalBody`
      else:
        `assemblerSym`.code.add `rexBody`
  elif adr.kind != nnkNilLit:
    let adr0 = nnkBracketExpr.newTree(adr, newLit 0)
    let rexBody = body.rexifyBody(adr0)
    procBody.add quote do:
      when (`adr0` in (type `adr0`)(0b0_000) .. (type `adr0`)(0b0_111)):
        `assemblerSym`.code.add `canonicalBody`
      else:
        `assemblerSym`.code.add `rexBody`
  else: # instruction depends on flags like JE/JNE
    procBody.add quote do:
      `assemblerSym`.code.add `canonicalBody`

proc newInlineProc*(
    name = newEmptyNode();
    params: openArray[NimNode] = [newEmptyNode()];
    body: NimNode = newStmtList(),
    procType = nnkProcDef): NimNode =
  ## We define our own newProc shortcut with
  ## pragmas and export marker
  assert procType in RoutineNodes
  result = newNimNode(procType).add(
    nnkPostFix.newTree(ident"*", name),
    newEmptyNode(),                          ## pattern matching for term-rewriting macros
    newEmptyNode(),                          ## Generics
    nnkFormalParams.newTree(params),         ## params
    nnkPragma.newTree(ident"inline", ident"noSideEffect"),
    newEmptyNode(),                          ## Always empty
    body)

macro op_generator*(instructions: untyped): untyped =
  # echo instructions.treeRepr
  result = newStmtList()

  # a: Assembler[X86_64] is the first param
  # of all procs
  let assemblerSym = newIdentNode"a"
  let assemblerDef = assembler(assemblerSym)

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
        var bodySpecs = overload[1][0]

        # Aliases
        let params = overload[0]
        let bytecode = overload[1][0]

        # Track used params
        var useLabels = false
        var dst, src, adr, imm: NimNode

        # Let's create the proc parameters
        var procParams = @[newEmptyNode()] # void return type
        procParams.add assemblerDef        # a: Assembler[X86_64]

        # Workaround typechecking imm64: uint64 or pointer
        # we generate a pointer immediate proc separate from
        # the int procs
        var genPointerVersion = false

        # After the 32 and 64-bit immediate have been generated
        # we can generate a static int for literal that will dispatch
        # compile-time immediate to the best proc
        var genLiteralDispatch = false

        # 1. Iterate on the raw param
        for i, arg in params:
          if arg.kind == nnkCall:
            # type(64) case
            procParams.add types(arg)
          else:
            # dstXX, srcXX, immXX, adr, label case
            let mnemo = $arg
            let pre = mnemo[0..2]  # mnemo prefix

            if pre == "dst":
              dst = arg
              let suf = mnemo[3..^1]
              procParams.add arg.reg("RegX86_" & suf)
            elif pre == "src":
              src = arg
              let suf = mnemo[3..^1]
              procParams.add arg.reg("RegX86_" & suf)
            elif pre == "imm":
              imm = arg
              let suf = mnemo[3..^1]
              procParams.add arg.imm("uint" & suf, "int" & suf)
              if suf == "64":
                genPointerVersion = true
              if suf == "32" and params[i-1].eqIdent"dst64":
                genLiteralDispatch = true
            elif pre == "adr":
              adr = arg
              procParams.add arg.adr()
            elif pre == "lab":
              useLabels = true
              procParams.add arg.label()
              # 32-bit Placeholder for rel32 (jump target) or disp32 (RIP-relative addressing)
              bodySpecs.expectKind nnkBracket
              bodySpecs.add newLit 0x00, newLit 0x00, newLit 0x00, newLit 0x00
            else:
              error "Wrong mnemonic \"" & mnemo & "\"." &
                    " Only dst, src, imm (+size suffix)," &
                    " adr and label are allowed.", arg

        # 2. create the body
        var procBody = newStmtList()
        procBody.add procComment

        if bodySpecs.kind == nnkBracket:
          procBody.processSpecs(assemblerSym, dst, src, adr, bodySpecs)
        elif bodySpecs.kind == nnkInfix:
          assert not useLabels
          assert bodySpecs[0].eqIdent"&"
          procBody.processSpecs(assemblerSym, dst, src, adr, bodySpecs[1])
          procBody.add quote do:
            `assemblerSym`.code.add cast[array[sizeof(`imm`), byte]](`imm`)
        else:
          error "Unexpected opcode body", bodySpecs

        if useLabels:
          let label = newIdentNode"label"
          procBody.add quote do:
            `assemblerSym`.add_target `label`

        # 3. Put everything together
        result.add newInlineProc(procName, procParams, procBody)

        if genPointerVersion:
          # We replace the imm64: uint64 or int64 by imm64: pointer
          assert procParams[^1][0].eqIdent"imm64"
          # Deal with shallow copy semantics of NimNodes
          var procParamsPointer = procParams
          var pnode = procParamsPointer[^1].copy()
          pnode[1] = bindSym"pointer"
          procParamsPointer[^1] = pnode
          # Add the proc with pointer immediates
          result.add newInlineProc(procName, procParamsPointer, procBody)

        if genLiteralDispatch:
          # We generate a dispatch template that will check if the int literal
          # needs 64-bit opcode or 32-bit is enough
          result.add quote do:
            template `procName`*(a: var Assembler[X86_64], dst64: static RegX86_64, literal: static int) =
              when low(int32) < literal and literal < high(int32):
                a.`procName`(dst64, int32 literal)
              else:
                a.`procName`(dst64, int64 literal)

        # Reset state
        genPointerVersion = false
        genLiteralDispatch = false
        useLabels = false
        dst = nil
        src = nil
        adr = nil
        imm = nil

  # Check code generated
  # echo result.toStrLit()
