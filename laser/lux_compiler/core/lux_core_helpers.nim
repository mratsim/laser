# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  random,
  # Internal
  ./lux_types

# ###########################################
#
#         Lux Core Helpers
#
# ###########################################

var luxNodeRngCT {.compileTime.} = initRand(0x42)
  ## Workaround for having no UUID for LuxNodes
  ## at compile-time - https://github.com/nim-lang/RFCs/issues/131

var luxNodeRngRT = initRand(0x42)
  ## Runtime ID

proc genId*(): int =
  when nimvm:
    luxNodeRngCT.rand(high(int))
  else:
    luxNodeRngRT.rand(high(int))

const ScalarExpr = {
            IntImm, FloatImm, IntParam, FloatParam,
            IntMut, FloatMut, IntLVal, FloatLVal,
            BinOp,
            Access, Shape, Domain
          }

func checkScalarExpr*(targetOp: string, input: LuxNode) =
  # TODO - mapping LuxNode -> corresponding function
  # for better errors as LuxNode are "low-level"
  if input.kind notin ScalarExpr:
    raise newException(
      ValueError,
      "Invalid scalar expression \"" & $input.kind & "\"\n" &
      "Only the following LuxNodes are allowed:\n    " & $ScalarExpr &
      "\nfor building a \"" & targetOp & "\" function."
      # "\nOrigin: " & $node.lineInfo
    )

func checkMutable*(node: LuxNode) =
  # TODO
  discard

func checkTensor*(node: LuxNode) =
  if node.kind notin {InTensor, LValTensor, MutTensor}:
    raise newException(
      ValueError,
      "Invalid tensor node \"" & $node.kind & "\""
    )
