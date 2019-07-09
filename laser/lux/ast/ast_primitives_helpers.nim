import
  # Standard library
  random,
  # Internal
  ./ast_types

# ###########################################
#
#         Lux Primitive Helpers
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

proc assign*(lhs, rhs: LuxNode): LuxNode =
  # Generate the Assign node
  # This also scans the domain /loop nest needed
  # to generate the assignment
  LuxNode(
    id: genId(),
    kind: Assign,
    lval: lhs,
    rval: rhs
    # domains: TODO
  )

proc newSym*(symbol: string, rhs: LuxNode): LuxNode =
  # Declare and allocate a new AST symbol
  # This also scans the domain/loop nest needed
  # to generate the assignment
  assign(
    lhs = LuxNode(
        # 1. Declare unallocated lval
        id: genId(),
        kind: LValTensor,
        symLVal: symbol,
        version: 0,
        prev_version: nil
      ),
    rhs = rhs
  )

proc lvalify*(node: var LuxNode) =
  ## Allocate an expression result
  ## to a mutable memory location.
  ##
  ## Solve the case where we have:
  ##
  ## .. code::nim
  ##   var B = A
  ##   B += C
  ##
  ## B must be attributed a memory location (l-value)
  ## to be mutated
  ##
  ## Also in the case
  ## .. code::nim
  ##   var B = A
  ##   var B2 = A
  ##   B += C
  ##   B2 *= C
  ##
  ## we want to reuse the same computation A
  ## but B and B2 should have unique ID in the AST tree
  ##
  ## # Summary
  ##
  ## Unique:
  ##   - lval symbol
  ##
  ## Reused:
  ##   - operation/expression ID

  let lval_id = genId()
  let lval_symbol = "lval_" & $lval_id
  node = LuxNode(
    id: lval_id,
    kind: LValTensor, # TODO accept scalars
    symLVal: lval_symbol,
    version: 1,
    prev_version: newSym(lval_symbol, node)
  )

proc inputTensor*(paramId: int): LuxNode =
  LuxNode(
    id: genId(),
    kind: InTensor, symId: paramId
  )
