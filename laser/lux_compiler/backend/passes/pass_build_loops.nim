# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard Library
  tables,
  # Internal
  ../../core/lux_types,
  # Debug
  ../../core/lux_print

# ###########################################
#
#            Pass: Build Loop
#
# ###########################################

# Lower the high-level loop-nest representation
# to a low-level one.
#
# In the future this will take scheduling information
# like parallel, unroll or vectorize

proc passBuildLoop(ast: LuxNode, visited: var Table[Id, LuxNode]): LuxNode =
  proc inspect(node: LuxNode): LuxNode =
    case node.kind:
    of InTensor, IntParam, FloatParam:
      return node
    of MutTensor, LValTensor, IntMut, FloatMut, IntLVal, FloatLVal:
      new result
      result.kind = node.kind
      result.symLVal = node.symLval
      result.version = node.version
      result.prev_version = inspect(node.prev_version)
      return
    of IntImm, FloatImm:
      return node
    of Assign:
      if node.id in visited:
        return visited[node.id]
