# ################################################################
#
#                Example of using a parallel reduction
#                    primitives on a raw buffer
#
# ################################################################

import
  random, sequtils,
  ../laser/primitives/reductions

proc main() =
  let interval = -1f .. 1f
  let size = 10_000_000
  let buf = newSeqWith(size, rand(interval))

  echo reduce_max(buf[0].unsafeAddr, size)

main()
