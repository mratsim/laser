# Apache v2 License
# Mamy Ratsimbazafy

# ##########################################
import
  ../../laser/strided_iteration/foreach,
  ../../laser/tensor/[datatypes, initialization],
  ../../laser/dynamic_stack_arrays

proc buildTensorView*(x: seq[float32], M, N: int): Tensor[float32] =
  var size: int
  initTensorMetadata(result, size, [M, N])

  result.storage = CpuStorage[float32](
    raw_buffer: cast[ptr UncheckedArray[float32]](x[0].unsafeAddr),
    memalloc: nil,
    memowner: false
  )

proc transpose_naive_forEach*(dst: var Tensor[float32], src: Tensor[float32]) =
  forEachStrided d in dst, s in src:
    d = s
