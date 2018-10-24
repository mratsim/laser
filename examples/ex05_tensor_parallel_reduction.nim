import ../laser/strided_iteration/[reduce_each, foreach]
import ../laser/tensor/[datatypes, allocator, initialization]
import ../laser/[compiler_optim_hints, dynamic_stack_arrays]
import ../laser/openmp
import sequtils, macros

proc foo[T](x, y: Tensor[T]): T =
  const
    max_threads = 22
    cache_line = 64
    padding = cache_line div sizeof(T)
  # We support up to 22 simultaneous threads
  # We pad the value by 64 bytes to avoid
  # false sharing/cache invalidation
  withCompilerOptimHints()
  var partial_reduce{.align_variable.}: array[max_threads * padding, T]

  var nb_chunks: Natural
  reduceEach nb_chunks, xi in x, yi in y:
    partial_reduce[omp_get_thread_num() * padding] += xi + yi

  for idx in 0 ..< nb_chunks:
    echo partial_reduce[idx * padding]
    result += partial_reduce[idx * padding]

  echo partial_reduce

proc toTensor[T](s: seq[T]): Tensor[T] =
  var size: int
  initTensorMetadata(result, size, [s.len])
  allocCpuStorage(result.storage, size)
  result.copyFromRaw(s[0].unsafeAddr, s.len)

let a = toSeq(1..10001).toTensor
let b = toSeq(-10000 .. 0).toTensor

echo foo(a, b)
