import ../laser/strided_iteration/[reduce_each, foreach]
import ../laser/tensor/[datatypes, allocator, initialization]
import ../laser/[compiler_optim_hints, dynamic_stack_arrays]
import ../laser/openmp
import sequtils

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
    result += partial_reduce[idx * padding]
  echo "foo partial_reduce: ", partial_reduce

proc toTensor[T](s: seq[T]): Tensor[T] =
  var size: int
  initTensorMetadata(result, size, [s.len])
  allocCpuStorage(result.storage, size)
  result.copyFromRaw(s[0].unsafeAddr, s.len)

let a = toSeq(1..10001).toTensor
let b = toSeq(-10000 .. 0).toTensor

echo foo(a, b)

# The following doesn't work because Nim OpenMP clause
# uses "#pragma omp parallel for" directly
# instead of "#pragma omp for"
#
# This means that each thread will do the same work
# Final result will be nb_threads * real_result
#
# See feature request: https://github.com/nim-lang/Nim/issues/9490

# proc foo_alt[T](x, y: Tensor[T]): T =
#   const
#     max_threads = 22
#     cache_line = 64
#     padding = cache_line div sizeof(T)
#   withCompilerOptimHints()
#   var nb_chunks: Natural
#   omp_parallel:
#     var local_sum = T(0)
#     reduceEach nb_chunks, xi in x, yi in y:
#       local_sum += xi + yi
#     omp_critical:
#       result += local_sum
#
# echo foo_alt(a, b)
