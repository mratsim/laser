import ../laser/strided_iteration/foreach_reduce
import ../laser/tensor/[datatypes, initialization]
import ../laser/[compiler_optim_hints, dynamic_stack_arrays]
import sequtils

proc dot[T](x, y: Tensor[T]): T =
  const
    MAX_THREADS = 22
    CACHE_LINE = 64
  # We support up to 22 simultaneous threads
  # We pad the value by 64 bytes to avoid
  # false sharing/cache invalidation
  withCompilerOptimHints()
  var
    nb_chunks: Natural
    partial_reduce{.align_variable.}: array[MAX_THREADS * CACHE_LINE div sizeof(T), T]
  forEachReduce nb_chunks, chunk_id, xi in x, yi in y:
    partial_reduce[CACHE_LINE * chunk_id] += xi * yi
