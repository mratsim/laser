# ################################################################
#
#                Example of using the fine-grained
#                 parallel forEachStaged iterator
#     for parallel reduction on arbitrary number of tensors
#
# ################################################################

import ../laser/strided_iteration/[foreach, foreach_staged]
import ../laser/tensor/[datatypes, allocator, initialization]
import ../laser/[compiler_optim_hints, dynamic_stack_arrays]
import ../laser/[openmp, cpuinfo]
import sequtils

proc reduction_localsum_critical[T](x, y: Tensor[T]): T =
  forEachStaged xi in x, yi in y:
    openmp_config:
      use_openmp: true
      use_simd: false
      nowait: true
      omp_grain_size: OMP_MEMORY_BOUND_GRAIN_SIZE
    iteration_kind:
      {contiguous, strided} # Default, "contiguous", "strided" are also possible
    before_loop:
      var local_sum = 0.T
    in_loop:
      local_sum += xi + yi
    after_loop:
      omp_critical:
        result += local_sum

proc reduction_localsum_omp_atomic[T](x, y: Tensor[T]): T =
  forEachStaged xi in x, yi in y:
    openmp_config:
      use_simd: false
      nowait: true
    iteration_kind:
      contiguous
    before_loop:
      var local_sum = 0.T
    in_loop:
      local_sum += xi + yi
    after_loop:
      {.emit: "#pragma omp atomic".}
      {.emit: "`result` += `local_sum`;".}

proc reduction_localsum_system_atomic[T](x, y: Tensor[T]): T =
  forEachStaged xi in x, yi in y:
    openmp_config:
      use_simd: false
      nowait: true
    iteration_kind:
      contiguous
    before_loop:
      var local_sum = 0.T
    in_loop:
      local_sum += xi + yi
    after_loop:
      result.atomicInc local_sum # This requires --threads:on

proc reduction_padding[T](x, y: Tensor[T]): T =
  let cache_line_size = int cpuinfo_get_l1d_caches().line_size
  let padding = cache_line_size div sizeof(T)
  var buffer: seq[T]

  forEachStaged xi in x, yi in y:
    openmp_config:
      use_simd: false
      nowait: true
    iteration_kind:
      contiguous
    before_loop:
      omp_master:
        buffer = newSeq[T](omp_get_num_threads() * padding)
      omp_barrier()
      omp_flush(buffer)
    in_loop:
      buffer[omp_get_thread_num() * padding] += xi + yi

  for idx in countup(0, buffer.len - 1, padding):
    result += buffer[idx]

proc toTensor[T](s: seq[T]): Tensor[T] =
  var size: int
  initTensorMetadata(result, size, [s.len])
  allocCpuStorage(result.storage, size)
  result.copyFromRaw(s[0].unsafeAddr, s.len)

let a = toSeq(1..10001).toTensor
let b = toSeq(-10000 .. 0).toTensor

echo reduction_localsum_critical(a, b)
echo reduction_localsum_omp_atomic(a, b)
echo reduction_localsum_system_atomic(a, b)
echo reduction_padding(a, b)
