# Vectorization and multiple SIMD supports

To support multiple SIMD paths with the library, something akin to GCC Function Multi Versioning would be very useful.

Note that this is not supported at all in Microsoft MSVC.

To reduce the cost of dispatching, the dispatch selection should be done very early after function call.
To avoid code duplication, only the inner loop with changed code
should be dispatched.

## Tradeoffs

### Indirect dispatch

To reconcile both needs (dispatch in outer part to avoid redundant code execution and dispatch in the inner part to avoid code duplication),
we can use a variable holding the SIMDized function pointer, if nil it will check the best implementation and store a pointer to it.
The inner loop will only refer to this function pointer.

Those inner loop functions can be tagged `hot` to move them
to hot locations in the binary and reduce the overhead of
repeated function calls in a loop.

#### Danger

The main issues are:
- register pressure
- stack overhead at function entry/cleanup with lots of arguments
- compiler cannot propagate optimizations like loop hoisting.
  This is particularly relevant to array indexing.
  For example to index a convolution through a NCHW tensor (Batch, Color, Height, Width):
  `w + Width * (h + Height * (c + Color * BatchSize))`
- Harder to predict. Note that there is no prefetch for instructions (https://stackoverflow.com/questions/48571263/bring-code-into-the-l1-instruction-cache-without-executing-it)

#### Benches

- https://gist.github.com/rianhunter/0be8dc116b120ad5fdd4
- https://stackoverflow.com/questions/10757167/do-function-pointers-force-an-instruction-pipeline-to-clear

Note that a function with an inner vectorized loop would be much more costly that the example given.

## Flexibility

A `dispatch_at` parameter might be useful to choose at which root/loop_level to do the SIMD detection/dispatch.

## References

- https://lwn.net/Articles/691932/
- https://gcc.gnu.org/wiki/FunctionMultiVersioning

## Limitations and caveats

- https://hannes.hauswedell.net/post/2017/12/09/fmv/
