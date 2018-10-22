# Sum reduction benchmark

See Stepanov test https://blogs.fau.de/hager/archives/7658
and Agner Fog instructions table: https://www.agner.org/optimize/instruction_tables.pdf


On Xeon E5-2660v2 Ivy Bridge at fixed 2.2GHz:
  - 8.8 GFlop/s for optimal code
  - 2.93 GFlop/s with vectorization and no unrolling
  - 2.2 GFlop/s with 3 way unrolling and no vectorization
  - 733 MFlop/s when standard compliant (i.e. no FP add reordering)

Note:
  Before Skylake "FP Add" latency was 3 so unrolling should be 3
  After Skylake latency is 4 but 2 adds per second can be started (throughput 2)
