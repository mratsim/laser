# ################################################################
#
#      Example of accessing CPU information at runtime
#
# ################################################################

import ../laser/cpuinfo

echo "cpuinfo_get_processors - ", cpuinfo_get_processors()[]
echo "cpuinfo_get_cores - ", cpuinfo_get_cores()[]
echo "cpuinfo_get_clusters - ", cpuinfo_get_clusters()[]
echo "cpuinfo_get_packages - ", cpuinfo_get_packages()[]
echo "cpuinfo_get_l1i_caches - ", cpuinfo_get_l1i_caches()[]
echo "cpuinfo_get_l1d_caches - ", cpuinfo_get_l1d_caches()[]
echo "cpuinfo_get_l2_caches - ", cpuinfo_get_l2_caches()[]
echo "cpuinfo_get_l3_caches - ", cpuinfo_get_l3_caches()[]
# echo "cpuinfo_get_l4_caches - ", cpuinfo_get_l4_caches()[]
when false:
  echo "cpuinfo_get_get_processor0 - ", cpuinfo_get_processor(index = 0)[]
  echo "cpuinfo_get_get_core0 - ", cpuinfo_get_core(index = 0)[]
  echo "cpuinfo_get_get_cluster0 - ", cpuinfo_get_cluster(index = 0)[]
  echo "cpuinfo_get_get_package0 - ", cpuinfo_get_package(index = 0)[]
  echo "cpuinfo_get_l1i_cache0 - ", cpuinfo_get_l1i_cache(index = 0)[]
  echo "cpuinfo_get_l1d_cache0 - ", cpuinfo_get_l1d_cache(index = 0)[]
  echo "cpuinfo_get_l2_cache0 - ", cpuinfo_get_l2_cache(index = 0)[]
  echo "cpuinfo_get_l3_cache0 - ", cpuinfo_get_l3_cache(index = 0)[]
  # echo "cpuinfo_get_l4_cache - ", cpuinfo_get_l4_cache(index = 0)[]
echo "cpuinfo_get_processors_count - ", cpuinfo_get_processors_count()
echo "cpuinfo_get_cores_count - ", cpuinfo_get_cores_count()
echo "cpuinfo_get_clusters_count - ", cpuinfo_get_clusters_count()
echo "cpuinfo_get_packages_count - ", cpuinfo_get_packages_count()
echo "cpuinfo_get_l1i_caches_count - ", cpuinfo_get_l1i_caches_count()
echo "cpuinfo_get_l1d_caches_count - ", cpuinfo_get_l1d_caches_count()
echo "cpuinfo_get_l2_caches_count - ", cpuinfo_get_l2_caches_count()
echo "cpuinfo_get_l3_caches_count - ", cpuinfo_get_l3_caches_count()
# echo "cpuinfo_get_l4_caches_count - ", cpuinfo_get_l4_caches_count()
when defined(linux):
  echo "cpuinfo_get_current_processor - ", cpuinfo_get_current_processor()[]
  echo "cpuinfo_get_current_core - ", cpuinfo_get_current_core()[]

# Result on i5-5257U (mobile broadwell with HyperThreading)

# cpuinfo_get_processors - (smt_id: 0, core: ..., cluster: ..., package: ..., cache: ...)
# cpuinfo_get_cores - (processor_start: 0, processor_count: 2, core_id: 0, cluster: ..., package: ..., vendor: cpuinfo_vendor_intel, uarch: cpuinfo_uarch_broadwell, cpuid: 198356, frequency: 0)
# cpuinfo_get_clusters - (processor_start: 0, processor_count: 4, core_start: 0, core_count: 2, cluster_id: 0, package: ..., vendor: cpuinfo_vendor_intel, uarch: cpuinfo_uarch_broadwell, cpuid: 198356, frequency: 0)
# cpuinfo_get_packages - (name: ['I', 'n', 't', 'e', 'l', ' ', 'C', 'o', 'r', 'e', ' ', 'i', '5', '-', '5', '2', '5', '7', 'U', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00'], processor_start: 0, processor_count: 4, core_start: 0, core_count: 2, cluster_start: 0, cluster_count: 1)
# cpuinfo_get_l1i_caches - (size: 32768, associativity: 8, sets: 64, partitions: 1, line_size: 64, flags: 0, processor_start: 0, processor_count: 2)
# cpuinfo_get_l1d_caches - (size: 32768, associativity: 8, sets: 64, partitions: 1, line_size: 64, flags: 0, processor_start: 0, processor_count: 2)
# cpuinfo_get_l2_caches - (size: 262144, associativity: 8, sets: 512, partitions: 1, line_size: 64, flags: 1, processor_start: 0, processor_count: 2)
# cpuinfo_get_l3_caches - (size: 3145728, associativity: 12, sets: 4096, partitions: 1, line_size: 64, flags: 7, processor_start: 0, processor_count: 4)
# cpuinfo_get_processors_count - 4
# cpuinfo_get_cores_count - 2
# cpuinfo_get_clusters_count - 1
# cpuinfo_get_packages_count - 1
# cpuinfo_get_l1i_caches_count - 2
# cpuinfo_get_l1d_caches_count - 2
# cpuinfo_get_l2_caches_count - 2
# cpuinfo_get_l3_caches_count - 1
