#!/usr/bin/env bash
# Private libipt for ptrecon:  bash third_party/libipt/build.sh PREFIX     (installs PREFIX/{include,lib}; never system-wide)
# libipt 2.0.6 (libipt-2.0.6.tar.gz = github.com/intel/libipt v2.0.6) + ptracer.patch:
#   * pt_image_find asks a read-memory callback (JIT code) before its section walk;
#   * a large section's block cache is calloc'ed instead of malloc + memset;
#   * a taken conditional branch's displacement is cached per block-cache entry;
#   * ptracer_ild_insn exports libipt's length decoder for one instruction.
# Built without libipt's section mutexes (FEATURE_THREADS=OFF): ptrecon's parallelism is fork().
# pt_capture2 and ptpktscan keep using the system libipt.
set -eu
P="$(realpath -m "$1")"; H="$(cd "$(dirname "$0")" && pwd)"
S="$P/src"; rm -rf "$S"; mkdir -p "$S"
tar -C "$S" -xzf "$H/libipt-2.0.6.tar.gz"
patch -d "$S/libipt-2.0.6" -p1 --quiet < "$H/ptracer.patch"
cmake -S "$S/libipt-2.0.6" -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DFEATURE_THREADS=OFF \
      -DPT_VERSION_EXT="-ptracer-bcalloc" -DCMAKE_INSTALL_PREFIX="$P" > /dev/null
cmake --build "$S/build" -j "${JOBS:-$(nproc)}" > /dev/null
cmake --install "$S/build" > /dev/null
rm -rf "$S"
ls "$P/lib"/libipt.so.2
