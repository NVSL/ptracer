# PTracer offline stage: ptrecon (Intel PT decode + VEX skeleton interpretation), pt_capture2
# (perf_event Intel PT capture with a sideband), ptpktscan (packet counts of a saved AUX file).
#
# Dependencies: libipt, Zydis (static library from the E9Patch tree) and libpyvex (from the Python
# environment that has pyvex installed).  ptrecon links the patched private libipt that
# third_party/libipt/build.sh installs into $(LIBIPT) (ptracer/build.sh does that first; rpath, never
# system-wide); pt_capture2 and ptpktscan link the system libipt.  Every path below can be overridden
# on the command line or in the environment:
#   PY     python interpreter whose `pyvex' package supplies libpyvex (default: python3)
#   PYVEX  the pyvex package directory (default: asked from $(PY))
#   ZYDIS  the Zydis source tree with libZydis.a built (default: ../../third_party/e9patch/contrib/zydis)
#   LIBIPT the private libipt prefix (default: ./libipt)
PY    ?= python3
PYVEX ?= $(shell $(PY) -c 'import pyvex,os;print(os.path.dirname(pyvex.__file__))')
ZYDIS ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../third_party/e9patch/contrib/zydis)
LIBIPT ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/libipt)
CXX   ?= g++
CC    ?= gcc
CXXFLAGS := -std=c++17 -O2 -g -Wall -Wno-unused-function -I$(PYVEX)/include -I$(ZYDIS)/include -I$(ZYDIS)/dependencies/zycore/include -DZYDIS_STATIC_BUILD -DZYCORE_STATIC_BUILD -I$(LIBIPT)/include -DPTRECON_LIBIPT_PATCHED
LDFLAGS  := -L$(PYVEX)/lib -lpyvex -Wl,-rpath,$(PYVEX)/lib $(ZYDIS)/libZydis.a -L$(LIBIPT)/lib -Wl,-rpath,$(LIBIPT)/lib -lipt

all: ptrecon pt_capture2 ptpktscan hifirecon tests/vextest tests/gtstream_test

HDRS := vexlift.h vexinterp.h vexval.h ptdecode.h decode_progress.h recon.h jitcode.h e9phase.h json.h trace_format.h \
        gtstream.h recwriter.h shadowmap.h cjit.h ../runtime/rt/ptgt_format.h

# ptrecon compiles hot skeleton programs to native code at run time with gcc (cjit.cpp; PTRECON_CJIT=0 disables it).
ptrecon: main.o recon.o recon_seed.o recon_par.o mtdemux.o ptdecode.o pipedec.o e9phase.o jitcode.o vexinterp.o vexlift.o cjit.o
	$(CXX) -o $@ $^ $(LDFLAGS)

# hifirecon: the HiFi (Pin JIT) reconstructor.  It replays the Pintool's control-flow log and
# critical values through the same VEX interpreter (no Intel PT decode), so it needs only the
# interpreter objects.
hifirecon: hifirecon.cpp vexinterp.o vexlift.o cjit.o $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ hifirecon.cpp vexinterp.o vexlift.o cjit.o $(LDFLAGS)

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# pt_capture2: raw perf_event_open Intel PT capture of one command (per-task, or per-CPU with
# --cpu), with the draining AUX writer, the /proc/PID/maps sideband and the spare-TCB hand-off
# to new threads of a buffer-sink runtime (../runtime/rt/ptlog_abi.h).
pt_capture2: pt_capture2.c ../runtime/rt/ptlog_abi.h
	$(CC) -O2 -Wall -Wno-stringop-truncation -o $@ $< -lipt -lpthread

# ptpktscan: count PT packet types in an AUX file that is already on disk.  A capture taken with
# --no-decode writes the AUX and no packet summary, and a capture can have lost = 0 /
# truncated = 0 (the drain kept up) and still be full of on-chip OVF packets (the packet
# generator did not).  OVF is trace loss, so it has to be counted.
ptpktscan: ptpktscan.c
	$(CC) -O2 -Wall -o $@ $< -lipt

# Hardware-free unit tests: the VEX lifter/interpreter on hand-assembled bytes, and the
# ground-truth stream reader on synthetic records.
tests/vextest: tests/vextest.cpp vexinterp.o vexlift.o cjit.o
	$(CXX) $(CXXFLAGS) -o $@ $< vexinterp.o vexlift.o cjit.o $(LDFLAGS)

tests/gtstream_test: tests/gtstream_test.cpp gtstream.h ../runtime/rt/ptgt_format.h
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -o $@ $<

test: tests/vextest tests/gtstream_test
	tests/gtstream_test && tests/vextest

clean:
	rm -f *.o ptrecon pt_capture2 ptpktscan hifirecon tests/vextest tests/gtstream_test
.PHONY: all clean test
