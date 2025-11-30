# --- Compiler and Flags ---
CXX := g++
CXXFLAGS := -std=c++17 -O3 -pthread
HIPFLAGS := $(CXXFLAGS) -ffast-math --offload-arch=gfx908

# --- Target Names ---
EXECUTABLES := hw5 nbody

# --- Primary Targets ---
.PHONY: all
all: hw5

# --- GPU Targets ---
hw5: hw5.cpp
	hipcc $(HIPFLAGS) -o hw5 hw5.cpp

.PHONY: hw5

# --- Sequential Target ---
# nbody: nbody.cc
# 	$(CXX) $(CXXFLAGS) -o nbody nbody.cc

# .PHONY: seq
# seq: nbody

# --- Cleanup ---
.PHONY: clean
clean:
	rm -f $(EXECUTABLES)
