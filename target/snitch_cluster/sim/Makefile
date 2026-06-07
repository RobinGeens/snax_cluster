# memsim — standalone host build (no podman, no license).
# Uses the C++17 compiler on PATH (CXX, default g++).
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter
SRC      := $(wildcard src/*.cpp)
HDR      := $(wildcard src/*.hpp)
BIN      := ../bin/snitch_cluster.memsim

.PHONY: all clean
all: $(BIN)

$(BIN): $(SRC) $(HDR)
	@mkdir -p ../bin
	$(CXX) $(CXXFLAGS) $(SRC) -o $@
	@echo "built $@"

clean:
	rm -f $(BIN)
