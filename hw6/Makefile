# USAGE: make all
#
# Change the variables to point to the compiler location.

NVCC_FLAGS := -std=c++11
NVCC := /usr/local/cuda/bin/nvcc
LD_LIBRARY_PATH := /usr/local/cuda/lib64

all: hw6

hw6: 
	$(NVCC) -o hw6 hw6.cu $(NVCC_FLAGS)

clean:
	rm -rf hw6
