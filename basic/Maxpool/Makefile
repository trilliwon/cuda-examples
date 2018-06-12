# USAGE: make all
#
# Change the variables to point to the compiler location.

NVCC_FLAGS := -std=c++11
NVCC := /usr/local/cuda/bin/nvcc
LD_LIBRARY_PATH := /usr/local/cuda/lib64
TARGET = maxpool

all: $(TARGET) 

$(TARGET): 
	$(NVCC) -o $(TARGET) $(TARGET).cu $(NVCC_FLAGS)

clean:
	rm -rf $(TARGET) 
