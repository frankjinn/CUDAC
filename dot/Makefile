CC=g++
NVCC=nvcc

all: dot
	@./dot

dot: dot.o
	@nvcc -o dot dot.o

dot.o: dot.cu
	@nvcc -std=c++11 -c dot.cu

clean:
	@echo "removing all .o files"
	@rm -rf *.o

.PHONY: all run