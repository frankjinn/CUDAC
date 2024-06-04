CC=g++
NVCC=nvcc

all: ripple
	@./ripple

ripple: ripple.o
	@nvcc -o ripple ripple.o -lGL -lGLU -lglut

ripple.o: ripple.cu
	@nvcc -std=c++11 -c ripple.cu

clean:
	@echo "removing all .o files"
	@rm -rf *.o

.PHONY: all run