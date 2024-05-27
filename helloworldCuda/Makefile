CC=g++
NVCC=nvcc

all: helloworld
	@./helloworld

helloworld: helloworld.o
	@nvcc -o helloworld helloworld.o

helloworld.o: helloworld.cu
	@nvcc -std=c++11 -c helloworld.cu

clean:
	@echo "removing all .o files"
	@rm -rf *.o

.PHONY: all run