CC=g++
NVCC=nvcc

all: rayTrace
	@./rayTrace

rayTrace: rayTrace.o
	@nvcc -o rayTrace rayTrace.o -lGL -lGLU -lglut

rayTrace.o: rayTrace.cu
	@nvcc -std=c++11 -c rayTrace.cu

clean:
	@echo "removing all .o files"
	@rm -rf *.o

.PHONY: all run