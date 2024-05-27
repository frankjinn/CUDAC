CC=g++
NVCC=nvcc

all: juliaSetVisual
	@./juliaSetVisual

juliaSetVisual: juliaSetVisual.o
	@nvcc -o juliaSetVisual juliaSetVisual.o -lGL -lGLU -lglut

juliaSetVisual.o: juliaSetVisual.cu
	@nvcc -std=c++11 -c juliaSetVisual.cu

clean:
	@echo "removing all .o files"
	@rm -rf *.o

.PHONY: all run