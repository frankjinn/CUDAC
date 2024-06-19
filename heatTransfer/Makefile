CC=g++
NVCC=nvcc

all: heatTransfer
	@./heatTransfer

heatTransfer: heatTransfer.o
	@nvcc -o heatTransfer heatTransfer.o -lGL -lGLU -lglut

heatTransfer.o: heatTransfer.cu
	@nvcc -std=c++11 -c heatTransfer.cu

clean:
	@echo "removing all .o files"
	@rm -rf *.o

.PHONY: all run