CC=g++
NVCC=nvcc
FILENAME := add
all: ${FILENAME}
	@./${FILENAME}

${FILENAME}: ${FILENAME}.o
	@nvcc -o ${FILENAME} ${FILENAME}.o

${FILENAME}.o: ${FILENAME}.cu
	@nvcc -std=c++11 -c ${FILENAME}.cu

clean:
	@echo "removing all .o files"
	@rm -rf *.o

.PHONY: all run