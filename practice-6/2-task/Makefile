# Makefile для матричного умножения OpenCL (macOS Apple Silicon)

CC = clang
CFLAGS = -Wall -O0
OPENCL_FLAGS = -framework OpenCL

TARGET = matrix_multiply

.PHONY: all clean run

all: $(TARGET)

$(TARGET): matrix_multiply.c matrix_mul_kernel.cl
	$(CC) $(CFLAGS) $(OPENCL_FLAGS) matrix_multiply.c -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
