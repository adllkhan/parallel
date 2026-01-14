# Makefile для OpenCL программы (macOS Apple Silicon)

CC = clang
CFLAGS = -Wall -O0  # -O0 для корректного измерения времени CPU
OPENCL_FLAGS = -framework OpenCL

# Цели
TARGET = opencl_vector_add

.PHONY: all clean run

all: $(TARGET)

$(TARGET): opencl_vector_add.c kernel.cl
	$(CC) $(CFLAGS) $(OPENCL_FLAGS) opencl_vector_add.c -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
