# Makefile для Assignment 4
# Гибридные и распределённые параллельные вычисления

# Компиляторы
NVCC = nvcc
MPICXX = mpic++

# Флаги
NVCCFLAGS = -O2 -arch=sm_50
CUDAOMP = -Xcompiler -fopenmp  # Для гибридной версии

# Исполняемые файлы
TASK1 = task1
TASK2 = task2
TASK3 = task3
TASK4 = task4

# CUDA задачи
CUDA_TASKS = $(TASK1) $(TASK2) $(TASK3)

# Правило по умолчанию - собрать CUDA задачи
all: cuda
	@echo ""
	@echo "✓ CUDA задачи скомпилированы!"
	@echo ""
	@echo "Запуск:"
	@echo "  ./$(TASK1) - Сумма массива"
	@echo "  ./$(TASK2) - Префиксная сумма"
	@echo "  ./$(TASK3) - Гибридная CPU+GPU"
	@echo ""
	@echo "Для MPI задачи:"
	@echo "  make mpi"
	@echo "  mpirun -np 4 ./$(TASK4)"

# Собрать CUDA задачи
cuda: $(CUDA_TASKS)

# Собрать MPI задачу
mpi: $(TASK4)
	@echo ""
	@echo "✓ MPI задача скомпилирована!"
	@echo ""
	@echo "Запуск:"
	@echo "  mpirun -np 2 ./$(TASK4)  # 2 процесса"
	@echo "  mpirun -np 4 ./$(TASK4)  # 4 процесса"
	@echo "  mpirun -np 8 ./$(TASK4)  # 8 процессов"

# Task 1: Сумма массива на GPU
$(TASK1): task1_array_sum.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Task 2: Префиксная сумма с shared memory
$(TASK2): task2_prefix_sum.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Task 3: Гибридная CPU+GPU
$(TASK3): task3_hybrid.cu
	$(NVCC) $(NVCCFLAGS) $(CUDAOMP) -o $@ $<

# Task 4: MPI распределенные вычисления
$(TASK4): task4_mpi.cpp
	$(MPICXX) -O2 -o $@ $<

# Очистка
clean:
	rm -f $(TASK1) $(TASK2) $(TASK3) $(TASK4)
	rm -f *.o
	@echo "✓ Очищено!"

# Запуск CUDA задач
run1: $(TASK1)
	./$(TASK1)

run2: $(TASK2)
	./$(TASK2)

run3: $(TASK3)
	./$(TASK3)

# Запуск MPI задачи с разным количеством процессов
run4-2: $(TASK4)
	@echo "=== Запуск с 2 процессами ==="
	mpirun -np 2 ./$(TASK4)

run4-4: $(TASK4)
	@echo "=== Запуск с 4 процессами ==="
	mpirun -np 4 ./$(TASK4)

run4-8: $(TASK4)
	@echo "=== Запуск с 8 процессами ==="
	mpirun -np 8 ./$(TASK4)

# Запуск всех MPI тестов
run4-all: $(TASK4)
	@echo ""
	@echo "=========================================="
	@echo "Тест 1: 2 процесса"
	@echo "=========================================="
	mpirun -np 2 ./$(TASK4)
	@echo ""
	@echo "=========================================="
	@echo "Тест 2: 4 процесса"
	@echo "=========================================="
	mpirun -np 4 ./$(TASK4)
	@echo ""
	@echo "=========================================="
	@echo "Тест 3: 8 процессов"
	@echo "=========================================="
	mpirun -np 8 ./$(TASK4)

# Справка
help:
	@echo "Доступные команды:"
	@echo ""
	@echo "Сборка:"
	@echo "  make        - собрать CUDA задачи (1-3)"
	@echo "  make cuda   - собрать CUDA задачи"
	@echo "  make mpi    - собрать MPI задачу (4)"
	@echo "  make clean  - удалить исполняемые файлы"
	@echo ""
	@echo "Запуск:"
	@echo "  make run1   - запустить Task 1 (сумма массива)"
	@echo "  make run2   - запустить Task 2 (prefix sum)"
	@echo "  make run3   - запустить Task 3 (гибридная)"
	@echo "  make run4-2 - запустить Task 4 с 2 процессами"
	@echo "  make run4-4 - запустить Task 4 с 4 процессами"
	@echo "  make run4-8 - запустить Task 4 с 8 процессами"
	@echo "  make run4-all - запустить Task 4 со всеми конфигурациями"
	@echo ""
	@echo "Требования:"
	@echo "  - NVIDIA CUDA Toolkit (для задач 1-3)"
	@echo "  - OpenMP (для задачи 3)"
	@echo "  - MPI (Open MPI или MPICH) (для задачи 4)"

.PHONY: all cuda mpi clean run1 run2 run3 run4-2 run4-4 run4-8 run4-all help
