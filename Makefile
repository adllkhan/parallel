# Makefile для Assignment 2
# Компиляция задач по параллельным вычислениям

# Компиляторы
CXX = g++
NVCC = nvcc

# Флаги компиляции
CXXFLAGS = -Wall -O2
OPENMP_FLAGS = -fopenmp

# Имена исполняемых файлов
TASK2 = task2_openmp
TASK3 = task3_selection_sort
TASK4 = task4_cuda_sort

# Правило по умолчанию - собрать OpenMP задачи
all: openmp

# Собрать только OpenMP задачи (Task 2 и Task 3)
openmp: $(TASK2) $(TASK3)
	@echo ""
	@echo "OpenMP задачи скомпилированы успешно!"
	@echo "Запуск:"
	@echo "  ./$(TASK2)"
	@echo "  ./$(TASK3)"

# Собрать CUDA задачу (Task 4)
cuda: $(TASK4)
	@echo ""
	@echo "CUDA задача скомпилирована успешно!"
	@echo "Запуск: ./$(TASK4)"

# Task 2: Поиск минимума и максимума
$(TASK2): task2_openmp.cpp
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $<

# Task 3: Сортировка выбором
$(TASK3): task3_selection_sort.cpp
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $<

# Task 4: CUDA сортировка слиянием
$(TASK4): task4_cuda_merge_sort.cu
	$(NVCC) -o $@ $<

# Очистка
clean:
	rm -f $(TASK2) $(TASK3) $(TASK4)
	rm -f *.o
	@echo "Очищено!"

# Запуск Task 2
run2: $(TASK2)
	./$(TASK2)

# Запуск Task 3
run3: $(TASK3)
	./$(TASK3)

# Запуск Task 4
run4: $(TASK4)
	./$(TASK4)

# Справка
help:
	@echo "Доступные команды:"
	@echo "  make         - скомпилировать OpenMP задачи (Task 2 и 3)"
	@echo "  make openmp  - скомпилировать OpenMP задачи"
	@echo "  make cuda    - скомпилировать CUDA задачу (Task 4)"
	@echo "  make clean   - удалить исполняемые файлы"
	@echo "  make run2    - запустить Task 2"
	@echo "  make run3    - запустить Task 3"
	@echo "  make run4    - запустить Task 4"
	@echo "  make help    - показать эту справку"

.PHONY: all openmp cuda clean run2 run3 run4 help
