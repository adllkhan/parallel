# Makefile для Assignment 3
# Архитектура GPU и оптимизация CUDA-программ

# Компилятор CUDA
NVCC = nvcc

# Флаги компиляции
NVCCFLAGS = -O2 -arch=sm_50

# Имена исполняемых файлов
TASK1 = task1
TASK2 = task2
TASK3 = task3
TASK4 = task4

# Все задачи
ALL_TASKS = $(TASK1) $(TASK2) $(TASK3) $(TASK4)

# Правило по умолчанию
all: $(ALL_TASKS)
	@echo ""
	@echo "✓ Все задачи скомпилированы!"
	@echo ""
	@echo "Запуск:"
	@echo "  ./$(TASK1) - Глобальная vs разделяемая память"
	@echo "  ./$(TASK2) - Влияние размера блока"
	@echo "  ./$(TASK3) - Коалесцированный доступ"
	@echo "  ./$(TASK4) - Оптимизация конфигурации"

# Task 1: Глобальная vs разделяемая память
$(TASK1): task1_global_vs_shared.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Task 2: Влияние размера блока
$(TASK2): task2_block_size.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Task 3: Коалесцированный доступ
$(TASK3): task3_coalesced_access.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Task 4: Оптимизация конфигурации
$(TASK4): task4_optimization.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Очистка
clean:
	rm -f $(ALL_TASKS)
	rm -f *.o
	@echo "✓ Очищено!"

# Запуск всех задач
run-all: all
	@echo "=========================================="
	@echo "Запуск Task 1: Глобальная vs Shared память"
	@echo "=========================================="
	./$(TASK1)
	@echo ""
	@echo "=========================================="
	@echo "Запуск Task 2: Влияние размера блока"
	@echo "=========================================="
	./$(TASK2)
	@echo ""
	@echo "=========================================="
	@echo "Запуск Task 3: Коалесцированный доступ"
	@echo "=========================================="
	./$(TASK3)
	@echo ""
	@echo "=========================================="
	@echo "Запуск Task 4: Оптимизация конфигурации"
	@echo "=========================================="
	./$(TASK4)

# Отдельные запуски
run1: $(TASK1)
	./$(TASK1)

run2: $(TASK2)
	./$(TASK2)

run3: $(TASK3)
	./$(TASK3)

run4: $(TASK4)
	./$(TASK4)

# Справка
help:
	@echo "Доступные команды:"
	@echo "  make         - скомпилировать все задачи"
	@echo "  make clean   - удалить исполняемые файлы"
	@echo "  make run-all - запустить все задачи"
	@echo "  make run1    - запустить Task 1"
	@echo "  make run2    - запустить Task 2"
	@echo "  make run3    - запустить Task 3"
	@echo "  make run4    - запустить Task 4"
	@echo "  make help    - показать эту справку"
	@echo ""
	@echo "Требования:"
	@echo "  - NVIDIA CUDA Toolkit"
	@echo "  - GPU с compute capability >= 5.0"

.PHONY: all clean run-all run1 run2 run3 run4 help
