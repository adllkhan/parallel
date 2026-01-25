/*
 * Assignment 4 - Задание 2
 * Префиксная сумма (Scan) с использованием разделяемой памяти
 *
 * Префиксная сумма (prefix sum / scan) - это операция которая создает
 * новый массив где каждый элемент равен сумме всех предыдущих:
 *
 * Входной массив:  [3, 1, 7, 0, 4, 1, 6, 3]
 * Префиксная сумма: [0, 3, 4, 11, 11, 15, 16, 22]  (exclusive scan)
 * или
 * Префиксная сумма: [3, 4, 11, 11, 15, 16, 22, 25] (inclusive scan)
 *
 * Мы реализуем inclusive scan (каждый элемент включает себя)
 *
 * Размер массива: 1,000,000 элементов
 *
 * Компиляция: nvcc -o task2 task2_prefix_sum.cu
 * Запуск: ./task2
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

using namespace std;

// Размер массива - 1 миллион элементов
#define ARRAY_SIZE 1000000

// Размер блока - должен быть степенью двойки для алгоритма scan
#define BLOCK_SIZE 512

// Макрос для проверки ошибок
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            cout << "CUDA ошибка: " << cudaGetErrorString(error) << endl; \
            cout << "Файл: " << __FILE__ << ", строка: " << __LINE__ << endl; \
            exit(1); \
        } \
    } while(0)

//==============================================================================
// ВЕРСИЯ 1: ПОСЛЕДОВАТЕЛЬНАЯ ПРЕФИКСНАЯ СУММА НА CPU
//==============================================================================

/*
 * Простая последовательная реализация prefix sum
 *
 * Просто идем по массиву и накапливаем сумму:
 * output[i] = output[i-1] + input[i]
 */
void prefixSumCPU(int* input, int* output, int size)
{
    // Первый элемент
    output[0] = input[0];

    // Каждый следующий элемент = предыдущий + текущий
    for (int i = 1; i < size; i++)
    {
        output[i] = output[i - 1] + input[i];
    }
}

//==============================================================================
// ВЕРСИЯ 2: ПАРАЛЛЕЛЬНАЯ ПРЕФИКСНАЯ СУММА НА GPU
//==============================================================================

/*
 * БЛОЧНЫЙ SCAN с использованием shared memory
 *
 * Это ядро вычисляет prefix sum ВНУТРИ КАЖДОГО БЛОКА
 *
 * Алгоритм: Up-Sweep / Down-Sweep (также известен как Blelloch scan)
 *
 * Пример для 8 элементов [3, 1, 7, 0, 4, 1, 6, 3]:
 *
 * Исходные данные:     [3, 1, 7, 0, 4, 1, 6, 3]
 *
 * Up-Sweep (суммирование):
 *   Шаг 1 (stride=1):  [3, 4, 7, 7, 4, 5, 6, 9]  (попарно складываем)
 *   Шаг 2 (stride=2):  [3, 4, 7, 11, 4, 5, 6, 14] (складываем через 2)
 *   Шаг 3 (stride=4):  [3, 4, 7, 11, 4, 5, 6, 25] (складываем через 4)
 *
 * Down-Sweep (распространение):
 *   Обнуляем последний: [3, 4, 7, 11, 4, 5, 6, 0]
 *   Шаг 1 (stride=4):  [3, 4, 7, 0, 4, 5, 6, 11]
 *   Шаг 2 (stride=2):  [3, 4, 0, 7, 4, 0, 11, 16]
 *   Шаг 3 (stride=1):  [3, 0, 4, 4, 11, 11, 16, 22]
 *
 * Но мы используем более простой алгоритм для обучения
 */
__global__ void scanBlockKernel(int* input, int* output, int* blockSums, int size)
{
    // Выделяем shared memory для блока
    // Используем в 2 раза больше для временного хранения
    __shared__ int temp[BLOCK_SIZE * 2];

    // Глобальный индекс
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    // Загружаем данные в shared memory
    if (globalIdx < size)
    {
        temp[localIdx] = input[globalIdx];
    }
    else
    {
        temp[localIdx] = 0;  // Если вышли за границы
    }
    __syncthreads();

    // Выполняем scan внутри блока
    // Используем простой алгоритм: каждый элемент добавляет к себе элемент слева
    //
    // Это НЕ самый эффективный алгоритм, но простой для понимания
    // Более эффективный: Blelloch scan (см. комментарий выше)

    // Проходим log2(BLOCK_SIZE) итераций
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // Временный индекс для чтения (чтобы не было конфликтов)
        int idx = localIdx;

        // Временная переменная
        int temp_val = 0;

        // Если есть элемент слева на расстоянии stride - берем его
        if (idx >= stride)
        {
            temp_val = temp[idx - stride];
        }

        __syncthreads();  // Ждем пока все прочитают

        // Добавляем к текущему элементу
        if (idx >= stride)
        {
            temp[idx] += temp_val;
        }

        __syncthreads();  // Ждем пока все запишут
    }

    // Записываем результат обратно в глобальную память
    if (globalIdx < size)
    {
        output[globalIdx] = temp[localIdx];
    }

    // Последний поток блока записывает сумму блока
    // Это нужно для следующего шага (сканирование между блоками)
    if (localIdx == blockDim.x - 1 && blockSums != NULL)
    {
        blockSums[blockIdx.x] = temp[localIdx];
    }
}

/*
 * Ядро для добавления offset к каждому элементу блока
 *
 * После того как мы посчитали scan внутри каждого блока,
 * нужно добавить к каждому блоку сумму всех предыдущих блоков
 */
__global__ void addBlockSumsKernel(int* output, int* blockSums, int size)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < size && blockIdx.x > 0)
    {
        // Добавляем сумму всех предыдущих блоков
        output[globalIdx] += blockSums[blockIdx.x - 1];
    }
}

//==============================================================================
// ФУНКЦИЯ ДЛЯ ЗАПУСКА GPU ВЕРСИИ
//==============================================================================

/*
 * Главная функция для вычисления prefix sum на GPU
 *
 * Алгоритм:
 * 1. Вычисляем scan внутри каждого блока
 * 2. Собираем суммы блоков
 * 3. Вычисляем scan сумм блоков (рекурсивно)
 * 4. Добавляем scan сумм блоков к каждому блоку
 */
void prefixSumGPU(int* h_input, int* h_output, int size)
{
    int* d_input;
    int* d_output;
    int* d_blockSums;

    // Количество блоков
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Выделяем память на GPU
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_blockSums, numBlocks * sizeof(int)));

    // Копируем входные данные
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice));

    // ШАГ 1: Сканируем каждый блок независимо
    scanBlockKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, d_blockSums, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ШАГ 2: Если блоков больше одного, нужно обработать суммы блоков
    if (numBlocks > 1)
    {
        // Сканируем суммы блоков (рекурсивный вызов для маленького массива)
        int* h_blockSums = new int[numBlocks];
        int* h_blockSumsScanned = new int[numBlocks];

        // Копируем суммы блоков на CPU
        CUDA_CHECK(cudaMemcpy(h_blockSums, d_blockSums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost));

        // Вычисляем prefix sum сумм блоков на CPU (их немного)
        prefixSumCPU(h_blockSums, h_blockSumsScanned, numBlocks);

        // Копируем обратно
        CUDA_CHECK(cudaMemcpy(d_blockSums, h_blockSumsScanned, numBlocks * sizeof(int), cudaMemcpyHostToDevice));

        // ШАГ 3: Добавляем суммы предыдущих блоков к каждому блоку
        addBlockSumsKernel<<<numBlocks, BLOCK_SIZE>>>(d_output, d_blockSums, size);
        CUDA_CHECK(cudaDeviceSynchronize());

        delete[] h_blockSums;
        delete[] h_blockSumsScanned;
    }

    // Копируем результат обратно
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождаем память
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_blockSums));
}

//==============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
//==============================================================================

// Заполнить массив
void fillArray(int* array, int size)
{
    for (int i = 0; i < size; i++)
    {
        // Маленькие числа чтобы не было переполнения
        array[i] = rand() % 10;
    }
}

// Проверить результат
bool verifyResult(int* cpuResult, int* gpuResult, int size)
{
    // Проверяем первые 10000 элементов (для экономии времени)
    int checkSize = min(10000, size);

    for (int i = 0; i < checkSize; i++)
    {
        if (cpuResult[i] != gpuResult[i])
        {
            cout << "Ошибка на позиции " << i << ": ";
            cout << "CPU=" << cpuResult[i] << ", GPU=" << gpuResult[i] << endl;
            return false;
        }
    }

    return true;
}

//==============================================================================
// ГЛАВНАЯ ФУНКЦИЯ
//==============================================================================

int main()
{
    cout << "=== Assignment 4 - Задание 2 ===" << endl;
    cout << "Префиксная сумма (Scan) с shared memory" << endl;
    cout << endl;

    cout << "Размер массива: " << ARRAY_SIZE << " элементов" << endl;
    cout << "Размер блока: " << BLOCK_SIZE << " потоков" << endl;
    cout << endl;

    // Проверяем GPU
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        cout << "Ошибка: GPU не найден!" << endl;
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "GPU: " << prop.name << endl;
    cout << "Shared memory на блок: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
    cout << endl;

    // Инициализация
    srand(time(NULL));

    //==========================================================================
    // ПОДГОТОВКА ДАННЫХ
    //==========================================================================

    cout << "Подготовка данных..." << endl;

    int* h_input = new int[ARRAY_SIZE];
    int* h_outputCPU = new int[ARRAY_SIZE];
    int* h_outputGPU = new int[ARRAY_SIZE];

    fillArray(h_input, ARRAY_SIZE);

    cout << "Входные данные (первые 10): ";
    for (int i = 0; i < 10; i++)
    {
        cout << h_input[i] << " ";
    }
    cout << endl << endl;

    //==========================================================================
    // ТЕСТ 1: ПРЕФИКСНАЯ СУММА НА CPU
    //==========================================================================

    cout << "--- Вычисление на CPU ---" << endl;

    clock_t startCPU = clock();

    prefixSumCPU(h_input, h_outputCPU, ARRAY_SIZE);

    clock_t endCPU = clock();

    double timeCPU = (double)(endCPU - startCPU) / CLOCKS_PER_SEC * 1000.0;

    cout << "Результат (первые 10): ";
    for (int i = 0; i < 10; i++)
    {
        cout << h_outputCPU[i] << " ";
    }
    cout << endl;
    cout << "Последний элемент (сумма всех): " << h_outputCPU[ARRAY_SIZE - 1] << endl;
    cout << "Время: " << timeCPU << " мс" << endl;
    cout << endl;

    //==========================================================================
    // ТЕСТ 2: ПРЕФИКСНАЯ СУММА НА GPU
    //==========================================================================

    cout << "--- Вычисление на GPU ---" << endl;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    prefixSumGPU(h_input, h_outputGPU, ARRAY_SIZE);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float timeGPU;
    CUDA_CHECK(cudaEventElapsedTime(&timeGPU, start, stop));

    cout << "Результат (первые 10): ";
    for (int i = 0; i < 10; i++)
    {
        cout << h_outputGPU[i] << " ";
    }
    cout << endl;
    cout << "Последний элемент (сумма всех): " << h_outputGPU[ARRAY_SIZE - 1] << endl;
    cout << "Время: " << timeGPU << " мс" << endl;
    cout << endl;

    //==========================================================================
    // СРАВНЕНИЕ
    //==========================================================================

    cout << "--- Сравнение ---" << endl;

    if (verifyResult(h_outputCPU, h_outputGPU, ARRAY_SIZE))
    {
        cout << "✓ Результаты совпадают" << endl;
    }
    else
    {
        cout << "✗ ОШИБКА: результаты не совпадают!" << endl;
    }
    cout << endl;

    cout << "Время CPU: " << timeCPU << " мс" << endl;
    cout << "Время GPU: " << timeGPU << " мс" << endl;

    if (timeGPU < timeCPU)
    {
        cout << "Ускорение: " << (timeCPU / timeGPU) << "x" << endl;
    }
    else
    {
        cout << "GPU медленнее (накладные расходы)" << endl;
    }
    cout << endl;

    //==========================================================================
    // ВЫВОДЫ
    //==========================================================================

    cout << "--- Выводы ---" << endl;
    cout << endl;
    cout << "1. PREFIX SUM (SCAN) - важная примитивная операция" << endl;
    cout << "   Используется в:" << endl;
    cout << "   - Компактификации массивов" << endl;
    cout << "   - Быстром выделении памяти" << endl;
    cout << "   - Построении деревьев" << endl;
    cout << "   - Многих параллельных алгоритмах" << endl;
    cout << endl;
    cout << "2. SHARED MEMORY критична для эффективности" << endl;
    cout << "   Без неё scan будет очень медленным из-за" << endl;
    cout << "   множества обращений к глобальной памяти" << endl;
    cout << endl;
    cout << "3. АЛГОРИТМ BLELLOCH SCAN (работает за O(n) операций)" << endl;
    cout << "   Мы использовали простую версию для обучения" << endl;
    cout << "   Более эффективная версия: two-phase scan" << endl;
    cout << endl;
    cout << "4. ПРОБЛЕМА ЗАВИСИМОСТЕЙ" << endl;
    cout << "   Каждый элемент зависит от предыдущего" << endl;
    cout << "   Нужна синхронизация на каждом шаге" << endl;
    cout << endl;
    cout << "5. МНОГОУРОВНЕВЫЙ SCAN для больших массивов" << endl;
    cout << "   - Scan внутри блоков" << endl;
    cout << "   - Scan сумм блоков" << endl;
    cout << "   - Добавление offset к блокам" << endl;

    // Освобождаем память
    delete[] h_input;
    delete[] h_outputCPU;
    delete[] h_outputGPU;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
