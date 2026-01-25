/*
 * Assignment 4 - Задание 3
 * Гибридная программа CPU + GPU
 *
 * Обрабатываем массив частично на CPU и частично на GPU ОДНОВРЕМЕННО
 *
 * Задача: Возвести каждый элемент массива в квадрат
 * - Первая половина массива обрабатывается на CPU
 * - Вторая половина массива обрабатывается на GPU
 *
 * Это демонстрирует:
 * 1. Как разделить работу между CPU и GPU
 * 2. Параллельное выполнение на CPU и GPU
 * 3. Когда гибридный подход эффективен
 *
 * Размер массива: 10,000,000 элементов (чтобы увидеть разницу)
 *
 * Компиляция: nvcc -o task3 task3_hybrid.cu -Xcompiler -fopenmp
 * Запуск: ./task3
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <omp.h>  // Для параллелизма на CPU

using namespace std;

// Размер массива - 10 миллионов для более четкой картины
#define ARRAY_SIZE 10000000

// Размер блока для GPU
#define BLOCK_SIZE 256

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
// ФУНКЦИЯ ОБРАБОТКИ (возведение в квадрат)
//==============================================================================

/*
 * Простая функция для обработки одного элемента
 * Просто возводим в квадрат
 */
inline float processElement(float x)
{
    return x * x;
}

//==============================================================================
// ВЕРСИЯ 1: ТОЛЬКО CPU (последовательная)
//==============================================================================

/*
 * Обработка всего массива на CPU последовательно
 * Просто проходим по массиву и обрабатываем каждый элемент
 */
void processCPUSequential(float* input, float* output, int size)
{
    for (int i = 0; i < size; i++)
    {
        output[i] = processElement(input[i]);
    }
}

//==============================================================================
// ВЕРСИЯ 2: ТОЛЬКО CPU (параллельная с OpenMP)
//==============================================================================

/*
 * Обработка всего массива на CPU параллельно
 * Используем OpenMP для распараллеливания
 */
void processCPUParallel(float* input, float* output, int size)
{
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        output[i] = processElement(input[i]);
    }
}

//==============================================================================
// ВЕРСИЯ 3: ТОЛЬКО GPU
//==============================================================================

/*
 * CUDA ядро для обработки на GPU
 * Каждый поток обрабатывает один элемент
 */
__global__ void processGPUKernel(float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Каждый поток обрабатывает несколько элементов если нужно
    while (idx < size)
    {
        output[idx] = input[idx] * input[idx];  // Возводим в квадрат
        idx += blockDim.x * gridDim.x;
    }
}

/*
 * Функция запуска обработки на GPU
 */
void processGPU(float* h_input, float* h_output, int size)
{
    float* d_input;
    float* d_output;

    // Выделяем память на GPU
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));

    // Копируем данные на GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));

    // Запускаем ядро
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    processGPUKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, size);

    // Копируем результат обратно
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));

    // Освобождаем память
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

//==============================================================================
// ВЕРСИЯ 4: ГИБРИДНАЯ (CPU + GPU)
//==============================================================================

/*
 * Гибридная обработка: CPU и GPU работают ОДНОВРЕМЕННО
 *
 * Схема:
 * 1. Делим массив на две части
 * 2. CPU обрабатывает первую половину (используя OpenMP)
 * 3. GPU обрабатывает вторую половину (асинхронно)
 * 4. Ждем завершения обоих
 *
 * Важно: используем CUDA streams для асинхронного выполнения
 */
void processHybrid(float* h_input, float* h_output, int size)
{
    // Делим массив пополам
    int cpuSize = size / 2;      // Первая половина - CPU
    int gpuSize = size - cpuSize; // Вторая половина - GPU

    cout << "  Разделение: CPU=" << cpuSize << " элементов, GPU=" << gpuSize << " элементов" << endl;

    float* d_input;
    float* d_output;

    // Выделяем память на GPU только для второй половины
    CUDA_CHECK(cudaMalloc(&d_input, gpuSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, gpuSize * sizeof(float)));

    // Создаем CUDA stream для асинхронного выполнения
    // Stream позволяет копированию и вычислениям выполняться параллельно с CPU
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ЗАПУСКАЕМ GPU (асинхронно)
    // Копируем вторую половину на GPU
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input + cpuSize, gpuSize * sizeof(float),
                                cudaMemcpyHostToDevice, stream));

    // Запускаем ядро на GPU (асинхронно)
    int numBlocks = (gpuSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    processGPUKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(d_input, d_output, gpuSize);

    // Копируем результат обратно (асинхронно)
    CUDA_CHECK(cudaMemcpyAsync(h_output + cpuSize, d_output, gpuSize * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));

    // ОДНОВРЕМЕННО ЗАПУСКАЕМ CPU
    // Пока GPU работает, CPU обрабатывает первую половину
    #pragma omp parallel for
    for (int i = 0; i < cpuSize; i++)
    {
        h_output[i] = processElement(h_input[i]);
    }

    // Ждем завершения GPU
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Освобождаем ресурсы
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

//==============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
//==============================================================================

// Заполнить массив
void fillArray(float* array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = (float)(rand() % 100) / 10.0f;  // Числа от 0 до 9.9
    }
}

// Проверить результат
bool verifyResult(float* expected, float* actual, int size)
{
    int checkSize = min(10000, size);

    for (int i = 0; i < checkSize; i++)
    {
        float diff = abs(expected[i] - actual[i]);
        if (diff > 0.001f)
        {
            cout << "Ошибка на позиции " << i << ": ";
            cout << "expected=" << expected[i] << ", actual=" << actual[i] << endl;
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
    cout << "=== Assignment 4 - Задание 3 ===" << endl;
    cout << "Гибридные вычисления CPU + GPU" << endl;
    cout << endl;

    cout << "Размер массива: " << ARRAY_SIZE << " элементов" << endl;
    cout << "Операция: возведение в квадрат" << endl;
    cout << "Потоков OpenMP: " << omp_get_max_threads() << endl;
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
    cout << endl;

    srand(time(NULL));

    //==========================================================================
    // ПОДГОТОВКА ДАННЫХ
    //==========================================================================

    cout << "Подготовка данных..." << endl;

    float* h_input = new float[ARRAY_SIZE];
    float* h_outputCPU = new float[ARRAY_SIZE];
    float* h_outputGPU = new float[ARRAY_SIZE];
    float* h_outputHybrid = new float[ARRAY_SIZE];

    fillArray(h_input, ARRAY_SIZE);

    cout << "Входные данные (первые 5): ";
    for (int i = 0; i < 5; i++)
    {
        cout << h_input[i] << " ";
    }
    cout << endl << endl;

    //==========================================================================
    // ТЕСТ 1: ТОЛЬКО CPU (параллельно с OpenMP)
    //==========================================================================

    cout << "--- Тест 1: Только CPU (OpenMP) ---" << endl;

    double startCPU = omp_get_wtime();

    processCPUParallel(h_input, h_outputCPU, ARRAY_SIZE);

    double endCPU = omp_get_wtime();
    double timeCPU = (endCPU - startCPU) * 1000.0;

    cout << "Результат (первые 5): ";
    for (int i = 0; i < 5; i++)
    {
        cout << h_outputCPU[i] << " ";
    }
    cout << endl;
    cout << "Время: " << timeCPU << " мс" << endl;
    cout << endl;

    //==========================================================================
    // ТЕСТ 2: ТОЛЬКО GPU
    //==========================================================================

    cout << "--- Тест 2: Только GPU ---" << endl;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    processGPU(h_input, h_outputGPU, ARRAY_SIZE);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float timeGPU;
    CUDA_CHECK(cudaEventElapsedTime(&timeGPU, start, stop));

    cout << "Результат (первые 5): ";
    for (int i = 0; i < 5; i++)
    {
        cout << h_outputGPU[i] << " ";
    }
    cout << endl;
    cout << "Время: " << timeGPU << " мс" << endl;

    // Проверяем корректность
    if (verifyResult(h_outputCPU, h_outputGPU, ARRAY_SIZE))
    {
        cout << "✓ Результат GPU корректен" << endl;
    }
    cout << endl;

    //==========================================================================
    // ТЕСТ 3: ГИБРИДНАЯ ВЕРСИЯ (CPU + GPU)
    //==========================================================================

    cout << "--- Тест 3: Гибридная (CPU + GPU) ---" << endl;

    CUDA_CHECK(cudaEventRecord(start));
    double startHybrid = omp_get_wtime();

    processHybrid(h_input, h_outputHybrid, ARRAY_SIZE);

    double endHybrid = omp_get_wtime();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    double timeHybrid = (endHybrid - startHybrid) * 1000.0;

    cout << "Результат (первые 5): ";
    for (int i = 0; i < 5; i++)
    {
        cout << h_outputHybrid[i] << " ";
    }
    cout << endl;
    cout << "Время: " << timeHybrid << " мс" << endl;

    // Проверяем корректность
    if (verifyResult(h_outputCPU, h_outputHybrid, ARRAY_SIZE))
    {
        cout << "✓ Результат гибридной версии корректен" << endl;
    }
    cout << endl;

    //==========================================================================
    // СРАВНЕНИЕ
    //==========================================================================

    cout << "=== Сравнение ===" << endl;
    cout << endl;

    cout << "Время CPU:     " << timeCPU << " мс" << endl;
    cout << "Время GPU:     " << timeGPU << " мс" << endl;
    cout << "Время Hybrid:  " << timeHybrid << " мс" << endl;
    cout << endl;

    // Находим лучший вариант
    double bestTime = min(timeCPU, min((double)timeGPU, timeHybrid));
    string best = "CPU";
    if ((double)timeGPU == bestTime) best = "GPU";
    if (timeHybrid == bestTime) best = "Hybrid";

    cout << "Лучший результат: " << best << " (" << bestTime << " мс)" << endl;
    cout << endl;

    cout << "Ускорение относительно CPU:" << endl;
    cout << "  GPU:    " << (timeCPU / timeGPU) << "x" << endl;
    cout << "  Hybrid: " << (timeCPU / timeHybrid) << "x" << endl;
    cout << endl;

    //==========================================================================
    // ВЫВОДЫ
    //==========================================================================

    cout << "=== Выводы ===" << endl;
    cout << endl;
    cout << "1. ГИБРИДНЫЙ ПОДХОД эффективен когда:" << endl;
    cout << "   - Задача достаточно большая" << endl;
    cout << "   - CPU и GPU имеют сопоставимую производительность" << endl;
    cout << "   - Можно разделить работу без зависимостей" << endl;
    cout << endl;
    cout << "2. CUDA STREAMS позволяют:" << endl;
    cout << "   - Копировать данные асинхронно" << endl;
    cout << "   - Запускать ядра параллельно с CPU" << endl;
    cout << "   - Перекрывать вычисления и передачу данных" << endl;
    cout << endl;
    cout << "3. OPENMP на CPU:" << endl;
    cout << "   - Простой способ параллелизма" << endl;
    cout << "   - Хорош для CPU части гибридной программы" << endl;
    cout << "   - Использует все ядра CPU" << endl;
    cout << endl;
    cout << "4. КОГДА GPU ЛУЧШЕ:" << endl;
    cout << "   - Очень большие данные" << endl;
    cout << "   - Простые операции" << endl;
    cout << "   - Данные уже на GPU" << endl;
    cout << endl;
    cout << "5. КОГДА HYBRID ЛУЧШЕ:" << endl;
    cout << "   - Средние объемы данных" << endl;
    cout << "   - Когда CPU простаивает пока GPU работает" << endl;
    cout << "   - Когда нужно максимально использовать всё железо" << endl;
    cout << endl;
    cout << "6. НАКЛАДНЫЕ РАСХОДЫ:" << endl;
    cout << "   - Копирование CPU ↔ GPU" << endl;
    cout << "   - Запуск ядра" << endl;
    cout << "   - Синхронизация" << endl;
    cout << "   Для маленьких задач могут превысить выигрыш!" << endl;

    // Освобождаем память
    delete[] h_input;
    delete[] h_outputCPU;
    delete[] h_outputGPU;
    delete[] h_outputHybrid;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
