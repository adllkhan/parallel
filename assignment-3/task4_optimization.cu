/*
 * Assignment 3 - Задание 4
 * Оптимизация конфигурации сетки и блоков
 *
 * Берем задачу из Задания 2 (сложение векторов) и подбираем
 * оптимальные параметры конфигурации.
 *
 * Тестируем разные комбинации размера блока и сетки.
 *
 * Компиляция: nvcc -o task4 task4_optimization.cu
 * Запуск: ./task4
 */

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Размер массива
#define ARRAY_SIZE 10000000  // 10 миллионов для более четких различий

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
// CUDA ЯДРО ДЛЯ СЛОЖЕНИЯ ВЕКТОРОВ
//==============================================================================

/*
 * Простое ядро для сложения векторов
 * C[i] = A[i] + B[i]
 *
 * Это простая операция, но она хорошо показывает
 * влияние конфигурации сетки и блоков
 */
__global__ void vectorAdd(float* A, float* B, float* C, int size)
{
    // Вычисляем глобальный индекс
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем границы
    if (idx < size)
    {
        // Складываем элементы
        C[idx] = A[idx] + B[idx];
    }
}

//==============================================================================
// ОПТИМИЗИРОВАННОЕ ЯДРО (с развертыванием цикла)
//==============================================================================

/*
 * Оптимизированное ядро которое обрабатывает несколько элементов на поток
 *
 * Идея: каждый поток обрабатывает не 1, а несколько элементов.
 * Это называется "loop unrolling" или развертывание цикла.
 *
 * Преимущества:
 * - Меньше потоков = меньше накладных расходов
 * - Лучше использование регистров
 * - Больше независимых операций для конвейера GPU
 */
__global__ void vectorAddOptimized(float* A, float* B, float* C, int size)
{
    // Вычисляем начальный индекс для этого потока
    // Каждый поток обрабатывает 4 элемента
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    // Обрабатываем 4 элемента подряд
    // Это развертывание цикла (loop unrolling)
    if (idx < size)
    {
        C[idx] = A[idx] + B[idx];
    }
    if (idx + 1 < size)
    {
        C[idx + 1] = A[idx + 1] + B[idx + 1];
    }
    if (idx + 2 < size)
    {
        C[idx + 2] = A[idx + 2] + B[idx + 2];
    }
    if (idx + 3 < size)
    {
        C[idx + 3] = A[idx + 3] + B[idx + 3];
    }
}

//==============================================================================
// СТРУКТУРА ДЛЯ ХРАНЕНИЯ КОНФИГУРАЦИИ
//==============================================================================

struct KernelConfig
{
    int blockSize;      // Количество потоков в блоке
    int numBlocks;      // Количество блоков
    string description; // Описание конфигурации

    // Конструктор
    KernelConfig(int bs, int nb, string desc)
        : blockSize(bs), numBlocks(nb), description(desc) {}
};

//==============================================================================
// ФУНКЦИЯ ДЛЯ ТЕСТИРОВАНИЯ КОНФИГУРАЦИИ
//==============================================================================

/*
 * Тестирует одну конфигурацию и возвращает время выполнения
 */
float testConfiguration(KernelConfig config, float* d_A, float* d_B, float* d_C, int size, bool optimized)
{
    // Создаем события для измерения времени
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Запускаем несколько раз и берем среднее
    int runs = 10;
    float totalTime = 0;

    for (int i = 0; i < runs; i++)
    {
        CUDA_CHECK(cudaEventRecord(start));

        // Запускаем нужное ядро
        if (optimized)
        {
            // Для оптимизированного ядра нужно меньше блоков
            // так как каждый поток обрабатывает 4 элемента
            int adjustedBlocks = (size + config.blockSize * 4 - 1) / (config.blockSize * 4);
            vectorAddOptimized<<<adjustedBlocks, config.blockSize>>>(d_A, d_B, d_C, size);
        }
        else
        {
            // Обычное ядро
            vectorAdd<<<config.numBlocks, config.blockSize>>>(d_A, d_B, d_C, size);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
        totalTime += time;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalTime / runs;
}

//==============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
//==============================================================================

void fillArray(float* arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = (float)(rand() % 100);
    }
}

bool verifyResults(float* A, float* B, float* C, int size)
{
    // Проверяем первые 1000 элементов
    for (int i = 0; i < min(1000, size); i++)
    {
        float expected = A[i] + B[i];
        if (abs(C[i] - expected) > 0.001f)
        {
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
    cout << "=== Assignment 3 - Задание 4 ===" << endl;
    cout << "Оптимизация конфигурации сетки и блоков" << endl;
    cout << endl;

    cout << "Размер массива: " << ARRAY_SIZE << " элементов" << endl;
    cout << "Операция: сложение векторов C = A + B" << endl;
    cout << endl;

    // Проверяем GPU
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        cout << "Ошибка: GPU не найден!" << endl;
        return 1;
    }

    // Получаем информацию о GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    cout << "GPU: " << prop.name << endl;
    cout << "Мультипроцессоров: " << prop.multiProcessorCount << endl;
    cout << "Максимум потоков на блок: " << prop.maxThreadsPerBlock << endl;
    cout << "Максимум блоков на грид (x): " << prop.maxGridSize[0] << endl;
    cout << "Размер warp: " << prop.warpSize << endl;
    cout << endl;

    // Инициализация
    srand(time(NULL));

    //==========================================================================
    // ПОДГОТОВКА ДАННЫХ
    //==========================================================================

    cout << "Подготовка данных..." << endl;

    // Выделяем память на CPU
    float* h_A = new float[ARRAY_SIZE];
    float* h_B = new float[ARRAY_SIZE];
    float* h_C = new float[ARRAY_SIZE];

    fillArray(h_A, ARRAY_SIZE);
    fillArray(h_B, ARRAY_SIZE);

    // Выделяем память на GPU
    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, ARRAY_SIZE * sizeof(float)));

    // Копируем данные на GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    cout << "Данные готовы!" << endl;
    cout << endl;

    //==========================================================================
    // СОЗДАЕМ РАЗНЫЕ КОНФИГУРАЦИИ ДЛЯ ТЕСТИРОВАНИЯ
    //==========================================================================

    // Будем тестировать эти конфигурации
    KernelConfig configs[] = {
        // ПЛОХИЕ конфигурации
        KernelConfig(32, (ARRAY_SIZE + 31) / 32, "Очень маленький блок (32)"),
        KernelConfig(64, (ARRAY_SIZE + 63) / 64, "Маленький блок (64)"),

        // СРЕДНИЕ конфигурации
        KernelConfig(128, (ARRAY_SIZE + 127) / 128, "Средний блок (128)"),
        KernelConfig(256, (ARRAY_SIZE + 255) / 256, "Хороший блок (256)"),

        // БОЛЬШИЕ конфигурации
        KernelConfig(512, (ARRAY_SIZE + 511) / 512, "Большой блок (512)"),
        KernelConfig(1024, (ARRAY_SIZE + 1023) / 1024, "Максимальный блок (1024)"),
    };

    int numConfigs = 6;

    //==========================================================================
    // ТЕСТИРОВАНИЕ НЕОПТИМИЗИРОВАННОЙ ВЕРСИИ
    //==========================================================================

    cout << "========================================" << endl;
    cout << "НЕОПТИМИЗИРОВАННАЯ ВЕРСИЯ" << endl;
    cout << "========================================" << endl;
    cout << endl;

    float times[6];
    float bestTime = 99999;
    int bestConfig = 0;

    for (int i = 0; i < numConfigs; i++)
    {
        cout << "Тест " << (i + 1) << ": " << configs[i].description << endl;
        cout << "  Блоков: " << configs[i].numBlocks;
        cout << ", Потоков в блоке: " << configs[i].blockSize << endl;

        // Вычисляем общее количество потоков
        long long totalThreads = (long long)configs[i].numBlocks * configs[i].blockSize;
        cout << "  Всего потоков: " << totalThreads << endl;

        // Запускаем тест
        float time = testConfiguration(configs[i], d_A, d_B, d_C, ARRAY_SIZE, false);
        times[i] = time;

        cout << "  Время: " << time << " мс" << endl;

        // Проверяем результат (только для первого теста)
        if (i == 0)
        {
            CUDA_CHECK(cudaMemcpy(h_C, d_C, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            if (verifyResults(h_A, h_B, h_C, ARRAY_SIZE))
            {
                cout << "  Результат корректен" << endl;
            }
            else
            {
                cout << "  ОШИБКА в результате!" << endl;
            }
        }

        // Отслеживаем лучший результат
        if (time < bestTime)
        {
            bestTime = time;
            bestConfig = i;
        }

        cout << endl;
    }

    //==========================================================================
    // СРАВНЕНИЕ НЕОПТИМИЗИРОВАННЫХ КОНФИГУРАЦИЙ
    //==========================================================================

    cout << "--- Сравнение неоптимизированных конфигураций ---" << endl;
    cout << endl;

    cout << "Лучшая конфигурация: " << configs[bestConfig].description << endl;
    cout << "Время: " << bestTime << " мс" << endl;
    cout << endl;

    cout << "Сравнительная таблица:" << endl;
    cout << "----------------------------------------------------------------" << endl;
    for (int i = 0; i < numConfigs; i++)
    {
        float relative = times[i] / bestTime;
        cout << configs[i].description << ": " << times[i] << " мс";

        if (i == bestConfig)
        {
            cout << " ← ЛУЧШИЙ";
        }
        else
        {
            cout << " (x" << relative << " медленнее)";
        }

        cout << endl;
    }
    cout << "----------------------------------------------------------------" << endl;
    cout << endl;

    //==========================================================================
    // ТЕСТИРОВАНИЕ ОПТИМИЗИРОВАННОЙ ВЕРСИИ
    //==========================================================================

    cout << "========================================" << endl;
    cout << "ОПТИМИЗИРОВАННАЯ ВЕРСИЯ" << endl;
    cout << "(каждый поток обрабатывает 4 элемента)" << endl;
    cout << "========================================" << endl;
    cout << endl;

    // Тестируем только лучшую конфигурацию с оптимизацией
    cout << "Используем лучшую конфигурацию: " << configs[bestConfig].description << endl;
    cout << endl;

    float optimizedTime = testConfiguration(configs[bestConfig], d_A, d_B, d_C, ARRAY_SIZE, true);

    cout << "Неоптимизированное: " << bestTime << " мс" << endl;
    cout << "Оптимизированное:   " << optimizedTime << " мс" << endl;
    cout << endl;

    float speedup = bestTime / optimizedTime;
    cout << "Ускорение: " << speedup << "x" << endl;

    if (speedup > 1.0f)
    {
        cout << "Оптимизация дала ускорение " << ((speedup - 1) * 100) << "%" << endl;
    }
    else
    {
        cout << "Оптимизация не дала ускорения (возможно bottleneck в памяти)" << endl;
    }

    cout << endl;

    //==========================================================================
    // ВЫВОДЫ И РЕКОМЕНДАЦИИ
    //==========================================================================

    cout << "=== Выводы ===" << endl;
    cout << endl;

    cout << "1. РАЗМЕР БЛОКА:" << endl;
    cout << "   Оптимальный размер для этой задачи: " << configs[bestConfig].blockSize << endl;
    cout << "   - Слишком маленький (32-64): плохая утилизация GPU" << endl;
    cout << "   - Слишком большой (1024): может не хватить ресурсов" << endl;
    cout << "   - Оптимальный (128-256): баланс между ресурсами и загрузкой" << endl;
    cout << endl;

    cout << "2. КОЛИЧЕСТВО БЛОКОВ:" << endl;
    cout << "   Нужно достаточно блоков чтобы загрузить все мультипроцессоры." << endl;
    cout << "   Минимум: " << prop.multiProcessorCount * 2 << " блоков (2x число SM)" << endl;
    cout << "   У нас: " << configs[bestConfig].numBlocks << " блоков" << endl;
    cout << endl;

    cout << "3. ПРАВИЛО КРАТНОСТИ WARP:" << endl;
    cout << "   Размер блока должен быть кратен " << prop.warpSize << " (размер warp)" << endl;
    cout << "   Иначе часть warp будет простаивать." << endl;
    cout << endl;

    cout << "4. ОПТИМИЗАЦИЯ ЧЕРЕЗ LOOP UNROLLING:" << endl;
    cout << "   Когда каждый поток обрабатывает несколько элементов," << endl;
    cout << "   уменьшаются накладные расходы и лучше используется конвейер." << endl;
    cout << endl;

    cout << "5. ОБЩИЕ РЕКОМЕНДАЦИИ:" << endl;
    cout << "   - Начинайте с 256 потоков на блок" << endl;
    cout << "   - Делайте размер кратным 32 (размер warp)" << endl;
    cout << "   - Тестируйте разные конфигурации для своей задачи" << endl;
    cout << "   - Используйте CUDA Occupancy Calculator" << endl;
    cout << "   - Профилируйте с помощью nvprof или Nsight" << endl;

    // Освобождаем память
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
