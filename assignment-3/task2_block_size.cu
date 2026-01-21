/*
 * Assignment 3 - Задание 2
 * Исследование влияния размера блока на производительность
 *
 * Программа складывает два массива поэлементно: C[i] = A[i] + B[i]
 * Тестируем разные размеры блоков и смотрим как это влияет на скорость
 *
 * Компиляция: nvcc -o task2 task2_block_size.cu
 * Запуск: ./task2
 */

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Размер массивов
#define ARRAY_SIZE 1000000

// Макрос для проверки ошибок CUDA
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
// CUDA ЯДРО ДЛЯ СЛОЖЕНИЯ МАССИВОВ
//==============================================================================

/*
 * Ядро которое складывает два массива
 *
 * Что делает каждый поток:
 * 1. Вычисляет свой индекс в массиве
 * 2. Проверяет что не вышел за границы
 * 3. Читает элементы из массивов A и B
 * 4. Складывает их
 * 5. Записывает результат в массив C
 *
 * Это простая операция, но она показывает как размер блока влияет на производительность
 */
__global__ void vectorAdd(float* A, float* B, float* C, int size)
{
    // Вычисляем глобальный индекс потока
    // Формула: индекс_блока * размер_блока + индекс_потока_в_блоке
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем границы массива
    // Это нужно потому что количество потоков может не делиться нацело на размер массива
    if (idx < size)
    {
        // Читаем из глобальной памяти два элемента
        float a = A[idx];
        float b = B[idx];

        // Складываем
        float c = a + b;

        // Записываем результат
        C[idx] = c;
    }
}

//==============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
//==============================================================================

// Заполнить массив случайными числами
void fillArray(float* arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        // Случайное число от 0 до 99
        arr[i] = (float)(rand() % 100);
    }
}

// Проверить что сложение выполнено правильно
bool verifyResults(float* A, float* B, float* C, int size)
{
    // Проверяем первые 1000 элементов (для экономии времени)
    int checkCount = min(1000, size);

    for (int i = 0; i < checkCount; i++)
    {
        // Ожидаемый результат
        float expected = A[i] + B[i];

        // Проверяем с учетом погрешности
        if (abs(C[i] - expected) > 0.001f)
        {
            cout << "Ошибка на позиции " << i << ": ";
            cout << A[i] << " + " << B[i] << " = " << C[i];
            cout << " (ожидалось " << expected << ")" << endl;
            return false;
        }
    }

    return true;
}

//==============================================================================
// ФУНКЦИЯ ДЛЯ ТЕСТИРОВАНИЯ ОДНОГО РАЗМЕРА БЛОКА
//==============================================================================

/*
 * Функция запускает сложение массивов с заданным размером блока
 * и измеряет время выполнения
 */
float testBlockSize(int blockSize, float* d_A, float* d_B, float* d_C, int arraySize)
{
    // Вычисляем сколько блоков нам нужно
    // Округляем вверх чтобы обработать все элементы
    int numBlocks = (arraySize + blockSize - 1) / blockSize;

    // Создаем события CUDA для точного измерения времени GPU
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Начинаем измерение
    CUDA_CHECK(cudaEventRecord(start));

    // Запускаем ядро
    // <<<количество_блоков, потоков_в_блоке>>>
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, arraySize);

    // Заканчиваем измерение
    CUDA_CHECK(cudaEventRecord(stop));

    // Ждем завершения всех операций
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Вычисляем время
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Освобождаем события
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return milliseconds;
}

//==============================================================================
// ГЛАВНАЯ ФУНКЦИЯ
//==============================================================================

int main()
{
    cout << "=== Assignment 3 - Задание 2 ===" << endl;
    cout << "Влияние размера блока на производительность" << endl;
    cout << endl;

    cout << "Размер массивов: " << ARRAY_SIZE << " элементов" << endl;
    cout << "Операция: C[i] = A[i] + B[i]" << endl;
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
    cout << "Максимальный размер блока: " << prop.maxThreadsPerBlock << endl;
    cout << "Размер warp: " << prop.warpSize << " потоков" << endl;
    cout << endl;

    // Инициализируем генератор случайных чисел
    srand(time(NULL));

    //==========================================================================
    // ПОДГОТОВКА ДАННЫХ
    //==========================================================================

    cout << "Подготовка данных..." << endl;

    // Выделяем память на CPU
    float* h_A = new float[ARRAY_SIZE];
    float* h_B = new float[ARRAY_SIZE];
    float* h_C = new float[ARRAY_SIZE];

    // Заполняем массивы
    fillArray(h_A, ARRAY_SIZE);
    fillArray(h_B, ARRAY_SIZE);

    // Выделяем память на GPU
    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, ARRAY_SIZE * sizeof(float)));

    // Копируем входные данные на GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    cout << "Данные готовы!" << endl;
    cout << endl;

    //==========================================================================
    // ТЕСТИРОВАНИЕ РАЗНЫХ РАЗМЕРОВ БЛОКОВ
    //==========================================================================

    cout << "=== Тестирование разных размеров блоков ===" << endl;
    cout << endl;

    // Будем тестировать эти размеры блоков
    // Важно: размеры должны быть кратны 32 (размер warp)
    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    int numTests = 6;

    // Массив для хранения времени выполнения
    float times[6];

    // Тестируем каждый размер блока
    for (int i = 0; i < numTests; i++)
    {
        int blockSize = blockSizes[i];

        // Проверяем что размер блока не превышает максимум для GPU
        if (blockSize > prop.maxThreadsPerBlock)
        {
            cout << "Размер блока " << blockSize << " - СЛИШКОМ БОЛЬШОЙ для этого GPU" << endl;
            times[i] = -1;
            continue;
        }

        // Вычисляем количество блоков
        int numBlocks = (ARRAY_SIZE + blockSize - 1) / blockSize;

        cout << "--- Тест " << (i + 1) << ": Размер блока = " << blockSize << " ---" << endl;
        cout << "Количество блоков: " << numBlocks << endl;

        // Запускаем тест 5 раз и берем среднее время
        // Это дает более стабильные результаты
        float totalTime = 0;
        int runs = 5;

        for (int run = 0; run < runs; run++)
        {
            float time = testBlockSize(blockSize, d_A, d_B, d_C, ARRAY_SIZE);
            totalTime += time;
        }

        float avgTime = totalTime / runs;
        times[i] = avgTime;

        cout << "Среднее время: " << avgTime << " мс" << endl;

        // Проверяем правильность результата (только для первого теста)
        if (i == 0)
        {
            CUDA_CHECK(cudaMemcpy(h_C, d_C, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            if (verifyResults(h_A, h_B, h_C, ARRAY_SIZE))
            {
                cout << "Результаты корректны!" << endl;
            }
            else
            {
                cout << "ОШИБКА в вычислениях!" << endl;
            }
        }

        cout << endl;
    }

    //==========================================================================
    // АНАЛИЗ РЕЗУЛЬТАТОВ
    //==========================================================================

    cout << "=== Анализ результатов ===" << endl;
    cout << endl;

    // Находим лучший результат
    float bestTime = times[0];
    int bestBlockSize = blockSizes[0];

    for (int i = 1; i < numTests; i++)
    {
        if (times[i] > 0 && times[i] < bestTime)
        {
            bestTime = times[i];
            bestBlockSize = blockSizes[i];
        }
    }

    cout << "Лучший результат: блок размером " << bestBlockSize << " потоков" << endl;
    cout << "Время: " << bestTime << " мс" << endl;
    cout << endl;

    // Показываем сравнительную таблицу
    cout << "Сравнительная таблица:" << endl;
    cout << "---------------------------------------------" << endl;
    cout << "Размер блока | Время (мс) | Относительно" << endl;
    cout << "---------------------------------------------" << endl;

    for (int i = 0; i < numTests; i++)
    {
        if (times[i] > 0)
        {
            float relative = times[i] / bestTime;
            cout << "    " << blockSizes[i] << "      |   " << times[i] << "   |   ";

            if (blockSizes[i] == bestBlockSize)
            {
                cout << "ЛУЧШИЙ";
            }
            else
            {
                cout << "x" << relative;
            }

            cout << endl;
        }
    }
    cout << "---------------------------------------------" << endl;
    cout << endl;

    //==========================================================================
    // ВЫВОДЫ
    //==========================================================================

    cout << "=== Выводы ===" << endl;
    cout << endl;
    cout << "1. РАЗМЕР WARP (32 потока):" << endl;
    cout << "   Размер блока должен быть кратен 32, иначе часть warp" << endl;
    cout << "   будет простаивать." << endl;
    cout << endl;
    cout << "2. СЛИШКОМ МАЛЕНЬКИЙ БЛОК:" << endl;
    cout << "   Если блок меньше 128 потоков, GPU недоиспользуется." << endl;
    cout << "   Накладные расходы на управление блоками становятся заметны." << endl;
    cout << endl;
    cout << "3. СЛИШКОМ БОЛЬШОЙ БЛОК:" << endl;
    cout << "   Если блок больше 512-1024, может не хватить регистров" << endl;
    cout << "   и shared memory. Меньше блоков может выполняться одновременно." << endl;
    cout << endl;
    cout << "4. ОПТИМАЛЬНЫЙ РАЗМЕР:" << endl;
    cout << "   Обычно 128-256 потоков - золотая середина." << endl;
    cout << "   Но нужно тестировать для конкретной задачи!" << endl;
    cout << endl;
    cout << "5. ДЛЯ ЭТОЙ ЗАДАЧИ:" << endl;
    cout << "   Лучший результат показал блок размером " << bestBlockSize << " потоков." << endl;

    // Освобождаем память
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
