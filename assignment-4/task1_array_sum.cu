/*
 * Assignment 4 - Задание 1
 * Вычисление суммы элементов массива на GPU
 *
 * Программа вычисляет сумму всех элементов массива двумя способами:
 * 1. Последовательно на CPU (обычный цикл)
 * 2. Параллельно на GPU с использованием глобальной памяти
 *
 * Размер массива: 100,000 элементов
 *
 * Компиляция: nvcc -o task1 task1_array_sum.cu
 * Запуск: ./task1
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

using namespace std;

// Размер массива - 100 тысяч элементов
#define ARRAY_SIZE 100000

// Размер блока потоков
#define BLOCK_SIZE 256

// Макрос для проверки ошибок CUDA
// Если функция CUDA вернула ошибку - выводим сообщение и завершаем программу
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
// ВЕРСИЯ 1: ПОСЛЕДОВАТЕЛЬНАЯ СУММА НА CPU
//==============================================================================

/*
 * Обычная функция суммирования на CPU
 * Просто проходим по массиву и складываем все элементы
 */
double sumCPU(float* array, int size)
{
    double sum = 0.0;  // Используем double для точности

    // Обычный цикл - обрабатываем элементы по одному
    for (int i = 0; i < size; i++)
    {
        sum += array[i];
    }

    return sum;
}

//==============================================================================
// ВЕРСИЯ 2: ПАРАЛЛЕЛЬНАЯ СУММА НА GPU (ШАГ 1 - ЧАСТИЧНЫЕ СУММЫ)
//==============================================================================

/*
 * Ядро GPU которое вычисляет ЧАСТИЧНЫЕ суммы
 *
 * Проблема: мы не можем просто взять и сложить 100,000 чисел одним ядром
 * Решение: делаем это в два шага:
 *
 * Шаг 1 (это ядро): Каждый блок суммирует свою часть массива
 *   - Блок 0 суммирует элементы 0-255
 *   - Блок 1 суммирует элементы 256-511
 *   - И так далее...
 *   - Получаем массив частичных сумм (по одной на блок)
 *
 * Шаг 2 (следующее ядро): Суммируем эти частичные суммы
 *
 * Это называется "двухуровневая редукция"
 */
__global__ void sumReductionKernel(float* input, float* output, int size)
{
    // Вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Временная сумма для этого потока
    float sum = 0.0f;

    // Каждый поток суммирует несколько элементов
    // Это нужно если элементов больше чем потоков
    // Например: 100,000 элементов / 256 потоков в блоке = ~390 блоков
    // Каждый поток обрабатывает: 100,000 / (256 * 390) ≈ 1 элемент
    // Но для больших массивов может быть больше
    while (idx < size)
    {
        sum += input[idx];
        // Переходим к следующему элементу для этого потока
        // Прыгаем на количество всех потоков
        idx += blockDim.x * gridDim.x;
    }

    // Теперь нужно сложить суммы всех потоков В БЛОКЕ
    // Используем атомарную операцию для записи в глобальную память
    // atomicAdd безопасно складывает даже если несколько потоков
    // пишут в одно место одновременно
    //
    // output[blockIdx.x] - каждый блок пишет в свою ячейку
    atomicAdd(&output[blockIdx.x], sum);
}

//==============================================================================
// ВЕРСИЯ 2: ПАРАЛЛЕЛЬНАЯ СУММА НА GPU (ШАГ 2 - ФИНАЛЬНАЯ СУММА)
//==============================================================================

/*
 * Ядро которое суммирует частичные суммы от каждого блока
 *
 * После первого ядра у нас есть массив частичных сумм
 * (по одной от каждого блока)
 * Теперь нужно сложить эти частичные суммы
 *
 * Это ядро запускаем с одним блоком
 */
__global__ void sumFinalKernel(float* partialSums, float* result, int numBlocks)
{
    int idx = threadIdx.x;

    float sum = 0.0f;

    // Каждый поток суммирует свою часть частичных сумм
    while (idx < numBlocks)
    {
        sum += partialSums[idx];
        idx += blockDim.x;  // Переходим к следующему элементу
    }

    // Суммируем результаты всех потоков в блоке
    atomicAdd(result, sum);
}

//==============================================================================
// ФУНКЦИЯ ДЛЯ ЗАПУСКА GPU ВЕРСИИ
//==============================================================================

/*
 * Главная функция которая запускает вычисление суммы на GPU
 * Управляет двухшаговым процессом редукции
 */
double sumGPU(float* h_array, int size)
{
    // Указатели на память GPU
    float* d_input;      // Входной массив на GPU
    float* d_partial;    // Массив частичных сумм
    float* d_result;     // Финальный результат

    // Вычисляем сколько блоков нам нужно
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Выделяем память на GPU
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, numBlocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

    // Копируем входные данные на GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_array, size * sizeof(float), cudaMemcpyHostToDevice));

    // Обнуляем массивы результатов
    CUDA_CHECK(cudaMemset(d_partial, 0, numBlocks * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

    // ШАГ 1: Вычисляем частичные суммы для каждого блока
    sumReductionKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_partial, size);
    CUDA_CHECK(cudaDeviceSynchronize());  // Ждем завершения

    // ШАГ 2: Суммируем частичные суммы
    // Запускаем один блок который сложит все частичные суммы
    sumFinalKernel<<<1, BLOCK_SIZE>>>(d_partial, d_result, numBlocks);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Копируем результат обратно на CPU
    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    // Освобождаем память GPU
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_result));

    return (double)result;
}

//==============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
//==============================================================================

// Заполнить массив случайными числами
void fillArray(float* array, int size)
{
    for (int i = 0; i < size; i++)
    {
        // Случайные числа от 0 до 9
        array[i] = (float)(rand() % 10);
    }
}

//==============================================================================
// ГЛАВНАЯ ФУНКЦИЯ
//==============================================================================

int main()
{
    cout << "=== Assignment 4 - Задание 1 ===" << endl;
    cout << "Вычисление суммы элементов массива" << endl;
    cout << endl;

    cout << "Размер массива: " << ARRAY_SIZE << " элементов" << endl;
    cout << "Размер блока: " << BLOCK_SIZE << " потоков" << endl;
    cout << endl;

    // Проверяем наличие GPU
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        cout << "Ошибка: GPU не найден!" << endl;
        return 1;
    }

    // Информация о GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "GPU: " << prop.name << endl;
    cout << endl;

    // Инициализируем генератор случайных чисел
    srand(time(NULL));

    //==========================================================================
    // ПОДГОТОВКА ДАННЫХ
    //==========================================================================

    cout << "Подготовка данных..." << endl;

    // Выделяем память на CPU
    float* h_array = new float[ARRAY_SIZE];

    // Заполняем массив случайными числами
    fillArray(h_array, ARRAY_SIZE);

    cout << "Первые 10 элементов: ";
    for (int i = 0; i < 10; i++)
    {
        cout << h_array[i] << " ";
    }
    cout << endl << endl;

    //==========================================================================
    // ТЕСТ 1: СУММА НА CPU
    //==========================================================================

    cout << "--- Вычисление на CPU ---" << endl;

    // Засекаем время
    clock_t startCPU = clock();

    double sumCPU_result = sumCPU(h_array, ARRAY_SIZE);

    clock_t endCPU = clock();

    double timeCPU = (double)(endCPU - startCPU) / CLOCKS_PER_SEC * 1000.0;

    cout << "Сумма: " << sumCPU_result << endl;
    cout << "Время: " << timeCPU << " мс" << endl;
    cout << endl;

    //==========================================================================
    // ТЕСТ 2: СУММА НА GPU
    //==========================================================================

    cout << "--- Вычисление на GPU ---" << endl;

    // Создаем события CUDA для точного измерения времени
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Засекаем время
    CUDA_CHECK(cudaEventRecord(start));

    double sumGPU_result = sumGPU(h_array, ARRAY_SIZE);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float timeGPU;
    CUDA_CHECK(cudaEventElapsedTime(&timeGPU, start, stop));

    cout << "Сумма: " << sumGPU_result << endl;
    cout << "Время: " << timeGPU << " мс" << endl;
    cout << endl;

    //==========================================================================
    // СРАВНЕНИЕ РЕЗУЛЬТАТОВ
    //==========================================================================

    cout << "--- Сравнение ---" << endl;

    // Проверяем что результаты совпадают (с учетом погрешности вычислений)
    double difference = abs(sumCPU_result - sumGPU_result);
    double tolerance = 0.01 * sumCPU_result;  // Допускаем погрешность 1%

    if (difference < tolerance)
    {
        cout << "✓ Результаты совпадают (разница: " << difference << ")" << endl;
    }
    else
    {
        cout << "✗ ОШИБКА: результаты не совпадают!" << endl;
        cout << "  CPU: " << sumCPU_result << endl;
        cout << "  GPU: " << sumGPU_result << endl;
        cout << "  Разница: " << difference << endl;
    }
    cout << endl;

    // Сравниваем время
    cout << "Время CPU: " << timeCPU << " мс" << endl;
    cout << "Время GPU: " << timeGPU << " мс" << endl;

    if (timeGPU < timeCPU)
    {
        double speedup = timeCPU / timeGPU;
        cout << "GPU быстрее в " << speedup << " раз" << endl;
    }
    else
    {
        double slowdown = timeGPU / timeCPU;
        cout << "GPU медленнее в " << slowdown << " раз" << endl;
        cout << "(для маленьких задач накладные расходы могут превышать выгоду)" << endl;
    }
    cout << endl;

    //==========================================================================
    // ВЫВОДЫ
    //==========================================================================

    cout << "--- Выводы ---" << endl;
    cout << endl;
    cout << "1. РЕДУКЦИЯ (суммирование) - типичная задача для GPU" << endl;
    cout << "   Мы использовали двухуровневую редукцию:" << endl;
    cout << "   - Шаг 1: каждый блок суммирует свою часть" << endl;
    cout << "   - Шаг 2: суммируем результаты блоков" << endl;
    cout << endl;
    cout << "2. АТОМАРНЫЕ ОПЕРАЦИИ (atomicAdd)" << endl;
    cout << "   Позволяют безопасно писать в одно место из разных потоков" << endl;
    cout << "   Но они медленнее обычных операций!" << endl;
    cout << endl;
    cout << "3. ДЛЯ МАЛЕНЬКИХ МАССИВОВ GPU может быть МЕДЛЕННЕЕ CPU" << endl;
    cout << "   Накладные расходы:" << endl;
    cout << "   - Копирование данных CPU → GPU → CPU" << endl;
    cout << "   - Запуск ядра" << endl;
    cout << "   - Синхронизация" << endl;
    cout << endl;
    cout << "4. GPU ВЫГОДЕН для больших массивов (миллионы элементов)" << endl;
    cout << "   Параллелизм компенсирует накладные расходы" << endl;
    cout << endl;
    cout << "5. ОПТИМИЗАЦИЯ: использовать shared memory вместо atomicAdd" << endl;
    cout << "   (см. Задание 2)" << endl;

    // Освобождаем память
    delete[] h_array;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
