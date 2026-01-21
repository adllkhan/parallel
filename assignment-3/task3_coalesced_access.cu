/*
 * Assignment 3 - Задание 3
 * Коалесцированный vs некоалесцированный доступ к памяти
 *
 * Программа копирует массив двумя способами:
 * 1. Коалесцированный доступ - соседние потоки читают соседние адреса (БЫСТРО)
 * 2. Некоалесцированный доступ - соседние потоки читают разные адреса (МЕДЛЕННО)
 *
 * Компиляция: nvcc -o task3 task3_coalesced_access.cu
 * Запуск: ./task3
 */

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Размер массива
#define ARRAY_SIZE 1000000

// Размер блока
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
// ВЕРСИЯ 1: КОАЛЕСЦИРОВАННЫЙ ДОСТУП (ПРАВИЛЬНЫЙ СПОСОБ)
//==============================================================================

/*
 * Ядро с коалесцированным доступом к памяти
 *
 * Что такое коалесцированный доступ:
 * - Соседние потоки (в одном warp) читают соседние адреса в памяти
 * - GPU может объединить эти обращения в одну транзакцию памяти
 * - Это ОЧЕНЬ эффективно!
 *
 * Пример для потоков 0-3:
 *   Поток 0 читает input[0]
 *   Поток 1 читает input[1]
 *   Поток 2 читает input[2]
 *   Поток 3 читает input[3]
 *   → GPU делает ОДНО обращение к памяти для всех 4 потоков!
 *
 * Каждый поток:
 * 1. Вычисляет свой индекс
 * 2. Читает элемент по этому индексу (соседние потоки → соседние адреса)
 * 3. Умножает на 2 (простая обработка)
 * 4. Записывает по тому же индексу
 */
__global__ void copyCoalesced(float* input, float* output, int size)
{
    // Вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем границы
    if (idx < size)
    {
        // КОАЛЕСЦИРОВАННЫЙ доступ:
        // Поток 0 читает input[0]
        // Поток 1 читает input[1]
        // Поток 2 читает input[2]
        // и т.д. - все адреса идут подряд!
        //
        // GPU объединяет эти обращения в минимальное количество
        // транзакций памяти (обычно 1-2 транзакции на warp)
        float value = input[idx];

        // Простая обработка - умножаем на 2
        value = value * 2.0f;

        // Записываем результат (тоже коалесцированно)
        output[idx] = value;
    }
}

//==============================================================================
// ВЕРСИЯ 2: НЕКОАЛЕСЦИРОВАННЫЙ ДОСТУП (ПЛОХОЙ СПОСОБ)
//==============================================================================

/*
 * Ядро с некоалесцированным доступом к памяти
 *
 * Что такое некоалесцированный доступ:
 * - Соседние потоки читают НЕсоседние адреса
 * - GPU не может объединить обращения
 * - Каждое обращение = отдельная транзакция памяти
 * - Это МЕДЛЕННО!
 *
 * Мы создаем некоалесцированный доступ используя stride (шаг):
 *
 * Пример со stride=32 для потоков 0-3:
 *   Поток 0 читает input[0]
 *   Поток 1 читает input[32]
 *   Поток 2 читает input[64]
 *   Поток 3 читает input[96]
 *   → GPU делает 4 ОТДЕЛЬНЫХ обращения к памяти!
 *
 * Stride выбран равным 32 (размер warp) чтобы гарантировать
 * что соседние потоки обращаются к разным блокам памяти
 */
__global__ void copyStrided(float* input, float* output, int size)
{
    // Вычисляем глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Stride (шаг) - делаем его равным размеру warp (32)
    // Это гарантирует что соседние потоки читают из разных
    // кэш-линий памяти
    int stride = 32;

    // Вычисляем индекс с которым будем работать
    // Используем stride чтобы "разбросать" доступы
    int block = tid / stride;  // Номер блока потоков
    int offset = tid % stride;  // Смещение внутри блока

    // Новый индекс: сначала смещение, потом блоки
    // Это создает паттерн доступа где соседние потоки
    // читают далеко друг от друга
    int idx = offset * (size / stride) + block;

    // Проверяем границы
    if (idx < size)
    {
        // НЕКОАЛЕСЦИРОВАННЫЙ доступ:
        // Поток 0 читает input[0]
        // Поток 1 читает input[31250]  (size/32)
        // Поток 2 читает input[62500]
        // и т.д. - адреса далеко друг от друга!
        //
        // GPU НЕ может объединить эти обращения
        // Каждое обращение = отдельная транзакция (МЕДЛЕННО)
        float value = input[idx];

        // Та же обработка
        value = value * 2.0f;

        // Записываем (тоже некоалесцированно)
        output[idx] = value;
    }
}

//==============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
//==============================================================================

// Заполнить массив
void fillArray(float* arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = (float)(i % 100);
    }
}

// Проверить результат
bool verifyResult(float* input, float* output, int size)
{
    // Проверяем первые 1000 элементов
    int checkCount = min(1000, size);

    for (int i = 0; i < checkCount; i++)
    {
        float expected = input[i] * 2.0f;

        if (abs(output[i] - expected) > 0.001f)
        {
            cout << "Ошибка на позиции " << i << ": ";
            cout << output[i] << " != " << expected << endl;
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
    cout << "=== Assignment 3 - Задание 3 ===" << endl;
    cout << "Коалесцированный vs некоалесцированный доступ к памяти" << endl;
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

    // Информация о GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "GPU: " << prop.name << endl;
    cout << "Размер warp: " << prop.warpSize << " потоков" << endl;
    cout << "Глобальная память: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
    cout << endl;

    //==========================================================================
    // ПОДГОТОВКА ДАННЫХ
    //==========================================================================

    cout << "Подготовка данных..." << endl;

    // Память на CPU
    float* h_input = new float[ARRAY_SIZE];
    float* h_output1 = new float[ARRAY_SIZE];  // Результат коалесцированного
    float* h_output2 = new float[ARRAY_SIZE];  // Результат некоалесцированного

    // Заполняем входной массив
    fillArray(h_input, ARRAY_SIZE);

    // Память на GPU
    float* d_input;
    float* d_output;

    CUDA_CHECK(cudaMalloc(&d_input, ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, ARRAY_SIZE * sizeof(float)));

    // Копируем на GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Количество блоков
    int numBlocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cout << "Количество блоков: " << numBlocks << endl;
    cout << endl;

    //==========================================================================
    // ТЕСТ 1: КОАЛЕСЦИРОВАННЫЙ ДОСТУП
    //==========================================================================

    cout << "--- Тест 1: Коалесцированный доступ ---" << endl;
    cout << "Паттерн: соседние потоки → соседние адреса" << endl;
    cout << "Пример: поток 0 → input[0], поток 1 → input[1], ..." << endl;
    cout << endl;

    // События для измерения времени
    cudaEvent_t start1, stop1;
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));

    // Запускаем несколько раз для точности
    int runs = 10;
    float totalTime1 = 0;

    for (int i = 0; i < runs; i++)
    {
        CUDA_CHECK(cudaEventRecord(start1));

        // Запускаем ядро с коалесцированным доступом
        copyCoalesced<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, ARRAY_SIZE);

        CUDA_CHECK(cudaEventRecord(stop1));
        CUDA_CHECK(cudaEventSynchronize(stop1));

        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start1, stop1));
        totalTime1 += time;
    }

    float avgTime1 = totalTime1 / runs;

    // Копируем результат
    CUDA_CHECK(cudaMemcpy(h_output1, d_output, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    cout << "Среднее время: " << avgTime1 << " мс" << endl;

    if (verifyResult(h_input, h_output1, ARRAY_SIZE))
    {
        cout << "Результат корректен" << endl;
    }
    else
    {
        cout << "ОШИБКА в результате!" << endl;
    }

    cout << endl;

    //==========================================================================
    // ТЕСТ 2: НЕКОАЛЕСЦИРОВАННЫЙ ДОСТУП
    //==========================================================================

    cout << "--- Тест 2: Некоалесцированный доступ ---" << endl;
    cout << "Паттерн: соседние потоки → далекие адреса" << endl;
    cout << "Пример: поток 0 → input[0], поток 1 → input[31250], ..." << endl;
    cout << endl;

    // Очищаем выходной буфер
    CUDA_CHECK(cudaMemset(d_output, 0, ARRAY_SIZE * sizeof(float)));

    // События для измерения времени
    cudaEvent_t start2, stop2;
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));

    // Запускаем несколько раз
    float totalTime2 = 0;

    for (int i = 0; i < runs; i++)
    {
        CUDA_CHECK(cudaEventRecord(start2));

        // Запускаем ядро с некоалесцированным доступом
        copyStrided<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, ARRAY_SIZE);

        CUDA_CHECK(cudaEventRecord(stop2));
        CUDA_CHECK(cudaEventSynchronize(stop2));

        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start2, stop2));
        totalTime2 += time;
    }

    float avgTime2 = totalTime2 / runs;

    // Копируем результат
    CUDA_CHECK(cudaMemcpy(h_output2, d_output, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    cout << "Среднее время: " << avgTime2 << " мс" << endl;

    if (verifyResult(h_input, h_output2, ARRAY_SIZE))
    {
        cout << "Результат корректен" << endl;
    }
    else
    {
        cout << "ОШИБКА в результате!" << endl;
    }

    cout << endl;

    //==========================================================================
    // СРАВНЕНИЕ
    //==========================================================================

    cout << "=== Сравнение ===" << endl;
    cout << endl;

    cout << "Коалесцированный доступ:     " << avgTime1 << " мс" << endl;
    cout << "Некоалесцированный доступ:   " << avgTime2 << " мс" << endl;
    cout << endl;

    float slowdown = avgTime2 / avgTime1;
    cout << "Замедление: " << slowdown << "x" << endl;
    cout << "Некоалесцированный доступ медленнее на " << ((slowdown - 1) * 100) << "%" << endl;
    cout << endl;

    //==========================================================================
    // ВИЗУАЛИЗАЦИЯ ПАТТЕРНОВ ДОСТУПА
    //==========================================================================

    cout << "=== Визуализация паттернов доступа ===" << endl;
    cout << endl;

    cout << "Коалесцированный (первые 8 потоков warp):" << endl;
    cout << "  Поток 0 → Адрес 0" << endl;
    cout << "  Поток 1 → Адрес 1" << endl;
    cout << "  Поток 2 → Адрес 2" << endl;
    cout << "  Поток 3 → Адрес 3" << endl;
    cout << "  ..." << endl;
    cout << "  → Все адреса последовательные!" << endl;
    cout << "  → GPU делает 1-2 транзакции памяти на весь warp" << endl;
    cout << endl;

    cout << "Некоалесцированный (первые 4 потока warp):" << endl;
    cout << "  Поток 0 → Адрес 0" << endl;
    cout << "  Поток 1 → Адрес " << (ARRAY_SIZE / 32) << endl;
    cout << "  Поток 2 → Адрес " << (2 * ARRAY_SIZE / 32) << endl;
    cout << "  Поток 3 → Адрес " << (3 * ARRAY_SIZE / 32) << endl;
    cout << "  ..." << endl;
    cout << "  → Адреса далеко друг от друга!" << endl;
    cout << "  → GPU делает 32 отдельные транзакции на warp" << endl;
    cout << endl;

    //==========================================================================
    // ВЫВОДЫ
    //==========================================================================

    cout << "=== Выводы ===" << endl;
    cout << endl;
    cout << "1. КОАЛЕСЦИРОВАННЫЙ ДОСТУП - ключ к производительности GPU!" << endl;
    cout << "   Разница может достигать 10-20x." << endl;
    cout << endl;
    cout << "2. ПРАВИЛО: соседние потоки должны читать соседние адреса." << endl;
    cout << "   Формула: idx = blockIdx.x * blockDim.x + threadIdx.x" << endl;
    cout << endl;
    cout << "3. GPU объединяет обращения соседних потоков (warp) в одну" << endl;
    cout << "   или несколько транзакций памяти." << endl;
    cout << endl;
    cout << "4. ПЛОХИЕ ПАТТЕРНЫ:" << endl;
    cout << "   - Обращения со stride (шагом)" << endl;
    cout << "   - Случайные обращения" << endl;
    cout << "   - Обращения с разными индексами в условных операторах" << endl;
    cout << endl;
    cout << "5. ХОРОШИЕ ПАТТЕРНЫ:" << endl;
    cout << "   - Последовательные обращения (idx, idx+1, idx+2, ...)" << endl;
    cout << "   - Выровненные обращения (кратны размеру кэш-линии)" << endl;
    cout << endl;
    cout << "6. При разработке CUDA программ ВСЕГДА думайте о паттерне" << endl;
    cout << "   доступа к памяти!" << endl;

    // Освобождаем память
    delete[] h_input;
    delete[] h_output1;
    delete[] h_output2;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));

    return 0;
}
