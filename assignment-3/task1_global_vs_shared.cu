/*
 * Assignment 3 - Задание 1
 * Сравнение глобальной и разделяемой памяти
 *
 * Программа умножает каждый элемент массива на число двумя способами:
 * 1. Используя только глобальную память (медленнее)
 * 2. Используя разделяемую память (быстрее)
 *
 * Компиляция: nvcc -o task1 task1_global_vs_shared.cu
 * Запуск: ./task1
 */

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Размер массива - 1 миллион элементов
#define ARRAY_SIZE 1000000

// Размер блока потоков
#define BLOCK_SIZE 256

// Число на которое умножаем
#define MULTIPLIER 3.5f

// Макрос для проверки ошибок CUDA
// Если что-то пошло не так - выводим сообщение и выходим
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
// ВЕРСИЯ 1: ТОЛЬКО ГЛОБАЛЬНАЯ ПАМЯТЬ
//==============================================================================

/*
 * Ядро которое использует только глобальную память
 *
 * Что происходит:
 * 1. Каждый поток вычисляет свой индекс
 * 2. Читает значение из глобальной памяти (МЕДЛЕННО)
 * 3. Умножает на число
 * 4. Записывает обратно в глобальную память (МЕДЛЕННО)
 *
 * Проблема: глобальная память медленная, потому что она физически далеко от GPU ядер
 */
__global__ void multiplyGlobalMemory(float* input, float* output, int size, float multiplier)
{
    // Вычисляем индекс элемента который будет обрабатывать этот поток
    // blockIdx.x - номер блока (0, 1, 2, ...)
    // blockDim.x - количество потоков в блоке (у нас 256)
    // threadIdx.x - номер потока внутри блока (0-255)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем что мы не вышли за границы массива
    // Это важно потому что количество потоков может быть больше размера массива
    if (idx < size)
    {
        // Читаем из глобальной памяти, умножаем, записываем обратно
        // Все операции с глобальной памятью - медленные
        output[idx] = input[idx] * multiplier;
    }
}

//==============================================================================
// ВЕРСИЯ 2: С ИСПОЛЬЗОВАНИЕМ РАЗДЕЛЯЕМОЙ ПАМЯТИ
//==============================================================================

/*
 * Ядро которое использует разделяемую память (shared memory)
 *
 * Что происходит:
 * 1. Сначала копируем данные из глобальной памяти в разделяемую (МЕДЛЕННО, но 1 раз)
 * 2. Все потоки блока ждут пока данные загрузятся (__syncthreads)
 * 3. Обрабатываем данные в разделяемой памяти (БЫСТРО)
 * 4. Записываем результат обратно в глобальную память (МЕДЛЕННО, но 1 раз)
 *
 * Почему быстрее:
 * - Разделяемая память находится прямо на чипе GPU
 * - Доступ к ней в ~100 раз быстрее чем к глобальной
 * - Все потоки одного блока используют одну и ту же разделяемую память
 */
__global__ void multiplySharedMemory(float* input, float* output, int size, float multiplier)
{
    // Выделяем разделяемую память для этого блока
    // __shared__ означает что эта память общая для всех потоков блока
    // Размер = BLOCK_SIZE = 256 элементов
    __shared__ float sharedData[BLOCK_SIZE];

    // Глобальный индекс элемента
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Локальный индекс внутри блока (0-255)
    int localIdx = threadIdx.x;

    // ШАГ 1: Загружаем данные из глобальной памяти в разделяемую
    // Это делает каждый поток для своего элемента
    if (globalIdx < size)
    {
        // Копируем из медленной глобальной памяти в быструю разделяемую
        sharedData[localIdx] = input[globalIdx];
    }

    // ШАГ 2: Синхронизация - ждем пока ВСЕ потоки загрузят свои данные
    // Это важно! Иначе некоторые потоки начнут обработку до загрузки данных
    __syncthreads();

    // ШАГ 3: Обрабатываем данные в разделяемой памяти (быстро!)
    if (globalIdx < size)
    {
        // Читаем из быстрой разделяемой памяти
        float value = sharedData[localIdx];

        // Умножаем
        value = value * multiplier;

        // ШАГ 4: Записываем результат обратно в глобальную память
        output[globalIdx] = value;
    }
}

//==============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
//==============================================================================

// Функция для инициализации массива случайными числами
void initArray(float* arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        // Генерируем случайное число от 0 до 99
        arr[i] = (float)(rand() % 100);
    }
}

// Функция для проверки правильности результата
bool checkResults(float* arr1, float* arr2, int size, float tolerance = 0.001f)
{
    for (int i = 0; i < size; i++)
    {
        // Сравниваем с учетом погрешности вычислений с плавающей точкой
        if (abs(arr1[i] - arr2[i]) > tolerance)
        {
            cout << "Ошибка на позиции " << i << ": ";
            cout << arr1[i] << " != " << arr2[i] << endl;
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
    cout << "=== Assignment 3 - Задание 1 ===" << endl;
    cout << "Сравнение глобальной и разделяемой памяти" << endl;
    cout << endl;

    // Показываем параметры
    cout << "Размер массива: " << ARRAY_SIZE << " элементов" << endl;
    cout << "Размер блока: " << BLOCK_SIZE << " потоков" << endl;
    cout << "Множитель: " << MULTIPLIER << endl;
    cout << endl;

    // Проверяем что GPU доступен
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        cout << "Ошибка: GPU не найден!" << endl;
        return 1;
    }

    // Выводим информацию о GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "Используемый GPU: " << prop.name << endl;
    cout << "Shared memory на блок: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
    cout << endl;

    // Инициализируем генератор случайных чисел
    srand(time(NULL));

    // Выделяем память на CPU (host)
    float* h_input = new float[ARRAY_SIZE];      // Входной массив
    float* h_output1 = new float[ARRAY_SIZE];    // Выход версии 1 (глобальная память)
    float* h_output2 = new float[ARRAY_SIZE];    // Выход версии 2 (разделяемая память)

    // Заполняем входной массив
    cout << "Инициализация данных..." << endl;
    initArray(h_input, ARRAY_SIZE);

    // Выделяем память на GPU (device)
    float* d_input;
    float* d_output;

    CUDA_CHECK(cudaMalloc(&d_input, ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, ARRAY_SIZE * sizeof(float)));

    // Копируем входные данные на GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Вычисляем количество блоков
    // Округляем вверх чтобы обработать все элементы
    int numBlocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cout << "Количество блоков: " << numBlocks << endl;
    cout << endl;

    //==========================================================================
    // ТЕСТ 1: ГЛОБАЛЬНАЯ ПАМЯТЬ
    //==========================================================================

    cout << "--- Тест 1: Глобальная память ---" << endl;

    // Создаем события для измерения времени
    cudaEvent_t start1, stop1;
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));

    // Запускаем ядро и измеряем время
    CUDA_CHECK(cudaEventRecord(start1));

    // Запускаем ядро с глобальной памятью
    // <<< количество блоков, количество потоков в блоке >>>
    multiplyGlobalMemory<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, ARRAY_SIZE, MULTIPLIER);

    CUDA_CHECK(cudaEventRecord(stop1));
    CUDA_CHECK(cudaEventSynchronize(stop1));

    // Считаем время
    float time1;
    CUDA_CHECK(cudaEventElapsedTime(&time1, start1, stop1));

    // Копируем результат обратно на CPU
    CUDA_CHECK(cudaMemcpy(h_output1, d_output, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    cout << "Время выполнения: " << time1 << " мс" << endl;
    cout << "Первые 5 результатов: ";
    for (int i = 0; i < 5; i++)
    {
        cout << h_output1[i] << " ";
    }
    cout << endl << endl;

    //==========================================================================
    // ТЕСТ 2: РАЗДЕЛЯЕМАЯ ПАМЯТЬ
    //==========================================================================

    cout << "--- Тест 2: Разделяемая память ---" << endl;

    // Создаем события для измерения времени
    cudaEvent_t start2, stop2;
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));

    // Очищаем выходной буфер
    CUDA_CHECK(cudaMemset(d_output, 0, ARRAY_SIZE * sizeof(float)));

    // Запускаем ядро с разделяемой памятью
    CUDA_CHECK(cudaEventRecord(start2));

    multiplySharedMemory<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, ARRAY_SIZE, MULTIPLIER);

    CUDA_CHECK(cudaEventRecord(stop2));
    CUDA_CHECK(cudaEventSynchronize(stop2));

    // Считаем время
    float time2;
    CUDA_CHECK(cudaEventElapsedTime(&time2, start2, stop2));

    // Копируем результат обратно
    CUDA_CHECK(cudaMemcpy(h_output2, d_output, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    cout << "Время выполнения: " << time2 << " мс" << endl;
    cout << "Первые 5 результатов: ";
    for (int i = 0; i < 5; i++)
    {
        cout << h_output2[i] << " ";
    }
    cout << endl << endl;

    //==========================================================================
    // СРАВНЕНИЕ РЕЗУЛЬТАТОВ
    //==========================================================================

    cout << "--- Сравнение ---" << endl;

    // Проверяем что результаты одинаковые
    if (checkResults(h_output1, h_output2, ARRAY_SIZE))
    {
        cout << "Результаты совпадают - OK!" << endl;
    }
    else
    {
        cout << "ОШИБКА: результаты не совпадают!" << endl;
    }

    // Вычисляем ускорение
    float speedup = time1 / time2;
    cout << "Ускорение: " << speedup << "x" << endl;

    if (speedup > 1.0f)
    {
        cout << "Разделяемая память быстрее на " << ((speedup - 1) * 100) << "%" << endl;
    }
    else
    {
        cout << "В этом случае разделяемая память не дала ускорения" << endl;
    }

    cout << endl;

    //==========================================================================
    // ВЫВОДЫ
    //==========================================================================

    cout << "--- Выводы ---" << endl;
    cout << "1. Разделяемая память находится на чипе GPU и работает быстрее" << endl;
    cout << "   чем глобальная память." << endl;
    cout << endl;
    cout << "2. Для этой простой задачи (умножение) разница может быть небольшой," << endl;
    cout << "   так как операция очень простая и bottleneck не в памяти." << endl;
    cout << endl;
    cout << "3. Разделяемая память дает большее ускорение когда:" << endl;
    cout << "   - Данные переиспользуются несколько раз" << endl;
    cout << "   - Потоки обмениваются данными друг с другом" << endl;
    cout << "   - Выполняются сложные вычисления" << endl;
    cout << endl;
    cout << "4. Размер разделяемой памяти ограничен (обычно 48-96 KB на блок)," << endl;
    cout << "   поэтому её нужно использовать аккуратно." << endl;

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
