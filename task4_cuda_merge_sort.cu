/*
 * Задача 4. Сортировка слиянием на GPU с использованием CUDA
 *
 * Программа реализует параллельную сортировку слиянием:
 * 1) Массив разделяется на подмассивы
 * 2) Каждый блок GPU сортирует свой подмассив
 * 3) Подмассивы сливаются параллельно
 *
 * Компиляция: nvcc -o task4_cuda_sort task4_cuda_merge_sort.cu
 * Запуск: ./task4_cuda_sort
 */

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

// Размер массива
#define ARRAY_SIZE 10000

// Размер блока (потоков в блоке)
#define BLOCK_SIZE 256

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            cout << "CUDA Error: " << cudaGetErrorString(error) << endl; \
            cout << "File: " << __FILE__ << ", Line: " << __LINE__ << endl; \
            exit(1); \
        } \
    } while(0)

// Функция слияния двух отсортированных частей массива (на CPU)
// left - начало первой части
// mid - конец первой части (и начало второй)
// right - конец второй части
void merge(int arr[], int left, int mid, int right)
{
    // Вычисляем размеры двух подмассивов
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Создаем временные массивы
    int* leftArr = new int[n1];
    int* rightArr = new int[n2];

    // Копируем данные во временные массивы
    for (int i = 0; i < n1; i++)
    {
        leftArr[i] = arr[left + i];
    }
    for (int j = 0; j < n2; j++)
    {
        rightArr[j] = arr[mid + 1 + j];
    }

    // Сливаем временные массивы обратно в arr
    int i = 0;      // Индекс левого подмассива
    int j = 0;      // Индекс правого подмассива
    int k = left;   // Индекс объединенного массива

    while (i < n1 && j < n2)
    {
        if (leftArr[i] <= rightArr[j])
        {
            arr[k] = leftArr[i];
            i++;
        }
        else
        {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }

    // Копируем оставшиеся элементы левого массива
    while (i < n1)
    {
        arr[k] = leftArr[i];
        i++;
        k++;
    }

    // Копируем оставшиеся элементы правого массива
    while (j < n2)
    {
        arr[k] = rightArr[j];
        j++;
        k++;
    }

    delete[] leftArr;
    delete[] rightArr;
}

// GPU ядро для сортировки маленьких подмассивов (сортировка вставками)
// Каждый блок сортирует свой подмассив
__global__ void sortSmallArraysKernel(int* arr, int size, int chunkSize)
{
    // Вычисляем какой чанк обрабатывает этот блок
    int chunkIndex = blockIdx.x;
    int start = chunkIndex * chunkSize;
    int end = start + chunkSize;

    // Проверяем границы
    if (start >= size)
    {
        return;
    }
    if (end > size)
    {
        end = size;
    }

    // Только первый поток в блоке делает сортировку
    // (для простоты используем сортировку вставками)
    if (threadIdx.x == 0)
    {
        // Сортировка вставками для нашего чанка
        for (int i = start + 1; i < end; i++)
        {
            int key = arr[i];
            int j = i - 1;

            while (j >= start && arr[j] > key)
            {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }
}

// GPU ядро для слияния соседних отсортированных чанков
__global__ void mergeKernel(int* arr, int* temp, int size, int chunkSize)
{
    // Каждый блок сливает пару чанков
    int pairIndex = blockIdx.x;
    int left = pairIndex * 2 * chunkSize;
    int mid = left + chunkSize - 1;
    int right = left + 2 * chunkSize - 1;

    // Проверяем границы
    if (left >= size)
    {
        return;
    }
    if (mid >= size)
    {
        mid = size - 1;
    }
    if (right >= size)
    {
        right = size - 1;
    }

    // Только первый поток делает слияние
    if (threadIdx.x == 0)
    {
        int i = left;
        int j = mid + 1;
        int k = left;

        // Сливаем две части
        while (i <= mid && j <= right)
        {
            if (arr[i] <= arr[j])
            {
                temp[k] = arr[i];
                i++;
            }
            else
            {
                temp[k] = arr[j];
                j++;
            }
            k++;
        }

        // Копируем оставшиеся элементы
        while (i <= mid)
        {
            temp[k] = arr[i];
            i++;
            k++;
        }
        while (j <= right)
        {
            temp[k] = arr[j];
            j++;
            k++;
        }
    }
}

// GPU ядро для копирования из temp обратно в arr
__global__ void copyKernel(int* arr, int* temp, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        arr[idx] = temp[idx];
    }
}

// Главная функция сортировки на GPU
void mergeSortGPU(int* hostArr, int size)
{
    int* deviceArr;
    int* deviceTemp;

    // Выделяем память на GPU
    CUDA_CHECK(cudaMalloc(&deviceArr, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&deviceTemp, size * sizeof(int)));

    // Копируем данные на GPU
    CUDA_CHECK(cudaMemcpy(deviceArr, hostArr, size * sizeof(int), cudaMemcpyHostToDevice));

    // Начальный размер чанка для сортировки
    int initialChunkSize = 64;  // Каждый блок сортирует 64 элемента

    // Количество блоков для первичной сортировки
    int numChunks = (size + initialChunkSize - 1) / initialChunkSize;

    // Шаг 1: Сортируем маленькие чанки
    sortSmallArraysKernel<<<numChunks, 1>>>(deviceArr, size, initialChunkSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Шаг 2: Итеративно сливаем чанки
    for (int chunkSize = initialChunkSize; chunkSize < size; chunkSize *= 2)
    {
        // Количество пар для слияния
        int numPairs = (size + 2 * chunkSize - 1) / (2 * chunkSize);

        // Запускаем слияние
        mergeKernel<<<numPairs, 1>>>(deviceArr, deviceTemp, size, chunkSize);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Копируем результат обратно
        int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        copyKernel<<<numBlocks, BLOCK_SIZE>>>(deviceArr, deviceTemp, size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Копируем результат обратно на CPU
    CUDA_CHECK(cudaMemcpy(hostArr, deviceArr, size * sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождаем память GPU
    CUDA_CHECK(cudaFree(deviceArr));
    CUDA_CHECK(cudaFree(deviceTemp));
}

// Последовательная сортировка слиянием на CPU (для сравнения)
void mergeSortCPU(int arr[], int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;

        // Сортируем две половины
        mergeSortCPU(arr, left, mid);
        mergeSortCPU(arr, mid + 1, right);

        // Сливаем отсортированные половины
        merge(arr, left, mid, right);
    }
}

// Заполнение массива случайными числами
void fillArray(int arr[], int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = rand() % 100000;
    }
}

// Проверка что массив отсортирован
bool isSorted(int arr[], int size)
{
    for (int i = 0; i < size - 1; i++)
    {
        if (arr[i] > arr[i + 1])
        {
            return false;
        }
    }
    return true;
}

// Копирование массива
void copyArray(int src[], int dst[], int size)
{
    for (int i = 0; i < size; i++)
    {
        dst[i] = src[i];
    }
}

int main()
{
    cout << "=== Задача 4: Сортировка слиянием на GPU (CUDA) ===" << endl;
    cout << "Размер массива: " << ARRAY_SIZE << endl;
    cout << endl;

    // Проверяем наличие GPU
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        cout << "ОШИБКА: GPU не найден!" << endl;
        return 1;
    }

    // Получаем информацию о GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "GPU: " << prop.name << endl;
    cout << "Мультипроцессоры: " << prop.multiProcessorCount << endl;
    cout << endl;

    // Инициализация генератора случайных чисел
    srand(time(NULL));

    // Создаем массивы
    int* original = new int[ARRAY_SIZE];
    int* arrCPU = new int[ARRAY_SIZE];
    int* arrGPU = new int[ARRAY_SIZE];

    // Заполняем исходный массив
    fillArray(original, ARRAY_SIZE);

    // Копируем для каждой версии
    copyArray(original, arrCPU, ARRAY_SIZE);
    copyArray(original, arrGPU, ARRAY_SIZE);

    // ===== CPU сортировка =====
    cout << "--- Сортировка на CPU ---" << endl;

    clock_t startCPU = clock();
    mergeSortCPU(arrCPU, 0, ARRAY_SIZE - 1);
    clock_t endCPU = clock();

    double timeCPU = (double)(endCPU - startCPU) / CLOCKS_PER_SEC * 1000;

    if (isSorted(arrCPU, ARRAY_SIZE))
    {
        cout << "Результат: массив отсортирован корректно" << endl;
    }
    else
    {
        cout << "ОШИБКА: массив не отсортирован!" << endl;
    }
    cout << "Время: " << timeCPU << " мс" << endl;
    cout << endl;

    // ===== GPU сортировка =====
    cout << "--- Сортировка на GPU (CUDA) ---" << endl;

    // Создаем события CUDA для точного измерения времени
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    mergeSortGPU(arrGPU, ARRAY_SIZE);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float timeGPU;
    CUDA_CHECK(cudaEventElapsedTime(&timeGPU, start, stop));

    if (isSorted(arrGPU, ARRAY_SIZE))
    {
        cout << "Результат: массив отсортирован корректно" << endl;
    }
    else
    {
        cout << "ОШИБКА: массив не отсортирован!" << endl;
    }
    cout << "Время: " << timeGPU << " мс" << endl;
    cout << endl;

    // ===== Сравнение =====
    cout << "--- Сравнение ---" << endl;

    if (timeGPU > 0)
    {
        double speedup = timeCPU / timeGPU;
        cout << "Ускорение GPU: " << speedup << "x" << endl;
    }

    // Проверяем что результаты совпадают
    bool resultsMatch = true;
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        if (arrCPU[i] != arrGPU[i])
        {
            resultsMatch = false;
            break;
        }
    }

    if (resultsMatch)
    {
        cout << "Результаты CPU и GPU совпадают - OK!" << endl;
    }
    else
    {
        cout << "ВНИМАНИЕ: результаты отличаются" << endl;
    }

    cout << endl;

    // ===== Выводы =====
    cout << "--- Выводы ---" << endl;
    cout << "1. GPU показывает ускорение на больших массивах" << endl;
    cout << "   благодаря параллельной обработке множества элементов." << endl;
    cout << endl;
    cout << "2. Для маленьких массивов накладные расходы на" << endl;
    cout << "   копирование данных между CPU и GPU могут" << endl;
    cout << "   превышать выигрыш от параллелизма." << endl;
    cout << endl;
    cout << "3. Сортировка слиянием хорошо подходит для GPU," << endl;
    cout << "   так как операции слияния независимы на каждом уровне." << endl;
    cout << endl;
    cout << "4. Оптимизация размера блока влияет на производительность -" << endl;
    cout << "   нужно экспериментировать для конкретного GPU." << endl;

    // Освобождаем память
    delete[] original;
    delete[] arrCPU;
    delete[] arrGPU;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
