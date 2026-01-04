/*
 * Задача 3. Параллельная сортировка выбором с OpenMP
 *
 * Программа реализует сортировку выбором:
 * 1) Последовательную версию
 * 2) Параллельную версию с OpenMP
 *
 * Тестируется на массивах размером 1000 и 10000 элементов.
 *
 * Компиляция: g++ -fopenmp -o task3_selection_sort task3_selection_sort.cpp
 * Запуск: ./task3_selection_sort
 */

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <omp.h>

using namespace std;

// Функция для заполнения массива случайными числами
void fillArray(int arr[], int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = rand() % 10000;
    }
}

// Функция для копирования массива
void copyArray(int source[], int dest[], int size)
{
    for (int i = 0; i < size; i++)
    {
        dest[i] = source[i];
    }
}

// Функция для проверки что массив отсортирован
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

// Последовательная сортировка выбором
// Идея: находим минимальный элемент и ставим его на нужное место
void selectionSortSequential(int arr[], int size)
{
    for (int i = 0; i < size - 1; i++)
    {
        // Считаем что минимальный элемент на позиции i
        int minIndex = i;

        // Ищем минимальный элемент в оставшейся части
        for (int j = i + 1; j < size; j++)
        {
            if (arr[j] < arr[minIndex])
            {
                minIndex = j;
            }
        }

        // Меняем местами если нашли меньший элемент
        if (minIndex != i)
        {
            int temp = arr[i];
            arr[i] = arr[minIndex];
            arr[minIndex] = temp;
        }
    }
}

// Параллельная сортировка выбором с OpenMP
// Распараллеливаем поиск минимального элемента
void selectionSortParallel(int arr[], int size)
{
    for (int i = 0; i < size - 1; i++)
    {
        int minIndex = i;
        int minValue = arr[i];

        // Параллельный поиск минимума
        // Каждый поток ищет минимум в своей части массива
        #pragma omp parallel
        {
            int localMinIndex = i;
            int localMinValue = arr[i];

            // Распределяем итерации между потоками
            #pragma omp for nowait
            for (int j = i + 1; j < size; j++)
            {
                if (arr[j] < localMinValue)
                {
                    localMinValue = arr[j];
                    localMinIndex = j;
                }
            }

            // Критическая секция для обновления глобального минимума
            #pragma omp critical
            {
                if (localMinValue < minValue)
                {
                    minValue = localMinValue;
                    minIndex = localMinIndex;
                }
            }
        }

        // Меняем местами
        if (minIndex != i)
        {
            int temp = arr[i];
            arr[i] = arr[minIndex];
            arr[minIndex] = temp;
        }
    }
}

// Функция для тестирования производительности
void testPerformance(int size)
{
    cout << "========================================" << endl;
    cout << "Размер массива: " << size << " элементов" << endl;
    cout << "========================================" << endl;

    // Создаем массивы
    int* original = new int[size];
    int* arrSeq = new int[size];
    int* arrPar = new int[size];

    // Заполняем исходный массив
    fillArray(original, size);

    // Копируем для каждой версии
    copyArray(original, arrSeq, size);
    copyArray(original, arrPar, size);

    // ===== Последовательная сортировка =====
    cout << endl << "Последовательная сортировка выбором:" << endl;

    double startSeq = omp_get_wtime();
    selectionSortSequential(arrSeq, size);
    double endSeq = omp_get_wtime();

    double timeSeq = endSeq - startSeq;

    if (isSorted(arrSeq, size))
    {
        cout << "  Результат: массив отсортирован корректно" << endl;
    }
    else
    {
        cout << "  ОШИБКА: массив не отсортирован!" << endl;
    }
    cout << "  Время: " << timeSeq * 1000 << " мс" << endl;

    // ===== Параллельная сортировка =====
    cout << endl << "Параллельная сортировка выбором (OpenMP):" << endl;
    cout << "  Количество потоков: " << omp_get_max_threads() << endl;

    double startPar = omp_get_wtime();
    selectionSortParallel(arrPar, size);
    double endPar = omp_get_wtime();

    double timePar = endPar - startPar;

    if (isSorted(arrPar, size))
    {
        cout << "  Результат: массив отсортирован корректно" << endl;
    }
    else
    {
        cout << "  ОШИБКА: массив не отсортирован!" << endl;
    }
    cout << "  Время: " << timePar * 1000 << " мс" << endl;

    // ===== Сравнение =====
    cout << endl << "Сравнение:" << endl;

    if (timePar > 0)
    {
        double speedup = timeSeq / timePar;
        cout << "  Ускорение: " << speedup << "x" << endl;

        if (speedup > 1)
        {
            cout << "  Вывод: параллельная версия быстрее" << endl;
        }
        else
        {
            cout << "  Вывод: последовательная версия быстрее" << endl;
            cout << "         (накладные расходы на синхронизацию)" << endl;
        }
    }

    // Освобождаем память
    delete[] original;
    delete[] arrSeq;
    delete[] arrPar;
}

int main()
{
    cout << "=== Задача 3: Сортировка выбором с OpenMP ===" << endl;
    cout << endl;

    // Инициализация генератора случайных чисел
    srand(time(NULL));

    // Тест для 1000 элементов
    testPerformance(1000);

    cout << endl;

    // Тест для 10000 элементов
    testPerformance(10000);

    cout << endl;

    // ===== Общие выводы =====
    cout << "========================================" << endl;
    cout << "Общие выводы:" << endl;
    cout << "========================================" << endl;
    cout << endl;
    cout << "1. Сортировка выбором имеет сложность O(n^2), поэтому" << endl;
    cout << "   время сильно растет с увеличением размера массива." << endl;
    cout << endl;
    cout << "2. Параллелизация внутреннего цикла (поиск минимума)" << endl;
    cout << "   дает умеренное ускорение, но ограничена тем, что" << endl;
    cout << "   внешний цикл остается последовательным." << endl;
    cout << endl;
    cout << "3. Сортировка выбором плохо подходит для параллелизации," << endl;
    cout << "   так как каждая итерация зависит от предыдущей." << endl;
    cout << endl;
    cout << "4. Для лучшей параллельной производительности лучше" << endl;
    cout << "   использовать алгоритмы как quicksort или mergesort." << endl;

    return 0;
}
