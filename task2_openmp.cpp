/*
 * Задача 2. Работа с массивами и OpenMP
 *
 * Программа создает массив из 10000 случайных чисел и находит
 * минимальное и максимальное значения двумя способами:
 * 1) Последовательно
 * 2) Параллельно с OpenMP
 *
 * Компиляция: g++ -fopenmp -o task2_openmp task2_openmp.cpp
 * Запуск: ./task2_openmp
 */

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// Размер массива
const int ARRAY_SIZE = 10000;

// Функция для заполнения массива случайными числами
void fillArrayWithRandomNumbers(int arr[], int size)
{
    // Используем srand для инициализации генератора
    srand(time(NULL));

    for (int i = 0; i < size; i++)
    {
        // Генерируем числа от 0 до 99999
        arr[i] = rand() % 100000;
    }
}

// Последовательный поиск минимума и максимума
void findMinMaxSequential(int arr[], int size, int &minVal, int &maxVal)
{
    // Начинаем с первого элемента
    minVal = arr[0];
    maxVal = arr[0];

    // Проходим по всему массиву
    for (int i = 1; i < size; i++)
    {
        if (arr[i] < minVal)
        {
            minVal = arr[i];
        }
        if (arr[i] > maxVal)
        {
            maxVal = arr[i];
        }
    }
}

// Параллельный поиск минимума и максимума с OpenMP
void findMinMaxParallel(int arr[], int size, int &minVal, int &maxVal)
{
    // Начинаем с первого элемента
    minVal = arr[0];
    maxVal = arr[0];

    // Используем reduction для поиска min и max
    // reduction(min:minVal) - каждый поток находит свой минимум, потом они объединяются
    // reduction(max:maxVal) - то же самое для максимума
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (int i = 1; i < size; i++)
    {
        if (arr[i] < minVal)
        {
            minVal = arr[i];
        }
        if (arr[i] > maxVal)
        {
            maxVal = arr[i];
        }
    }
}

int main()
{
    cout << "=== Задача 2: Поиск минимума и максимума ===" << endl;
    cout << "Размер массива: " << ARRAY_SIZE << endl;
    cout << endl;

    // Создаем массив
    int* numbers = new int[ARRAY_SIZE];

    // Заполняем случайными числами
    fillArrayWithRandomNumbers(numbers, ARRAY_SIZE);

    // Показываем сколько потоков используется
    cout << "Количество потоков OpenMP: " << omp_get_max_threads() << endl;
    cout << endl;

    // Переменные для результатов
    int minSeq, maxSeq;
    int minPar, maxPar;

    // ===== Последовательная версия =====
    cout << "--- Последовательная версия ---" << endl;

    double startSeq = omp_get_wtime();  // Засекаем время
    findMinMaxSequential(numbers, ARRAY_SIZE, minSeq, maxSeq);
    double endSeq = omp_get_wtime();

    double timeSeq = endSeq - startSeq;

    cout << "Минимум: " << minSeq << endl;
    cout << "Максимум: " << maxSeq << endl;
    cout << "Время: " << timeSeq * 1000 << " мс" << endl;
    cout << endl;

    // ===== Параллельная версия =====
    cout << "--- Параллельная версия (OpenMP) ---" << endl;

    double startPar = omp_get_wtime();
    findMinMaxParallel(numbers, ARRAY_SIZE, minPar, maxPar);
    double endPar = omp_get_wtime();

    double timePar = endPar - startPar;

    cout << "Минимум: " << minPar << endl;
    cout << "Максимум: " << maxPar << endl;
    cout << "Время: " << timePar * 1000 << " мс" << endl;
    cout << endl;

    // ===== Сравнение результатов =====
    cout << "--- Сравнение ---" << endl;

    // Проверяем что результаты совпадают
    if (minSeq == minPar && maxSeq == maxPar)
    {
        cout << "Результаты совпадают - OK!" << endl;
    }
    else
    {
        cout << "ОШИБКА: результаты не совпадают!" << endl;
    }

    // Считаем ускорение
    if (timePar > 0)
    {
        double speedup = timeSeq / timePar;
        cout << "Ускорение: " << speedup << "x" << endl;
    }

    cout << endl;

    // ===== Выводы =====
    cout << "--- Выводы ---" << endl;
    cout << "1. Для массива из 10000 элементов параллельная версия" << endl;
    cout << "   может работать медленнее из-за накладных расходов" << endl;
    cout << "   на создание потоков." << endl;
    cout << endl;
    cout << "2. OpenMP упрощает распараллеливание - достаточно добавить" << endl;
    cout << "   одну директиву #pragma omp parallel for." << endl;
    cout << endl;
    cout << "3. Директива reduction автоматически объединяет результаты" << endl;
    cout << "   из разных потоков." << endl;
    cout << endl;
    cout << "4. Для больших массивов (1000000+) параллельная версия" << endl;
    cout << "   будет значительно быстрее." << endl;

    // Освобождаем память
    delete[] numbers;

    return 0;
}
