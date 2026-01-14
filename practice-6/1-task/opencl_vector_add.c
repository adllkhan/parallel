#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <mach/mach_time.h>
#else
#include <CL/cl.h>
#endif

#define ARRAY_SIZE 16777216  // 16M элементов для заметного измерения времени

// Функция для получения времени в секундах
double get_time() {
#ifdef __APPLE__
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    return (double)mach_absolute_time() * timebase.numer / timebase.denom / 1e9;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}

// Функция для чтения файла ядра
char* read_kernel_file(const char* filename, size_t* length) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Ошибка: не удалось открыть файл %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    *length = ftell(file);
    rewind(file);

    char* source = (char*)malloc(*length + 1);
    if (!source) {
        fclose(file);
        return NULL;
    }

    fread(source, 1, *length, file);
    source[*length] = '\0';
    fclose(file);

    return source;
}

int main() {
    cl_int err;

    // Данные для вычислений (выделяем в куче из-за большого размера)
    float* A = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float* B = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float* C = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float* C_cpu = (float*)malloc(ARRAY_SIZE * sizeof(float));  // Для сравнения с CPU

    if (!A || !B || !C || !C_cpu) {
        fprintf(stderr, "Ошибка выделения памяти\n");
        return 1;
    }

    // Инициализация массивов
    for (int i = 0; i < ARRAY_SIZE; i++) {
        A[i] = (float)i;
        B[i] = (float)(i * 2);
    }

    printf("=== OpenCL Vector Addition ===\n");
    printf("Размер массива: %d элементов (%.2f MB)\n\n", ARRAY_SIZE,
           (float)(ARRAY_SIZE * sizeof(float)) / (1024 * 1024));

    // ========================================
    // Измерение времени на CPU (последовательное выполнение)
    // ========================================

    double cpu_start = get_time();
    for (int i = 0; i < ARRAY_SIZE; i++) {
        C_cpu[i] = A[i] + B[i];
    }
    double cpu_end = get_time();
    double cpu_time = cpu_end - cpu_start;

    // Используем результат, чтобы компилятор не удалил вычисления
    volatile float dummy = C_cpu[ARRAY_SIZE / 2];
    (void)dummy;

    printf("CPU (последовательно): %.6f сек\n\n", cpu_time);

    // ========================================
    // Шаг 1: Инициализация платформы и устройства
    // ========================================

    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Ошибка получения платформы: %d\n", err);
        return 1;
    }

    // Получение информации о платформе
    char platform_name[256];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    printf("Платформа: %s\n", platform_name);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        // Если GPU недоступен, пробуем CPU
        printf("GPU не найден, используем CPU...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Ошибка получения устройства: %d\n", err);
            return 1;
        }
    }

    // Получение информации об устройстве
    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Устройство: %s\n\n", device_name);

    // ========================================
    // Шаг 2: Создание контекста и командной очереди
    // ========================================

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Ошибка создания контекста: %d\n", err);
        return 1;
    }
    printf("Контекст создан успешно\n");

    // Создание командной очереди (используем deprecated функцию для совместимости)
#ifdef CL_VERSION_2_0
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
#else
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Ошибка создания очереди: %d\n", err);
        clReleaseContext(context);
        return 1;
    }
    printf("Командная очередь создана успешно\n");

    // ========================================
    // Шаг 3: Загрузка и компиляция ядра
    // ========================================

    size_t kernel_length;
    char* kernel_source = read_kernel_file("kernel.cl", &kernel_length);
    if (!kernel_source) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    cl_program program = clCreateProgramWithSource(context, 1,
                                                   (const char**)&kernel_source,
                                                   &kernel_length, &err);
    free(kernel_source);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "Ошибка создания программы: %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    // Компиляция программы
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Ошибка компиляции программы: %d\n", err);

        // Вывод лога ошибок компиляции
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Лог компиляции:\n%s\n", log);
        free(log);

        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    printf("Ядро скомпилировано успешно\n");

    // Создание объекта ядра
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Ошибка создания ядра: %d\n", err);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    printf("Ядро создано успешно\n\n");

    // ========================================
    // Шаг 4: Подготовка данных (буферы)
    // ========================================

    size_t buffer_size = ARRAY_SIZE * sizeof(float);

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    buffer_size, A, &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    buffer_size, B, &err);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    buffer_size, NULL, &err);

    if (!bufferA || !bufferB || !bufferC) {
        fprintf(stderr, "Ошибка создания буферов\n");
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    printf("Буферы созданы успешно\n");

    // Установка аргументов ядра
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

    // ========================================
    // Шаг 5: Выполнение ядра и считывание результатов
    // ========================================

    size_t global_size = ARRAY_SIZE;

    printf("Запуск ядра с %zu work-items...\n", global_size);

    // Измерение времени выполнения OpenCL (только ядро)
    double opencl_start = get_time();

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Ошибка запуска ядра: %d\n", err);
        clReleaseMemObject(bufferA);
        clReleaseMemObject(bufferB);
        clReleaseMemObject(bufferC);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    // Ожидание завершения выполнения
    clFinish(queue);

    double opencl_kernel_time = get_time() - opencl_start;

    // Считывание результатов
    double read_start = get_time();
    err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, buffer_size, C, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Ошибка чтения результатов: %d\n", err);
    }
    double read_time = get_time() - read_start;

    printf("Ядро выполнено успешно!\n\n");

    // ========================================
    // Сравнение времени выполнения
    // ========================================

    printf("=== Сравнение времени выполнения ===\n");
    printf("CPU (последовательно):    %.6f сек\n", cpu_time);
    printf("OpenCL (только ядро):     %.6f сек\n", opencl_kernel_time);
    printf("OpenCL (чтение данных):   %.6f сек\n", read_time);
    printf("OpenCL (ядро + чтение):   %.6f сек\n", opencl_kernel_time + read_time);
    printf("\n");

    if (opencl_kernel_time > 0) {
        printf("Ускорение (только ядро):  %.2fx\n", cpu_time / opencl_kernel_time);
        printf("Ускорение (с передачей):  %.2fx\n", cpu_time / (opencl_kernel_time + read_time));
    }
    printf("\n");

    // Вывод результатов (первые и последние 5 элементов)
    printf("=== Результаты ===\n");
    printf("Первые 5 элементов:\n");
    for (int i = 0; i < 5; i++) {
        printf("  A[%d] + B[%d] = %.1f + %.1f = %.1f\n", i, i, A[i], B[i], C[i]);
    }
    printf("...\n");
    printf("Последние 5 элементов:\n");
    for (int i = ARRAY_SIZE - 5; i < ARRAY_SIZE; i++) {
        printf("  A[%d] + B[%d] = %.1f + %.1f = %.1f\n", i, i, A[i], B[i], C[i]);
    }

    // Проверка корректности
    int errors = 0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (C[i] != A[i] + B[i]) {
            errors++;
        }
    }
    printf("\nПроверка: %s (%d ошибок)\n", errors == 0 ? "PASSED" : "FAILED", errors);

    // ========================================
    // Освобождение ресурсов
    // ========================================

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Освобождение памяти хоста
    free(A);
    free(B);
    free(C);
    free(C_cpu);

    printf("\nРесурсы освобождены. Программа завершена.\n");

    return 0;
}
