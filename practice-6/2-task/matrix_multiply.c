#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <mach/mach_time.h>
#else
#include <CL/cl.h>
#endif

// Размеры матриц: A[N x M], B[M x K], C[N x K]
#define N 512
#define M 512
#define K 512

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

// Последовательное умножение матриц на CPU
void matrix_multiply_cpu(const float* A, const float* B, float* C,
                         int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < m; l++) {
                sum += A[i * m + l] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

// Проверка корректности результатов
int verify_results(const float* C_gpu, const float* C_cpu, int n, int k) {
    int errors = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < n * k; i++) {
        float diff = fabs(C_gpu[i] - C_cpu[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3) {  // Допустимая погрешность для float
            errors++;
            if (errors <= 5) {
                printf("  Ошибка в позиции %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                       i, C_gpu[i], C_cpu[i], diff);
            }
        }
    }

    printf("Максимальная разница: %.6f\n", max_diff);
    return errors;
}

int main() {
    cl_int err;

    printf("=== OpenCL Matrix Multiplication ===\n");
    printf("Размеры матриц: A[%d x %d] * B[%d x %d] = C[%d x %d]\n\n",
           N, M, M, K, N, K);

    // Выделение памяти для матриц
    size_t size_A = N * M * sizeof(float);
    size_t size_B = M * K * sizeof(float);
    size_t size_C = N * K * sizeof(float);

    float* A = (float*)malloc(size_A);
    float* B = (float*)malloc(size_B);
    float* C_gpu = (float*)malloc(size_C);
    float* C_cpu = (float*)malloc(size_C);

    if (!A || !B || !C_gpu || !C_cpu) {
        fprintf(stderr, "Ошибка выделения памяти\n");
        return 1;
    }

    // Инициализация матриц случайными значениями
    srand(42);
    for (int i = 0; i < N * M; i++) {
        A[i] = (float)(rand() % 100) / 10.0f;
    }
    for (int i = 0; i < M * K; i++) {
        B[i] = (float)(rand() % 100) / 10.0f;
    }

    // ========================================
    // CPU: Последовательное умножение
    // ========================================

    printf("Выполнение на CPU...\n");
    double cpu_start = get_time();
    matrix_multiply_cpu(A, B, C_cpu, N, M, K);
    double cpu_time = get_time() - cpu_start;
    printf("CPU время: %.6f сек\n\n", cpu_time);

    // ========================================
    // OpenCL: Инициализация
    // ========================================

    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Ошибка получения платформы: %d\n", err);
        return 1;
    }

    char platform_name[256];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    printf("Платформа: %s\n", platform_name);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("GPU не найден, используем CPU...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Ошибка получения устройства: %d\n", err);
            return 1;
        }
    }

    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Устройство: %s\n\n", device_name);

    // Создание контекста и очереди
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Ошибка создания контекста: %d\n", err);
        return 1;
    }

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

    // ========================================
    // OpenCL: Загрузка и компиляция ядра
    // ========================================

    size_t kernel_length;
    char* kernel_source = read_kernel_file("matrix_mul_kernel.cl", &kernel_length);
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

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Ошибка компиляции программы: %d\n", err);
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

    cl_kernel kernel = clCreateKernel(program, "matrix_multiply", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Ошибка создания ядра: %d\n", err);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    printf("Ядро скомпилировано успешно\n");

    // ========================================
    // OpenCL: Создание буферов
    // ========================================

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    size_A, A, &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    size_B, B, &err);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    size_C, NULL, &err);

    if (!bufferA || !bufferB || !bufferC) {
        fprintf(stderr, "Ошибка создания буферов\n");
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    // Установка аргументов ядра
    int n_val = N, m_val = M, k_val = K;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &n_val);
    clSetKernelArg(kernel, 4, sizeof(int), &m_val);
    clSetKernelArg(kernel, 5, sizeof(int), &k_val);

    // ========================================
    // OpenCL: Выполнение ядра
    // ========================================

    // Глобальный размер соответствует размеру результирующей матрицы C[N x K]
    size_t global_size[2] = {N, K};
    size_t local_size[2] = {16, 16};  // Размер рабочей группы

    // Округление глобального размера до кратного локальному
    global_size[0] = ((N + local_size[0] - 1) / local_size[0]) * local_size[0];
    global_size[1] = ((K + local_size[1] - 1) / local_size[1]) * local_size[1];

    printf("Глобальный размер: %zu x %zu\n", global_size[0], global_size[1]);
    printf("Локальный размер:  %zu x %zu\n\n", local_size[0], local_size[1]);

    printf("Выполнение на GPU...\n");
    double gpu_start = get_time();

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size,
                                 0, NULL, NULL);
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

    clFinish(queue);
    double gpu_kernel_time = get_time() - gpu_start;

    // Чтение результатов
    double read_start = get_time();
    err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, size_C, C_gpu, 0, NULL, NULL);
    double read_time = get_time() - read_start;

    if (err != CL_SUCCESS) {
        fprintf(stderr, "Ошибка чтения результатов: %d\n", err);
    }

    printf("GPU время (ядро):   %.6f сек\n", gpu_kernel_time);
    printf("GPU время (чтение): %.6f сек\n", read_time);
    printf("GPU время (всего):  %.6f сек\n\n", gpu_kernel_time + read_time);

    // ========================================
    // Проверка корректности
    // ========================================

    printf("=== Проверка корректности ===\n");
    int errors = verify_results(C_gpu, C_cpu, N, K);
    printf("Результат: %s (%d ошибок)\n\n", errors == 0 ? "PASSED" : "FAILED", errors);

    // ========================================
    // Сравнение производительности
    // ========================================

    printf("=== Сравнение производительности ===\n");
    printf("CPU время:              %.6f сек\n", cpu_time);
    printf("GPU время (ядро):       %.6f сек\n", gpu_kernel_time);
    printf("GPU время (с чтением):  %.6f сек\n", gpu_kernel_time + read_time);
    printf("\n");
    printf("Ускорение (только ядро): %.2fx\n", cpu_time / gpu_kernel_time);
    printf("Ускорение (с передачей): %.2fx\n", cpu_time / (gpu_kernel_time + read_time));

    // ========================================
    // Вывод примера результатов
    // ========================================

    printf("\n=== Пример результатов (C[0][0..4]) ===\n");
    for (int j = 0; j < 5 && j < K; j++) {
        printf("C[0][%d] = %.4f (GPU) vs %.4f (CPU)\n", j, C_gpu[j], C_cpu[j]);
    }

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

    free(A);
    free(B);
    free(C_gpu);
    free(C_cpu);

    printf("\nРесурсы освобождены. Программа завершена.\n");

    return 0;
}
