__kernel void matrix_multiply(__global const float* A,
                              __global const float* B,
                              __global float* C,
                              const int N,
                              const int M,
                              const int K) {
    // Получаем глобальные индексы (строка и столбец результирующей матрицы)
    int row = get_global_id(0);  // Строка в C (и A)
    int col = get_global_id(1);  // Столбец в C (и B)

    // Проверка границ
    if (row >= N || col >= K) return;

    // Вычисление элемента C[row][col]
    float sum = 0.0f;
    for (int i = 0; i < M; i++) {
        sum += A[row * M + i] * B[i * K + col];
    }

    C[row * K + col] = sum;
}
