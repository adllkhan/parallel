__kernel void vector_add(__global const float* A,
                         __global const float* B,
                         __global float* C) {
    int id = get_global_id(0);  // Определение глобального ID
    C[id] = A[id] + B[id];      // Выполнение операции сложения
}
