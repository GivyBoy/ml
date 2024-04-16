#include <stdio.h>

__global__ void hello(){
    printf("Hello World from GPU!\n");
}

// __global__ indicates that the function runs on the GPU

int main(int argc, char **argv) {
    hello<<<1,2>>>(); // <<<M, T>>> - M represents the number of blocks and T represents the number of threads in each block
    cudaDeviceSynchronize();
    return 0;
}