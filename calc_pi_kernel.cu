#include <assert.h>
#include <cutil_inline.h>

#ifdef __DEVICE_EMULATION__
#define DIM    64
#else
#define DIM    512
#endif

#include "my_macros.hpp"


// Kernel definition
__global__ void
calc_pi_kernel(
    const uint64 ITERATIONS,
    const fp_type *domain_x_d,
    const fp_type *domain_y_d,
    uint32 *hits_d)
{
    __shared__ uint32 hits[DIM];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int i = bid * blockDim.x + tid;

    if(i >= ITERATIONS)
        return;
    
    fp_type distance = sqrt(domain_x_d[i]*domain_x_d[i] + domain_y_d[i]*domain_y_d[i]);
    hits[tid] = 0;
    if(distance < 1)
        hits[tid] = 1;
    __syncthreads();
    
// parallel reduction summation
    if (tid < 256)
        hits[tid] += hits[tid + 256];
    __syncthreads();
    if (tid < 128)
        hits[tid] += hits[tid + 128];
    __syncthreads();
    if (tid < 64)
        hits[tid] += hits[tid +  64];
    __syncthreads();
    if (tid < 32) {
        hits[tid] += hits[tid +  32];
        hits[tid] += hits[tid +  16];
        hits[tid] += hits[tid +   8];
        hits[tid] += hits[tid +   4];
        hits[tid] += hits[tid +   2];
        hits[tid] += hits[tid +   1];
    }
    //__syncthreads();
    
// copy per block result to global memory
    if(tid == 0)
        hits_d[bid] = hits[0];
 }


fp_type calc_pi_gpu(
        const uint64 ITERATIONS,
        const fp_type *domain_x,
        const fp_type *domain_y,
              double *time_kernel_cumulative)
{
    // set up device memory pointers
    fp_type *domain_x_d = NULL;
    fp_type *domain_y_d = NULL;
    uint32 *hits_d = NULL;
    uint32 *hits_h = NULL;
    
    // set up kernel parameters
    dim3 grid = ceil(ITERATIONS / (fp_type)DIM);
    dim3 threads(DIM, 1, 1);
    assert(threads.x <= DIM);    // max_threads_per_block
    printf("ITERATIONS=%llu: Launching kernel with %u blocks and %u threads...\n", 
                ITERATIONS, grid.x*grid.y*grid.z, threads.x*threads.y*threads.z);    

    // allocate memory on device 
    cutilSafeCall( cudaMalloc( (void**)&domain_x_d, ITERATIONS * sizeof(fp_type) ) );
    cutilSafeCall( cudaMalloc( (void**)&domain_y_d, ITERATIONS * sizeof(fp_type) ) );
    cutilSafeCall( cudaMalloc( (void**)&hits_d,     grid.x * sizeof(uint32) ) );
    hits_h = (uint32*)malloc(grid.x * sizeof(uint32));
    if(domain_x_d == NULL || domain_y_d == NULL || hits_d == NULL || hits_h == NULL) {
        fprintf(stderr, "%s:%d Error allocating memory\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }
    
    cutilSafeCall( cudaMemcpy( domain_x_d, domain_x, ITERATIONS * sizeof(fp_type), cudaMemcpyHostToDevice ) );
    cutilSafeCall( cudaMemcpy( domain_y_d, domain_y, ITERATIONS * sizeof(fp_type), cudaMemcpyHostToDevice ) );

    // create timer
    unsigned int timer_kernel = 0;
    cutilCheckError(cutCreateTimer(&timer_kernel));
    cutilCheckError(cutStartTimer(timer_kernel));  // start timer

    // launch the kernel
    calc_pi_kernel <<< grid, threads, DIM * sizeof(uint32) >>> 
    (
        ITERATIONS,
        domain_x_d, 
        domain_y_d,
        hits_d
    );
    cutilCheckMsg("Kernel execution failed");
    cudaThreadSynchronize();
    
    // read the timer
    cutilCheckError(cutStopTimer(timer_kernel));
    double time_kernel_this = cutGetTimerValue(timer_kernel);
    *time_kernel_cumulative += time_kernel_this;

    // copy Mnext_d (the result of kernel) to host main memory
    cutilSafeCall( cudaMemcpy( hits_h, hits_d, grid.x * sizeof(uint32), cudaMemcpyDeviceToHost ) );

// final summation in CPU
    uint64 hits = 0;
    fp_type pi_mc = 0;
    for(uint64 i = 0; i < grid.x; i++) {
        hits += hits_h[i];
        if(i == grid.x - 1)
            pi_mc = 4.0 * (fp_type)hits / (i+1)/512.0;
        else
            pi_mc = 4.0 * (fp_type)hits / (i+1)/512.0;
        //printf("Block-id=%llu : hits = %d, PI = %.8f\n", i, hits_h[i], pi_mc);
    }
    
    // reclaim memory
    cutilSafeCall( cudaFree(domain_x_d) );
    cutilSafeCall( cudaFree(domain_y_d) );
    cutilSafeCall( cudaFree(hits_d) );
    free(hits_h);
    
    return pi_mc;
}
