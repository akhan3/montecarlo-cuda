// TODO: implement matfile saving on the fly to save memory

#include <cstdlib>
#include <time.h>
#include <cutil_inline.h>

#include "my_macros.hpp"


// generates a uniformly distributed random number between a and b
inline fp_type rand_atob(fp_type a, fp_type b) {
    fp_type r = rand() / (fp_type)RAND_MAX;
    r = a + (b-a) * r;
    return r;
}


fp_type calc_pi_gpu(
        const uint64 ITERATIONS,
        const fp_type *domain_x,
        const fp_type *domain_y,
              double  *time_kernel_cumulative);


fp_type calc_pi_cpu(
        const uint64 ITERATIONS,
        const fp_type *domain_x,
        const fp_type *domain_y,
              fp_type *distance)
{
    const int verbose = 0;
// open file to write results    
    FILE *cpu_log = fopen("mc_pi_cpu.txt", "w");
    if(cpu_log == NULL) {
        fprintf(stderr, "%s:%d Error opening/creating file mc_pi_cpu.txt\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }
// Monte-Carlo loop
    uint64 hits = 0;
    fp_type pi_mc = 0;
    for(uint64 i = 0; i < ITERATIONS; i++) {
        distance[i] = sqrt(domain_x[i]*domain_x[i] + domain_y[i]*domain_y[i]);
        if(distance[i] < 1)
            hits++;
        pi_mc = 4.0 * (fp_type)hits / (i+1);
        if(verbose)
            printf("Iter#%llu: PI = %.8f\n", i+1, pi_mc);
        fprintf(cpu_log, "%f\n", pi_mc);
    }
    fclose(cpu_log);
    return pi_mc;
}


int calc_pi(uint64 ITERATIONS) 
{
    fp_type *domain_x = (fp_type*)malloc(ITERATIONS * sizeof(fp_type));
    fp_type *domain_y = (fp_type*)malloc(ITERATIONS * sizeof(fp_type));
    fp_type *distance = (fp_type*)malloc(ITERATIONS * sizeof(fp_type));
    if(domain_x == NULL || domain_y == NULL || distance == NULL) {
        fprintf(stderr, "%s:%d Error allocating memory\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }
    
// generate random inputs in the domain
    for(uint64 i = 0; i < ITERATIONS; i++) {
        domain_x[i] = rand_atob(0, 1);
        domain_y[i] = rand_atob(0, 1);
    }

    int status = 0;
    
// call the RK solver routine on GPU
    NEWLINE; printf("Simulating on GPU...\n"); 
    // create timer
    unsigned int timer_gpu = 0;
    double time_kernel_cumulative = 0;
    cutilCheckError(cutCreateTimer(&timer_gpu));
    cutilCheckError(cutStartTimer(timer_gpu));  // start timer
    // launch solver
    fp_type pi_gpu = calc_pi_gpu(ITERATIONS, domain_x, domain_y, &time_kernel_cumulative);
    // read the timer
    cutilCheckError(cutStopTimer(timer_gpu));
    double time_gpu = cutGetTimerValue(timer_gpu);
    NEWLINE;
    //printf("Time taken by all (%d) kernel launches = %f ms (%.0f%%)\n", fieldlength-1, time_kernel_cumulative, 100*time_kernel_cumulative/time_gpu);
    printf("Time taken by the kernel = %f ms (%.2f%%)\n", time_kernel_cumulative, 100*time_kernel_cumulative/time_gpu);
    printf("Time taken by memory trasfers and sync overhead = %f ms (%.2f%%)\n", time_gpu - time_kernel_cumulative, 100*(1-time_kernel_cumulative/time_gpu));
    SEPARATOR;
    printf("Calculated value of PI = %.8f after %llu iterations\n", pi_gpu, ITERATIONS);
    printf("Time taken by Monte Carlo on GPU = %f ms\n", time_gpu);
    SEPARATOR;

// call the RK solver routine on CPU
    NEWLINE; printf("Simulating on CPU...\n"); 
    // create timer
    unsigned int timer_cpu = 0;
    cutilCheckError(cutCreateTimer(&timer_cpu));
    cutilCheckError(cutStartTimer(timer_cpu));  // start timer
    // launch solver
    fp_type pi_cpu = calc_pi_cpu(ITERATIONS, domain_x, domain_y, distance);
    // read the timer
    cutilCheckError(cutStopTimer(timer_cpu));
    double time_cpu = cutGetTimerValue(timer_cpu);
    SEPARATOR;
    printf("Calculated value of PI = %.8f after %llu iterations\n", pi_cpu, ITERATIONS);
    printf("Time taken by Monte Carlo on CPU = %f ms\n", time_cpu);    
    SEPARATOR;
            
    // print speed-up info
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    SEPARATOR2;
    printf("Speed-up factor (overall) = %.2f (\"%s\" with %d cores)\n", 
        time_cpu/time_gpu, deviceProp.name, 8 * deviceProp.multiProcessorCount);
    printf("Speed-up factor (compute) = %.2f (\"%s\" with %d cores)\n", 
        time_cpu/time_kernel_cumulative, deviceProp.name, 8 * deviceProp.multiProcessorCount);
    SEPARATOR2;

// reclaim memory
    free(domain_x);
    free(domain_y);
    free(distance);

    return status;
}

int main(int argc, char **argv) {
    uint64 ITERATIONS = 999999;
    unsigned int SEED = (unsigned int)time(NULL);

    if(argc >= 2) {
        sscanf(argv[1], "%llu", &ITERATIONS);
        if(argc >= 3)
            sscanf(argv[2], "%u", &SEED);
    }

    srand(SEED);
    int status = calc_pi(ITERATIONS);

    if(status == 0)
        fprintf(stdout, "Successfully completed!\n");
    else
        fprintf(stderr, "Error occured!\n");
    printf("SEED: %u\n", SEED);
    return (status == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
