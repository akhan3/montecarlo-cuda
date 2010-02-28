// TODO: implement matfile saving on the fly to save memory

#include <cstdlib>
#include <cutil_inline.h>

#include "my_macros.hpp"
#include "Vector3.hpp"
#include "Matrix3.hpp"
#include "sim_constants.hpp"
#include "save_matfile.hpp"


// generates a uniformly distributed random number between a and b
fp_type rand_atob(fp_type a, fp_type b) {
    fp_type r = rand() / (fp_type)RAND_MAX;
    r = a + (b-a) * r;
    return r;
}

// implements Landau-Lifshitz-Gilbert differential equation
HOSTDEVICE 
Vector3 LLG_Mprime(
            const fp_type t, 
            const Vector3 &M, 
            const Vector3 &Hcoupling,
            const Vector3 &Hext,
            const Vector3 N, 
            const fp_type c, 
            const fp_type alfa, 
            const fp_type Ms);


int rk4solver_cuda(
        const int fieldlength, 
        const fp_type dt, 
        const fp_type *timepoints, 
        Vector3 *M,
        double *time_kernel_cumulative);


int rk4solver(
        const int fieldlength, 
        const fp_type dt, 
        const fp_type *timepoints, 
        Vector3 *M)
{
    Vector3 *Hcoupling = (Vector3*)malloc(numdots * sizeof(Vector3));
    if(Hcoupling == NULL) {
        fprintf(stderr, "%s:%d Error allocating memory\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    // Time-marching loop
    for(int i = 0; i <= fieldlength-2; i++) 
    {
        const fp_type t = timepoints[i];
        //printf("t = %g\n", t);
        printf("%d ", i); fflush(stdout);
        const Vector3 Hext = Hext_function(t);

        // determine coupling field from neighbouring dots
        for(int n = 0; n < numdots; n++) 
        {
            Hcoupling[n] = c0 * (   ((n-numdots_x >= 0)      ? M[i*numdots + n-numdots_x] : Vector3(0,0,0))     // top
                                  + ((n+numdots_x < numdots) ? M[i*numdots + n+numdots_x] : Vector3(0,0,0))     // bottom
                                  + ((n%numdots_x != 0)      ? M[i*numdots + n-1]         : Vector3(0,0,0))     // left
                                  + (((n+1)%numdots_x != 0)  ? M[i*numdots + n+1]         : Vector3(0,0,0)) );  // right
        }
        // evaluate one-step of RK for all dots
        for(int n = 0; n < numdots; n++) 
        {
            Vector3 k1 = LLG_Mprime(t        , M[i*numdots + n]             , Hcoupling[n], Hext, N, c, alfa, Ms);
            Vector3 k2 = LLG_Mprime(t + dt/2 , M[i*numdots + n] + (dt/2)*k1 , Hcoupling[n], Hext, N, c, alfa, Ms);
            Vector3 k3 = LLG_Mprime(t + dt/2 , M[i*numdots + n] + (dt/2)*k2 , Hcoupling[n], Hext, N, c, alfa, Ms);
            Vector3 k4 = LLG_Mprime(t + dt   , M[i*numdots + n] + dt*k3     , Hcoupling[n], Hext, N, c, alfa, Ms);
            Vector3 Mprime = 1/6.0 * (k1 + 2*k2 + 2*k3 + k4);
            M[(i+1)*numdots + n] = M[i*numdots + n] + dt * Mprime;
        }
    }
    NEWLINE;
    free(Hcoupling);
    return 0;
}


int solve_array() {
    const fp_type ftime = timestep * ceil((fp_type)finaltime / timestep);
    const int fieldlength = ftime / timestep + 1;
    fp_type *timepoints = (fp_type*)malloc(fieldlength * sizeof(fp_type));
    if(timepoints == NULL) {
        fprintf(stderr, "%s:%d Error allocating memory\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }
    for(int i = 0; i < fieldlength; i++)
        timepoints[i] = i * ftime / (fieldlength-1);
    const fp_type dt = timepoints[1] - timepoints[0];
    
    // initialize M for all dots at each timepoint
    // M[t][dotindex]
    Vector3 *M = (Vector3*)malloc(fieldlength * numdots * sizeof(Vector3));
    if(M == NULL) {
        fprintf(stderr, "%s:%d Error allocating memory\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }
    for(int y = 0; y < numdots_y; y++) {
        for(int x = 0; x < numdots_x; x++) {
            int n = y*numdots_x + x;
            fp_type theta = rand_atob(0, 2*M_PI);   // angle from x-axis in xy-plane
            fp_type phi = rand_atob(0, 2*M_PI);     // angle from z-axis
            //if(y%2 && x%2 || !(y%2) && !(x%2)) phi = 0; else phi = M_PI;
            M[0*numdots + n] = Ms * Vector3(sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi));
        }
    }
    printf("%.2f MB of memory is required for the simulation of %dx%d dots for %d time points (%gs at %gs stepping)\n",
                fieldlength * numdots * sizeof(Vector3)/1024.0/1024.0,
                numdots_y, numdots_x, fieldlength, ftime, timestep);
    
    int status = 0;
    
// call the RK solver routine on GPU
    NEWLINE; printf("Simulating on GPU...\n"); 
    // create timer
    unsigned int timer_gpu = 0;
    double time_kernel_cumulative = 0;
    cutilCheckError(cutCreateTimer(&timer_gpu));
    cutilCheckError(cutStartTimer(timer_gpu));  // start timer
    // launch solver
    status |= rk4solver_cuda(fieldlength, dt, timepoints, M, &time_kernel_cumulative);
    // read the timer
    cutilCheckError(cutStopTimer(timer_gpu));
    double time_gpu = cutGetTimerValue(timer_gpu);
    printf("Time taken by memory trasfers and sync overhead = %f ms\n", time_gpu - time_kernel_cumulative);
    SEPARATOR;
    printf("Time taken by RK4 solver on GPU = %f ms\n", time_gpu);
    SEPARATOR;
    printf("M(t = 0, dot0) = "); M[0*numdots + 0].print();
    printf("M(t = %g, dot0) = ", ftime); M[(fieldlength-1)*numdots + 0].print();
    if(save_matfiles) {
        char matfile_name[100];
        sprintf(matfile_name, "%s/%s_results_cuda.mat", matfiles_dir, sim_id);
        status |= save_matfile(matfile_name, fieldlength, numdots_y, numdots_x, M, timepoints, 1);
    }

// call the RK solver routine on CPU
    NEWLINE; printf("Simulating on CPU...\n"); 
    // create timer
    unsigned int timer_cpu = 0;
    cutilCheckError(cutCreateTimer(&timer_cpu));
    cutilCheckError(cutStartTimer(timer_cpu));  // start timer
    // launch solver
    status |= rk4solver(fieldlength, dt, timepoints, M);
    // read the timer
    cutilCheckError(cutStopTimer(timer_cpu));
    double time_cpu = cutGetTimerValue(timer_cpu);
    SEPARATOR;
    printf("Time taken by RK4 solver on CPU = %f ms\n", time_cpu);
    SEPARATOR;
    printf("M(t = 0, dot0) = "); M[0*numdots + 0].print();
    printf("M(t = %g, dot0) = ", ftime); M[(fieldlength-1)*numdots + 0].print();
    if(save_matfiles) {
        char matfile_name[100];
        sprintf(matfile_name, "%s/%s_results.mat", matfiles_dir, sim_id);
        status |= save_matfile(matfile_name, fieldlength, numdots_y, numdots_x, M, timepoints, 1);
    }
            
    // reclaim memory
    free(timepoints);
    free(M);
    printf("%.2f MB of memory was required for the simulation of %dx%d dots for %d time points (%gs at %gs stepping)\n",
                fieldlength * numdots * sizeof(Vector3)/1024.0/1024.0,
                numdots_y, numdots_x, fieldlength, ftime, timestep);
                
    // print speed-up info
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    SEPARATOR2;
    printf("Speed-up factor = %.2f (%s with %d CUDA cores)\n", 
        time_cpu/time_gpu, deviceProp.name, 8 * deviceProp.multiProcessorCount);
    SEPARATOR2;
    return status;
}

int main(int argc, char **argv) {
    int debug = 0, pinned = 0, matlab = 1;
    if( !strcmp(argv[argc-1], "debug") )
        debug = 1;
    if( !strcmp(argv[argc-1], "pinned") )
        pinned = 1;
    if( !strcmp(argv[argc-1], "no-matlab") )
        matlab = 0;

    srand((unsigned int)time(NULL));
    int status = solve_array();

    if(status == 0)
        fprintf(stdout, "Successfully completed!\n");
    else
        fprintf(stderr, "Error occured!\n");
    return (status == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
