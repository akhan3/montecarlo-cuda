#include <assert.h>
#include <cutil_inline.h>

#include "my_macros.hpp"
#include "Vector3.hpp"
#include "Matrix3.hpp"
#include "sim_constants.hpp"

// Function prototypes
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
            const fp_type Ms)
{
    Vector3 Heff = Hext - N * M + Hcoupling;
    Vector3 McrossH = M.cross(Heff);
    Vector3 Mprime = -c * McrossH - (alfa * c / Ms) * M.cross(McrossH);
    return Mprime;
}


// declare device constant memory
//__device__ __constant__ fp_type c_d;

// Kernel definition
__global__ void
rk4solver_kernel(
    Vector3 *Hcoupling, 
    const Vector3 Hext, 
    const Vector3 *M, 
    Vector3 *Mnext,
    const fp_type t,
    const fp_type dt,

    const int numdots_y,
    const int numdots_x,
    const int numdots,
    const fp_type c0,

    const Vector3 N, 
    const fp_type c, 
    const fp_type alfa, 
    const fp_type Ms)
{
    const int n = threadIdx.x;

    // determine coupling field from neighbouring dots
    //for(int n = 0; n < numdots; n++) {
    Hcoupling[n] = c0 * (   ((n-numdots_x >= 0)      ? M[n-numdots_x] : Vector3(0,0,0))     // top
                          + ((n+numdots_x < numdots) ? M[n+numdots_x] : Vector3(0,0,0))     // bottom
                          + ((n%numdots_x != 0)      ? M[n-1]         : Vector3(0,0,0))     // left
                          + (((n+1)%numdots_x != 0)  ? M[n+1]         : Vector3(0,0,0)) );  // right

    // evaluate one-step of RK for all dots
    //for(int n = 0; n < numdots; n++) {
    Vector3 k1 = LLG_Mprime(t        , M[n]             , Hcoupling[n], Hext, N, c, alfa, Ms);
    Vector3 k2 = LLG_Mprime(t + dt/2 , M[n] + (dt/2)*k1 , Hcoupling[n], Hext, N, c, alfa, Ms);
    Vector3 k3 = LLG_Mprime(t + dt/2 , M[n] + (dt/2)*k2 , Hcoupling[n], Hext, N, c, alfa, Ms);
    Vector3 k4 = LLG_Mprime(t + dt   , M[n] + dt*k3     , Hcoupling[n], Hext, N, c, alfa, Ms);
    Vector3 Mprime = 1/6.0 * (k1 + 2*k2 + 2*k3 + k4);
    Mnext[n] = M[n] + dt * Mprime;
}


int rk4solver_cuda(
        int fieldlength, 
        fp_type dt, 
        fp_type *timepoints, 
        Vector3 *M)
{
    // set up device memory pointers
    static Vector3 *Hcoupling_d = NULL;
    static Vector3 *Mcurr_d = NULL;
    static Vector3 *Mnext_d = NULL;

    // allocate memory on device for Hcoupling_d, Mcurr_d, Mnext_d
    cutilSafeCall( cudaMalloc( (void**)&Hcoupling_d,    numdots * sizeof(Vector3) ) );
    cutilSafeCall( cudaMalloc( (void**)&Mcurr_d,        numdots * sizeof(Vector3) ) );
    cutilSafeCall( cudaMalloc( (void**)&Mnext_d,        numdots * sizeof(Vector3) ) );
    assert(Hcoupling_d != NULL && Mnext_d != NULL);

    // copy constants to device constant memory
    //cutilSafeCall( cudaMemcpyToSymbol( &c_d, &c, sizeof(fp_type) ) );
    
    // Time-marching loop
    for(int i = 0; i <= fieldlength-2; i++) 
    {
        const fp_type t = timepoints[i];
        printf("t = %g\n", t);
                
        // copy current value of M to device global memory
        printf("%f MB copied\n", numdots * sizeof(Vector3) / (1024.0*1024.0));
        cutilSafeCall( cudaMemcpy( Mcurr_d, &M[i*numdots], numdots * sizeof(Vector3), cudaMemcpyHostToDevice ) );

        // set up kernel parameters
        dim3 grid = 1;
        dim3 threads(numdots, 1, 1);
        assert(threads.x <= 512);    // max_threads_per_block
        printf("numdots_x=%u, numdots_y=%u, numdots=%u, threads.x=%u, threads.y=%u, grid.x=%u\n",
                numdots_x,    numdots_y,    numdots,    threads.x,    threads.y,    grid.x);
        printf("launching kernel with %u blocks and %u threads...\n", 
                    grid.x*grid.y*grid.z, 
                    threads.x*threads.y*threads.z);
            
        // launch the kernel
        rk4solver_kernel <<<grid, threads>>> (
            Hcoupling_d, 
            Hext_function(t), 
            Mcurr_d, 
            Mnext_d,
            t,
            dt,
            numdots_y,
            numdots_x,
            numdots,
            c0,
            N,
            c,
            alfa,
            Ms
        );
        cutilCheckMsg("Kernel execution failed");
        cudaThreadSynchronize();
        
        // copy Mnext_d (the result of kernel) to host main memory
        cutilSafeCall( cudaMemcpy( &M[(i+1)*numdots], Mnext_d, numdots * sizeof(Vector3), cudaMemcpyDeviceToHost ) );
    }
    return 0;
}
