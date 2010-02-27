#include "my_macros.hpp"
#include "Vector3.hpp"
#include "Matrix3.hpp"
#include "sim_constants.hpp"

__global__ void
rk4solver_kernel(int fieldlength, fp_type dt, fp_type *timepoints, Vector3 *M)
{
    for(int n = 0; n < numdots; n++) 
    {
        Vector3 k1 = LLG_Mprime(timepoints[i]        , M[i*numdots + n]             , Hcoupling[n]);
        Vector3 k2 = LLG_Mprime(timepoints[i] + dt/2 , M[i*numdots + n] + (dt/2)*k1 , Hcoupling[n]);
        Vector3 k3 = LLG_Mprime(timepoints[i] + dt/2 , M[i*numdots + n] + (dt/2)*k2 , Hcoupling[n]);
        Vector3 k4 = LLG_Mprime(timepoints[i] + dt   , M[i*numdots + n] + dt*k3     , Hcoupling[n]);
        Vector3 Mprime = 1/6.0 * (k1 + 2*k2 + 2*k3 + k4);
        M[(i+1)*numdots + n] = M[i*numdots + n] + dt * Mprime;
    }
}

    
int rk4solver_cuda(int fieldlength, fp_type dt, fp_type *timepoints, Vector3 *M)
{
    Vector3 *Hcoupling = (Vector3*)malloc(numdots * sizeof(Vector3));
    if(Hcoupling == NULL) {
        fprintf(stderr, "%s:%d Error allocating memory\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }
    for(int i = 0; i <= fieldlength-2; i++) {
        printf("t = %g\n", timepoints[i]);
        // determine coupling field from neighbouring dots
        for(int n = 0; n < numdots; n++) {
            Hcoupling[n] = c0 * (   ((n-numdots_x >= 0)      ? M[i*numdots + n-numdots_x] : Vector3(0,0,0))     // top
                                  + ((n+numdots_x < numdots) ? M[i*numdots + n+numdots_x] : Vector3(0,0,0))     // bottom
                                  + ((n%numdots_x != 0)      ? M[i*numdots + n-1]         : Vector3(0,0,0))     // left
                                  + (((n+1)%numdots_x != 0)  ? M[i*numdots + n+1]         : Vector3(0,0,0)) );  // right
        }
        // evaluate one-step of RK for all dots
        for(int n = 0; n < numdots; n++) {
            Vector3 k1 = LLG_Mprime(timepoints[i]        , M[i*numdots + n]             , Hcoupling[n]);
            Vector3 k2 = LLG_Mprime(timepoints[i] + dt/2 , M[i*numdots + n] + (dt/2)*k1 , Hcoupling[n]);
            Vector3 k3 = LLG_Mprime(timepoints[i] + dt/2 , M[i*numdots + n] + (dt/2)*k2 , Hcoupling[n]);
            Vector3 k4 = LLG_Mprime(timepoints[i] + dt   , M[i*numdots + n] + dt*k3     , Hcoupling[n]);
            Vector3 Mprime = 1/6.0 * (k1 + 2*k2 + 2*k3 + k4);
            M[(i+1)*numdots + n] = M[i*numdots + n] + dt * Mprime;
        }
    }
    free(Hcoupling);
    return 0;
}
