#ifndef _MYMACROS_H_
#define _MYMACROS_H_

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define HOST
#define DEVICE
#define HOSTDEVICE
#endif

#define NEWLINE printf("\n")
#define TAB printf("    ")
#define SEPARATOR printf("--------------------------------------------------------------------------------\n")
#define SEPARATOR2 printf("================================================================================\n")

// floating point precision to use
// typedef float fp_type;
typedef float fp_type;


#endif // #ifndef _MYMACROS_H_
