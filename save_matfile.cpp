#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mat.h>
#include <engine.h>

#include "my_macros.hpp"
#include "Vector3.hpp"
#include "Matrix3.hpp"
#include "sim_constants.hpp"
#include "save_matfile.hpp"


int save_matfile(
            const char *file,           // .MAT filename
            const int fieldlength,
            const int rows,
            const int cols,
            const Vector3 *M_ptr,       // pointer to M
            const fp_type *t_ptr,       // pointer to time
            const int debug   )
{
    if(debug) printf("Writing file %s ...\n", file);

    mwSize dims[4] = {3, rows, cols, fieldlength};    // Fortran/MATLAB style
    mxArray *pmat_M = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    if (pmat_M == NULL) {
        fprintf(stderr, "%s:%d Error allocating memory\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    //memcpy( (void*)(mxGetPr(pmat_M)), (void*)M_ptr, fieldlength*rows*cols*sizeof(*M_ptr));
    for(int p = 0; p < fieldlength; p++)
        for(int r = 0; r < rows; r++)       // loop is in row-major-order
            for(int c = 0; c < cols; c++)
                ((Vector3*)mxGetPr(pmat_M))[p*cols*rows + c*rows + r] = M_ptr[p*rows*cols + r*cols + c];

    mxArray *pmat_t = mxCreateNumericMatrix(1, fieldlength, mxSINGLE_CLASS, mxREAL);
    mxArray *pmat_Hext = mxCreateNumericMatrix(3, fieldlength, mxSINGLE_CLASS, mxREAL);
    mxArray *pmat_fieldlength = mxCreateDoubleScalar(fieldlength);
    mxArray *pmat_numdots_x = mxCreateDoubleScalar(numdots_x);
    mxArray *pmat_numdots_y = mxCreateDoubleScalar(numdots_y);
    mxArray *pmat_Ms = mxCreateDoubleScalar(Ms);
    if (pmat_fieldlength == NULL || pmat_numdots_x == NULL || pmat_numdots_y == NULL || pmat_Ms == NULL || pmat_t == NULL || pmat_Hext == NULL) {
        fprintf(stderr, "%s:%d Error allocating memory\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    memcpy( (void*)(mxGetPr(pmat_t)), (void*)t_ptr, fieldlength*sizeof(*t_ptr));
    for(int p = 0; p < fieldlength; p++)
        ((Vector3*)mxGetPr(pmat_Hext))[p] = Hext_function(t_ptr[p]);
    
    // open the MAT file
    MATFile *pmatfile = matOpen(file, "w7.3");
    if (pmatfile == NULL) {
        fprintf(stderr, "%s:%d Error opening file %s\n", __FILE__, __LINE__, file);
        return EXIT_FAILURE;
    }

    if( matPutVariable(pmatfile, "M", pmat_M) )
        return EXIT_FAILURE;
    if( matPutVariable(pmatfile, "t", pmat_t) )
        return EXIT_FAILURE;
    if( matPutVariable(pmatfile, "Hext", pmat_Hext) )
        return EXIT_FAILURE;
    if( matPutVariable(pmatfile, "fieldlength", pmat_fieldlength) )
        return EXIT_FAILURE;
    if( matPutVariable(pmatfile, "numdots_x", pmat_numdots_x) )
        return EXIT_FAILURE;
    if( matPutVariable(pmatfile, "numdots_y", pmat_numdots_y) )
        return EXIT_FAILURE;
    if( matPutVariable(pmatfile, "Ms", pmat_Ms) )
        return EXIT_FAILURE;

    // done with MAT file, so close it
    if (matClose(pmatfile) != 0) {
        fprintf(stderr, "Error closing file %s\n",file);
        return EXIT_FAILURE;
    }

    // reclaim memory
    mxDestroyArray(pmat_M);
    mxDestroyArray(pmat_fieldlength);
    mxDestroyArray(pmat_numdots_x);
    mxDestroyArray(pmat_numdots_y);
    mxDestroyArray(pmat_Ms);
    mxDestroyArray(pmat_t);
    mxDestroyArray(pmat_Hext);

    return EXIT_SUCCESS;
}
