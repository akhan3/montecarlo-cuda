#include <stdlib.h>
#include "sim_constants.hpp"

const fp_type finaltime = 1E-9;
const fp_type timestep = 2E-12;

const int numdots_y     = 31;   // rows
const int numdots_x     = 31;   // columns
const int numdots       = numdots_x * numdots_y;

const fp_type alfa      = 0.05;     // damping coefficient
const fp_type c         = 2.21E5;   // gamma (LLG gyromagneticx ratio)
const fp_type c0        = -0.01;     // coupling matrix between dots
//const Matrix3 N(0.4, 0.4, 0.2);     // demagnetization tensor
const Vector3 N(0.4, 0.4, 0.2);     // demagnetization tensor (must be declared as a matrix if non-zero off-diagonal elements)
const fp_type Ms        = 8.6E5;    // saturation magnetization of permalloy

const int save_matfiles = 0;
const char matfiles_dir[] = "./results_matfiles";
const char *sim_id = (char*)malloc(100);
int dummy = sprintf((char*)sim_id, "sim_%dx%d_dots_%g_coupling_%gs_step_%gs", numdots_y, numdots_x, c0, finaltime, timestep);

// external field as a function of time
Vector3 Hext(fp_type t) {
    fp_type Hmax = 0.5E6; // maximum of external field
    //return Hmax * Vector3(  0.1*exp(-2.0*t/finaltime) * sin(5.0E+10*t),
                            //0.2*exp(-2.0*t/finaltime) * sin(5.0E+10*t),
                            //1.0*exp(-2.0*t/finaltime) * sin(5.0E+10*t)  );
    //return Hmax * Vector3(0.1,0.1,1);
    return Hmax * Vector3(0,0,0);
}
