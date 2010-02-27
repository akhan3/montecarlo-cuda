#ifndef _SIM_CONSTANTS_H_
#define _SIM_CONSTANTS_H_

#include "my_macros.hpp"
#include "Vector3.hpp"
#include "Matrix3.hpp"

extern const fp_type finaltime;
extern const fp_type timestep;

extern const int numdots_y;   // rows
extern const int numdots_x;   // columns
extern const int numdots;

extern const fp_type alfa;     // damping coefficient
extern const fp_type c;   // gamma (LLG gyromagneticx ratio)
extern const fp_type c0;     // coupling matrix between dots
extern const Vector3 N;     // demagnetization tensor (must be declared as a matrix if non-zero off-diagonal elements)
extern const fp_type Ms;    // saturation magnetization of permalloy

extern const int save_matfiles;
extern const char matfiles_dir[];
extern const char *sim_id;

// external field as a function of time
Vector3 Hext(fp_type t);

#endif // #ifndef _SIM_CONSTANTS_H_
