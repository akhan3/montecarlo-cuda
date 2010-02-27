#include "Matrix3.hpp"

HOSTDEVICE Matrix3::Matrix3() {
    memset((void*)elem, 0.0, 3*3*sizeof(fp_type));
}

HOSTDEVICE Matrix3::Matrix3(const fp_type x, const fp_type y, const fp_type z) {
    memset((void*)elem, 0.0, 3*3*sizeof(fp_type));
    elem[0][0] = x;
    elem[1][1] = y;
    elem[2][2] = z;
}

HOSTDEVICE Matrix3::Matrix3(const fp_type* elem_in) {
    memcpy((fp_type*)elem, elem_in, 3*3*sizeof(fp_type));
}

HOSTDEVICE Vector3 Matrix3::operator*(const Vector3 &b) const {
    return Vector3( elem[0][0] * b.elem[0] + elem[0][1] * b.elem[1] + elem[0][2] * b.elem[2],
                    elem[1][0] * b.elem[0] + elem[1][1] * b.elem[1] + elem[1][2] * b.elem[2],
                    elem[2][0] * b.elem[0] + elem[2][1] * b.elem[1] + elem[2][2] * b.elem[2]    );
}

HOST void Matrix3::print() const {
    printf("[ %g,\t %g,\t %g  \n", elem[0][0], elem[0][1], elem[0][2]);
    printf("  %g,\t %g,\t %g  \n", elem[1][0], elem[1][1], elem[1][2]);
    printf("  %g,\t %g,\t %g ]\n", elem[2][0], elem[2][1], elem[2][2]);
}
