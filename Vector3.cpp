#ifndef _VECTOR3_CU_
#define _VECTOR3_CU_

#include "Vector3.hpp"

HOSTDEVICE Vector3::Vector3() {
     elem[0] = 0.0;
     elem[1] = 0.0;
     elem[2] = 0.0;
}

HOSTDEVICE Vector3::Vector3(const fp_type x, const fp_type y, const fp_type z) {
     elem[0] = x;
     elem[1] = y;
     elem[2] = z;
}

HOSTDEVICE Vector3::Vector3(const fp_type* elem_in) {
     elem[0] = elem_in[0];
     elem[1] = elem_in[1];
     elem[2] = elem_in[2];
}

HOSTDEVICE Vector3 Vector3::operator+(const Vector3 &b) const {
    return Vector3( elem[0] + b.elem[0],
                    elem[1] + b.elem[1],
                    elem[2] + b.elem[2] );
}

HOSTDEVICE Vector3 Vector3::operator-(const Vector3 &b) const {
    return Vector3( elem[0] - b.elem[0],
                    elem[1] - b.elem[1],
                    elem[2] - b.elem[2] );
}

HOSTDEVICE Vector3 Vector3::operator/(const Vector3 &b) const {
    return Vector3( elem[0] / b.elem[0],
                    elem[1] / b.elem[1],
                    elem[2] / b.elem[2] );
}

HOSTDEVICE Vector3 Vector3::operator*(const fp_type k) const {
    return Vector3(k * elem[0], k * elem[1], k * elem[2]);
}

HOSTDEVICE Vector3 Vector3::operator*(const Vector3 &b) const {
    return Vector3( elem[0] * b.elem[0],
                    elem[1] * b.elem[1],
                    elem[2] * b.elem[2] );
}

HOSTDEVICE fp_type Vector3::dot(const Vector3 &b) const {
    return elem[0] * b.elem[0] + elem[1] * b.elem[1] + elem[2] * b.elem[2];
}

HOSTDEVICE Vector3 Vector3::cross(const Vector3 &b) const {
    return Vector3( elem[1] * b.elem[2] - elem[2] * b.elem[1],
                    elem[2] * b.elem[0] - elem[0] * b.elem[2],
                    elem[0] * b.elem[1] - elem[1] * b.elem[0]   );
}

HOSTDEVICE fp_type Vector3::magnitude() const {
    return sqrt(elem[0] * elem[0] + elem[1] * elem[1] + elem[2] * elem[2]);
}

HOST void Vector3::print() const {
    printf("[%g,\t %g,\t %g]\n", elem[0], elem[1], elem[2]);
}

#endif // #ifndef  _VECTOR3_CU_
