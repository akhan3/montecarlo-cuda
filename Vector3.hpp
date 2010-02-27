#ifndef _VECTOR3_H_
#define _VECTOR3_H_

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "my_macros.hpp"

struct Vector3 {
// data members
    fp_type elem[3];
// ctor's
    Vector3();
    Vector3(const fp_type x, const fp_type y, const fp_type z);
    Vector3(const fp_type* elem_in);
// member functions
    Vector3 operator+(const Vector3 &b) const;
    Vector3 operator-(const Vector3 &b) const;
    Vector3 operator/(const Vector3 &b) const;
    Vector3 operator*(const fp_type k) const;
    friend Vector3 operator*(const fp_type k, const Vector3 &vec);
    Vector3 operator*(const Vector3 &b) const;  // element-by-element multiplication
    fp_type dot(const Vector3 &b) const;        // dot product
    Vector3 cross(const Vector3 &b) const;      // cross product
    fp_type magnitude() const;
    friend std::ostream& operator<<(std::ostream& output, const Vector3 &vec);
    void print() const;
};

// friend Vector3 operator*(const fp_type k, const Vector3 &vec);
HOSTDEVICE inline Vector3 operator*(const fp_type k, const Vector3 &vec) {
    return vec * k;
}

// friend ostream& operator<<(ostream& output, const Vector3 &vec);
HOSTDEVICE inline std::ostream& operator<<(std::ostream& output, const Vector3 &vec) {
    output << vec.elem[0] << ", " << vec.elem[1] << ", " << vec.elem[2];
    return output;
}

#endif // #ifndef  _VECTOR3_H_
