#ifndef _MATRIX3_H_
#define _MATRIX3_H_

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "my_macros.hpp"
#include "Vector3.hpp"

struct Matrix3 {
// data members
    fp_type elem[3][3];
// ctor's
    Matrix3();
    Matrix3(const fp_type x, const fp_type y, const fp_type z);
    Matrix3(const fp_type* elem_in);
// member functions
    Vector3 operator*(const Vector3 &b) const;
    void print() const;
};

#endif // #ifndef  _MATRIX3_H_
