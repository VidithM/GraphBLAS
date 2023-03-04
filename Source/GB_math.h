//------------------------------------------------------------------------------
// GB_math.h: definitions for complex types, and mathematical operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_MATH_H
#define GB_MATH_H

//------------------------------------------------------------------------------
// complex macros
//------------------------------------------------------------------------------

#if GB_COMPILER_MSC

    //--------------------------------------------------------------------------
    // Microsoft Visual Studio compiler with its own complex type
    //--------------------------------------------------------------------------

    // complex-complex multiply: z = x*y where both x and y are complex
    #define GB_FC32_MUL_DEFN                                                \
           "GB_FC32_mul(x,y) (_FCmulcc (x, y))"
    #define GB_FC32_mul(x,y) (_FCmulcc (x, y))

    #define GB_FC64_MUL_DEFN                                                \
           "GB_FC64_mul(x,y) ( _Cmulcc (x, y))"
    #define GB_FC64_mul(x,y) ( _Cmulcc (x, y))

    // complex-complex addition: z = x+y where both x and y are complex
    #define GB_FC32_ADD_DEFN                                                \
       "    GB_FC32_add(x,y) "                                              \
       "GxB_CMPLXF (crealf (x) + crealf (y), cimagf (x) + cimagf (y))"
    #define GB_FC32_add(x,y)                                                \
        GxB_CMPLXF (crealf (x) + crealf (y), cimagf (x) + cimagf (y))

    #define GB_FC64_ADD_DEFN                                                \
       "    GB_FC64_add(x,y) "                                              \
       "GxB_CMPLX  (creal  (x) + creal  (y), cimag  (x) + cimag  (y))"
    #define GB_FC64_add(x,y)                                                \
        GxB_CMPLX  (creal  (x) + creal  (y), cimag  (x) + cimag  (y))

    // complex-complex subtraction: z = x-y where both x and y are complex
    #define GB_FC32_MINUS_DEFN                                              \
       "    GB_FC32_minus(x,y) "                                            \
       "GxB_CMPLXF (crealf (x) - crealf (y), cimagf (x) - cimagf (y))"
    #define GB_FC32_minus(x,y)                                              \
        GxB_CMPLXF (crealf (x) - crealf (y), cimagf (x) - cimagf (y))

    #define GB_FC64_MINUS_DEFN                                              \
       "    GB_FC64_minus(x,y) "                                            \
       "GxB_CMPLX  (creal  (x) - creal  (y), cimag  (x) - cimag  (y))"
    #define GB_FC64_minus(x,y)                                              \
        GxB_CMPLX  (creal  (x) - creal  (y), cimag  (x) - cimag  (y))

    // complex negation: z = -x
    #define GB_FC32_AINV_DEFN                                               \
           "GB_FC32_ainv(x) GxB_CMPLXF (-crealf (x), -cimagf (x))"
    #define GB_FC32_ainv(x) GxB_CMPLXF (-crealf (x), -cimagf (x))

    #define GB_FC64_AINV_DEFN                                               \
           "GB_FC64_ainv(x) GxB_CMPLX  (-creal  (x), -cimag  (x))"
    #define GB_FC64_ainv(x) GxB_CMPLX  (-creal  (x), -cimag  (x))

#else

    //--------------------------------------------------------------------------
    // native complex type support
    //--------------------------------------------------------------------------

    // complex-complex multiply: z = x*y where both x and y are complex
    #define GB_FC32_MUL_DEFN                                                \
           "GB_FC32_mul(x,y) ((x) * (y))"
    #define GB_FC32_mul(x,y) ((x) * (y))

    #define GB_FC64_MUL_DEFN                                                \
           "GB_FC64_mul(x,y) ((x) * (y))"
    #define GB_FC64_mul(x,y) ((x) * (y))

    // complex-complex addition: z = x+y where both x and y are complex
    #define GB_FC32_ADD_DEFN                                                \
           "GB_FC32_add(x,y) ((x) + (y))"
    #define GB_FC32_add(x,y) ((x) + (y))

    #define GB_FC64_ADD_DEFN                                                \
           "GB_FC64_add(x,y) ((x) + (y))"
    #define GB_FC64_add(x,y) ((x) + (y))

    // complex-complex subtraction: z = x-y where both x and y are complex
    #define GB_FC32_MINUS_DEFN                                              \
           "GB_FC32_minus(x,y) ((x) - (y))"
    #define GB_FC32_minus(x,y) ((x) - (y))

    #define GB_FC64_MINUS_DEFN                                              \
           "GB_FC64_minus(x,y) ((x) - (y))"
    #define GB_FC64_minus(x,y) ((x) - (y))

    // complex negation
    #define GB_FC32_AINV_DEFN                                               \
           "GB_FC32_ainv(x) (-(x))"
    #define GB_FC32_ainv(x) (-(x))

    #define GB_FC64_AINV_DEFN                                               \
           "GB_FC64_ainv(x) (-(x))"
    #define GB_FC64_ainv(x) (-(x))

#endif

#define GB_GUARD_GB_FC32_MUL_DEFINED
#define GB_GUARD_GB_FC64_MUL_DEFINED
#define GB_GUARD_GB_FC32_ADD_DEFINED
#define GB_GUARD_GB_FC64_ADD_DEFINED
#define GB_GUARD_GB_FC32_MINUS_DEFINED
#define GB_GUARD_GB_FC64_MINUS_DEFINED
#define GB_GUARD_GB_FC32_AINV_DEFINED
#define GB_GUARD_GB_FC64_AINV_DEFINED

// complex comparators
#define GB_GUARD_GB_FC32_EQ_DEFINED
#define GB_FC32_EQ_DEFN                                                     \
       "GB_FC32_eq(x,y) ((crealf(x) == crealf(y)) && (cimagf(x) == cimagf(y)))"
#define GB_FC32_eq(x,y) ((crealf(x) == crealf(y)) && (cimagf(x) == cimagf(y)))

#define GB_GUARD_GB_FC64_EQ_DEFINED
#define GB_FC64_EQ_DEFN                                                     \
       "GB_FC64_eq(x,y) ((creal (x) == creal (y)) && (cimag (x) == cimag (y)))"
#define GB_FC64_eq(x,y) ((creal (x) == creal (y)) && (cimag (x) == cimag (y)))

#define GB_GUARD_GB_FC32_NE_DEFINED
#define GB_FC32_NE_DEFN                                                     \
       "GB_FC32_ne(x,y) ((crealf(x) != crealf(y)) || (cimagf(x) != cimagf(y)))"
#define GB_FC32_ne(x,y) ((crealf(x) != crealf(y)) || (cimagf(x) != cimagf(y)))

#define GB_GUARD_GB_FC64_NE_DEFINED
#define GB_FC64_NE_DEFN                                                     \
       "GB_FC64_ne(x,y) ((creal (x) != creal (y)) || (cimag (x) != cimag (y)))"
#define GB_FC64_ne(x,y) ((creal (x) != creal (y)) || (cimag (x) != cimag (y)))

#define GB_GUARD_GB_FC32_ISEQ_DEFINED
#define GB_FC32_ISEQ_DEFN                                                   \
       "GB_FC32_iseq(x,y) GB_CMPLX32 ((float)  GB_FC32_eq (x,y), 0)"
#define GB_FC32_iseq(x,y) GB_CMPLX32 ((float)  GB_FC32_eq (x,y), 0)

#define GB_GUARD_GB_FC64_ISEQ_DEFINED
#define GB_FC64_ISEQ_DEFN                                                   \
       "GB_FC64_iseq(x,y) GB_CMPLX64  ((double) GB_FC64_eq (x,y), 0)"
#define GB_FC64_iseq(x,y) GB_CMPLX64  ((double) GB_FC64_eq (x,y), 0)

#define GB_GUARD_GB_FC32_ISNE_DEFINED
#define GB_FC32_ISNE_DEFN                                                   \
       "GB_FC32_isne(x,y) GB_CMPLX32 ((float)  GB_FC32_ne (x,y), 0)"
#define GB_FC32_isne(x,y) GB_CMPLX32 ((float)  GB_FC32_ne (x,y), 0)

#define GB_GUARD_GB_FC64_ISNE_DEFINED
#define GB_FC64_ISNE_DEFN                                                   \
       "GB_FC64_isne(x,y) GB_CMPLX64  ((double) GB_FC64_ne (x,y), 0)"
#define GB_FC64_isne(x,y) GB_CMPLX64  ((double) GB_FC64_ne (x,y), 0)

#define GB_GUARD_GB_FC32_EQ0_DEFINED
#define GB_FC32_EQ0_DEFN                                                    \
       "GB_FC32_eq0(x) ((crealf (x) == 0) && (cimagf (x) == 0))"
#define GB_FC32_eq0(x) ((crealf (x) == 0) && (cimagf (x) == 0))

#define GB_GUARD_GB_FC64_EQ0_DEFINED
#define GB_FC64_EQ0_DEFN                                                    \
       "GB_FC64_eq0(x) ((creal  (x) == 0) && (cimag  (x) == 0))"
#define GB_FC64_eq0(x) ((creal  (x) == 0) && (cimag  (x) == 0))

#define GB_GUARD_GB_FC32_NE0_DEFINED
#define GB_FC32_NE0_DEFN                                                    \
       "GB_FC32_ne0(x) ((crealf (x) != 0) || (cimagf (x) != 0))"
#define GB_FC32_ne0(x) ((crealf (x) != 0) || (cimagf (x) != 0))

#define GB_GUARD_GB_FC64_NE0_DEFINED
#define GB_FC64_NE0_DEFN                                                    \
       "GB_FC64_ne0(x) ((creal  (x) != 0) || (cimag  (x) != 0))"
#define GB_FC64_ne0(x) ((creal  (x) != 0) || (cimag  (x) != 0))

//------------------------------------------------------------------------------
// min, max, and NaN handling
//------------------------------------------------------------------------------

// For floating-point computations, SuiteSparse:GraphBLAS relies on the IEEE
// 754 standard for the basic operations (+ - / *).  Comparator also
// work as they should; any compare with NaN is always false, even
// eq(NaN,NaN) is false.  This follows the IEEE 754 standard.

// For integer MIN and MAX, SuiteSparse:GraphBLAS relies on one compator:
// z = min(x,y) = (x < y) ? x : y
// z = max(x,y) = (x > y) ? x : y

// However, this is not suitable for floating-point x and y.  Compares with
// NaN always return false, so if either x or y are NaN, then z = y, for both
// min(x,y) and max(x,y).

// The ANSI C11 fmin, fminf, fmax, and fmaxf functions have the 'omitnan'
// behavior.  These are used in SuiteSparse:GraphBLAS v2.3.0 and later.

// for integers only:
#define GB_IABS(x) (((x) >= 0) ? (x) : (-(x)))

// suitable for integers, and non-NaN floating point:
#include "GB_imin.h"

// ceiling of a/b for two integers a and b
#include "GB_iceil.h"

//------------------------------------------------------------------------------
// integer division
//------------------------------------------------------------------------------

// Integer division is done carefully so that GraphBLAS does not terminate the
// user's application on divide-by-zero.  To compute x/0: if x is zero, the
// result is zero (like NaN).  if x is negative, the result is the negative
// integer with biggest magnitude (like -infinity).  if x is positive, the
// result is the biggest positive integer (like +infinity).

    inline
    int8_t GB_idiv_int8 (int8_t x, int8_t y)
    {
        // returns x/y when x and y are int8_t
        if (y == -1)
        {
            // INT32_MIN/(-1) causes floating point exception; avoid it
            return (-x) ;
        }
        else if (y == 0)
        {
            // zero divided by zero gives 'integer Nan'
            // x/0 where x is nonzero: result is integer -Inf or +Inf
            return ((x == 0) ? 0 : ((x < 0) ? INT8_MIN : INT8_MAX)) ;
        }
        else
        {
            // normal case for signed integer division
            return (x / y) ;
        }
    }

#define GB_GUARD_GB_IDIV_INT8_DEFINED
#define GB_IDIV_INT8_DEFN                                                   \
   "int8_t GB_idiv_int8 (int8_t x, int8_t y)                            \n" \
   "{                                                                   \n" \
   "    if (y == -1)                                                    \n" \
   "    {                                                               \n" \
   "        return (-x) ;                                               \n" \
   "    }                                                               \n" \
   "    else if (y == 0)                                                \n" \
   "    {                                                               \n" \
   "        return ((x == 0) ? 0 : ((x < 0) ? INT8_MIN : INT8_MAX)) ;   \n" \
   "    }                                                               \n" \
   "    else                                                            \n" \
   "    {                                                               \n" \
   "        return (x / y) ;                                            \n" \
   "    }                                                               \n" \
   "}"

    inline
    int16_t GB_idiv_int16 (int16_t x, int16_t y)
    {
        // returns x/y when x and y are int16_t
        if (y == -1)
        {
            // INT32_MIN/(-1) causes floating point exception; avoid it
            return (-x) ;
        }
        else if (y == 0)
        {
            // zero divided by zero gives 'integer Nan'
            // x/0 where x is nonzero: result is integer -Inf or +Inf
            return ((x == 0) ? 0 : ((x < 0) ? INT16_MIN : INT16_MAX)) ;
        }
        else
        {
            // normal case for signed integer division
            return (x / y) ;
        }
    }

#define GB_GUARD_GB_IDIV_INT16_DEFINED
#define GB_IDIV_INT16_DEFN                                                  \
   "int16_t GB_idiv_int16 (int16_t x, int16_t y)                        \n" \
   "{                                                                   \n" \
   "    if (y == -1)                                                    \n" \
   "    {                                                               \n" \
   "        return (-x) ;                                               \n" \
   "    }                                                               \n" \
   "    else if (y == 0)                                                \n" \
   "    {                                                               \n" \
   "        return ((x == 0) ? 0 : ((x < 0) ? INT16_MIN : INT16_MAX)) ; \n" \
   "    }                                                               \n" \
   "    else                                                            \n" \
   "    {                                                               \n" \
   "        return (x / y) ;                                            \n" \
   "    }                                                               \n" \
   "}"

    inline
    int32_t GB_idiv_int32 (int32_t x, int32_t y)
    {
        // returns x/y when x and y are int32_t
        if (y == -1)
        {
            // INT32_MIN/(-1) causes floating point exception; avoid it
            return (-x) ;
        }
        else if (y == 0)
        {
            // zero divided by zero gives 'integer Nan'
            // x/0 where x is nonzero: result is integer -Inf or +Inf
            return ((x == 0) ? 0 : ((x < 0) ? INT32_MIN : INT32_MAX)) ;
        }
        else
        {
            // normal case for signed integer division
            return (x / y) ;
        }
    }

#define GB_GUARD_GB_IDIV_INT32_DEFINED
#define GB_IDIV_INT32_DEFN                                                  \
   "int32_t GB_idiv_int32 (int32_t x, int32_t y)                        \n" \
   "{                                                                   \n" \
   "    if (y == -1)                                                    \n" \
   "    {                                                               \n" \
   "        return (-x) ;                                               \n" \
   "    }                                                               \n" \
   "    else if (y == 0)                                                \n" \
   "    {                                                               \n" \
   "        return ((x == 0) ? 0 : ((x < 0) ? INT32_MIN : INT32_MAX)) ; \n" \
   "    }                                                               \n" \
   "    else                                                            \n" \
   "    {                                                               \n" \
   "        return (x / y) ;                                            \n" \
   "    }                                                               \n" \
   "}"

    inline
    int64_t GB_idiv_int64 (int64_t x, int64_t y)
    {
        // returns x/y when x and y are int64_t
        if (y == -1)
        {
            // INT32_MIN/(-1) causes floating point exception; avoid it
            return (-x) ;
        }
        else if (y == 0)
        {
            // zero divided by zero gives 'integer Nan'
            // x/0 where x is nonzero: result is integer -Inf or +Inf
            return ((x == 0) ? 0 : ((x < 0) ? INT64_MIN : INT64_MAX)) ;
        }
        else
        {
            // normal case for signed integer division
            return (x / y) ;
        }
    }

#define GB_GUARD_GB_IDIV_INT64_DEFINED
#define GB_IDIV_INT64_DEFN                                                  \
   "int64_t GB_idiv_int64 (int64_t x, int64_t y)                        \n" \
   "{                                                                   \n" \
   "    if (y == -1)                                                    \n" \
   "    {                                                               \n" \
   "        return (-x) ;                                               \n" \
   "    }                                                               \n" \
   "    else if (y == 0)                                                \n" \
   "    {                                                               \n" \
   "        return ((x == 0) ? 0 : ((x < 0) ? INT64_MIN : INT64_MAX)) ; \n" \
   "    }                                                               \n" \
   "    else                                                            \n" \
   "    {                                                               \n" \
   "        return (x / y) ;                                            \n" \
   "    }                                                               \n" \
   "}"

    inline
    uint8_t GB_idiv_uint8 (uint8_t x, uint8_t y)
    {
        if (y == 0)
        {
            // x/0:  0/0 is integer Nan, otherwise result is +Inf
            return ((x == 0) ? 0 : UINT8_MAX) ;
        }
        else
        {
            // normal case for unsigned integer division
            return (x / y) ;
        }
    }

#define GB_GUARD_GB_IDIV_UINT8_DEFINED
#define GB_IDIV_UINT8_DEFN                                                  \
   "uint8_t GB_idiv_uint8 (uint8_t x, uint8_t y)                        \n" \
   "{                                                                   \n" \
   "    if (y == 0)                                                     \n" \
   "    {                                                               \n" \
   "        return ((x == 0) ? 0 : UINT8_MAX) ;                         \n" \
   "    }                                                               \n" \
   "    else                                                            \n" \
   "    {                                                               \n" \
   "        return (x / y) ;                                            \n" \
   "    }                                                               \n" \
   "}"

    inline
    uint16_t GB_idiv_uint16 (uint16_t x, uint16_t y)
    {
        if (y == 0)
        {
            // x/0:  0/0 is integer Nan, otherwise result is +Inf
            return ((x == 0) ? 0 : UINT16_MAX) ;
        }
        else
        {
            // normal case for unsigned integer division
            return (x / y) ;
        }
    }

#define GB_GUARD_GB_IDIV_UINT16_DEFINED
#define GB_IDIV_UINT16_DEFN                                                 \
   "uint16_t GB_idiv_uint16 (uint16_t x, uint16_t y)                    \n" \
   "{                                                                   \n" \
   "    if (y == 0)                                                     \n" \
   "    {                                                               \n" \
   "        return ((x == 0) ? 0 : UINT16_MAX) ;                        \n" \
   "    }                                                               \n" \
   "    else                                                            \n" \
   "    {                                                               \n" \
   "        return (x / y) ;                                            \n" \
   "    }                                                               \n" \
   "}"

    inline
    uint32_t GB_idiv_uint32 (uint32_t x, uint32_t y)
    {
        if (y == 0)
        {
            // x/0:  0/0 is integer Nan, otherwise result is +Inf
            return ((x == 0) ? 0 : UINT32_MAX) ;
        }
        else
        {
            // normal case for unsigned integer division
            return (x / y) ;
        }
    }

#define GB_GUARD_GB_IDIV_UINT32_DEFINED
#define GB_IDIV_UINT32_DEFN                                                 \
   "uint32_t GB_idiv_uint32 (uint32_t x, uint32_t y)                    \n" \
   "{                                                                   \n" \
   "    if (y == 0)                                                     \n" \
   "    {                                                               \n" \
   "        return ((x == 0) ? 0 : UINT32_MAX) ;                        \n" \
   "    }                                                               \n" \
   "    else                                                            \n" \
   "    {                                                               \n" \
   "        return (x / y) ;                                            \n" \
   "    }                                                               \n" \
   "}"

    inline
    uint64_t GB_idiv_uint64 (uint64_t x, uint64_t y)
    {
        if (y == 0)
        {
            // x/0:  0/0 is integer Nan, otherwise result is +Inf
            return ((x == 0) ? 0 : UINT64_MAX) ;
        }
        else
        {
            // normal case for unsigned integer division
            return (x / y) ;
        }
    }

#define GB_GUARD_GB_IDIV_UINT64_DEFINED
#define GB_IDIV_UINT64_DEFN                                                 \
   "uint64_t GB_idiv_uint64 (uint64_t x, uint64_t y)                    \n" \
   "{                                                                   \n" \
   "    if (y == 0)                                                     \n" \
   "    {                                                               \n" \
   "        return ((x == 0) ? 0 : UINT64_MAX) ;                        \n" \
   "    }                                                               \n" \
   "    else                                                            \n" \
   "    {                                                               \n" \
   "        return (x / y) ;                                            \n" \
   "    }                                                               \n" \
   "}"

//------------------------------------------------------------------------------
// complex division
//------------------------------------------------------------------------------

#if 1

    // complex division is problematic.  It is not supported at all on MS
    // Visual Studio.  With other compilers, complex division exists but it has
    // different NaN and Inf behavior as compared with MATLAB, which causes the
    // tests to fail.  As a result, the built-in complex division is not used,
    // even if the compiler supports it.

    // Three cases below are from ACM Algo 116, R. L. Smith, 1962.

    inline
    GxB_FC64_t GB_FC64_div (GxB_FC64_t x, GxB_FC64_t y)
    {
        double xr = creal (x) ;
        double xi = cimag (x) ;
        double yr = creal (y) ;
        double yi = cimag (y) ;
        int yr_class = fpclassify (yr) ;
        int yi_class = fpclassify (yi) ;
        if (yi_class == FP_ZERO)
        {
            // (zr,zi) = (xr,xi) / (yr,0)
            return (GB_CMPLX64 (xr / yr, cimag (x) / yr)) ;
        }
        else if (yr_class == FP_ZERO)
        {
            // (zr,zi) = (xr,xi) / (0,yi) = (xi,-xr) / (yi,0)
            return (GB_CMPLX64 (xi / yi, -xr / yi)) ;
        }
        else if (yi_class == FP_INFINITE && yr_class == FP_INFINITE)
        {
            // Using Smith's method for a very special case
            double r = (signbit (yr) == signbit (yi)) ? (1) : (-1) ;
            double d = yr + r * yi ;
            return (GB_CMPLX64 ((xr + xi * r) / d, (xi - xr * r) / d)) ;
        }
        else if (fabs (yr) >= fabs (yi))
        {
            // Smith's method (1st case)
            double r = yi / yr ;
            double d = yr + r * yi ;
            return (GB_CMPLX64 ((xr + xi * r) / d, (xi - xr * r) / d)) ;
        }
        else
        {
            // Smith's method (2nd case)
            double r = yr / yi ;
            double d = r * yr + yi ;
            return (GB_CMPLX64 ((xr * r + xi) / d, (xi * r - xr) / d)) ;
        }
    }

   #define GB_FC64_DIV_DEFN                                                 \
   "GxB_FC64_t GB_FC64_div (GxB_FC64_t x, GxB_FC64_t y)                 \n" \
   "{                                                                   \n" \
   "    double xr = creal (x) ;                                         \n" \
   "    double xi = cimag (x) ;                                         \n" \
   "    double yr = creal (y) ;                                         \n" \
   "    double yi = cimag (y) ;                                         \n" \
   "    int yr_class = fpclassify (yr) ;                                \n" \
   "    int yi_class = fpclassify (yi) ;                                \n" \
   "    if (yi_class == FP_ZERO)                                        \n" \
   "    {                                                               \n" \
   "        return (GB_CMPLX64 (xr / yr, xi / yr)) ;                    \n" \
   "    }                                                               \n" \
   "    else if (yr_class == FP_ZERO)                                   \n" \
   "    {                                                               \n" \
   "        return (GB_CMPLX64 (xi / yi, -xr / yi)) ;                   \n" \
   "    }                                                               \n" \
   "    else if (yi_class == FP_INFINITE && yr_class == FP_INFINITE)    \n" \
   "    {                                                               \n" \
   "        double r = (signbit (yr) == signbit (yi)) ? (1) : (-1) ;    \n" \
   "        double d = yr + r * yi ;                                    \n" \
   "        return (GB_CMPLX64 ((xr + xi * r) / d, (xi - xr * r) / d)) ;\n" \
   "    }                                                               \n" \
   "    else if (fabs (yr) >= fabs (yi))                                \n" \
   "    {                                                               \n" \
   "        double r = yi / yr ;                                        \n" \
   "        double d = yr + r * yi ;                                    \n" \
   "        return (GB_CMPLX64 ((xr + xi * r) / d, (xi - xr * r) / d)) ;\n" \
   "    }                                                               \n" \
   "    else                                                            \n" \
   "    {                                                               \n" \
   "        double r = yr / yi ;                                        \n" \
   "        double d = r * yr + yi ;                                    \n" \
   "        return (GB_CMPLX64 ((xr * r + xi) / d, (xi * r - xr) / d)) ;\n" \
   "    }                                                               \n" \
   "}"

    inline
    GxB_FC32_t GB_FC32_div (GxB_FC32_t x, GxB_FC32_t y)
    {
        // single complex division: cast double complex, do the division,
        // and then cast back to single complex.
        double xr = (double) crealf (x) ;
        double xi = (double) cimagf (x) ;
        double yr = (double) crealf (y) ;
        double yi = (double) cimagf (y) ;
        GxB_FC64_t zz = GB_FC64_div (GB_CMPLX64 (xr, xi), GB_CMPLX64 (yr, yi)) ;
        return (GB_CMPLX32 ((float) creal (zz), (float) cimag (zz))) ;
    }

   #define GB_FC32_DIV_DEFN                                                 \
   "GxB_FC32_t GB_FC32_div (GxB_FC32_t x, GxB_FC32_t y)                 \n" \
   "{                                                                   \n" \
   "    double xr = (double) crealf (x) ;                               \n" \
   "    double xi = (double) cimagf (x) ;                               \n" \
   "    double yr = (double) crealf (y) ;                               \n" \
   "    double yi = (double) cimagf (y) ;                               \n" \
   "    GxB_FC64_t zz ;                                                 \n" \
   "    zz = GB_FC64_div (GB_CMPLX64 (xr, xi), GB_CMPLX64 (yr, yi)) ;   \n" \
   "    return (GB_CMPLX32 ((float) creal (zz), (float) cimag (zz))) ;  \n" \
   "}"

#else

    // built-in complex division:  this works (except for MS Visual Studio) but
    // it gives unpredictable results, particularly when considering Inf and
    // NaN behavior.

    inline
    GxB_FC64_t GB_FC64_div (GxB_FC64_t x, GxB_FC64_t y)
    {
        return (x / y) ;
    }

   #define GB_FC64_DIV_DEFN                                                 \
   "GxB_FC64_t GB_FC64_div (GxB_FC64_t x, GxB_FC64_t y)                 \n" \
   "{                                                                   \n" \
   "    return (x / y) ;                                                \n" \
   "}"

    inline
    GxB_FC32_t GB_FC32_div (GxB_FC32_t x, GxB_FC32_t y)
    {
        return (x / y) ;
    }

   #define GB_FC32_DIV_DEFN                                                 \
   "GxB_FC32_t GB_FC32_div (GxB_FC32_t x, GxB_FC32_t y)                 \n" \
   "{                                                                   \n" \
   "    return (x / y) ;                                                \n" \
   "}"

#endif

#define GB_GUARD_GB_FC32_DIV_DEFINED
#define GB_GUARD_GB_FC64_DIV_DEFINED

//------------------------------------------------------------------------------
// z = x^y: wrappers for pow, powf, cpow, and cpowf
//------------------------------------------------------------------------------

//      if x or y are NaN, then z is NaN
//      if y is zero, then z is 1
//      if (x and y are complex but with zero imaginary parts, and
//          (x >= 0 or if y is an integer, NaN, or Inf)), then z is real
//      else use the built-in C library function, z = pow (x,y)

    inline
    float GB_powf (float x, float y)
    {
        int xr_class = fpclassify (x) ;
        int yr_class = fpclassify (y) ;
        if (xr_class == FP_NAN || yr_class == FP_NAN)
        {
            // z is nan if either x or y are nan
            return (NAN) ;
        }
        if (yr_class == FP_ZERO)
        {
            // z is 1 if y is zero
            return (1) ;
        }
        // otherwise, z = powf (x,y)
        return (powf (x, y)) ;
    }

   #define GB_GUARD_GB_POWF_DEFINED
   #define GB_POWF_DEFN                                                     \
   "float GB_powf (float x, float y)                                    \n" \
   "{                                                                   \n" \
   "    int xr_class = fpclassify (x) ;                                 \n" \
   "    int yr_class = fpclassify (y) ;                                 \n" \
   "    if (xr_class == FP_NAN || yr_class == FP_NAN)                   \n" \
   "    {                                                               \n" \
   "        return (NAN) ;                                              \n" \
   "    }                                                               \n" \
   "    if (yr_class == FP_ZERO)                                        \n" \
   "    {                                                               \n" \
   "        return (1) ;                                                \n" \
   "    }                                                               \n" \
   "    return (powf (x, y)) ;                                          \n" \
   "}"

    inline
    double GB_pow (double x, double y)
    {
        int xr_class = fpclassify (x) ;
        int yr_class = fpclassify (y) ;
        if (xr_class == FP_NAN || yr_class == FP_NAN)
        {
            // z is nan if either x or y are nan
            return (NAN) ;
        }
        if (yr_class == FP_ZERO)
        {
            // z is 1 if y is zero
            return (1) ;
        }
        // otherwise, z = pow (x,y)
        return (pow (x, y)) ;
    }

   #define GB_GUARD_GB_POW_DEFINED
   #define GB_POW_DEFN                                                      \
   "float GB_pow (float x, float y)                                     \n" \
   "{                                                                   \n" \
   "    int xr_class = fpclassify (x) ;                                 \n" \
   "    int yr_class = fpclassify (y) ;                                 \n" \
   "    if (xr_class == FP_NAN || yr_class == FP_NAN)                   \n" \
   "    {                                                               \n" \
   "        return (NAN) ;                                              \n" \
   "    }                                                               \n" \
   "    if (yr_class == FP_ZERO)                                        \n" \
   "    {                                                               \n" \
   "        return (1) ;                                                \n" \
   "    }                                                               \n" \
   "    return (pow (x, y)) ;                                           \n" \
   "}"

    inline
    GxB_FC32_t GB_cpowf (GxB_FC32_t x, GxB_FC32_t y)
    {
        float xr = crealf (x) ;
        float yr = crealf (y) ;
        int xr_class = fpclassify (xr) ;
        int yr_class = fpclassify (yr) ;
        int xi_class = fpclassify (cimagf (x)) ;
        int yi_class = fpclassify (cimagf (y)) ;
        if (xi_class == FP_ZERO && yi_class == FP_ZERO)
        {
            // both x and y are real; see if z should be real
            if (xr >= 0 || yr_class == FP_NAN ||
                yr_class == FP_INFINITE || yr == truncf (yr))
            {
                // z is real if x >= 0, or if y is an integer, NaN, or Inf
                return (GB_CMPLX32 (GB_powf (xr, yr), 0)) ;
            }
        }
        if (xr_class == FP_NAN || xi_class == FP_NAN ||
            yr_class == FP_NAN || yi_class == FP_NAN)
        {
            // z is (nan,nan) if any part of x or y are nan
            return (GB_CMPLX32 (NAN, NAN)) ;
        }
        if (yr_class == FP_ZERO && yi_class == FP_ZERO)
        {
            // z is (1,0) if y is (0,0)
            return (GxB_CMPLXF (1, 0)) ;
        }
        return (cpowf (x, y)) ;
    }

   #define GB_GUARD_GB_CPOWF_DEFINED
   #define GB_CPOWF_DEFN                                                    \
   "GxB_FC32_t GB_cpowf (GxB_FC32_t x, GxB_FC32_t y)                    \n" \
   "{                                                                   \n" \
   "    float xr = crealf (x) ;                                         \n" \
   "    float yr = crealf (y) ;                                         \n" \
   "    int xr_class = fpclassify (xr) ;                                \n" \
   "    int yr_class = fpclassify (yr) ;                                \n" \
   "    int xi_class = fpclassify (cimagf (x)) ;                        \n" \
   "    int yi_class = fpclassify (cimagf (y)) ;                        \n" \
   "    if (xi_class == FP_ZERO && yi_class == FP_ZERO)                 \n" \
   "    {                                                               \n" \
   "        if (xr >= 0 || yr_class == FP_NAN ||                        \n" \
   "            yr_class == FP_INFINITE || yr == truncf (yr))           \n" \
   "        {                                                           \n" \
   "            return (GB_CMPLX32 (GB_powf (xr, yr), 0)) ;             \n" \
   "        }                                                           \n" \
   "    }                                                               \n" \
   "    if (xr_class == FP_NAN || xi_class == FP_NAN ||                 \n" \
   "        yr_class == FP_NAN || yi_class == FP_NAN)                   \n" \
   "    {                                                               \n" \
   "        return (GB_CMPLX32 (NAN, NAN)) ;                            \n" \
   "    }                                                               \n" \
   "    if (yr_class == FP_ZERO && yi_class == FP_ZERO)                 \n" \
   "    {                                                               \n" \
   "        return (GxB_CMPLXF (1, 0)) ;                                \n" \
   "    }                                                               \n" \
   "    return (cpowf (x, y)) ;                                         \n" \
   "}"

    inline
    GxB_FC64_t GB_cpow (GxB_FC64_t x, GxB_FC64_t y)
    {
        double xr = creal (x) ;
        double yr = creal (y) ;
        int xr_class = fpclassify (xr) ;
        int yr_class = fpclassify (yr) ;
        int xi_class = fpclassify (cimag (x)) ;
        int yi_class = fpclassify (cimag (y)) ;
        if (xi_class == FP_ZERO && yi_class == FP_ZERO)
        {
            // both x and y are real; see if z should be real
            if (xr >= 0 || yr_class == FP_NAN ||
                yr_class == FP_INFINITE || yr == trunc (yr))
            {
                // z is real if x >= 0, or if y is an integer, NaN, or Inf
                return (GB_CMPLX64 (GB_pow (xr, yr), 0)) ;
            }
        }
        if (xr_class == FP_NAN || xi_class == FP_NAN ||
            yr_class == FP_NAN || yi_class == FP_NAN)
        {
            // z is (nan,nan) if any part of x or y are nan
            return (GB_CMPLX64 (NAN, NAN)) ;
        }
        if (yr_class == FP_ZERO && yi_class == FP_ZERO)
        {
            // z is (1,0) if y is (0,0)
            return (GxB_CMPLX (1, 0)) ;
        }
        return (cpow (x, y)) ;
    }

   #define GB_GUARD_GB_CPOW_DEFINED
   #define GB_CPOW_DEFN                                                     \
   "GxB_FC64_t GB_cpow (GxB_FC64_t x, GxB_FC64_t y)                     \n" \
   "{                                                                   \n" \
   "    double xr = creal (x) ;                                         \n" \
   "    double yr = creal (y) ;                                         \n" \
   "    int xr_class = fpclassify (xr) ;                                \n" \
   "    int yr_class = fpclassify (yr) ;                                \n" \
   "    int xi_class = fpclassify (cimag (x)) ;                         \n" \
   "    int yi_class = fpclassify (cimag (y)) ;                         \n" \
   "    if (xi_class == FP_ZERO && yi_class == FP_ZERO)                 \n" \
   "    {                                                               \n" \
   "        if (xr >= 0 || yr_class == FP_NAN ||                        \n" \
   "            yr_class == FP_INFINITE || yr == trunc (yr))            \n" \
   "        {                                                           \n" \
   "            return (GB_CMPLX64 (GB_pow (xr, yr), 0)) ;              \n" \
   "        }                                                           \n" \
   "    }                                                               \n" \
   "    if (xr_class == FP_NAN || xi_class == FP_NAN ||                 \n" \
   "        yr_class == FP_NAN || yi_class == FP_NAN)                   \n" \
   "    {                                                               \n" \
   "        return (GB_CMPLX64 (NAN, NAN)) ;                            \n" \
   "    }                                                               \n" \
   "    if (yr_class == FP_ZERO && yi_class == FP_ZERO)                 \n" \
   "    {                                                               \n" \
   "        return (GxB_CMPLX (1, 0)) ;                                 \n" \
   "    }                                                               \n" \
   "    return (cpow (x, y)) ;                                          \n" \
   "}"

    inline
    int8_t GB_pow_int8 (int8_t x, int8_t y)
    {
        return (GB_cast_to_int8_t (GB_pow ((double) x, (double) y))) ;
    }

   #define GB_GUARD_GB_POW_INT8_DEFINED
   #define GB_POW_INT8_DEFN                                                 \
   "int8_t GB_pow_int8 (int8_t x, int8_t y)                             \n" \
   "{                                                                   \n" \
   "    return (GB_cast_to_int8_t (GB_pow ((double) x, (double) y))) ;  \n" \
   "}"

    inline
    int16_t GB_pow_int16 (int16_t x, int16_t y)
    {
        return (GB_cast_to_int16_t (GB_pow ((double) x, (double) y))) ;
    }

   #define GB_GUARD_GB_POW_INT16_DEFINED
   #define GB_POW_INT16_DEFN                                                \
   "int16_t GB_pow_int16 (int16_t x, int16_t y)                         \n" \
   "{                                                                   \n" \
   "    return (GB_cast_to_int16_t (GB_pow ((double) x, (double) y))) ; \n" \
   "}"

    inline
    int32_t GB_pow_int32 (int32_t x, int32_t y)
    {
        return (GB_cast_to_int32_t (GB_pow ((double) x, (double) y))) ;
    }

   #define GB_GUARD_GB_POW_INT32_DEFINED
   #define GB_POW_INT32_DEFN                                                \
   "int32_t GB_pow_int32 (int32_t x, int32_t y)                         \n" \
   "{                                                                   \n" \
   "    return (GB_cast_to_int32_t (GB_pow ((double) x, (double) y))) ; \n" \
   "}"

    inline
    int64_t GB_pow_int64 (int64_t x, int64_t y)
    {
        return (GB_cast_to_int64_t (GB_pow ((double) x, (double) y))) ;
    }

   #define GB_GUARD_GB_POW_INT64_DEFINED
   #define GB_POW_INT64_DEFN                                                \
   "int64_t GB_pow_int64 (int64_t x, int64_t y)                         \n" \
   "{                                                                   \n" \
   "    return (GB_cast_to_int64_t (GB_pow ((double) x, (double) y))) ; \n" \
   "}"

    inline
    uint8_t GB_pow_uint8 (uint8_t x, uint8_t y)
    {
        return (GB_cast_to_uint8_t (GB_pow ((double) x, (double) y))) ;
    }

   #define GB_GUARD_GB_POW_UINT8_DEFINED
   #define GB_POW_UINT8_DEFN                                                \
   "int8_t GB_pow_uint8 (int8_t x, int8_t y)                            \n" \
   "{                                                                   \n" \
   "    return (GB_cast_to_uint8_t (GB_pow ((double) x, (double) y))) ; \n" \
   "}"

    inline
    uint16_t GB_pow_uint16 (uint16_t x, uint16_t y)
    {
        return (GB_cast_to_uint16_t (GB_pow ((double) x, (double) y))) ;
    }

   #define GB_GUARD_GB_POW_UINT16_DEFINED
   #define GB_POW_UINT16_DEFN                                               \
   "int16_t GB_pow_uint16 (int16_t x, int16_t y)                        \n" \
   "{                                                                   \n" \
   "    return (GB_cast_to_uint16_t (GB_pow ((double) x, (double) y))) ;\n" \
   "}"

    inline
    uint32_t GB_pow_uint32 (uint32_t x, uint32_t y)
    {
        return (GB_cast_to_uint32_t (GB_pow ((double) x, (double) y))) ;
    }

   #define GB_GUARD_GB_POW_UINT32_DEFINED
   #define GB_POW_UINT32_DEFN                                               \
   "int32_t GB_pow_uint32 (int32_t x, int32_t y)                        \n" \
   "{                                                                   \n" \
   "    return (GB_cast_to_uint32_t (GB_pow ((double) x, (double) y))) ;\n" \
   "}"

    inline
    uint64_t GB_pow_uint64 (uint64_t x, uint64_t y)
    {
        return (GB_cast_to_uint64_t (GB_pow ((double) x, (double) y))) ;
    }

   #define GB_GUARD_GB_POW_UINT64_DEFINED
   #define GB_POW_UINT64_DEFN                                               \
   "int64_t GB_pow_uint64 (int64_t x, int64_t y)                        \n" \
   "{                                                                   \n" \
   "    return (GB_cast_to_uint64_t (GB_pow ((double) x, (double) y))) ;\n" \
   "}"

//------------------------------------------------------------------------------
// frexp for float and double
//------------------------------------------------------------------------------

    inline
    float GB_frexpxf (float x)
    {
        // ignore the exponent and just return the mantissa
        int exp_ignored ;
        return (frexpf (x, &exp_ignored)) ;
    }

    #define GB_GUARD_GB_FREXPXF_DEFINED
    #define GB_FREXPXF_DEFN                                                 \
    "float GB_frexpxf (float x)                                         \n" \
    "{                                                                  \n" \
    "    int exp_ignored ;                                              \n" \
    "    return (frexpf (x, &exp_ignored)) ;                            \n" \
    "}"

    inline
    float GB_frexpef (float x)
    {
        // ignore the mantissa and just return the exponent
        int exp ;
        (void) frexpf (x, &exp) ;
        return ((float) exp) ;
    }

    #define GB_GUARD_GB_FREXPEF_DEFINED
    #define GB_FREXPEF_DEFN                                                 \
    "float GB_frexpef (float x)                                         \n" \
    "{                                                                  \n" \
    "    int exp ;                                                      \n" \
    "    (void) frexpf (x, &exp) ;                                      \n" \
    "    return ((float) exp) ;                                         \n" \
    "}"

    inline
    double GB_frexpx (double x)
    {
        // ignore the exponent and just return the mantissa
        int exp_ignored ;
        return (frexp (x, &exp_ignored)) ;
    }

    #define GB_GUARD_GB_FREXPX_DEFINED
    #define GB_FREXPX_DEFN                                                  \
    "double GB_frexpx (double x)                                        \n" \
    "{                                                                  \n" \
    "    int exp_ignored ;                                              \n" \
    "    return (frexp (x, &exp_ignored)) ;                             \n" \
    "}"

    inline
    double GB_frexpe (double x)
    {
        // ignore the mantissa and just return the exponent
        int exp ;
        (void) frexp (x, &exp) ;
        return ((double) exp) ;
    }

    #define GB_GUARD_GB_FREXPE_DEFINED
    #define GB_FREXPE_DEFN                                                  \
    "double GB_frexpe (double x)                                        \n" \
    "{                                                                  \n" \
    "    int exp ;                                                      \n" \
    "    (void) frexp (x, &exp) ;                                       \n" \
    "    return ((double) exp) ;                                        \n" \
    "}"

//------------------------------------------------------------------------------
// signum functions
//------------------------------------------------------------------------------

    inline
    float GB_signumf (float x)
    {
        if (isnan (x)) return (x) ;
        return ((float) ((x < 0) ? (-1) : ((x > 0) ? 1 : 0))) ;
    }

    #define GB_GUARD_GB_SIGNUMF_DEFINED
    #define GB_SIGNUMF_DEFN                                                 \
    "float GB_signumf (float x)                                         \n" \
    "{                                                                  \n" \
    "    if (isnan (x)) return (x) ;                                    \n" \
    "    return ((float) ((x < 0) ? (-1) : ((x > 0) ? 1 : 0))) ;        \n" \
    "}"

    inline
    double GB_signum (double x)
    {
        if (isnan (x)) return (x) ;
        return ((double) ((x < 0) ? (-1) : ((x > 0) ? 1 : 0))) ;
    }

    #define GB_GUARD_GB_SIGNUM_DEFINED
    #define GB_SIGNUM_DEFN                                                  \
    "double GB_signum (double x)                                        \n" \
    "{                                                                  \n" \
    "    if (isnan (x)) return (x) ;                                    \n" \
    "    return ((double) ((x < 0) ? (-1) : ((x > 0) ? 1 : 0))) ;       \n" \
    "}"

    inline
    GxB_FC32_t GB_csignumf (GxB_FC32_t x)
    {
        if (crealf (x) == 0 && cimagf (x) == 0) return (GxB_CMPLXF (0,0)) ;
        float y = cabsf (x) ;
        return (GB_CMPLX32 (crealf (x) / y, cimagf (x) / y)) ;
    }

    #define GB_GUARD_GB_CSIGNUMF_DEFINED
    #define GB_CSIGNUMF_DEFN                                                   \
    "GxB_FC32_t GB_csignumf (GxB_FC32_t x)                                  \n"\
    "{                                                                      \n"\
    "    if (crealf (x) == 0 && cimagf (x) == 0) return (GxB_CMPLXF (0,0)) ;\n"\
    "    float y = cabsf (x) ;                                              \n"\
    "    return (GB_CMPLX32 (crealf (x) / y, cimagf (x) / y)) ;             \n"\
    "}"

    inline
    GxB_FC64_t GB_csignum (GxB_FC64_t x)
    {
        if (creal (x) == 0 && cimag (x) == 0) return (GxB_CMPLX (0,0)) ;
        double y = cabs (x) ;
        return (GB_CMPLX64 (creal (x) / y, cimag (x) / y)) ;
    }

    #define GB_GUARD_GB_CSIGNUM_DEFINED
    #define GB_CSIGNUM_DEFN                                                    \
    "GxB_FC64_t GB_csignum (GxB_FC64_t x)                                   \n"\
    "{                                                                      \n"\
    "    if (creal (x) == 0 && cimag (x) == 0) return (GxB_CMPLX (0,0)) ;   \n"\
    "    double y = cabs (x) ;                                              \n"\
    "    return (GB_CMPLX64 (creal (x) / y, cimag (x) / y)) ;               \n"\
    "}"

//------------------------------------------------------------------------------
// complex functions
//------------------------------------------------------------------------------

// The ANSI C11 math.h header defines the ceil, floor, round, trunc,
// exp2, expm1, log10, log1pm, or log2 functions for float and double,
// but the corresponding functions do not appear in the ANSI C11 complex.h.
// These functions are used instead, for float complex and double complex.

//------------------------------------------------------------------------------
// z = ceil (x) for float complex
//------------------------------------------------------------------------------

    inline
    GxB_FC32_t GB_cceilf (GxB_FC32_t x)
    {
        return (GB_CMPLX32 (ceilf (crealf (x)), ceilf (cimagf (x)))) ;
    }

    #define GB_GUARD_GB_CCEILF_DEFINED
    #define GB_CCEILF_DEFN                                                  \
    "GxB_FC32_t GB_cceilf (GxB_FC32_t x)                                \n" \
    "{                                                                  \n" \
        "return (GB_CMPLX32 (ceilf (crealf (x)), ceilf (cimagf (x)))) ; \n" \
    "}"

//------------------------------------------------------------------------------
// z = ceil (x) for double complex
//------------------------------------------------------------------------------

    inline
    GxB_FC64_t GB_cceil (GxB_FC64_t x)
    {
        return (GB_CMPLX64 (ceil (creal (x)), ceil (cimag (x)))) ;
    }

    #define GB_GUARD_GB_CCEIL_DEFINED
    #define GB_CCEIL_DEFN                                                   \
    "GxB_FC64_t GB_cceil (GxB_FC64_t x)                                 \n" \
    "{                                                                  \n" \
    "    return (GB_CMPLX64 (ceil (creal (x)), ceil (cimag (x)))) ;     \n" \
    "}"

//------------------------------------------------------------------------------
// z = floor (x) for float complex
//------------------------------------------------------------------------------

    inline
    GxB_FC32_t GB_cfloorf (GxB_FC32_t x)
    {
        return (GB_CMPLX32 (floorf (crealf (x)), floorf (cimagf (x)))) ;
    }

    #define GB_GUARD_GB_CFLOORF_DEFINED
    #define GB_CFLOORF_DEFN                                                    \
    "GxB_FC32_t GB_cfloorf (GxB_FC32_t x)                                   \n"\
    "{                                                                      \n"\
    "    return (GB_CMPLX32 (floorf (crealf (x)), floorf (cimagf (x)))) ;   \n"\
    "}"

//------------------------------------------------------------------------------
// z = floor (x) for double complex
//------------------------------------------------------------------------------

    inline
    GxB_FC64_t GB_cfloor (GxB_FC64_t x)
    {
        return (GB_CMPLX64 (floor (creal (x)), floor (cimag (x)))) ;
    }

    #define GB_GUARD_GB_CFLOOR_DEFINED
    #define GB_CFLOOR_DEFN                                                     \
    "GxB_FC64_t GB_cfloor (GxB_FC64_t x)                                    \n"\
    "{                                                                      \n"\
    "    return (GB_CMPLX64 (floor (creal (x)), floor (cimag (x)))) ;       \n"\
    "}"

//------------------------------------------------------------------------------
// z = round (x) for float complex
//------------------------------------------------------------------------------

    inline
    GxB_FC32_t GB_croundf (GxB_FC32_t x)
    {
        return (GB_CMPLX32 (roundf (crealf (x)), roundf (cimagf (x)))) ;
    }

    #define GB_GUARD_GB_CROUNDF_DEFINED
    #define GB_CROUNDF_DEFN                                                    \
    "GxB_FC32_t GB_croundf (GxB_FC32_t x)                                   \n"\
    "{                                                                      \n"\
    "    return (GB_CMPLX32 (roundf (crealf (x)), roundf (cimagf (x)))) ;   \n"\
    "}"

//------------------------------------------------------------------------------
// z = round (x) for double complex
//------------------------------------------------------------------------------

    inline
    GxB_FC64_t GB_cround (GxB_FC64_t x)
    {
        return (GB_CMPLX64 (round (creal (x)), round (cimag (x)))) ;
    }

    #define GB_GUARD_GB_CROUND_DEFINED
    #define GB_CROUND_DEFN                                                  \
    "GxB_FC64_t GB_cround (GxB_FC64_t x)                                \n" \
    "{                                                                  \n" \
    "    return (GB_CMPLX64 (round (creal (x)), round (cimag (x)))) ;   \n" \
    "}"

//------------------------------------------------------------------------------
// z = trunc (x) for float complex
//------------------------------------------------------------------------------

    inline
    GxB_FC32_t GB_ctruncf (GxB_FC32_t x)
    {
        return (GB_CMPLX32 (truncf (crealf (x)), truncf (cimagf (x)))) ;
    }

    #define GB_GUARD_GB_CTRUNCF_DEFINED
    #define GB_CTRUNCF_DEFN                                                    \
    "GxB_FC32_t GB_ctruncf (GxB_FC32_t x)                                   \n"\
    "{                                                                      \n"\
    "    return (GB_CMPLX32 (truncf (crealf (x)), truncf (cimagf (x)))) ;   \n"\
    "}

//------------------------------------------------------------------------------
// z = trunc (x) for double complex
//------------------------------------------------------------------------------

    inline
    GxB_FC64_t GB_ctrunc (GxB_FC64_t x)
    {
        return (GB_CMPLX64 (trunc (creal (x)), trunc (cimag (x)))) ;
    }

    #define GB_GUARD_GB_CTRUNC_DEFINED
    #define GB_CTRUNC_DEFN                                                  \
    "GxB_FC64_t GB_ctrunc (GxB_FC64_t x)                                \n" \
    "{                                                                  \n" \
    "    return (GB_CMPLX64 (trunc (creal (x)), trunc (cimag (x)))) ;   \n" \
    "}"

//------------------------------------------------------------------------------
// z = exp2 (x) for float complex
//------------------------------------------------------------------------------

inline GxB_FC32_t GB_cexp2f (GxB_FC32_t x)
{
    if (fpclassify (cimagf (x)) == FP_ZERO)
    {
        // x is real, use exp2f
        return (GB_CMPLX32 (exp2f (crealf (x)), 0)) ;
    }
    return (GB_cpowf (GxB_CMPLXF (2,0), x)) ;     // z = 2^x
}

//------------------------------------------------------------------------------
// z = exp2 (x) for double complex
//------------------------------------------------------------------------------

inline GxB_FC64_t GB_cexp2 (GxB_FC64_t x)
{
    if (fpclassify (cimag (x)) == FP_ZERO)
    {
        // x is real, use exp2
        return (GB_CMPLX64 (exp2 (creal (x)), 0)) ;
    }
    return (GB_cpow (GxB_CMPLX (2,0), x)) ;      // z = 2^x
}

//------------------------------------------------------------------------------
// z = expm1 (x) for double complex
//------------------------------------------------------------------------------

inline GxB_FC64_t GB_cexpm1 (GxB_FC64_t x)
{
    // FUTURE: GB_cexpm1 is not accurate
    // z = cexp (x) - 1
    GxB_FC64_t z = cexp (x) ;
    return (GB_CMPLX64 (creal (z) - 1, cimag (z))) ;
}

//------------------------------------------------------------------------------
// z = expm1 (x) for float complex
//------------------------------------------------------------------------------

inline GxB_FC32_t GB_cexpm1f (GxB_FC32_t x)
{
    // typecast to double and use GB_cexpm1
    GxB_FC64_t z = GB_CMPLX64 ((double) crealf (x), (double) cimagf (x)) ;
    z = GB_cexpm1 (z) ;
    return (GB_CMPLX32 ((float) creal (z), (float) cimag (z))) ;
}

//------------------------------------------------------------------------------
// z = log1p (x) for double complex
//------------------------------------------------------------------------------

inline GxB_FC64_t GB_clog1p (GxB_FC64_t x)
{
    // FUTURE: GB_clog1p is not accurate
    // z = clog (1+x)
    return (clog (GB_CMPLX64 (creal (x) + 1, cimag (x)))) ;
}

//------------------------------------------------------------------------------
// z = log1p (x) for float complex
//------------------------------------------------------------------------------

inline GxB_FC32_t GB_clog1pf (GxB_FC32_t x)
{
    // typecast to double and use GB_clog1p
    GxB_FC64_t z = GB_CMPLX64 ((double) crealf (x), (double) cimagf (x)) ;
    z = GB_clog1p (z) ;
    return (GB_CMPLX32 ((float) creal (z), (float) cimag (z))) ;
}

//------------------------------------------------------------------------------
// z = log10 (x) for float complex
//------------------------------------------------------------------------------

inline GxB_FC32_t GB_clog10f (GxB_FC32_t x)
{
    // z = log (x) / log (10)
    return (GB_FC32_div (clogf (x), GxB_CMPLXF (2.3025851f, 0))) ;
}

//------------------------------------------------------------------------------
// z = log10 (x) for double complex
//------------------------------------------------------------------------------

inline GxB_FC64_t GB_clog10 (GxB_FC64_t x)
{
    // z = log (x) / log (10)
    return (GB_FC64_div (clog (x), GxB_CMPLX (2.302585092994045901, 0))) ;
}

//------------------------------------------------------------------------------
// z = log2 (x) for float complex
//------------------------------------------------------------------------------

inline GxB_FC32_t GB_clog2f (GxB_FC32_t x)
{
    // z = log (x) / log (2)
    return (GB_FC32_div (clogf (x), GxB_CMPLXF (0.69314718f, 0))) ;
}

//------------------------------------------------------------------------------
// z = log2 (x) for double complex
//------------------------------------------------------------------------------

inline GxB_FC64_t GB_clog2 (GxB_FC64_t x)
{
    // z = log (x) / log (2)
    return (GB_FC64_div (clog (x), GxB_CMPLX (0.693147180559945286, 0))) ;
}

//------------------------------------------------------------------------------
// z = isinf (x) for float complex
//------------------------------------------------------------------------------

inline bool GB_cisinff (GxB_FC32_t x)
{
    return (isinf (crealf (x)) || isinf (cimagf (x))) ;
}

//------------------------------------------------------------------------------
// z = isinf (x) for double complex
//------------------------------------------------------------------------------

inline bool GB_cisinf (GxB_FC64_t x)
{
    return (isinf (creal (x)) || isinf (cimag (x))) ;
}

//------------------------------------------------------------------------------
// z = isnan (x) for float complex
//------------------------------------------------------------------------------

inline bool GB_cisnanf (GxB_FC32_t x)
{
    return (isnan (crealf (x)) || isnan (cimagf (x))) ;
}

//------------------------------------------------------------------------------
// z = isnan (x) for double complex
//------------------------------------------------------------------------------

inline bool GB_cisnan (GxB_FC64_t x)
{
    return (isnan (creal (x)) || isnan (cimag (x))) ;
}

//------------------------------------------------------------------------------
// z = isfinite (x) for float complex
//------------------------------------------------------------------------------

inline bool GB_cisfinitef (GxB_FC32_t x)
{
    return (isfinite (crealf (x)) && isfinite (cimagf (x))) ;
}

//------------------------------------------------------------------------------
// z = isfinite (x) for double complex
//------------------------------------------------------------------------------

inline bool GB_cisfinite (GxB_FC64_t x)
{
    return (isfinite (creal (x)) && isfinite (cimag (x))) ;
}

#endif

