#ifndef IUCORE_STRUCTS_FLOAT16_H
#define IUCORE_STRUCTS_FLOAT16_H

#include "half.hpp"
#include "/usr/local/cuda/samples/common/inc/helper_math.h"

//#include <cutil_math.h>
#include <vector_types.h>

struct __host__ __device__ float16_2
{
    half_float::half x, y;
};

struct __host__ __device__ float16_4
{
    half_float::half x, y, z, w;
};

static __inline__ __host__ __device__ float16_2 make_float16_2(half_float::half x,
                                                               half_float::half y)
{
  float16_2 t;
  t.x = x; t.y = y;
  return t;
}

static __inline__ __host__ __device__ float16_2 make_float16_2(float x,
                                                               float y)
{
  float16_2 t;
  t.x = half_float::half(x);
  t.y = half_float::half(y);
  return t;
}

static __inline__ __host__ __device__ float16_4 make_float16_4(half_float::half x,
                                                               half_float::half y,
                                                               half_float::half z,
                                                               half_float::half w)
{
  float16_4 t; t.x = x; t.y = y; t.z = z; t.w = w;
  return t;
}

static __inline__ __host__ __device__ float16_4 make_float16_4(float x,
                                                               float y,
                                                               float z,
                                                               float w)
{
  float16_4 t;

  t.x = half_float::half(x);
  t.y = half_float::half(y);
  t.z = half_float::half(z);
  t.w = half_float::half(w);

  return t;
}


static __inline__ __host__ __device__ float16_4 make_float16_4(float val)
{
  float16_4 t;

  t.x = half_float::half(val);
  t.y = half_float::half(val);
  t.z = half_float::half(val);
  t.w = half_float::half(val);

  return t;
}


////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float dot(float16_2 a, float16_2 b)
{
    return (float)(a.x * b.x + a.y * b.y);
}

inline __host__ __device__ float dot(float16_4 a, float16_4 b)
{
    return (float)(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float length(float16_2 v)
{
    return sqrtf((float)dot(v, v));
}
inline __host__ __device__ float length(float16_4 v)
{
    return sqrtf((float)dot(v, v));
}


////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float16_2 operator*(float16_2 a, float16_2 b)
{
    return make_float16_2(half_float::half((float)a.x * (float)b.x),
                          half_float::half((float)a.y * (float)b.y));
}
inline __host__ __device__ void operator*=(float16_2 &a, float16_2 b)
{
    a.x *= b.x; a.y *= b.y;
}
inline __host__ __device__ float16_2 operator*(float16_2 a, float b)
{
    return make_float16_2((float)a.x * b, (float)a.y * b);
}
inline __host__ __device__ float16_2 operator*(float b, float16_2 a)
{
    return make_float16_2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float16_2 &a, float b)
{
    a.x = half_float::half((float)a.x * b);
    a.y = half_float::half((float)a.y * b);
}

inline __host__ __device__ float16_4 operator*(float16_4 a, float16_4 b)
{
    return make_float16_4(half_float::half((float)a.x * (float)b.x),
                          half_float::half((float)a.y * (float)b.y),
                          half_float::half((float)a.z * (float)b.z),
                          half_float::half((float)a.w * (float)b.w));
}
inline __host__ __device__ void operator*=(float16_4 &a, float16_4 b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}
inline __host__ __device__ float16_4 operator*(float16_4 a, float b)
{
    return make_float16_4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ float16_4 operator*(float b, float16_4 a)
{
    return make_float16_4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float16_4 &a, float b)
{
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float16_2 normalize(float16_2 v)
{
    float invLen = rsqrtf((float)dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float16_4 normalize(float16_4 v)
{
    float invLen = rsqrtf((float)dot(v, v));
    return v * invLen;
}


////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float16_2 fmaxf(float16_2 a, float16_2 b)
{
    return make_float16_2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}
inline __host__ __device__ float16_4 fmaxf(float16_4 a, float16_4 b)
{
    return make_float16_4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}


////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float16_2 operator/(float16_2 a, float16_2 b)
{
    return make_float16_2(half_float::half((float)a.x / (float)b.x),
                          half_float::half((float)a.y / (float)b.y));
}
inline __host__ __device__ void operator/=(float16_2 &a, float16_2 b)
{
    a.x /= b.x; a.y /= b.y;
}
inline __host__ __device__ float16_2 operator/(float16_2 a, float b)
{
    return make_float16_2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float16_2 &a, float b)
{
    a.x /= b; a.y /= b;
}
inline __host__ __device__ float16_2 operator/(float b, float16_2 a)
{
    return make_float16_2(b / a.x, b / a.y);
}

inline __host__ __device__ float16_4 operator/(float16_4 a, float16_4 b)
{
    return make_float16_4(half_float::half((float)a.x / (float)b.x),
                          half_float::half((float)a.y / (float)b.y),
                          half_float::half((float)a.z / (float)b.z),
                          half_float::half((float)a.w / (float)b.w));
}
inline __host__ __device__ void operator/=(float16_4 &a, float16_4 b)
{
    a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
}
inline __host__ __device__ float16_4 operator/(float16_4 a, float b)
{
    return make_float16_4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
inline __host__ __device__ void operator/=(float16_4 &a, float b)
{
    a.x /= b; a.y /= b; a.z /= b; a.w /= b;
}
inline __host__ __device__ float16_4 operator/(float b, float16_4 a){
    return make_float16_4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float16_2 floorf(float16_2 v)
{
    return make_float16_2(half_float::half(floorf((float)v.x)),
                          half_float::half(floorf((float)v.y)));
}
inline __host__ __device__ float16_4 floorf(float16_4 v)
{
    return make_float16_4(half_float::half(floorf((float)v.x)),
                          half_float::half(floorf((float)v.y)),
                          half_float::half(floorf((float)v.z)),
                          half_float::half(floorf((float)v.w)));
}



#endif

