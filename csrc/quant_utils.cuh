#pragma once

#include <assert.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>
#include "attention/attention_dtypes.h"
#include "attention/dtype_float32.cuh"
#include "cutlass/numeric_conversion.h"
using namespace vllm;

// this function is for function matching, delete it after writing customized dispatch functions
inline __device__ uint8_t quant(double a)
{
    using Base = typename cutlass::float_e5m2_t::Base;
    uint8_t uint8 = Base::convert_float_to_fp8(static_cast<float>(a));
    return uint8;
}

inline __device__ uint8_t quant(float a)
{
    using Base = typename cutlass::float_e5m2_t::Base;
    uint8_t uint8 = Base::convert_float_to_fp8(a);
    return uint8;
}

inline __device__ uint16_t quant(float2 a)
{
    union {
        uint8_t uint8[2];
        uint16_t  uint16;
    };

    using Base = typename cutlass::float_e5m2_t::Base;
    uint8[0] = Base::convert_float_to_fp8(a.x);
    uint8[1] = Base::convert_float_to_fp8(a.y);
    return uint16;
}

inline __device__ uint32_t quant(float4 a)
{
    union {
        uint8_t  uint8[4];
        uint32_t uint32;
    };

    using Base = typename cutlass::float_e5m2_t::Base;
    uint8[0] = Base::convert_float_to_fp8(a.x);
    uint8[1] = Base::convert_float_to_fp8(a.y);
    uint8[2] = Base::convert_float_to_fp8(a.z);
    uint8[3] = Base::convert_float_to_fp8(a.w);
    return uint32;
}

// float16 to uint8
inline __device__ uint8_t quant(uint16_t a)
{
    using Base = typename cutlass::float_e5m2_t::Base;
    float  b = half_to_float(a);
    uint8_t uint8 = Base::convert_float_to_fp8(b);
    return uint8;
}

// float16x2 to uint8x2
inline __device__ uint16_t quant(uint32_t a)
{
    union {
        uint8_t uint8[2];
        uint16_t  uint16;
    };
    float2 b = half2_to_float2(a);
    using Base = typename cutlass::float_e5m2_t::Base;
    uint8[0] = Base::convert_float_to_fp8(b.x);
    uint8[1] = Base::convert_float_to_fp8(b.y);
    return uint16;
}

// float16x4 to int8x4
inline __device__ uint32_t quant(uint2 a)
{
    union {
        uint16_t uint16[2];
        uint32_t uint32;
    };

    uint16[0] = quant(a.x);
    uint16[1] = quant(a.y);
    return uint32;
}

// float16x8 to int8x8
inline __device__ uint64_t quant(uint4 a)
{
    union {
        uint16_t uint16[4];
        uint64_t uint64;
    };

    uint16[0] = quant(a.x);
    uint16[1] = quant(a.y);
    uint16[2] = quant(a.z);
    uint16[3] = quant(a.w);
    return uint64;
}

// uint8 to float32, then `vec_conversion` to target format
inline __device__ float dequant(uint8_t a)
{
    float b = static_cast<float>(cutlass::float_e5m2_t::bitcast(a));
    return b;
}

// int8x2 to float32x2
inline __device__ float2 dequant(uint16_t a)
{
    union {
        uint8_t  uint8[2];
        uint16_t uint16;
    };
    uint16 = a;

    float2 b;
    b.x = dequant(uint8[0]);
    b.y = dequant(uint8[1]);
    return b;
}

// int8x4 to float32x4
inline __device__ Float4_ dequant(uint32_t a)
{
    union {
        uint8_t  uint8[4];
        uint32_t uint32;
    };
    uint32 = a;

    Float4_ b;
    b.x.x = dequant(uint8[0]);
    b.x.y = dequant(uint8[1]);
    b.y.x = dequant(uint8[2]);
    b.y.y = dequant(uint8[3]);
    return b;
}

inline __device__ Float8_ dequant(uint64_t a)
{
    union {
        uint16_t uint16[4];
        uint64_t uint64;
    };
    uint64 = a;

    Float8_ b;
    b.x = dequant(uint16[0]);
    b.y = dequant(uint16[1]);
    b.z = dequant(uint16[2]);
    b.w = dequant(uint16[3]);
    return b;
}

template<typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(const Tin& x)
{
    return x;
}

template<>
__inline__ __device__ uint32_t vec_conversion<uint32_t, float2>(const float2& a)
{
    union {
        half2    float162;
        uint32_t uint32;
    };

    float162 = __float22half2_rn(a);
    return uint32;
}

template<>
__inline__ __device__ uint2 vec_conversion<uint2, Float4_>(const Float4_& a)
{
    uint2  b;
    float2 val;
    val.x = a.x.x;
    val.y = a.x.y;
    b.x   = vec_conversion<uint32_t, float2>(val);

    val.x = a.y.x;
    val.y = a.y.y;
    b.y   = vec_conversion<uint32_t, float2>(val);

    return b;
}

template<>
__inline__ __device__ float4 vec_conversion<float4, Float4_>(const Float4_& a)
{
    float4 b;
    b.x = a.x.x;
    b.y = a.x.y;
    b.z = a.y.x;
    b.w = a.y.y;
    return b;
}

template<>
__inline__ __device__ uint4 vec_conversion<uint4, Float8_>(const Float8_& a)
{
    uint4 b;
    b.x = vec_conversion<uint32_t, float2>(a.x);
    b.y = vec_conversion<uint32_t, float2>(a.y);
    b.z = vec_conversion<uint32_t, float2>(a.z);
    b.w = vec_conversion<uint32_t, float2>(a.w);
    return b;
}


template<>
__inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162, float2>(const float2 &a) {
    // return __float22bfloat162_rn(a);
    __nv_bfloat162 b;
    return b;
}

template<>
__inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, Float4_>(const Float4_ &a) {
    bf16_4_t b;
    // b.x = vec_conversion<__nv_bfloat162, float2>(a.x);
    // b.y = vec_conversion<__nv_bfloat162, float2>(a.y);
    return b;
}

template<>
__inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, Float8_>(const Float8_ &a) {
    bf16_8_t b;
    // b.x = vec_conversion<__nv_bfloat162, float2>(a.x);
    // b.y = vec_conversion<__nv_bfloat162, float2>(a.y);
    // b.z = vec_conversion<__nv_bfloat162, float2>(a.z);
    // b.w = vec_conversion<__nv_bfloat162, float2>(a.w);
    return b;
}