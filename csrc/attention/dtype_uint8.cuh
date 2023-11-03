#pragma once

#include <stdint.h>
#include "attention_generic.cuh"
#include "dtype_float32.cuh"

namespace vllm {
// define uint8  vector types for quantization of kv cache

template<>
struct Vec<uint8_t, 1> {
    using Type = uint8_t;
};

template<>
struct Vec<uint8_t, 2> {
    using Type = uint16_t;
};

template<>
struct Vec<uint8_t, 4> {
    using Type = uint32_t;
};

template<>
struct Vec<uint8_t, 8> {
    using Type = uint64_t;
};

template<>
struct FloatVecTemp<uint8_t> {
    using Type = float;
};

template<>
struct FloatVecTemp<uint16_t> {
    using Type = float2;
};

template<>
struct FloatVecTemp<uint32_t> {
    using Type = Float4_;
};

template<>
struct FloatVecTemp<uint64_t> {
    using Type = Float8_;
};
}