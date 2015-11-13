#ifndef TRAITS_H
#define TRAITS_H

#include <limits>
#include <vector>
#include <algorithm>

namespace pinot_tracker {
template <typename T>
struct DepthTraits {};

template <>
struct DepthTraits<uint16_t> {
  static inline bool valid(uint16_t depth) { return depth != 0; }
  static inline float toMeters(uint16_t depth) {
    return depth * 0.001f;
  }  // originally mm
  static inline uint16_t fromMeters(float depth) {
    return (depth * 1000.0f) + 0.5f;
  }
  static inline void initializeBuffer(std::vector<uint8_t>& buffer) {
  }  // Do nothing - already zero-filled
};

template <>
struct DepthTraits<float> {
  static inline bool valid(float depth) { return std::isfinite(depth); }
  static inline float toMeters(float depth) { return depth; }
  static inline float fromMeters(float depth) { return depth; }

  static inline void initializeBuffer(std::vector<uint8_t>& buffer) {
    float* start = reinterpret_cast<float*>(&buffer[0]);
    float* end = reinterpret_cast<float*>(&buffer[0] + buffer.size());
    std::fill(start, end, std::numeric_limits<float>::quiet_NaN());
  }
};
}

#endif  // TRAITS_H
