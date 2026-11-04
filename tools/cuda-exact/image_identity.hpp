#pragma once

// Host-side image admission for the optional translated backend. The expected
// bytes derive from the pinned ELF at build time; no production CPU pin changes.
#include "guest_image_identity.inc"
#include <openssl/sha.h>
#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace irisu::cuda_exact {
inline void verify_image(const unsigned char* image, std::size_t size) {
  if (!image || size != cuda_guest::image_bytes)
    throw std::invalid_argument("CUDA guest image size does not match compiled identity");
  unsigned char actual[SHA256_DIGEST_LENGTH];
  if (!SHA256(image, size, actual) ||
      !std::equal(actual, actual + SHA256_DIGEST_LENGTH, cuda_guest::image_sha256))
    throw std::invalid_argument("CUDA guest image SHA-256 does not match compiled identity");
}
}  // namespace irisu::cuda_exact
