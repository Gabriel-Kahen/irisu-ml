#include "image_identity.hpp"
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1], std::ios::binary);
  std::vector<unsigned char> image((std::istreambuf_iterator<char>(input)), {});
  try { irisu::cuda_exact::verify_image(image.data(), image.size()); }
  catch (...) { return 3; }
  unsigned rejected = 0;
  auto reject = [&](const unsigned char* data, std::size_t size) {
    try { irisu::cuda_exact::verify_image(data, size); }
    catch (const std::invalid_argument&) { ++rejected; }
  };
  reject(nullptr, image.size());
  reject(image.data(), image.size() - 1);
  reject(image.data(), image.size() + 1);
  image[image.size() / 2] ^= 1;
  reject(image.data(), image.size());
  if (rejected != 4) return 4;
  std::puts("{\"canonical_image\":true,\"rejected_invalid_images\":4}");
}
