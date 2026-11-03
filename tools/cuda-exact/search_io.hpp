#pragma once
#include "../../clone/cuda/search.hpp"
#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace irisu::cuda_search {
inline std::vector<unsigned char> read_bytes(const char* path) {
  std::ifstream stream(path,std::ios::binary);
  if(!stream) throw std::runtime_error(std::string("cannot open ")+path);
  return {(std::istreambuf_iterator<char>(stream)),{}};
}
inline std::uint32_t read_word(const std::vector<unsigned char>& data,std::size_t at) {
  if(at+4>data.size()) throw std::runtime_error("short input");
  return data[at]|(std::uint32_t(data[at+1])<<8)|(std::uint32_t(data[at+2])<<16)|(std::uint32_t(data[at+3])<<24);
}
struct Inputs {
  Options options{};
  std::vector<std::uint32_t> replay;
  std::vector<Candidate> candidates;
};
inline Inputs load_inputs(const char* replay_path,const char* specs_path,
                          const char* prefix,const char* horizon) {
  static_assert(std::endian::native==std::endian::little);
  const auto bytes=read_bytes(replay_path),specs=read_bytes(specs_path);
  if(bytes.size()<=52||(bytes.size()-52)%4||read_word(bytes,16)!=0)
    throw std::runtime_error("expected padded normal-mode replay");
  if(specs.empty()||specs.size()%sizeof(Candidate)||specs.size()/sizeof(Candidate)>4096)
    throw std::runtime_error("expected 1..4096 fixed 40-byte candidates");
  Inputs in;in.options.seed=read_word(bytes,0);
  const auto p=std::stoull(prefix),h=std::stoull(horizon);
  if(p>(bytes.size()-52)/4||!h||h>100000||p+h>UINT32_MAX)
    throw std::runtime_error("invalid prefix/horizon");
  in.options.prefix=p;in.options.horizon=h;
  for(std::size_t i=52;i<bytes.size();i+=4) in.replay.push_back(read_word(bytes,i));
  in.candidates.resize(specs.size()/sizeof(Candidate));
  std::memcpy(in.candidates.data(),specs.data(),specs.size());
  for(const auto& c:in.candidates) if(!valid(c,in.options.horizon))
    throw std::runtime_error("invalid sparse candidate");
  return in;
}
inline void emit_results(const std::vector<Result>& results,
                          const std::vector<std::uint32_t>& ranks) {
  if(std::fwrite(results.data(),sizeof(Result),results.size(),stdout)!=results.size()||
     std::fwrite(ranks.data(),sizeof(std::uint32_t),ranks.size(),stdout)!=ranks.size()||
     std::fflush(stdout)) throw std::runtime_error("cannot write search results");
}
} // namespace irisu::cuda_search
