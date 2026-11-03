// Compile/link against the authoritative ELF32 exact irisu_core; see the
// runnable command in docs/exact-runtime-recovery.md. This oracle never calls
// the generated GPU game implementation.
#include "irisu/simulator.hpp"
#include "game_digest.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::uint32_t word(const std::vector<unsigned char>& data, std::size_t at) {
  return std::uint32_t(data[at]) | (std::uint32_t(data[at+1]) << 8) |
         (std::uint32_t(data[at+2]) << 16) | (std::uint32_t(data[at+3]) << 24);
}

void emit(const irisu::game_digest::Frame& frame) {
  const std::uint64_t words[] = {
    frame.tick, frame.body_hash, frame.observation_hash, frame.event_hash,
    frame.event_count, frame.transition_hash, frame.body_count, frame.score,
    frame.gauge, frame.level, frame.highest_chain, frame.canonical_score,
  };
  unsigned char encoded[96];
  for (unsigned i=0; i<12; ++i)
    for (unsigned j=0; j<8; ++j) encoded[i*8+j] = static_cast<unsigned char>(words[i] >> (8*j));
  if (std::fwrite(encoded, sizeof(encoded), 1, stdout) != 1)
    throw std::runtime_error("cannot write game digest");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3)
      throw std::invalid_argument("usage: game-oracle padded-replay.rpy [maximum-steps]");
    std::ifstream stream(argv[1], std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open replay");
    const std::vector<unsigned char> data((std::istreambuf_iterator<char>(stream)), {});
    if (data.size() <= 52 || (data.size()-52) % 4 || word(data, 16) != 0)
      throw std::invalid_argument("expected nonempty padded v2.03 normal-mode replay");
    const std::size_t available = (data.size()-52)/4;
    const std::size_t steps = argc == 3 ? static_cast<std::size_t>(
        std::min<unsigned long long>(available, std::stoull(argv[2]))) : available;
    const std::uint16_t cw = 0x027f;
    __asm__ __volatile__("fldcw %0" : : "m"(cw));
    const auto begin = std::chrono::steady_clock::now();
    irisu::Simulator simulator;
    auto observation = simulator.reset(word(data, 0));
    irisu::StepResult result{};
    auto* dump=std::getenv("IRISU_GAME_BODY_DUMP_FILE");
    auto* dump_file=dump?std::fopen(dump,"w"):nullptr;
    if(dump&&!dump_file)throw std::runtime_error("cannot open body dump");
    if(dump_file)for(const auto& b:simulator.bodies())irisu::game_digest::dump_body(dump_file,0,b);
    emit(irisu::game_digest::make_frame(simulator.bodies(), observation,
                                      irisu::game_digest::basis, 0, result));
    std::uint64_t events = 0;
    for (std::size_t i=0; i<steps; ++i) {
      const auto input = word(data, 52+4*i);
      result = simulator.step({static_cast<irisu::ActionKind>(input & 3U),
                               static_cast<double>((input >> 2U) & 0x3ffU),
                               static_cast<double>((input >> 12U) & 0x1ffU), 1, i<2});
      observation = simulator.observation();
      if(dump_file)for(const auto& b:simulator.bodies())irisu::game_digest::dump_body(dump_file,observation.tick,b);
      emit(irisu::game_digest::make_frame(simulator.bodies(), observation,
             irisu::game_digest::native_events(result.events), result.events.size(), result));
      events += result.events.size();
    }
    if (steps == available &&
        (!result.terminated || result.truncated || !result.diagnostics.terminal_metadata_recorded ||
         result.diagnostics.latest_final_score != word(data, 8) ||
         result.diagnostics.latest_final_level != word(data, 4) ||
         result.diagnostics.latest_final_highest_chain != word(data, 12)))
      throw std::runtime_error("full replay canonical terminal disagrees with header");
    std::fprintf(stderr,
      "{\"backend\":\"native-msvc-cpu-oracle\",\"format\":\"game-digest-v1\",\"rows\":%llu,\"steps\":%llu,\"events\":%llu,\"tick\":%llu,\"live_score\":%lld,\"canonical_score\":%lld,\"level\":%u,\"highest_chain\":%u,\"config_hash\":%llu,\"elapsed_seconds\":%.9f}\n",
      static_cast<unsigned long long>(steps+1), static_cast<unsigned long long>(steps),
      static_cast<unsigned long long>(events), static_cast<unsigned long long>(observation.tick),
      static_cast<long long>(observation.score), static_cast<long long>(result.diagnostics.latest_final_score),
      observation.level, observation.highest_chain, static_cast<unsigned long long>(simulator.config_hash()),
      std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());
    if(dump_file)std::fclose(dump_file);
    return std::fflush(stdout) == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
