#include "search_io.hpp"
#include <cstdlib>

int main(int argc,char** argv) {
  try {
    if(argc<9||argc>12) throw std::runtime_error("usage: search-runner guest-image.bin replay.rpy specs.bin prefix horizon arena-mib keep target-score [--skip-compaction-check] [--host-prefix] [--isolated-warps]");
    const auto image=irisu::cuda_search::read_bytes(argv[1]);
    auto in=irisu::cuda_search::load_inputs(argv[2],argv[3],argv[4],argv[5]);
    const auto mib=std::stoul(argv[6]),keep=std::stoul(argv[7]);
    if(mib<1||mib>512||keep<1||keep>in.candidates.size()) throw std::runtime_error("invalid arena/keep");
    in.options.arena_bytes=mib*1024*1024;in.options.keep=keep;in.options.target_score=std::stoll(argv[8]);
    for(int i=9;i<argc;++i){const std::string option=argv[i];
      if(option=="--skip-compaction-check")in.options.validate_compaction=false;
      else if(option=="--host-prefix")in.options.host_prefix=true;
      else if(option=="--isolated-warps")in.options.isolated_warps=true;
      else throw std::runtime_error("unknown option");}
    std::vector<irisu::cuda_search::Result> results(in.candidates.size());
    std::vector<std::uint32_t> ranks(in.candidates.size());irisu::cuda_search::Timings times;
    irisu::cuda_search::run(in.options,image.data(),image.size(),in.replay.data(),in.replay.size(),
      in.candidates.data(),in.candidates.size(),results.data(),ranks.data(),&times);
    std::uint64_t ticks{},failed{};for(const auto& r:results){ticks+=r.simulated_ticks;failed+=bool(r.failed);}
    std::fprintf(stderr,"{\"identity\":\"experimental-cuda-game-search-v1\",\"promotion_eligible\":false,\"candidates\":%zu,\"prefix_ticks\":%u,\"horizon\":%u,\"simulated_ticks\":%llu,\"failed_candidates\":%llu,\"host_prefix\":%s,\"host_prefix_ms\":%.9f,\"prefix_ms\":%.9f,\"branch_ms\":%.9f,\"rollout_ms\":%.9f,\"rank_compact_ms\":%.9f,\"validation_ms\":%.9f,\"compaction_check_enabled\":%s,\"total_ms\":%.9f,\"allocated_bytes\":%llu,\"peak_guest_bytes\":%llu,\"compaction_errors\":%llu,\"winner\":%u}\n",
      results.size(),in.options.prefix,in.options.horizon,(unsigned long long)ticks,(unsigned long long)failed,
      in.options.host_prefix?"true":"false",times.host_prefix_ms,
      times.prefix_ms,times.branch_ms,times.rollout_ms,times.rank_compact_ms,times.validation_ms,
      in.options.validate_compaction?"true":"false",times.total_ms,
      (unsigned long long)times.allocation_bytes,(unsigned long long)times.peak_guest_bytes,
      (unsigned long long)times.compaction_errors,ranks[0]);
    if(failed||times.compaction_errors)return 2;
    irisu::cuda_search::emit_results(results,ranks);return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"search-runner: %s\n",e.what());return 1;}
}
