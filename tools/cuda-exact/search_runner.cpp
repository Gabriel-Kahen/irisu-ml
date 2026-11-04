#include "search_io.hpp"
#include <cstdlib>

int main(int argc,char** argv) {
  try {
    if(argc<9) throw std::runtime_error("usage: search-runner image replay specs prefix horizon arena-mib keep target [--host-prefix] [--isolated-warps] [--final-only] [--global-guest] [--block-threads N] [--stack-kib N] [--skip-compaction-check] [--word-stride 1|32]");
    const auto image=irisu::cuda_search::read_bytes(argv[1]);
    auto in=irisu::cuda_search::load_inputs(argv[2],argv[3],argv[4],argv[5]);
    const auto mib=std::stoul(argv[6]),keep=std::stoul(argv[7]);
    if(mib<1||mib>512||keep<1||keep>in.candidates.size()) throw std::runtime_error("invalid arena/keep");
    in.options.arena_bytes=mib*1024*1024;in.options.keep=keep;in.options.target_score=std::stoll(argv[8]);
    for(int i=9;i<argc;++i){const std::string option=argv[i];
      if(option=="--skip-compaction-check")in.options.validate_compaction=false;
      else if(option=="--host-prefix")in.options.host_prefix=true;
      else if(option=="--isolated-warps")in.options.isolated_warps=true;
      else if(option=="--final-only")in.options.trace_every_tick=false;
      else if(option=="--global-guest")in.options.cache_guest=false;
      else if(option=="--block-threads"&&i+1<argc){const auto n=std::stoul(argv[++i]);if(!n||n>256)throw std::runtime_error("invalid block size");in.options.block_threads=n;}
      else if(option=="--word-stride"&&i+1<argc){const auto n=std::stoul(argv[++i]);if(n!=1&&n!=32)throw std::runtime_error("invalid arena word stride");in.options.arena_word_stride=n;}
      else if(option=="--stack-kib"&&i+1<argc){const auto n=std::stoul(argv[++i]);if(n<16||n>128)throw std::runtime_error("invalid stack size");in.options.stack_bytes=n*1024;}
      else throw std::runtime_error("unknown option");}
    std::vector<irisu::cuda_search::Result> results(in.candidates.size());
    std::vector<std::uint32_t> ranks(in.candidates.size());irisu::cuda_search::Timings times;
    irisu::cuda_search::run(in.options,image.data(),image.size(),in.replay.data(),in.replay.size(),
      in.candidates.data(),in.candidates.size(),results.data(),ranks.data(),&times);
    std::uint64_t ticks{},failed{};for(const auto& r:results){ticks+=r.simulated_ticks;failed+=bool(r.failed);}
    std::fprintf(stderr,"{\"identity\":\"experimental-cuda-game-search-v2\",\"promotion_eligible\":false,\"candidates\":%zu,\"prefix_ticks\":%u,\"horizon\":%u,\"simulated_ticks\":%llu,\"failed_candidates\":%llu,\"trace_every_tick\":%s,\"cache_guest\":%s,\"block_threads\":%u,\"stack_kib\":%u,\"word_stride\":%u,\"host_prefix\":%s,\"host_prefix_ms\":%.9f,\"prefix_ms\":%.9f,\"branch_ms\":%.9f,\"rollout_ms\":%.9f,\"rank_compact_ms\":%.9f,\"validation_ms\":%.9f,\"compaction_check_enabled\":%s,\"total_ms\":%.9f,\"allocated_bytes\":%llu,\"peak_guest_bytes\":%llu,\"compaction_errors\":%llu,\"winner\":%u}\n",
      results.size(),in.options.prefix,in.options.horizon,(unsigned long long)ticks,(unsigned long long)failed,
      in.options.trace_every_tick?"true":"false",in.options.cache_guest?"true":"false",
      in.options.block_threads,in.options.stack_bytes/1024,in.options.arena_word_stride,in.options.host_prefix?"true":"false",times.host_prefix_ms,
      times.prefix_ms,times.branch_ms,times.rollout_ms,times.rank_compact_ms,times.validation_ms,
      in.options.validate_compaction?"true":"false",times.total_ms,
      (unsigned long long)times.allocation_bytes,(unsigned long long)times.peak_guest_bytes,
      (unsigned long long)times.compaction_errors,ranks[0]);
    if(failed||times.compaction_errors)return 2;
    irisu::cuda_search::emit_results(results,ranks);return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"search-runner: %s\n",e.what());return 1;}
}
