#include "search_io.hpp"
#include "../../clone/cuda/search_session.hpp"
#include <openssl/sha.h>
#include <sstream>

namespace search=irisu::cuda_search;
std::string digest(const std::vector<unsigned char>& bytes) {
  unsigned char hash[32];if(!SHA256(bytes.data(),bytes.size(),hash))throw std::runtime_error("cannot hash session input");
  const char* hex="0123456789abcdef";std::string result;
  for(auto byte:hash){result+=hex[byte>>4];result+=hex[byte&15];}return result;
}
void word(std::FILE* stream,std::uint32_t value) {
  const unsigned char bytes[]={static_cast<unsigned char>(value),static_cast<unsigned char>(value>>8),
    static_cast<unsigned char>(value>>16),static_cast<unsigned char>(value>>24)};
  if(std::fwrite(bytes,4,1,stream)!=1)throw std::runtime_error("cannot write session result");
}
int main(int argc,char** argv) {
  try {
    if(argc<12)throw std::runtime_error("usage: search-session image replay specs prefix horizon arena-mib beam depths target winner.rpy lineage.json [--host-prefix] [--isolated-warps] [--final-only] [--global-guest] [--block-threads N] [--stack-kib N] [--skip-compaction-check] [--no-protect-incumbent] [--greedy-beam]");
    const auto image=search::read_bytes(argv[1]);
    const auto source_replay=search::read_bytes(argv[2]),source_specs=search::read_bytes(argv[3]);
    auto in=search::load_inputs(argv[2],argv[3],argv[4],argv[5]);
    if(source_replay!=search::read_bytes(argv[2])||source_specs!=search::read_bytes(argv[3]))throw std::runtime_error("session input changed during load");
    search::SessionOptions options;options.game=in.options;
    const auto arena=std::stoul(argv[6]),beam=std::stoul(argv[7]),depths=std::stoul(argv[8]);
    if(!arena||arena>512||!beam||beam>64||!depths||depths>16)throw std::runtime_error("invalid bounded session dimensions");
    options.game.arena_bytes=arena*1024*1024;options.game.keep=beam;options.beam=beam;options.depths=depths;
    options.game.target_score=std::stoll(argv[9]);
    for(int i=12;i<argc;++i){const std::string flag=argv[i];
      if(flag=="--host-prefix")options.game.host_prefix=true;
      else if(flag=="--isolated-warps")options.game.isolated_warps=true;
      else if(flag=="--final-only")options.game.trace_every_tick=false;
      else if(flag=="--global-guest")options.game.cache_guest=false;
      else if(flag=="--block-threads"&&i+1<argc){const auto n=std::stoul(argv[++i]);if(!n||n>256)throw std::runtime_error("invalid session block size");options.game.block_threads=n;}
      else if(flag=="--stack-kib"&&i+1<argc){const auto n=std::stoul(argv[++i]);if(n<16||n>128)throw std::runtime_error("invalid session stack size");options.game.stack_bytes=n*1024;}
      else if(flag=="--skip-compaction-check")options.game.validate_compaction=false;
      else if(flag=="--no-protect-incumbent")options.protect_incumbent=false;
      else if(flag=="--greedy-beam")options.greedy_beam=true;
      else throw std::runtime_error("unknown session flag");}
    search::SessionOutput output;
    search::run_session(options,image.data(),image.size(),in.replay.data(),in.replay.size(),in.candidates.data(),in.candidates.size(),&output);
    std::size_t last_offset=0;for(unsigned i=0;i+1<depths;++i)last_offset+=output.round_counts[i];
    const auto& winner=output.results[last_offset+output.ranks[last_offset]];
    std::uint64_t failed=0;for(const auto& r:output.results)failed+=bool(r.failed);
    const auto& t=output.timings;
    const char* policy=options.greedy_beam
      ?(options.protect_incumbent?"canonical-rank-with-protected-zero-edit-lineage":"canonical-top-k")
      :(options.protect_incumbent?"global-best-live-beam-with-protected-zero-edit-lineage":"global-best-live-beam");
    std::fprintf(stderr,"{\"selection_policy\":\"%s\",\"protect_incumbent\":%s}\n",
      policy,
      options.protect_incumbent?"true":"false");
    std::fprintf(stderr,"{\"identity\":\"experimental-cuda-game-search-session-v4\",\"promotion_eligible\":false,\"depths\":%lu,\"beam\":%lu,\"fanout\":%zu,\"candidate_nodes\":%zu,\"actual_simulated_branches\":%llu,\"actual_simulated_ticks\":%llu,\"failed_candidates\":%llu,\"trace_every_tick\":%s,\"cache_guest\":%s,\"block_threads\":%u,\"stack_kib\":%u,\"host_prefix_ms\":%.9f,\"prefix_ms\":%.9f,\"branch_ms\":%.9f,\"rollout_ms\":%.9f,\"rank_compact_ms\":%.9f,\"validation_ms\":%.9f,\"total_ms\":%.9f,\"allocated_bytes\":%llu,\"compaction_errors\":%llu,\"winner_tick\":%llu,\"winner_live_score\":%llu,\"winner_canonical_score\":%llu,\"winner_canonical_level\":%llu,\"winner_canonical_chain\":%llu,\"winner_natural\":%llu}\n",
      depths,beam,in.candidates.size(),output.results.size(),(unsigned long long)output.actual_simulated_branches,(unsigned long long)output.actual_simulated_ticks,(unsigned long long)failed,
      options.game.trace_every_tick?"true":"false",options.game.cache_guest?"true":"false",options.game.block_threads,options.game.stack_bytes/1024,
      t.host_prefix_ms,t.prefix_ms,t.branch_ms,t.rollout_ms,t.rank_compact_ms,t.validation_ms,t.total_ms,
      (unsigned long long)t.allocation_bytes,(unsigned long long)t.compaction_errors,
      (unsigned long long)winner.final.tick,(unsigned long long)winner.final.score,(unsigned long long)winner.canonical_score,
      (unsigned long long)winner.canonical_level,(unsigned long long)output.winner_canonical_chain,(unsigned long long)winner.natural);
    if(failed||t.compaction_errors)return 2;
    // This is an experimental candidate artifact. Independent native replay
    // verification is mandatory before any score or runtime promotion.
    std::FILE* replay=std::fopen(argv[10],"wb");if(!replay)throw std::runtime_error("cannot create winner replay");
    word(replay,options.game.seed);word(replay,winner.natural?winner.canonical_level:winner.final.level);
    word(replay,winner.natural?winner.canonical_score:winner.final.score);
    word(replay,winner.natural?output.winner_canonical_chain:winner.final.highest_chain);word(replay,0);
    for(unsigned i=0;i<8;++i)word(replay,0);
    for(auto value:output.winner_words)word(replay,value);
    if(std::fclose(replay))throw std::runtime_error("cannot close winner replay");
    std::ofstream lineage(argv[11]);if(!lineage)throw std::runtime_error("cannot create lineage manifest");
    lineage<<"{\n  \"schema\":\"irisu-experimental-cuda-search-lineage-v4\",\n  \"promotion_eligible\":false,\n  \"native_replay_verified\":false,\n  \"replay_sha256\":\""<<digest(source_replay)<<"\",\n  \"specs_sha256\":\""<<digest(source_specs)<<"\",\n  \"image_sha256\":\""<<digest(image)<<"\",\n  \"seed\":"<<options.game.seed<<",\n  \"prefix\":"<<options.game.prefix<<",\n  \"horizon\":"<<options.game.horizon<<",\n  \"fanout\":"<<in.candidates.size()<<",\n  \"beam\":"<<beam<<",\n  \"candidate_specs\":[";
    for(unsigned c=0;c<in.candidates.size();++c){lineage<<(c?",":"")<<"[";
      for(unsigned m=0;m<in.candidates[c].count;++m)lineage<<(m?",":"")<<"["<<in.candidates[c].mutations[m].offset<<","<<in.candidates[c].mutations[m].input<<"]";
      lineage<<"]";}
    lineage<<"],\n  \"protect_incumbent\":"<<(options.protect_incumbent?"true":"false")
      <<",\n  \"greedy_beam\":"<<(options.greedy_beam?"true":"false")
      <<",\n  \"selection_policy\":\""<<policy
      <<"\",\n  \"winner_source\":\"best-final-canonical-leaf\",\n  \"winner_candidates\":[";
    for(unsigned i=0;i<depths;++i)lineage<<(i?",":"")<<output.winner_candidates[i];
    lineage<<"],\n  \"rounds\":[\n";
    std::size_t offset=0,selected_offset=0;
    for(unsigned d=0;d<depths;++d){if(d)lineage<<",\n";lineage<<"    {\"depth\":"<<d<<",\"children\":"<<output.round_counts[d]<<",\"selected_children\":[";
      for(unsigned j=0;j<output.round_kept[d];++j)lineage<<(j?",":"")<<output.selected_children[selected_offset+j];
      unsigned live=0;for(unsigned i=0;i<output.round_counts[d];++i)live+=output.lifecycle_flags[offset+i]==0;
      lineage<<"],\"protected_parent_rank\":"<<output.protected_ranks[d]<<",\"live_children\":"<<live<<"}";
      offset+=output.round_counts[d];selected_offset+=output.round_kept[d];}
    lineage<<"\n  ]\n}\n";
    if(!lineage)throw std::runtime_error("cannot write lineage manifest");
    word(stdout,4);word(stdout,depths);
    offset=0;selected_offset=0;for(unsigned d=0;d<depths;++d){const auto count=output.round_counts[d];word(stdout,count);word(stdout,output.round_kept[d]);
      word(stdout,search::result_magic);word(stdout,search::result_version);word(stdout,count);word(stdout,sizeof(search::Result));
      if(std::fwrite(output.results.data()+offset,sizeof(search::Result),count,stdout)!=count||
         std::fwrite(output.ranks.data()+offset,4,count,stdout)!=count)throw std::runtime_error("cannot write session rounds");
      if(std::fwrite(output.selected_children.data()+selected_offset,4,output.round_kept[d],stdout)!=output.round_kept[d])
        throw std::runtime_error("cannot write selected session children");
      word(stdout,output.protected_ranks[d]);
      if(std::fwrite(output.lifecycle_flags.data()+offset,4,count,stdout)!=count)throw std::runtime_error("cannot write lifecycle flags");
      offset+=count;selected_offset+=output.round_kept[d];}
    return std::fflush(stdout)?1:0;
  }catch(const std::exception& e){std::fprintf(stderr,"search-session: %s\n",e.what());return 1;}
}
