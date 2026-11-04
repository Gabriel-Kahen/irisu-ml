#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define IRISU_GUEST_HD __host__ __device__
#else
#define IRISU_GUEST_HD
#endif

namespace irisu::cuda_guest {

// A private 32-bit address space keeps pointers relocatable across device lanes.
// Unsupported instructions, FP domains and memory accesses invalidate the lane.
enum Error : std::uint32_t {
  ok, bad_address, out_of_memory, bad_pc, unsupported_fp, stack_fault,
  aborted, budget_exhausted, unsupported_instruction, invalid_division
};

struct Guest {
  std::uint32_t r[8]{}; // eax, ecx, edx, ebx, esp, ebp, esi, edi
  double fp[8]{}, fault_value{};
  std::uint32_t top{}, depth{}, status{}, control{0x027f}, flags{};
  std::uint32_t pc{}, error{}, fault_pc{}, heap{}, heap_limit{}, capacity{};
  std::uint64_t instructions{};
  std::uint32_t free_head{}, allocation_reuses{};
  unsigned char* memory{};
  // Storage transport only: guest addresses/capacity remain logical bytes.
  // A stride-32 cohort interleaves adjacent worlds' 32-bit words.
  std::uint32_t word_stride{1};

  IRISU_GUEST_HD std::size_t byte_offset(std::uint32_t a) const {
    return std::size_t(a & ~3u) * word_stride + (a & 3u);
  }
  IRISU_GUEST_HD unsigned char byte_at(std::uint32_t a) const {
    return memory[byte_offset(a)];
  }
  IRISU_GUEST_HD void byte_at(std::uint32_t a, unsigned char value) {
    memory[byte_offset(a)] = value;
  }

  IRISU_GUEST_HD void fail(Error e) {
    if (!error) { error=e; fault_pc=pc; fault_value=st(); }
  }
  IRISU_GUEST_HD bool valid(std::uint32_t a, unsigned n) {
    if (word_stride != 1 && word_stride != 32) { fail(bad_address); return false; }
    if (a > capacity || n > capacity-a) { fail(bad_address); return false; }
    return true;
  }
  template<class T> IRISU_GUEST_HD T load(std::uint32_t a) {
    T v{};
    if (valid(a,sizeof(T))) {
#ifdef __CUDA_ARCH__
      if constexpr (sizeof(T)==4) {
        if(!(a&3)){union{std::uint32_t word;T value;}u;u.word=*reinterpret_cast<const std::uint32_t*>(memory+byte_offset(a));return u.value;}
      } else if constexpr (sizeof(T)==8) {
        if(word_stride==1 && !(a&7)){union{std::uint64_t word;T value;}u;u.word=*reinterpret_cast<const std::uint64_t*>(memory+a);return u.value;}
        if(!(a&3)){
          union{std::uint64_t word;T value;}u;
          const auto low=*reinterpret_cast<const std::uint32_t*>(memory+byte_offset(a));
          const auto high=*reinterpret_cast<const std::uint32_t*>(memory+byte_offset(a+4));
          u.word=std::uint64_t(low)|(std::uint64_t(high)<<32);return u.value;
        }
      }
#endif
      auto* p=reinterpret_cast<unsigned char*>(&v);
      for (unsigned i=0;i<sizeof(T);++i) p[i]=byte_at(a+i);
    }
    return v;
  }
  template<class T> IRISU_GUEST_HD void store(std::uint32_t a, T v) {
    if (valid(a,sizeof(T))) {
#ifdef __CUDA_ARCH__
      if constexpr (sizeof(T)==4) {
        if(!(a&3)){union{T value;std::uint32_t word;}u;u.value=v;*reinterpret_cast<std::uint32_t*>(memory+byte_offset(a))=u.word;return;}
      } else if constexpr (sizeof(T)==8) {
        if(word_stride==1 && !(a&7)){union{T value;std::uint64_t word;}u;u.value=v;*reinterpret_cast<std::uint64_t*>(memory+a)=u.word;return;}
        if(!(a&3)){
          union{T value;std::uint64_t word;}u;u.value=v;
          *reinterpret_cast<std::uint32_t*>(memory+byte_offset(a))=std::uint32_t(u.word);
          *reinterpret_cast<std::uint32_t*>(memory+byte_offset(a+4))=std::uint32_t(u.word>>32);return;
        }
      }
#endif
      auto* p=reinterpret_cast<unsigned char*>(&v);
      for (unsigned i=0;i<sizeof(T);++i) byte_at(a+i,p[i]);
    }
  }
  IRISU_GUEST_HD void push(std::uint32_t v) { r[4]-=4; store(r[4],v); }
  IRISU_GUEST_HD std::uint32_t pop() { auto v=load<std::uint32_t>(r[4]); r[4]+=4; return v; }
  IRISU_GUEST_HD double& st(unsigned i=0) { return fp[(top+i)&7]; }
  IRISU_GUEST_HD void fpush(double v) {
    if (depth==8) { fail(stack_fault); return; }
    top=(top-1)&7; st()=v; ++depth;
  }
  IRISU_GUEST_HD void fpop() {
    if (!depth) { fail(stack_fault); return; }
    top=(top+1)&7; --depth;
  }
  IRISU_GUEST_HD std::uint32_t alloc(std::uint32_t n) {
    // Headers and links are guest offsets, so copying the arena preserves the
    // allocator exactly. First-fit recycling bounds long-running game memory.
    if (n>0xfffffff0u) { fail(out_of_memory); return 0; }
    n=(n+15)&~15u;
    if (!n) n=16;
    std::uint32_t previous=0, a=free_head;
    while(a&&!error) {
      if(!valid(a,16)||load<std::uint32_t>(a+8)!=0x49524953u || load<std::uint32_t>(a+12)!=1) {
        fail(bad_address); return 0;
      }
      const auto next=load<std::uint32_t>(a+4), size=load<std::uint32_t>(a);
      if(size>=n) {
        if(previous)store(previous+4,next);else free_head=next;
        store(a+12,0u);++allocation_reuses;
        fill(a+16,0,size);return a+16;
      }
      previous=a;a=next;
    }
    if(error)return 0;
    if (n>heap_limit || heap>heap_limit-n || heap_limit-heap-n<16) {
      fail(out_of_memory); return 0;
    }
    a=heap; heap+=n+16; store(a,n);store(a+4,0u);
    store(a+8,0x49524953u);store(a+12,0u);
    fill(a+16,0,n);
    return a+16;
  }
  IRISU_GUEST_HD void fill(std::uint32_t a,std::uint32_t value,std::uint32_t n) {
    if(!valid(a,n))return;
    value&=255;
    while(n&&(a&3)){byte_at(a++,value);--n;}
    const auto word=value*0x01010101u;
    while(n>=4){store<std::uint32_t>(a,word);a+=4;n-=4;}
    while(n--){byte_at(a++,value);}
  }
  IRISU_GUEST_HD void copy(std::uint32_t a,std::uint32_t b,std::uint32_t n) {
    if(!valid(a,n)||!valid(b,n))return;
    if(a<b){
      if((a&3)==(b&3)){
        while(n&&(a&3)){byte_at(a++,byte_at(b++));--n;}
        while(n>=4){store<std::uint32_t>(a,load<std::uint32_t>(b));a+=4;b+=4;n-=4;}
      }
      while(n--){byte_at(a++,byte_at(b++));}
    }else if(a>b){
      a+=n;b+=n;
      if((a&3)==(b&3)){
        while(n&&(a&3)){byte_at(--a,byte_at(--b));--n;}
        while(n>=4){a-=4;b-=4;store<std::uint32_t>(a,load<std::uint32_t>(b));n-=4;}
      }
      while(n--){byte_at(--a,byte_at(--b));}
    }
  }
  IRISU_GUEST_HD void release(std::uint32_t a) {
    if(!a)return;
    if(a<16 || a>heap || !valid(a-16,16)) {fail(bad_address);return;}
    a-=16;
    if(load<std::uint32_t>(a+8)!=0x49524953u || load<std::uint32_t>(a+12)!=0 ||
       load<std::uint32_t>(a)>heap-a-16) {fail(bad_address);return;}
    store(a+4,free_head);store(a+12,1u);free_head=a;
  }
  IRISU_GUEST_HD void compare_fp(double a,double b) {
    status &= ~0x4500u;
    if (a!=a || b!=b) status|=0x4500;
    else if (a<b) status|=0x100;
    else if (a==b) status|=0x4000;
  }
  IRISU_GUEST_HD bool condition(unsigned kind) const {
    const bool c=flags&1,p=flags&4,z=flags&64,s=flags&128,o=flags&2048;
    switch(kind) {
      case 0:return z; case 1:return !z; case 2:return c; case 3:return !c;
      case 4:return c||z; case 5:return !c&&!z; case 6:return s!=o;
      case 7:return s==o; case 8:return z||(s!=o); case 9:return !z&&(s==o);
      case 10:return p; case 11:return !p; case 12:return s; case 13:return !s;
      default:return false;
    }
  }
  IRISU_GUEST_HD std::uint32_t alu(unsigned op,std::uint32_t a,std::uint32_t b,unsigned bits) {
    const std::uint64_t mask=bits==32?0xffffffffull:((1ull<<bits)-1);
    const std::uint32_t sign=1u<<(bits-1);
    a&=mask; b&=mask;
    std::uint64_t wide{}; bool c=false,o=false;
    switch(op) {
      case 0: wide=std::uint64_t(a)+b; c=wide>mask; o=(~(a^b)&(a^wide)&sign)!=0; break;
      case 1: wide=std::uint64_t(a)-b; c=a<b; o=((a^b)&(a^wide)&sign)!=0; break;
      case 2: wide=a&b; break;
      case 3: wide=a|b; break;
      case 4: wide=a^b; break;
    }
    const auto v=std::uint32_t(wide&mask);
    unsigned p=v&255; p^=p>>4; p&=15;
    flags=(flags&~0x8c5u)|unsigned(c)|(unsigned(((0x9669u>>p)&1)!=0)<<2)
      |(unsigned(v==0)<<6)|(unsigned((v&sign)!=0)<<7)|(unsigned(o)<<11);
    return v;
  }
  IRISU_GUEST_HD std::uint32_t shift(unsigned op,std::uint32_t a,std::uint32_t b,unsigned bits) {
    b&=31;
    if (!b) return a;
    const auto mask=bits==32?0xffffffffu:((1u<<bits)-1);
    a&=mask;
    const bool c=b<=bits ? (op==0?((a>>(bits-b))&1):((a>>(b-1))&1)):false;
    const auto signed_value=std::int32_t(a<<(32-bits))>>(32-bits);
    std::uint32_t v=op==0 ? (a<<b)&mask : op==1 ? a>>b : std::uint32_t(signed_value>>b)&mask;
    alu(3,v,0,bits);
    flags=(flags&~1u)|c;
    if(b==1) {
      const bool o=op==0 ? ((v>>(bits-1))&1)!=c : op==1 ? (a>>(bits-1))&1 : false;
      flags=(flags&~2048u)|(unsigned(o)<<11);
    }
    return v;
  }
  IRISU_GUEST_HD std::uint32_t carry(unsigned op,std::uint32_t a,std::uint32_t b,unsigned bits) {
    const std::uint64_t mask=bits==32?0xffffffffull:((1ull<<bits)-1);
    const std::uint32_t sign=1u<<(bits-1), cin=flags&1;
    a&=mask;b&=mask;
    const auto wide=op==0?std::uint64_t(a)+b+cin:std::uint64_t(a)-b-cin;
    const auto v=std::uint32_t(wide&mask);
    const bool c=op==0?wide>mask:std::uint64_t(a)<std::uint64_t(b)+cin;
    const bool o=op==0?(~(a^b)&(a^v)&sign)!=0:((a^b)&(a^v)&sign)!=0;
    alu(3,v,0,bits);
    flags=(flags&~2049u)|unsigned(c)|(unsigned(o)<<11);
    return v;
  }
};

IRISU_GUEST_HD inline std::uint32_t float_word(float f) {
  union { float f; std::uint32_t u; } v{f}; return v.u;
}

} // namespace irisu::cuda_guest
