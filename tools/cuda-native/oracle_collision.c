// Exact hidden native circle collision oracle; build ELF32, link crypto + dl.
#include "oracle_native.h"
#define INPUT_MAGIC 0x31434e49u
#define CASE_WORDS 22u
typedef void (*CircleCollision)(void*,const void*,const void*,int);
static int finite_word(uint32_t value){return (value&0x7fffffffu)<0x7f800000u;}
int main(int argc,char** argv){
  if(argc!=3)oracle_die("usage: oracle-collision pinned.so cases.bin");
  NativeOracle oracle=oracle_open(argv[1]);
  CircleCollision collide=(CircleCollision)oracle_symbol(&oracle,"?b2CollideCircle@@YAXPAUb2Manifold@@PAVb2CircleShape@@1_N@Z",0x45f0);
  FILE* input=fopen(argv[2],"rb");uint32_t header[4];
  if(!input||fread(header,sizeof(header),1,input)!=1||header[0]!=INPUT_MAGIC||header[1]!=1||!header[2]||header[2]>1000000||header[3]!=CASE_WORDS)oracle_die("invalid collision packet header");
  const uint16_t cw=0x027f;__asm__ __volatile__("fninit; fldcw %0"::"m"(cw));
  uint64_t contacts=0,disjoint=0;unsigned shape_errors=0;
  for(unsigned index=0;index<header[2];++index){
    uint32_t words[CASE_WORDS],first[19],second[19],saved_first[19],saved_second[19],manifold[15];
    if(fread(words,sizeof(words),1,input)!=1)oracle_die("short collision case");
    for(unsigned j=0;j<6;++j)if(!finite_word(words[j]))oracle_die("nonfinite geometry outside stage contract");
    if((words[2]>>31&&(words[2]&0x7fffffffu))||(words[5]>>31&&(words[5]&0x7fffffffu))||words[6]>1)oracle_die("invalid radius/conservative flag");
    memset(first,0xcd,sizeof(first));memset(second,0xcd,sizeof(second));
    first[6]=words[0];first[7]=words[1];first[18]=words[2];second[6]=words[3];second[7]=words[4];second[18]=words[5];
    memcpy(saved_first,first,sizeof(first));memcpy(saved_second,second,sizeof(second));memcpy(manifold,words+7,sizeof(manifold));
    collide(manifold,first,second,words[6]);
    shape_errors+=memcmp(first,saved_first,sizeof(first))!=0||memcmp(second,saved_second,sizeof(second))!=0;
    if(manifold[14]>1)oracle_die("native circle returned invalid point count");
    contacts+=manifold[14]!=0;disjoint+=manifold[14]==0;
    if(fwrite(manifold,sizeof(manifold),1,stdout)!=1)oracle_die("cannot write manifold output");
  }
  if(fgetc(input)!=EOF||ferror(input))oracle_die("trailing/read-error collision packet");
  fclose(input);oracle_close(&oracle);
  fprintf(stderr,"{\"identity\":\"experimental-native-circle-stage-v1\",\"native_sha256\":\"%s\",\"cases\":%u,\"contacts\":%llu,\"disjoint\":%llu,\"shape_mutation_errors\":%u,\"manifold_bytes\":60}\n",ORACLE_PIN,header[2],(unsigned long long)contacts,(unsigned long long)disjoint,shape_errors);
  return shape_errors||fflush(stdout)?2:0;
}
