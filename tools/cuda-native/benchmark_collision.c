// Native ELF32 persistent-thread circle-stage baseline. Original native code,
// fresh manifold input each repetition, no per-case RPC/fork or output in timer.
#include "oracle_native.h"
#include <pthread.h>
#include <time.h>

typedef void (*CircleCollision)(void*,const void*,const void*,int);
typedef struct {uint32_t a[19],b[19],initial[15],conservative;} Case;
typedef struct {unsigned id,count,threads,runs;Case* input;uint32_t (*output)[15];CircleCollision collide;pthread_barrier_t* begin;pthread_barrier_t* end;} Worker;
static double milliseconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1000000.0;}
static int compare_double(const void* a,const void* b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static void* work(void* argument){
  Worker* w=argument;const uint16_t cw=0x027f;__asm__ __volatile__("fninit; fldcw %0"::"m"(cw));
  const unsigned first=(uint64_t)w->id*w->count/w->threads,last=(uint64_t)(w->id+1)*w->count/w->threads;
  for(unsigned run=0;run<w->runs;++run){pthread_barrier_wait(w->begin);
    for(unsigned i=first;i<last;++i){memcpy(w->output[i],w->input[i].initial,60);w->collide(w->output[i],w->input[i].a,w->input[i].b,w->input[i].conservative);}
    pthread_barrier_wait(w->end);
  }
  return NULL;
}
int main(int argc,char** argv){
  const double started=milliseconds();
  if(argc<3)oracle_die("usage: benchmark-collision native.so cases.bin [--threads N] [--warmups N] [--repeats N]");
  unsigned threads=1,warmups=1,repeats=3;
  for(int i=3;i<argc;++i){unsigned* option=NULL;
    if(!strcmp(argv[i],"--threads"))option=&threads;else if(!strcmp(argv[i],"--warmups"))option=&warmups;else if(!strcmp(argv[i],"--repeats"))option=&repeats;
    if(!option||++i>=argc)oracle_die("invalid benchmark option");*option=strtoul(argv[i],NULL,10);
  }
  if(!threads||threads>64||warmups>10||!repeats||repeats>20)oracle_die("invalid bounded benchmark dimensions");
  NativeOracle oracle=oracle_open(argv[1]);CircleCollision collide=(CircleCollision)oracle_symbol(&oracle,"?b2CollideCircle@@YAXPAUb2Manifold@@PAVb2CircleShape@@1_N@Z",0x45f0);
  const uint16_t cw=0x027f;__asm__ __volatile__("fninit; fldcw %0"::"m"(cw));
  FILE* file=fopen(argv[2],"rb");uint32_t header[4];
  if(!file||fread(header,16,1,file)!=1||header[0]!=0x31434e49u||header[1]!=1||!header[2]||header[2]>1000000||header[3]!=22)oracle_die("invalid circle packet");
  const unsigned count=header[2];Case* input=malloc((size_t)count*sizeof(Case));uint32_t (*output)[15]=malloc((size_t)count*60);
  if(!input||!output)oracle_die("cannot allocate benchmark corpus");
  for(unsigned i=0;i<count;++i){uint32_t words[22];if(fread(words,88,1,file)!=1)oracle_die("short case packet");
    for(unsigned j=0;j<6;++j)if((words[j]&0x7fffffffu)>=0x7f800000u)oracle_die("nonfinite input");
    if(words[6]>1||(words[2]>>31&&(words[2]&0x7fffffffu))||(words[5]>>31&&(words[5]&0x7fffffffu)))oracle_die("invalid circle input");
    memset(&input[i],0xcd,sizeof(Case));input[i].a[6]=words[0];input[i].a[7]=words[1];input[i].a[18]=words[2];
    input[i].b[6]=words[3];input[i].b[7]=words[4];input[i].b[18]=words[5];memcpy(input[i].initial,words+7,60);input[i].conservative=words[6];
  }
  if(fgetc(file)!=EOF||ferror(file))oracle_die("trailing case data");fclose(file);
  // Input immutability is independently checked by oracle_collision before
  // timing. Do not add an extra native pass to this process-wall baseline.
  pthread_barrier_t begin,end;if(pthread_barrier_init(&begin,NULL,threads+1)||pthread_barrier_init(&end,NULL,threads+1))oracle_die("cannot create thread barriers");
  pthread_t ids[64];Worker workers[64];
  for(unsigned i=0;i<threads;++i){workers[i]=(Worker){i,count,threads,warmups+repeats,input,output,collide,&begin,&end};
    if(pthread_create(ids+i,NULL,work,workers+i))oracle_die("cannot create native worker");}
  double samples[20];
  for(unsigned run=0;run<warmups+repeats;++run){const double start=milliseconds();pthread_barrier_wait(&begin);pthread_barrier_wait(&end);if(run>=warmups)samples[run-warmups]=milliseconds()-start;}
  for(unsigned i=0;i<threads;++i)pthread_join(ids[i],NULL);
  unsigned contacts=0;for(unsigned i=0;i<count;++i){if(output[i][14]>1)oracle_die("native invalid point count");contacts+=output[i][14]!=0;}
  double sorted[20];memcpy(sorted,samples,repeats*sizeof(double));qsort(sorted,repeats,sizeof(double),compare_double);
  fprintf(stderr,"{\"schema\":\"irisu-native-circle-benchmark-v1\",\"backend\":\"native-elf32\",\"threads\":%u,\"cases\":%u,\"contacts\":%u,\"warmups\":%u,\"repeats\":%u,\"median_stage_ms\":%.9f,\"elapsed_before_stdout_ms\":%.9f,\"samples_ms\":[",threads,count,contacts,warmups,repeats,sorted[repeats/2],milliseconds()-started);
  for(unsigned i=0;i<repeats;++i)fprintf(stderr,"%s%.9f",i?",":"",samples[i]);fprintf(stderr,"]}\n");
  const int failed=fwrite(output,60,count,stdout)!=count||fflush(stdout);oracle_close(&oracle);free(output);free(input);return failed?1:0;
}
