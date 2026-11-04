/* Bounded ELF32 CPU reference for the fixed-image GPU physics translator.
 * gcc -m32 -O2 -fno-fast-math physics_oracle.c -ldl -o physics-oracle
 * physics-oracle /absolute/path/to/exact-host.so [steps] [--jsonl] [--mutations]
 * Default stdout: little-endian uint32 words, step/body/(x,y,r,vx,vy).
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *library;
static void *(*world_create)(float, float, float, float, float, float);
static void (*world_destroy)(void *);
static void *(*create_box)(void *, float, float, float, float, float, float, float, float);
static void *(*create_triangle)(void *, float, float, float, float, float, float, float, float);
static void *(*create_circle)(void *, float, float, float, float, float, float);
static void (*destroy_body)(void *, void *);
static void (*world_step)(void *, float, int);
static float (*get_x)(void *, void *);
static float (*get_y)(void *, void *);
static float (*get_r)(void *, void *);
static void (*get_v)(void *, void *, float *, float *);
static void (*set_v)(void *, void *, float, float);
static void (*set_user_data)(void *, void *, void *);
static int (*get_contact)(void *, void **, void **);

static void load(void *target, size_t size, const char *name) {
  void *symbol = dlsym(library, name);
  if (!symbol || size != sizeof(symbol)) {
    fprintf(stderr, "cannot resolve %s: %s\n", name, dlerror());
    exit(1);
  }
  memcpy(target, &symbol, size);
}

static uint32_t bits(float value) {
  uint32_t word;
  memcpy(&word, &value, sizeof(word));
  return word;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: physics-oracle exact-host.so [steps] [--jsonl] [--mutations]\n");
    return 2;
  }
  unsigned steps = 600;
  int jsonl = 0, mutations = 0;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--jsonl")) jsonl = 1;
    else if (!strcmp(argv[i], "--mutations")) mutations = 1;
    else {
      char *end;
      unsigned long count = strtoul(argv[i], &end, 10);
      if (*end || !count || count > 100000) return 2;
      steps = (unsigned)count;
    }
  }
  const uint16_t cw = 0x027f;
  __asm__ __volatile__("fldcw %0" : : "m"(cw));
  library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    fprintf(stderr, "%s\n", dlerror());
    return 1;
  }
#define LOAD(local, suffix) load(&local, sizeof(local), "b2d_world_" suffix)
  LOAD(world_create, "create");
  LOAD(world_destroy, "destroy");
  LOAD(create_box, "create_box");
  LOAD(create_triangle, "create_triangle");
  LOAD(create_circle, "create_circle");
  LOAD(destroy_body, "destroy_body");
  LOAD(world_step, "step");
  LOAD(get_x, "get_x");
  LOAD(get_y, "get_y");
  LOAD(get_r, "get_r");
  LOAD(get_v, "get_v");
  LOAD(set_v, "set_v");
  LOAD(set_user_data, "set_user_data");
  LOAD(get_contact, "get_contact");
#undef LOAD
  void *world = world_create(-100, -100, 100, 100, 10, 1);
  void *floor = create_box(world, 20, 1, 0, 10, 0, 0, 0.3f, 0.1f);
  void *bodies[3] = {
      create_circle(world, 0.5f, 0, 0, 1, 0.3f, 0.1f),
      create_box(world, 1, 1, 0.3f, -2, 0, 1, 0.3f, 0.1f),
      create_triangle(world, 1, 1, -0.2f, -4, 0, 1, 0.3f, 0.1f),
  };
  if (!world || !floor || !bodies[0] || !bodies[1] || !bodies[2]) return 1;
  set_user_data(world, floor, (void *)(uintptr_t)1);
  for (unsigned body = 0; body < 3; ++body)
    set_user_data(world, bodies[body], (void *)(uintptr_t)(body + 2));
  for (unsigned step = 1; step <= steps; ++step) {
    if (mutations && step == 200) set_v(world, bodies[0], 1.25f, -3);
    if (mutations && step == 300) {
      destroy_body(world, bodies[1]);
      bodies[1] = create_box(world, 1, 1, 1, -3, 0.2f, 1, 0.3f, 0.1f);
      if (!bodies[1]) return 1;
      set_user_data(world, bodies[1], (void *)(uintptr_t)3);
    }
    world_step(world, 1.0f / 60.0f, 10);
    if (jsonl) printf("{\"tick\":%u,\"bodies\":[", step);
    for (unsigned body = 0; body < 3; ++body) {
      float vx, vy;
      get_v(world, bodies[body], &vx, &vy);
      uint32_t words[5] = {bits(get_x(world, bodies[body])),
                           bits(get_y(world, bodies[body])),
                           bits(get_r(world, bodies[body])), bits(vx), bits(vy)};
      if (jsonl) {
        if (body) putchar(',');
        printf("[%u,%u,%u,%u,%u]", words[0], words[1], words[2], words[3], words[4]);
      } else if (fwrite(words, sizeof(words), 1, stdout) != 1) return 1;
    }
    if (jsonl) {
      printf("],\"contacts\":[");
      void *first, *second;
      int comma = 0;
      while (get_contact(world, &first, &second)) {
        if (comma) putchar(',');
        printf("[%u,%u]", (unsigned)(uintptr_t)first, (unsigned)(uintptr_t)second);
        comma = 1;
      }
      puts("]}");
    }
  }
  world_destroy(world);
  dlclose(library);
  return ferror(stdout) ? 1 : 0;
}
