// Component oracle: execute the pinned original's pure level setters only.
// The Python driver authenticates the input; no game entrypoint is called.
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static uint16_t u16(const unsigned char *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static uint32_t u32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t u64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static void die(const char *message) { fprintf(stderr, "%s\n", message); exit(1); }

static float ex_speed = -0.05f;
static int reward = 120;
static void *get_config(void *map, void *type, unsigned size, unsigned length, const char *key) {
  (void)map; (void)type;
  if (size != 4) die("unexpected config width");
  if (length == 14 && memcmp(key, "ex_block_speed", 14) == 0) return &ex_speed;
  if (length == 19 && memcmp(key, "game_life_plus_unit", 19) == 0) return &reward;
  die("unexpected config request");
  return NULL;
}

static void jump(uintptr_t address, uintptr_t target) {
  unsigned char *p = (unsigned char *)address;
  const uint32_t displacement = (uint32_t)(target - address - 5);
  p[0] = 0xe9;
  memcpy(p + 1, &displacement, 4);
}

static void run_setter(uint32_t level, uintptr_t entry) {
  unsigned short control = 0x027f;
  __asm__ volatile("fninit; fldcw %0" : : "m"(control) : "memory");
  __asm__ volatile("call *%1" : "+a"(level) : "m"(entry)
                   : "ecx", "edx", "memory", "cc", "st", "st(1)", "st(2)", "st(3)");
}

int main(int argc, char **argv) {
  if (argc != 2) die("usage: level_probe pinned-irisu.exe");
  FILE *stream = fopen(argv[1], "rb");
  if (!stream) die("cannot open original image");
  if (fseek(stream, 0, SEEK_END)) die("cannot size original image");
  long length = ftell(stream);
  if (length < 0x400 || length > 1000000 || fseek(stream, 0, SEEK_SET)) die("invalid image size");
  unsigned char *file = malloc((size_t)length);
  if (!file || fread(file, 1, (size_t)length, stream) != (size_t)length) die("cannot read original image");
  fclose(stream);
  uint32_t pe = u32(file + 0x3c);
  if (pe + 24 > (uint32_t)length || memcmp(file + pe, "PE\0\0", 4)) die("not a PE image");
  const unsigned char *optional = file + pe + 24;
  if (u16(optional) != 0x10b || u32(optional + 28) != 0x400000) die("unexpected PE ABI/base");
  uint32_t image_size = u32(optional + 56);
  if (image_size > 0x100000) die("oversized image");
  void *image = mmap((void *)0x400000, image_size, PROT_READ | PROT_WRITE | PROT_EXEC,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  if (image == MAP_FAILED) { perror("mmap original-address image"); return 1; }
  const unsigned char *section = optional + u16(file + pe + 20);
  unsigned sections = u16(file + pe + 6);
  for (unsigned i = 0; i < sections; ++i, section += 40) {
    if (section + 40 > file + length) die("invalid section table");
    uint32_t rva = u32(section + 12), size = u32(section + 16), offset = u32(section + 20);
    if (rva > image_size || size > image_size - rva || offset > (uint32_t)length || size > (uint32_t)length - offset)
      die("invalid section extent");
    memcpy((unsigned char *)image + rva, file + offset, size);
  }
  free(file);
  jump(0x416268, (uintptr_t)&get_config); // Supply only the two shipped config values.
  *(unsigned char *)0x418964 = 0xc3;    // Disable debug logging; caller removes args.
  const unsigned char rng_stub[] = {0x31, 0xc0, 0xc3};
  memcpy((void *)0x40ffc4, rng_stub, sizeof rng_stub); // Normal shape draw irrelevant to scale.
  const unsigned char *globals = (const unsigned char *)0x436a10;
  puts("mode\tlevel\tdrain\tdescent_bits\tex_rise_bits\trot_penalty\tmaximum_color_id\tspawn_interval\tex_spawn_interval\tscore_scale_bits\tclear_reward\tshape_cutoff");
  for (unsigned mode = 0; mode < 2; ++mode) {
    for (unsigned level = 1; level <= 99; ++level) {
      run_setter(level, mode ? 0x415800 : 0x415624);
      printf("%u\t%u\t%u\t%08" PRIx32 "\t%08" PRIx32 "\t%u\t%u\t%u\t%u\t%016" PRIx64 "\t%u\t%u\n",
             mode, level, u32(globals), u32(globals + 4), u32(globals + 8), u32(globals + 12),
             u32(globals + 16), u32(globals + 20), u32(globals + 24), u64(globals + 32),
             u32(globals + 40), u32(globals + 44));
    }
  }
  munmap(image, image_size);
  return 0;
}
