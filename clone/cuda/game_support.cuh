#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>

#if defined(__CUDACC__)
#define GAME_HD __host__ __device__
#define GAME_NOINLINE __noinline__
#else
#define GAME_HD
#define GAME_NOINLINE __attribute__((noinline))
#endif

namespace irisu::cuda_game {

template<class T, std::size_t N> struct array {
  T values[N]{};
  GAME_HD constexpr T& operator[](std::size_t i) { return values[i]; }
  GAME_HD constexpr const T& operator[](std::size_t i) const { return values[i]; }
  GAME_HD constexpr T* begin() { return values; }
  GAME_HD constexpr const T* begin() const { return values; }
  GAME_HD constexpr T* end() { return values + N; }
  GAME_HD constexpr const T* end() const { return values + N; }
  GAME_HD constexpr T& front() { return values[0]; }
  GAME_HD constexpr const T& front() const { return values[0]; }
  GAME_HD constexpr T& back() { return values[N - 1]; }
  GAME_HD constexpr const T& back() const { return values[N - 1]; }
  GAME_HD constexpr T* data() { return values; }
  GAME_HD constexpr const T* data() const { return values; }
  GAME_HD constexpr std::size_t size() const { return N; }
  GAME_HD constexpr void fill(const T& value) { for (auto& item : values) item = value; }
};

template<class T> GAME_HD constexpr T min(T a, T b) { return a < b ? a : b; }
template<class T> GAME_HD constexpr T max(T a, T b) { return a < b ? b : a; }
template<class T> GAME_HD constexpr T clamp(T x, T low, T high) { return min(max(x, low), high); }
template<class T> GAME_HD constexpr T abs(T x) { return x < T{} ? -x : x; }
GAME_HD inline bool isfinite(double x) {
#ifdef __CUDA_ARCH__
  return ::isfinite(x);
#else
  return std::isfinite(x);
#endif
}
GAME_HD inline double trunc(double x) { return ::trunc(x); }
GAME_HD inline double sqrt(double x) { return ::sqrt(x); }

template<class To, class From> GAME_HD To bit_cast(const From& from) {
  static_assert(sizeof(To) == sizeof(From));
  To result;
  auto* out = reinterpret_cast<unsigned char*>(&result);
  const auto* in = reinterpret_cast<const unsigned char*>(&from);
  for (std::size_t i = 0; i < sizeof(To); ++i) out[i] = in[i];
  return result;
}

template<class T> GAME_HD constexpr T&& move(T& object) { return static_cast<T&&>(object); }
template<class T> GAME_HD constexpr void swap(T& a, T& b) { const T copy = a; a = b; b = copy; }
template<class It, class Predicate> GAME_HD It find_if(It first, It last, Predicate predicate) {
  while (first != last && !predicate(*first)) ++first;
  return first;
}
template<class It, class Predicate> GAME_HD It remove_if(It first, It last, Predicate predicate) {
  auto output = first;
  for (; first != last; ++first) if (!predicate(*first)) *output++ = *first;
  return output;
}
template<class It> GAME_HD It unique(It first, It last) {
  if (first == last) return last;
  auto output = first;
  for (auto current = first + 1; current != last; ++current)
    if (!(*output == *current)) *++output = *current;
  return output + 1;
}
template<class T> struct Less {
  GAME_HD bool operator()(const T& a, const T& b) const { return a < b; }
};
template<class It, class LessThan> GAME_HD void sort(It first, It last, LessThan less) {
  // Stable insertion sort preserves ties; rule sorting keys are unique (actor
  // slots), or deduplicated immediately afterwards (contact-pair keys).
  for (auto it = first; it != last; ++it) {
    const auto value = *it;
    auto position = it;
    while (position != first && less(value, *(position - 1))) {
      *position = *(position - 1);
      --position;
    }
    *position = value;
  }
}
template<class It> GAME_HD void sort(It first, It last) {
  sort(first, last, Less<typename std::remove_reference<decltype(*first)>::type>{});
}
template<class It, class T> GAME_HD bool binary_search(It first, It last, const T& value) {
  while (first != last) {
    const auto middle = first + (last - first) / 2;
    if (*middle < value) first = middle + 1;
    else if (value < *middle) last = middle;
    else return true;
  }
  return false;
}

struct Body;
struct Contact;
struct GroupState;
struct ScriptedOrigin;
template<class T> struct vector_capacity { static constexpr std::size_t value = 512; };
template<> struct vector_capacity<Contact> { static constexpr std::size_t value = 4096; };
template<> struct vector_capacity<GroupState> { static constexpr std::size_t value = 4096; };
template<> struct vector_capacity<std::uint64_t> { static constexpr std::size_t value = 4096; };

template<class T> struct vector {
  static constexpr auto capacity = vector_capacity<T>::value;
  // Construct only live elements. Zero/default-constructing every unused
  // contact slot at every tick overwhelms the physics work on a CUDA lane.
  static_assert(std::is_trivially_destructible<T>::value);
  alignas(T) unsigned char storage[sizeof(T) * capacity];
  std::uint32_t count{};
  bool overflow{};

  GAME_HD vector() {}
  // Keep representation copyable for cooperative device checkpoint transport.
  // Rule updates that need only live values call assign_live explicitly.
  GAME_HD void assign_live(const vector& other) {
    if (this == &other) return;
    count = 0;
    overflow = other.overflow;
    for (const auto& item : other) push_back(item);
  }
  GAME_HD T* begin() { return reinterpret_cast<T*>(storage); }
  GAME_HD const T* begin() const { return reinterpret_cast<const T*>(storage); }
  GAME_HD T* end() { return begin() + count; }
  GAME_HD const T* end() const { return begin() + count; }
  GAME_HD T* data() { return begin(); }
  GAME_HD const T* data() const { return begin(); }
  GAME_HD std::size_t size() const { return count; }
  GAME_HD bool empty() const { return count == 0; }
  GAME_HD T& operator[](std::size_t i) { return data()[i]; }
  GAME_HD const T& operator[](std::size_t i) const { return data()[i]; }
  GAME_HD T& front() { return data()[0]; }
  GAME_HD const T& front() const { return data()[0]; }
  GAME_HD T& back() { return data()[count ? count - 1 : 0]; }
  GAME_HD const T& back() const { return data()[count ? count - 1 : 0]; }
  GAME_HD void reserve(std::size_t n) { overflow |= n > capacity; }
  GAME_HD void clear() { count = 0; overflow = false; }
  GAME_HD void push_back(const T& value) {
    if (count == capacity) overflow = true;
    else ::new (static_cast<void*>(data() + count++)) T(value);
  }
  template<class... Args> GAME_HD void emplace_back(Args... args) { push_back(T{args...}); }
  GAME_HD void pop_back() { if (count) --count; else overflow = true; }
  GAME_HD void resize(std::size_t n) {
    if (n > capacity) { overflow = true; return; }
    while (count < n) ::new (static_cast<void*>(data() + count++)) T{};
    count = static_cast<std::uint32_t>(n);
  }
  GAME_HD T* erase(T* first, T* last) {
    const auto removed = last - first;
    for (auto it = last; it != end(); ++it) *(it - removed) = *it;
    count -= static_cast<std::uint32_t>(removed);
    return first;
  }
  GAME_HD T* erase(T* item) { return erase(item, item + 1); }
};

GAME_HD inline std::uint64_t fnv_byte(std::uint64_t hash, unsigned char value) {
  return (hash ^ value) * 1099511628211ULL;
}
template<class T> GAME_HD std::uint64_t fnv_integer(std::uint64_t hash, T value) {
  using U = typename std::make_unsigned<T>::type;
  const auto bits = static_cast<U>(value);
  for (unsigned i = 0; i < sizeof(T); ++i)
    hash = fnv_byte(hash, static_cast<unsigned char>(bits >> (i * 8)));
  return hash;
}
template<class T> GAME_HD std::uint64_t fnv_scalar(std::uint64_t hash, T value) {
  if constexpr (std::is_enum<T>::value)
    return fnv_integer(hash, static_cast<typename std::underlying_type<T>::type>(value));
  else if constexpr (std::is_same<T,bool>::value)
    return fnv_byte(hash, value ? 1 : 0);
  else if constexpr (std::is_same<T,double>::value)
    return fnv_integer(hash, bit_cast<std::uint64_t>(value));
  else if constexpr (std::is_same<T,float>::value)
    return fnv_integer(hash, bit_cast<std::uint32_t>(value));
  else return fnv_integer(hash, value);
}
struct Detail {
  std::uint64_t hash{14695981039346656037ULL};
  GAME_HD Detail(const char* text = "") {
    if (text) while (*text) hash = fnv_byte(hash, static_cast<unsigned char>(*text++));
  }
};

}  // namespace irisu::cuda_game
