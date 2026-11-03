#include "irisu/metsu/rules.hpp"

#include "irisu/floating_point.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace irisu::metsu {
namespace {

// Raw original setter stores, independently checked against the normal oracle.
// Reproduce with tools/metsu/dump_level_parameters.py; complete provenance and
// SHA-256 are in reference/metsu-level-parameters.tsv and its benchmark report.
constexpr std::array<std::uint32_t, 99> kDescentBits{
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f19999aU,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f19999aU,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f19999aU,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f800000U,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f19999aU,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f19999aU,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f19999aU,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f800000U,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f19999aU,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f19999aU,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f19999aU,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f800000U,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f19999aU,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f19999aU,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f19999aU,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU, 0x3e6147aeU, 0x3e800000U, 0x3f800000U,
    0x3e051eb8U, 0x3e23d70aU, 0x3e428f5cU,
};

constexpr std::array<std::uint64_t, 99> kScoreScaleBits{
    0x4010000000000000ULL, 0x4019fdf8bcce533dULL, 0x402142e81c889914ULL,
    0x40251cb453b9536cULL, 0x4028ae6d3fc447a1ULL, 0x402c0a88fb9717f1ULL,
    0x402f3c7995604382ULL, 0x403125fbee250664ULL, 0x40329f4504b4e7e6ULL,
    0x40340c28430012e7ULL, 0x40356e3d8cca85afULL, 0x4036c6c846ab7ca7ULL,
    0x403816ce65cfe3c4ULL, 0x40395f27eaf32c5bULL, 0x403aa089a39ba854ULL,
    0x403bdb8cdadbe120ULL, 0x403d10b4fc1eec63ULL, 0x403e4073cb226e4fULL,
    0x403f6b2c9b687921ULL, 0x4040489b672a7d8dULL, 0x4040d96fe46af637ULL,
    0x40416836416e33c9ULL, 0x4041f50d649383d7ULL, 0x4042801121978c6cULL,
    0x4047cbb14e7d141eULL, 0x4048754101d050d8ULL, 0x40491cde9666a004ULL,
    0x4049c2a1e8e36dfbULL, 0x404a66a0e27ff369ULL, 0x404b08efb19fd8dcULL,
    0x404ba9a0fa5234b0ULL, 0x404c48c6001f0ac0ULL, 0x404ce66eca3c5fe0ULL,
    0x404d82aa431018eeULL, 0x404e1d8653b98938ULL, 0x404eb70ffc3c0d3cULL,
    0x404f4f5368c94e3bULL, 0x404fe65c04944e75ULL, 0x40503e1a45427ed6ULL,
    0x405088738a0afa67ULL, 0x4050d23e934d33bfULL, 0x40511b7fdf906c82ULL,
    0x4051643bae8fa2d8ULL, 0x4051ac7606077528ULL, 0x4051f432b60c3066ULL,
    0x40523b755cf61b98ULL, 0x405282416af02770ULL, 0x4052c89a253388c8ULL,
    0x40530e82a8f9641cULL, 0x405731971dcc9e19ULL, 0x405784782564e276ULL,
    0x4057d6dcba15bf16ULL, 0x405828c7f62663c8ULL, 0x40587a3cd17d6f72ULL,
    0x4058cb3e23b9601fULL, 0x40591bcea61f2ceaULL, 0x40596bf0f562fd07ULL,
    0x4059bba7934e7fa2ULL, 0x405a0af4e84806cdULL, 0x405a59db44bd323aULL,
    0x405aa85ce273ab5dULL, 0x405af67be5c23289ULL, 0x405b443a5eb4022aULL,
    0x405b919a4a1859b0ULL, 0x405bde9d927fd3f7ULL, 0x405c2b4611290474ULL,
    0x405c77958eddb226ULL, 0x405cc38dc4c1e73aULL, 0x405d0f305d15efadULL,
    0x405d5a7ef3ec470cULL, 0x405da57b17d45f07ULL, 0x405df0264a7b1374ULL,
    0x405e3a8201418dd2ULL, 0x405e848fa5cb49e8ULL, 0x4061f859ad224152ULL,
    0x4062233396d0f980ULL, 0x40624de24898da1cULL, 0x406278667cea9859ULL,
    0x4062a2c0e8ba1e6eULL, 0x4062ccf23bb91f0bULL, 0x4062f6fb208e8423ULL,
    0x406320dc3d0aedb7ULL, 0x40634a96325a7120ULL, 0x406374299d33c5a5ULL,
    0x40639d97160507efULL, 0x4063c6df311e3abdULL, 0x4063f0027ed9a867ULL,
    0x406419018bc24652ULL, 0x406441dce0b838edULL, 0x40646a95031394caULL,
    0x4064932a74c5773cULL, 0x4064bb9db4778f4cULL, 0x4064e3ef3daa2dddULL,
    0x40650c1f88d0f29cULL, 0x4065342f0b6e2982ULL, 0x40655c1e382cebd6ULL,
    0x406583ed7efa15f0ULL, 0x4065ab9d4d1c222aULL, 0x4065d32e0d49f847ULL,
};

constexpr std::array<std::uint32_t, 99> kExIntervals{
    176U, 157U, 150U, 157U, 176U, 200U, 176U, 157U, 150U, 157U, 176U, 200U,
    175U, 157U, 150U, 157U, 176U, 200U, 175U, 157U, 150U, 157U, 175U, 200U,
    175U, 157U, 150U, 157U, 175U, 200U, 175U, 157U, 150U, 157U, 175U, 200U,
    176U, 157U, 150U, 157U, 176U, 200U, 176U, 157U, 150U, 157U, 175U, 200U,
    175U, 157U, 150U, 157U, 175U, 200U, 175U, 157U, 150U, 157U, 175U, 200U,
    176U, 157U, 150U, 157U, 175U, 200U, 176U, 157U, 150U, 157U, 176U, 200U,
    175U, 157U, 150U, 157U, 175U, 200U, 176U, 157U, 150U, 157U, 176U, 200U,
    175U, 157U, 150U, 157U, 175U, 200U, 176U, 157U, 150U, 157U, 176U, 200U,
    176U, 157U, 150U,
};

std::size_t index_for_level(std::uint32_t level) {
  if (level == 0 || level > 99) {
    throw std::invalid_argument("Metsu level parameters exist only for 1..99");
  }
  return level - 1U;
}

}  // namespace

MechanicsConfig default_config() {
  MechanicsConfig config;
  config.field_x = 94.0;
  config.field_y = 20.0;
  config.field_width = 420.0;
  config.field_height = 370.0;
  config.field_blank = 30.0;
  config.field_top_width = 450.0;
  config.field_bottom_height = 400.0;
  config.side_wall_top = 20.0;
  config.side_wall_bottom = 390.0;
  config.piece_sizes = {30.0, 48.0, 64.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0};
  config.piece_size_weights = {10, 40, 20, 0, 0, 0, 0, 0, 0, 0};
  config.piece_life_ticks = 10'000;
  config.rot_delay_ticks = 120;
  config.projectile_life_ticks = 1'200;
  config.gauge_max = 10'000;
  config.gauge_initial = 1'000;
  config.gauge_clear_unit = 120;
  config.rotten_penalty = 600;
  config.passive_gauge_decay_per_tick = 0;
  config.spawn_interval_ticks = 110;
  config.starting_colors = 6;
  config.maximum_colors = 9;
  config.qualifying_clears_per_level = 6;
  config.initial_rotten_count = 0;
  return config;
}

NormalLevelParameters level_parameters(std::uint32_t level) {
  const auto index = index_for_level(level);
  NormalLevelParameters result;
  result.passive_drain_unit = 0;
  result.scripted_descent_per_update = static_cast<double>(
      std::bit_cast<float>(kDescentBits[index]));
  result.rot_penalty = 600 + 50 * static_cast<std::int64_t>(level / 25U);
  result.maximum_color_id = 5U + level / 25U;
  result.spawn_interval_frames = 120U - 10U * (level % 8U);
  result.score_scale = std::bit_cast<double>(kScoreScaleBits[index]);
  result.clear_reward_unit = 120;
  return result;
}

std::uint32_t ex_spawn_interval(std::uint32_t level) {
  return kExIntervals[index_for_level(level)];
}

std::int64_t score_delta(std::uint32_t level, std::uint32_t group_num,
                         std::uint32_t group_chain, std::uint32_t size_slot) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  const auto parameters = level_parameters(level);
  const auto factor = std::max<std::int64_t>(1, static_cast<std::int64_t>(size_slot) - 4);
  double exact = static_cast<double>(group_num);
  exact *= 0.5;
  exact += 0.5;
  exact *= static_cast<double>(group_chain);
  exact *= parameters.score_scale;
  exact *= static_cast<double>(factor);
  if (!std::isfinite(exact) || exact >= 0x1p63) {
    throw std::overflow_error("Metsu score delta overflow");
  }
  return static_cast<std::int64_t>(exact);
}

}  // namespace irisu::metsu
