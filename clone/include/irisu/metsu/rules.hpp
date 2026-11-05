#pragma once

#include "irisu/config.hpp"
#include "irisu/normal_rules.hpp"

#include <cstdint>

namespace irisu::metsu {

MechanicsConfig default_config();
NormalLevelParameters level_parameters(std::uint32_t level);  // 1..99
std::uint32_t ex_spawn_interval(std::uint32_t level);  // 1..99
inline constexpr double ex_scripted_rise = static_cast<double>(-0.05F);
std::int64_t score_delta(std::uint32_t level, std::uint32_t group_num,
                         std::uint32_t group_chain, std::uint32_t size_slot);

}  // namespace irisu::metsu
