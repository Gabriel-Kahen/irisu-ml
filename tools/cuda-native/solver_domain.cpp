#include "../../clone/cuda_native/solver.hpp"
#include <cassert>
#include <limits>
namespace n=irisu::cuda_native;
int main(){
 n::BodyState bodies[2]{};n::ContactConstraint c{};c.body1=0;c.body2=1;c.point_count=1;
 assert(n::solver_velocity(bodies,2,&c,1));
 c.body2=2;assert(n::solver_velocity(bodies,2,&c,1).status==n::Status::invalid_body);
 c.body2=1;c.point_count=3;assert(n::solver_velocity(bodies,2,&c,1).status==n::Status::invalid_snapshot);
 c.point_count=1;c.normal.x=std::numeric_limits<float>::quiet_NaN();assert(n::solver_velocity(bodies,2,&c,1).status==n::Status::unsupported_numeric);
 n::Manifold manifold{};manifold.point_count=1;n::SolverContact contact{0,1,0,.3f,.1f};unsigned count=9;
 assert(n::solver_build_constraints(bodies,2,&manifold,1,&contact,1,&c,0,count).status==n::Status::capacity);
 assert(n::solver_build_constraints(bodies,2,&manifold,1,&contact,1,&c,1,count).status==n::Status::unsupported_numeric); // Both static: zero effective mass.
 n::math::Context m;const float z=m.store(-0.0);assert(n::math::finite(z)&&irisu::cuda_exact::bits_float(z)==0x80000000U&&m.ok());
 assert(!n::math::div(1,0).supported);assert(!n::math::sqrt(-1).supported);assert(!n::math::store(std::numeric_limits<double>::infinity()).supported);
}
