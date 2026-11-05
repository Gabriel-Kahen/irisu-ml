#include "../../clone/cuda_native/factory.hpp"
#include <bit>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
using namespace irisu::cuda_native;
int main(){try{
  unsigned checks=0;ShapeState a{},b{};a.radius=b.radius=1;b.position={1,0};
  Manifold initial{};initial.points[0].normal_impulse=7;initial.points[1].id=0xdeadbeef;
  const auto rejected=[&](ShapeState x,ShapeState y,Status expected){
    auto out=initial;const auto result=collide(out,x,y);
    if(result.status!=expected||std::memcmp(&out,&initial,sizeof(out)))
      throw std::runtime_error("collision rejection changed state or status");
    ++checks;
  };
  auto invalid=a;invalid.radius=-1;rejected(invalid,b,Status::invalid_shape);
  invalid=a;invalid.radius=std::numeric_limits<float>::infinity();rejected(invalid,b,Status::unsupported_numeric);
  invalid=a;invalid.position.x=std::numeric_limits<float>::quiet_NaN();rejected(invalid,b,Status::unsupported_numeric);
  invalid=a;invalid.position.x=std::numeric_limits<float>::max();auto distant=b;distant.position.x=-invalid.position.x;
  rejected(invalid,distant,Status::unsupported_numeric);
  invalid=a;invalid.kind=ShapeKind::polygon;rejected(invalid,b,Status::invalid_shape);
  invalid.vertex_count=9;rejected(invalid,b,Status::invalid_shape);
  invalid.vertex_count=3;invalid.normals[0].x=std::numeric_limits<float>::quiet_NaN();rejected(invalid,b,Status::unsupported_numeric);
  auto polygon=a;polygon.kind=ShapeKind::polygon;polygon.vertex_count=3;
  rejected(a,polygon,Status::unsupported_collision); // Registry owns native pair ordering.
  auto out=initial;auto far=b;far.position={3,0};
  if(!collide(out,a,far)||out.point_count!=0||out.points[0].normal_impulse!=7||out.points[1].id!=0xdeadbeef)
    throw std::runtime_error("no-contact result lost preserved fields");
  ++checks;
  BodyDefinition bd{};ShapeDefinition sd{};BodyState body{};ShapeState shape{};
  unsigned char before_body[sizeof(body)],before_shape[sizeof(shape)];
  body.position={17,23};shape.radius=19;std::memcpy(before_body,&body,sizeof(body));std::memcpy(before_shape,&shape,sizeof(shape));sd.radius=-1;
  if(construct_body_shape(body,shape,bd,sd).status!=Status::invalid_shape||
     std::memcmp(&body,before_body,sizeof(body))||std::memcmp(&shape,before_shape,sizeof(shape)))
    throw std::runtime_error("factory rejection changed physical state");
  ++checks;
  if(wrapper_definition(bd,sd,0,1,1,0,0,0,1,1,0,0).status!=Status::invalid_shape)
    throw std::runtime_error("zero magnification accepted");
  ++checks;
  std::printf("{\"pass\":true,\"domain_checks\":%u,\"transactional_collision_and_factory_rejection\":true}\n",checks);
  return 0;
}catch(const std::exception& e){std::fprintf(stderr,"domain: %s\n",e.what());return 1;}}
