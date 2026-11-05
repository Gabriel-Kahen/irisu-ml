#pragma once
#include <stdint.h>
#define SOLVER_INPUT_MAGIC 0x31534e49u
#define SOLVER_OUTPUT_MAGIC 0x31534e4fu
#define SOLVER_BODY_WORDS 12u
#define SOLVER_CONSTRAINT_WORDS 30u
#define SOLVER_MANIFOLD_WORDS 15u
#define SOLVER_CONTACT_WORDS 5u
/* Stages: construct=0, PreSolve=1, velocity=2, position=3, PostSolve=4.
 * Case metadata: stage, bodies, constraints/capacity, manifolds, contacts,
 * warmstart, beta floatbits. Body words: positionxy,rotation,R(col1xy,col2xy),
 * velocityxy,angular,invMass,invI. Constraint and manifold use native physical
 * word order, with manifold/body pointers replaced by packet indices.
 * All integers and floatbits are little-endian32; no process pointers travel.
 */
typedef struct SolverWireCase {uint32_t stage,bodies,constraints,manifolds,contacts,warm,beta;} SolverWireCase;
typedef struct SolverWireResult {uint32_t status,constraints,converged;} SolverWireResult;
