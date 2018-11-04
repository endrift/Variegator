#pragma once
#include "namespace.h"

#define POINTS 4
#define N 1024
#define BLOCKS 16

namespace Variegator {
namespace Cuda {

__device__ uint rand32();
__global__ void seed_lcg(ulong seed);
template<typename U> __global__ void flatten_board(BoardParams* params, float4* board, U* out);
template<typename T> __global__ void run(Xform<T>* xforms, unsigned n_xforms, BoardParams* board_params, Vec3<T>* points, float4* board, ulong steps);

}
}
