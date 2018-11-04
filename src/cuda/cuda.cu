#include "namespace.h"
#include "cuda/cuda.h"
#include <cfloat>

static VG_INLINE uint rand32() {
#ifdef __CUDA_ARCH__
	return Variegator::Cuda::rand32();
#else
	return rand();
#endif
}
#include "variations.h"

namespace Variegator {

namespace Cuda {

__device__ ulong lcg[N * BLOCKS];

__device__ uint rand32() {
	ulong bits = lcg[threadIdx.x + blockIdx.x * blockDim.x];
	bits *= 0x5DEECE66D;
	bits += 11;
	lcg[threadIdx.x + blockIdx.x * blockDim.x] = bits & 0xFFFFFFFFFFFFULL;
	return bits >> 16;
}

__global__ void seed_lcg(ulong seed) {
	for (unsigned i = 0; i < N * BLOCKS; ++i) {
		lcg[i] = seed + i;
	}
}

template<typename T>
__device__ void iterate(const Xform<T>* xforms, uint n_xforms, Vec3<T>* points) {
	const Xform<T>* x = &xforms[rand32() % n_xforms];

	Vec3<T> point;
	point.x = points[0].x * x->mtx[0].x + points[0].y * x->mtx[0].y + points[0].z * x->mtx[0].z + x->affine.x;
	point.y = points[0].x * x->mtx[1].x + points[0].y * x->mtx[1].y + points[0].z * x->mtx[1].z + x->affine.y;
	point.z = points[0].x * x->mtx[2].x + points[0].y * x->mtx[2].y + points[0].z * x->mtx[2].z + x->affine.z;

	Vec3<T> out;
	uint n_variations = x->n_variations;
	for (unsigned j = 0; j < n_variations; ++j) {
		const Variation<T>* v = &x->variations[j];
		Vec3<T> tmp = v->fn(x, v, point);
		out.x += v->weight.x * tmp.x;
		out.y += v->weight.y * tmp.y;
		out.z += v->weight.z * tmp.z;
	}
	points[0] = out;

	Vec3<T> color;
	color.x = (points[1].x + x->color.x) / 2.f;
	color.y = (points[1].y + x->color.y) / 2.f;
	color.z = (points[1].z + x->color.z) / 2.f;
	points[1] = color;
}

template<typename T>
__device__ void initialize_points(Vec3<T>* points) {
	points[0].x = (T) (int) rand32() / (T) 0x80000000;
	points[0].y = (T) (int) rand32() / (T) 0x80000000;
	points[0].z = (T) (int) rand32() / (T) 0x80000000;
	points[1].x = (T) rand32() / (T) 0x100000000LL;
	points[1].y = (T) rand32() / (T) 0x100000000LL;
	points[1].z = (T) rand32() / (T) 0x100000000LL;
}

template<typename T>
__device__ void update_board(const BoardParams* params, const Vec3<T>* points, float4* board) {
	int x = (int) (points[0].x * params->proj[0][0] + points[0].y * params->proj[1][0] + points[0].z * params->proj[2][0] + params->affine.x);
	int y = (int) (points[0].x * params->proj[0][1] + points[0].y * params->proj[1][1] + points[0].z * params->proj[2][1] + params->affine.y);
	if (x < 0 || x >= params->w || y < 0 || y >= params->h) {
		return;
	}
	Vec3<T> colors = points[1];
	uchar3 color = make_uchar3(colors.x * 256, colors.y * 256, colors.z * 256);
	atomicAdd(&board[x + y * params->w].x, params->palette[color.x].x);
	atomicAdd(&board[x + y * params->w].y, params->palette[color.y].y);
	atomicAdd(&board[x + y * params->w].z, params->palette[color.z].z);
	atomicAdd(&board[x + y * params->w].w, 1);
}

template<typename U>
__global__ void flatten_board(const BoardParams* params, const float4* board, U* out) {
	unsigned maxv = 0xFFFFFFFFU >> (32 - 8 * sizeof(U));
	float k1 = maxv * params->brightness * 268.f / 256.f;
	float k2 = 1.f / (params->quality);
	uint2 yy = {
		(blockIdx.y * blockDim.y + threadIdx.y) * params->h / (blockDim.y * gridDim.y),
		(blockIdx.y * blockDim.y + threadIdx.y + 1) * params->h / (blockDim.y * gridDim.y)
	};
	uint2 xx = {
		(blockIdx.x * blockDim.x + threadIdx.x) * params->w / (blockDim.x * gridDim.x),
		(blockIdx.x * blockDim.x + threadIdx.x + 1) * params->w / (blockDim.x * gridDim.x)
	};
	for (unsigned y = yy.x; y < yy.y; ++y) {
		for (unsigned x = xx.x; x < xx.y; ++x) {
			float4 dot = board[x + y * params->w];
			float alpha = log10f(dot.w * k2 + 1) * k1 / (dot.w + FLT_EPSILON);
			out[4 * (x + y * params->w) + 3] = min(maxv, (uint) alpha);
			out[4 * (x + y * params->w) + 0] = min(maxv, (uint) (dot.x * alpha));
			out[4 * (x + y * params->w) + 1] = min(maxv, (uint) (dot.y * alpha));
			out[4 * (x + y * params->w) + 2] = min(maxv, (uint) (dot.z * alpha));
		}
	}
}

template<typename T>
__global__ void run(Xform<T>* xforms, unsigned n_xforms, const BoardParams* board_params, Vec3<T>* points, float4* board, ulong steps) {
	if (threadIdx.x == 0) {
		for (unsigned i = 0; i < n_xforms; ++i) {
			for (unsigned j = 0; j < xforms[i].n_variations; ++j) {
				xforms[i].variations[j].fn = Variations<T>::find(xforms[i].variations[j].id);
			}
		}
	}
	__syncthreads();
	points = &points[(threadIdx.x + blockIdx.x * blockDim.x) * 2];
	initialize_points(points);
	for (unsigned i = 0; i < 20; ++i) {
		iterate(xforms, n_xforms, points);
	}
	for (ulong i = 0; i < steps; ++i) {
		iterate(xforms, n_xforms, points);
		update_board(board_params, points, board);
	}
}

template __global__ void run<float>(Xform<float>*, unsigned, const BoardParams*, Vec3<float>*, float4*, ulong);
template __global__ void run<double>(Xform<double>*, unsigned, const BoardParams*, Vec3<double>*, float4*, ulong);

template __global__ void flatten_board<unsigned short>(const BoardParams* params, const float4* board, unsigned short* out);
template __global__ void flatten_board<unsigned char>(const BoardParams* params, const float4* board, unsigned char* out);

}
}
