#include "namespace.h"
#include "cuda/cuda.h"

static VG_INLINE uint rand32() {
#ifdef __CUDA_ARCH__
	return Variegator::Cuda::rand32();
#else
	return rand();
#endif
}
#include "variations.h"

#define K1 1
#define K2 (1.f/8)

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
__device__ void iterate(Xform<T>* xforms, uint n_xforms, Vec3<T>* points) {
	Xform<T>* x = &xforms[rand32() % n_xforms];
	for (unsigned i = 0; i < POINTS; ++i) {
		Vec3<T> point;
		Vec3<T> color;
		point.x = points[i * 2].x * x->mtx[0].x + points[i * 2].y * x->mtx[0].y + points[i * 2].z * x->mtx[0].z + x->affine.x;
		point.y = points[i * 2].x * x->mtx[1].x + points[i * 2].y * x->mtx[1].y + points[i * 2].z * x->mtx[1].z + x->affine.y;
		point.z = points[i * 2].x * x->mtx[2].x + points[i * 2].y * x->mtx[2].y + points[i * 2].z * x->mtx[2].z + x->affine.z;
		color.x = (points[i * 2 + 1].x + x->color.x) / 2.f;
		color.y = (points[i * 2 + 1].y + x->color.y) / 2.f;
		color.z = (points[i * 2 + 1].z + x->color.z) / 2.f;
		points[i * 2 + 1] = color;

		Vec3<T> out;
		for (unsigned j = 0; j < x->n_variations; ++j) {
			Variation<T>* v = &x->variations[j];
			Vec3<T> tmp = v->fn(x, v, point);
			out.x += v->weight.x * tmp.x;
			out.y += v->weight.y * tmp.y;
			out.z += v->weight.z * tmp.z;
		}
		points[i * 2] = out;
	}
}

template<typename T>
__device__ void initialize_points(Vec3<T>* points) {
	for (unsigned i = 0; i < POINTS; ++i) {
		points[i * 2].x = (T) (int) rand32() / (T) 0x80000000;
		points[i * 2].y = (T) (int) rand32() / (T) 0x80000000;
		points[i * 2].z = (T) (int) rand32() / (T) 0x80000000;
		points[i * 2 + 1].x = (T) rand32() / (T) 0x100000000LL;
		points[i * 2 + 1].y = (T) rand32() / (T) 0x100000000LL;
		points[i * 2 + 1].z = (T) rand32() / (T) 0x100000000LL;
	}
}

template<typename T>
__device__ void update_board(BoardParams* params, Vec3<T>* points, float4* board) {
	for (unsigned i = 0; i < POINTS; ++i) {
		int x = (int) (points[i * 2].x * params->proj[0][0] + points[i * 2].y * params->proj[1][0] + points[i * 2].z * params->proj[2][0] + params->affine.x);
		int y = (int) (points[i * 2].x * params->proj[0][1] + points[i * 2].y * params->proj[1][1] + points[i * 2].z * params->proj[2][1] + params->affine.y);
		if (x < 0 || x >= params->w || y < 0 || y >= params->h) {
			continue;
		}
		Vec3<T> colors = points[i * 2 + 1];
		uchar3 color = make_uchar3(colors.x * 256, colors.y * 256, colors.z * 256);
		board[x + y * params->w].x += params->palette[color.x].x;
		board[x + y * params->w].y += params->palette[color.y].y;
		board[x + y * params->w].z += params->palette[color.z].z;
		board[x + y * params->w].w++;
	}
}

template<typename U>
__global__ void flatten_board(BoardParams* params, float4* board, U* out) {
	for (unsigned y = threadIdx.x * params->h / blockDim.x; y < (threadIdx.x + 1) * params->h / blockDim.x; ++y) {
		for (unsigned x = 0; x < params->w; ++x) {
			float4 dot = board[x + y * params->w];
			unsigned maxv = 0xFFFFFFFFU >> (32 - 8 * sizeof(U));
			float alpha = log1pf(dot.w * K1 / QUALITY) * K2 * maxv;
			out[4 * (x + y * params->w) + 3] = min(maxv, (uint) alpha);
			alpha /= dot.w + 1.f;
			out[4 * (x + y * params->w) + 0] = min(maxv, (uint) (dot.x * alpha));
			out[4 * (x + y * params->w) + 1] = min(maxv, (uint) (dot.y * alpha));
			out[4 * (x + y * params->w) + 2] = min(maxv, (uint) (dot.z * alpha));
		}
	}
}

template<typename T>
__global__ void run(Xform<T>* xforms, unsigned n_xforms, BoardParams* board_params, Vec3<T>* points, float4* board, ulong steps) {
	for (unsigned i = 0; i < n_xforms; ++i) {
		for (unsigned j = 0; j < xforms[i].n_variations; ++j) {
			xforms[i].variations[j].fn = Variations<T>::find(xforms[i].variations[j].id);
		}
	}
	points = &points[(threadIdx.x + blockIdx.x * blockDim.x) * POINTS * 2];
	initialize_points(points);
	for (unsigned i = 0; i < 20; ++i) {
		iterate(xforms, n_xforms, points);
	}
	for (ulong i = 0; i < steps; ++i) {
		iterate(xforms, n_xforms, points);
		update_board(board_params, points, board);
	}
}

template __global__ void run<float>(Xform<float>*, unsigned, BoardParams*, Vec3<float>*, float4*, ulong);
template __global__ void run<double>(Xform<double>*, unsigned, BoardParams*, Vec3<double>*, float4*, ulong);

template __global__ void flatten_board<unsigned short>(BoardParams* params, float4* board, unsigned short* out);
template __global__ void flatten_board<unsigned char>(BoardParams* params, float4* board, unsigned char* out);

}
}
