#pragma once

#ifdef __CUDA_ARCH__
#define VG_INLINE __device__ __forceinline__
#else
#define VG_INLINE inline
#endif

namespace Variegator {

template<typename T>
struct Vec3 {
	T x = 0;
	T y = 0;
	T z = 0;
};

template<typename T> struct Xform;
template<typename T>
struct Variation {
	int id;
	Vec3<T> weight;
	T params[4];
	Vec3<T> (*fn)(struct Xform<T>* x, Variation<T>* v, Vec3<T> point);
};

template<typename T>
struct Xform {
	Vec3<T> mtx[3];
	Vec3<T> affine;
	Vec3<float> color;
	unsigned n_variations;
	struct Variation<T> variations[32];
};

struct BoardParams {
	uint w;
	uint h;
	float proj[3][3];
	Vec3<float> affine;
	Vec3<float> palette[256];
};

}
