#pragma once
#include "namespace.h"

namespace Variegator {

template<typename T>
struct Variations {
	static VG_INLINE Vec3<T> linear(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		return point;
	}

	static VG_INLINE Vec3<T> sinusoidal(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		return { sinf(point.x), sinf(point.y) };
	}

	static VG_INLINE Vec3<T> sinusoidal3D(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		return { sinf(point.x), sinf(point.y), sinf(point.z) };
	}

	static VG_INLINE Vec3<T> spherical(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		T r2 = point.x * point.x + point.y * point.y;
		return { point.x / r2, point.y / r2 };
	}

	static VG_INLINE Vec3<T> spherical3D(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		T r3 = point.x * point.x + point.y * point.y + point.z * point.z;
		return { point.x / r3, point.y / r3, point.z / r3 };
	}

	static VG_INLINE Vec3<T> swirl(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		T r2 = point.x * point.x + point.y * point.y;
		T c = cosf(r2);
		T s = sinf(r2);
		return { point.x * s - point.y * c, point.x * c + point.y * s };
	}

	static VG_INLINE Vec3<T> horseshoe(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		T r = sqrtf(point.x * point.x + point.y * point.y);
		return { (point.x - point.y) * (point.x + point.y) / r, 2 * point.x * point.y / r };
	}

	static VG_INLINE Vec3<T> polar(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		T r = sqrtf(point.x * point.x + point.y * point.y);
		T theta = atan2f(point.x, point.y);
		return { theta / (T) M_PI, r - 1 };
	}

	static VG_INLINE Vec3<T> handkerchief(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		T r = sqrtf(point.x * point.x + point.y * point.y);
		T theta = atan2f(point.x, point.y);
		return { r * sinf(theta + r), r * cosf(theta - r) };
	}

	static VG_INLINE Vec3<T> heart(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		T r = sqrtf(point.x * point.x + point.y * point.y);
		T theta = atan2f(point.x, point.y);
		return { r * sinf(theta * r), -cosf(theta * r) };
	}

	static VG_INLINE Vec3<T> disc(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		T r = sqrtf(point.x * point.x + point.y * point.y);
		T theta = atan2f(point.x, point.y);
		return { theta / (T) M_PI * sinf((T) M_PI * r), theta / (T) M_PI * cosf((T) M_PI * r) };
	}

	static VG_INLINE Vec3<T> spiral(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		T r = sqrtf(point.x * point.x + point.y * point.y);
		T theta = atan2f(point.x, point.y);
		return { (cosf(theta) + sinf(r)) / r, (sinf(theta) - cosf(r)) / r };
	}

	static VG_INLINE Vec3<T> hyperbolic(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		T r = sqrtf(point.x * point.x + point.y * point.y);
		T theta = atan2f(point.x, point.y);
		return { sinf(theta) / r, r * cosf(theta) };
	}

	static VG_INLINE Vec3<T> diamond(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		T r = sqrtf(point.x * point.x + point.y * point.y);
		T theta = atan2f(point.x, point.y);
		return { sinf(theta) * cosf(r), cosf(theta) * sinf(r) };
	}

	static VG_INLINE Vec3<T> ex(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		T r = sqrtf(point.x * point.x + point.y * point.y);
		T theta = atan2f(point.x, point.y);
		T p0 = sin(theta + r);
		T p1 = cos(theta - r);
		p0 *= p0 * p0;
		p1 *= p1 * p1;
		return { r * (p0 + p1), r * (p0 - p1) };
	}

	static VG_INLINE Vec3<T> julia(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		T r = powf(point.x * point.x + point.y * point.y, .25f);
		T theta = atan2f(point.x, point.y);
		T omega = (rand32() & 1) * (T) M_PI;
		return { r * cos(theta / 2 + omega), r * sin(theta / 2 + omega) };
	}

	static VG_INLINE Vec3<T> bent(Xform<T>* x, Variation<T>* v, Vec3<T> point) {
		return { point.x >= 0 ? point.x : 2 * point.x, point.y >= 0 ? point.y : point.y / 2 };
	}

	typedef Vec3<T> (* Fn)(Xform<T>*, Variation<T>*, Vec3<T>);
	static VG_INLINE Fn find(unsigned id) {
		constexpr Fn t[]{
			linear,
			sinusoidal,
			spherical,
			swirl,
			horseshoe,
			polar,
			handkerchief,
			heart,
			disc,
			spiral,
			hyperbolic,
			diamond,
			ex,
			julia,
			bent,
			// waves,
			// fisheye,
			// popcorn,
			// exponential,
			// power,
			// cosine,
			// rings,
			// fan,
			// blob,
			// pdj,
			// fan2,
			// rings2,
			// eyefish,
			// bubble,
			// cylinder,
			// perspective,
			// noise,
			// julian,
			// juliascope,
			// blur,
			// gaussian,
			// radialblur,
			// pie,
			// ngon,
			// curl,
			// rectangles,
			// arch,
			// tangent,
			// square,
			// rays,
			// blade,
			// secant,
			// twintrian,
			// cross,
		};
		return t[id];
	};
};

}
