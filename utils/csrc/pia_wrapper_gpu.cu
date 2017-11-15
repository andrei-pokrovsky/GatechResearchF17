#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "pia_wrapper_gpu.h"

/* PIA */

typedef struct { float x, y; } point_t;

/*
  pia.c

  Calculate the area of intersection of a pair of polygons specified
  as and array of pairs of floats representing the polygon vertices.
  The return value is a float and accurate to float precision.
  Degenerate cases are avoided by working in exact arithmetic and
  "fudging" the exact coordinates to an extent smaller than float
  precision.

  -------

  This code is a derived from Norman Hardy's aip.c which can be
  downloaded from

    http://www.cap-lore.com/MathPhys/IP/

  In my view, an astonishing piece of algorithmic craftsmanship.

  My original intention was simply to convert it ANSI C (C89 at the
  time), but I found that I needed to reformat and "dumb it down"
  just so that I could understand what it was that I was converting.

  The main changes are:

  - renamed to pia.c to avoid any confusion with the original
  - added header file
  - renamed types *_t
  - pulled all nested function out, passing required variables by
    reference as needed
  - added lots of unnecessary parentheses to emphasis precedence
  - removed scope restraining blocks
  - lots of stylistic changes
  - some assumptions about the promotion of floats to doubles for
    intermediate calculation have been made explicit (these assumptions
    are true for gcc on x86, but are not guaranteed by standard and are
    false for gcc on amd64)
  - use integer types with explicit size from stdint.h, use size_t for
    array indices, used booleans from stdbool.h

  This is now C99 (according to gcc -Wall -std=c99)

  --------

  J.J. Green 2010, 2015
*/

#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct { point_t min, max; } box_t;

typedef struct { int32_t x, y; } ipoint_t;

typedef struct { int32_t mn, mx; } rng_t;

typedef struct {
    ipoint_t ip;
    rng_t rx, ry;
    int16_t in;
} vertex_t;

__device__ static void bd(float *X, float y) {
    if (*X >= y)
	*X = y;
}

__device__ static void bu(float *X, float y) {
    if (*X <= y)
	*X = y;
}

__device__ static void range(const point_t *x, size_t c, box_t *B) {
    for (int i = c - 1; i >= 0; --i) {
	bd(&(B->min.x), x[i].x);
	bu(&(B->max.x), x[i].x);
	bd(&(B->min.y), x[i].y);
	bu(&(B->max.y), x[i].y);
    }
}

__device__ static int64_t area(ipoint_t a, ipoint_t p, ipoint_t q) {
    return (int64_t)p.x * q.y - (int64_t)p.y * q.x +
	   (int64_t)a.x * (p.y - q.y) + (int64_t)a.y * (q.x - p.x);
}

__device__ static float dma(double a, double b, double c) { return a * b + c; }

/*
  Allowing this function to be inlined causes a unit-test failure (the
  'cross' test) for gcc 4.4.7 on x86 Linux, I'm not sure why, possibly
  a gcc bug?

  19/12/2016, I find that this issue is not present in gcc 4.9.2, so
  this suppression of inlining is restricted to versions earlier than
  this.
*/

__device__ static void fit(const point_t *__restrict__ x, size_t cx,
			   vertex_t *__restrict__ ix, int32_t fudge, float sclx,
			   float scly, float mid, box_t B) {
    int32_t c = cx;

    while (c--) {
	ix[c].ip.x =
	    ((int32_t)dma(x[c].x - B.min.x, sclx, -mid) & ~7) | fudge | (c & 1);

	ix[c].ip.y = ((int32_t)dma(x[c].y - B.min.y, scly, -mid) & ~7) | fudge;
    }

    ix[0].ip.y += (cx & 1);
    ix[cx].ip = ix[0].ip;

    c = cx;

    while (c--) {
	if (ix[c].ip.x < ix[c + 1].ip.x)
	    ix[c].rx = (rng_t){ix[c].ip.x, ix[c + 1].ip.x};
	else
	    ix[c].rx = (rng_t){ix[c + 1].ip.x, ix[c].ip.x};

	if (ix[c].ip.y < ix[c + 1].ip.y)
	    ix[c].ry = (rng_t){ix[c].ip.y, ix[c + 1].ip.y};
	else
	    ix[c].ry = (rng_t){ix[c + 1].ip.y, ix[c].ip.y};

	ix[c].in = 0;
    }
}

__device__ static void contrib(ipoint_t f, ipoint_t t, int16_t w,
			       int64_t *__restrict__ s) {
    *s += (int64_t)w * (t.x - f.x) * (t.y + f.y) / 2;
}

__device__ static bool ovl(rng_t p, rng_t q) {
    return (p.mn < q.mx) && (q.mn < p.mx);
}

__device__ static void
cross(vertex_t *__restrict__ a, const vertex_t *__restrict__ b,
      vertex_t *__restrict__ c, const vertex_t *__restrict__ d, double a1,
      double a2, double a3, double a4, int64_t *__restrict__ s) {
    float r1 = a1 / ((float)a1 + a2), r2 = a3 / ((float)a3 + a4);

    ipoint_t pA = {a->ip.x + r1 * (b->ip.x - a->ip.x),
		   a->ip.y + r1 * (b->ip.y - a->ip.y)};
    contrib(pA, b->ip, 1, s);

    ipoint_t pB = {c->ip.x + r2 * (d->ip.x - c->ip.x),
		   c->ip.y + r2 * (d->ip.y - c->ip.y)};
    contrib(d->ip, pB, 1, s);

    ++a->in;
    --c->in;
}

__device__ static void inness(const vertex_t *__restrict__ P, size_t cP,
			      const vertex_t *__restrict__ Q, size_t cQ,
			      int64_t *__restrict__ s) {
    int16_t S = 0;
    size_t c = cQ;

    while (c--) {
	ipoint_t p = P[0].ip;

	if ((Q[c].rx.mn < p.x) && (p.x < Q[c].rx.mx)) {
	    bool positive = (0 < area(p, Q[c].ip, Q[c + 1].ip));

	    if (positive == (Q[c].ip.x < Q[c + 1].ip.x))
		S += (positive ? -1 : 1);
	}
    }

    for (size_t j = 0; j < cP; ++j) {
	if (S)
	    contrib(P[j].ip, P[j + 1].ip, S, s);
	S += P[j].in;
    }
}

__device__ float pia_area(const point_t *__restrict__ a,
			  const point_t *__restrict__ b) {

    const size_t na = 4, nb = 4;

    box_t B = {{FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX}};

    range(a, na, &B);
    range(b, nb, &B);

    const float gamut = 500000000.0, mid = gamut / 2.0;
    float rngx = B.max.x - B.min.x, sclx = gamut / rngx,
	  rngy = B.max.y - B.min.y, scly = gamut / rngy;
    vertex_t ipa[na + 1], ipb[nb + 1];

    fit(a, na, ipa, 0, sclx, scly, mid, B);
    fit(b, nb, ipb, 2, sclx, scly, mid, B);

    double ascale = (double)sclx * (double)scly;
    int64_t s = 0;

    for (size_t j = 0; j < na; ++j) {
	for (size_t k = 0; k < nb; ++k) {
	    if (ovl(ipa[j].rx, ipb[k].rx) && ovl(ipa[j].ry, ipb[k].ry)) {
		int64_t a1 = -area(ipa[j].ip, ipb[k].ip, ipb[k + 1].ip),
			a2 = area(ipa[j + 1].ip, ipb[k].ip, ipb[k + 1].ip);

		bool o = (a1 < 0);

		if (o == (a2 < 0)) {
		    int64_t a3 = area(ipb[k].ip, ipa[j].ip, ipa[j + 1].ip),
			    a4 = -area(ipb[k + 1].ip, ipa[j].ip, ipa[j + 1].ip);

		    if ((a3 < 0) == (a4 < 0)) {
			if (o)
			    cross(ipa + j, ipa + j + 1, ipb + k, ipb + k + 1,
				  a1, a2, a3, a4, &s);
			else
			    cross(ipb + k, ipb + k + 1, ipa + j, ipa + j + 1,
				  a3, a4, a1, a2, &s);
		    }
		}
	    }
	}
    }

    inness(ipa, na, ipb, nb, &s);
    inness(ipb, nb, ipa, na, &s);

    return s / ascale;
}

/* END PIA */

__device__ void convert_to_rect(float x, float y, float w, float d, float t,
				point_t *__restrict__ output) {

    const float sinval = sin(t);
    const float cosval = cos(t);

    const float bx_x = w * cosval;
    const float bx_y = d * -sinval;

    const float by_x = w * sinval;
    const float by_y = d * cosval;

    output[0].x = x + bx_x + by_x;
    output[0].y = y + bx_y + by_y;

    output[1].x = x - bx_x + by_x;
    output[1].y = y - bx_y + by_y;

    output[2].x = x - bx_x - by_x;
    output[2].y = y - bx_y - by_y;

    output[3].x = x + bx_x - by_x;
    output[3].y = y + bx_y - by_y;

    /* printf("(%f, %f), (%f, %f), (%f, %f), (%f, %f)\n", output[0].x,
       output[0].y,
	   output[1].x, output[1].y, output[2].x, output[2].y, output[3].x,
	   output[3].y); */
}

// preds(b, n, bpp, 7) boxes(b, nb, 7)
// boxes format: [w, d, h, theta, fx, fy, cz]
// preds format: [w, d, h, theta, fx, fy, cz]
// output: iou_out (b, nb, n, bpp)
__global__ void pia_kernel(int b, int n, int bpp, int nb,
			   const float *__restrict__ preds,
			   const float *__restrict__ boxes,
			   float *__restrict__ iou_out) {

    int batch_index = blockIdx.x;
    preds += batch_index * n * bpp * 7;
    boxes += batch_index * nb * 7;

    iou_out += batch_index * nb * n * bpp;

    int index = threadIdx.x;
    int stride = blockDim.x;

    float scratch[4 * 2 * 2];
    point_t *predicted_rect = (point_t *)scratch;
    point_t *true_rect = ((point_t *)scratch) + 4;

    for (int j = 0; j < nb; ++j) {
	float b_fx = boxes[j * 7 + 4];
	float b_fy = boxes[j * 7 + 5];
	float b_cz = boxes[j * 7 + 6];

	float b_theta = boxes[j * 7 + 3];

	float b_w = max(boxes[j * 7 + 0], 0.0);
	float b_d = max(boxes[j * 7 + 1], 0.0);
	float b_h = max(boxes[j * 7 + 2], 0.0);

	float true_rect_vol = (2.0 * b_w) * (2.0 * b_d) * (2.0 * b_h);

	for (int i = index; i < n; i += stride) {
	    for (int k = 0; k < bpp; ++k) {
		float p_fx = preds[i * (bpp * 7) + k * 7 + 4];
		float p_fy = preds[i * (bpp * 7) + k * 7 + 5];
		float p_cz = preds[i * (bpp * 7) + k * 7 + 6];

		float p_theta = preds[i * (bpp * 7) + k * 7 + 3];

		float p_w = max(preds[i * (bpp * 7) + k * 7 + 0], 0.0);
		float p_d = max(preds[i * (bpp * 7) + k * 7 + 1], 0.0);
		float p_h = max(preds[i * (bpp * 7) + k * 7 + 2], 0.0);

		float predicted_rect_vol = (2.0 * p_w) * (2.0 * p_d) * (2.0 * p_h);

		float z_overlap =
		    min(p_cz + p_h, b_cz + b_h) - max(p_cz - p_h, b_cz - b_h);

		if (z_overlap > 0.0) {
		    convert_to_rect(p_fx, p_fy, p_w, p_d, p_theta,
				    predicted_rect);
		    convert_to_rect(b_fx, b_fy, b_w, b_d, b_theta, true_rect);

		    float overlap_area = pia_area(predicted_rect, true_rect);

		    float intersection = z_overlap * overlap_area;
		    float uni =
			predicted_rect_vol + true_rect_vol - intersection;
		    float iou = intersection / uni;
		    iou_out[j * (n * bpp) + i * bpp + k] = iou;
		} else {
		    iou_out[j * (n * bpp) + i * bpp + k] = 0.0;
		}
	    }
	}
    }
}

void pia_kernel_wrapper(int b, int n, int bpp, int nb, const float *preds,
			const float *boxes, float *iou_out,
			cudaStream_t stream) {

    cudaError_t err;
    pia_kernel<<<b, opt_n_threads(n) / 2, 0, stream>>>(b, n, bpp, nb, preds,
						       boxes, iou_out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
	fprintf(stderr, "PIA CUDA kernel failed : %s\n",
		cudaGetErrorString(err));
	exit(-1);
    }
}
