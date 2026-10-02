/*
 * src/prh.h
 * Orientation maths for prhpredict: Mark Johnson's prhpredictor.m (WHOI),
 * method 1. Line references are to that file.
 *
 * -DSINGLE_PRECISION switches every calculation here to float.
 *
 * Created: 2026-10-01
 * Authors: Maxence Morel Dierckx, Claude Opus 5.5
 */

#ifndef PRH_H
#define PRH_H


#include <cmath>


#ifdef SINGLE_PRECISION
typedef float real;
#else
typedef double real;
#endif
#define R(x) static_cast<real>(x)


static const real PI = R(3.14159265358979323846);


// Scatter matrix storage order.
enum { XX = 0, YY, ZZ, XY, XZ, YZ };


// Diving segment Ak2 (line 96), summed in the tag frame. Rotating the sums at
// solve time equals rotating each sample (line 189), so p0/r0 can come later.
typedef struct {
    real  S[6];             // sum of a a^T
    real  D[3];             // sum of dz a
    float prev_depth;
    int   have_prev;
    int   n;
} Segment;


// makeT.m: T = H*P*R, applied aw = T*a.
static inline void make_T(real p, real r, real h, real T[3][3])
{
    const real cp = std::cos(p), sp = std::sin(p);
    const real cr = std::cos(r), sr = std::sin(r);
    const real ch = std::cos(h), sh = std::sin(h);
    T[0][0] = ch * cp; T[0][1] = -sh * cr - ch * sp * sr; T[0][2] =  sh * sr - ch * sp * cr;
    T[1][0] = sh * cp; T[1][1] =  ch * cr - sh * sp * sr; T[1][2] = -ch * sr - sh * sp * cr;
    T[2][0] = sp;      T[2][1] =  cp * sr;                T[2][2] =  cp * cr;
}


// a2pr.m
static inline void a2pr(const real a[3], real *p, real *r)
{
    const real n = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    const real x = (n > R(0)) ? a[0] / n : R(0);
    *p = std::asin(x < R(-1) ? R(-1) : (x > R(1) ? R(1) : x));
    *r = std::atan2(a[1], a[2]);
}


// u^T S v for the packed symmetric S.
static inline real quad(const real S[6], const real u[3], const real v[3])
{
    return u[0] * (S[XX] * v[0] + S[XY] * v[1] + S[XZ] * v[2])
         + u[1] * (S[XY] * v[0] + S[YY] * v[1] + S[YZ] * v[2])
         + u[2] * (S[XZ] * v[0] + S[YZ] * v[1] + S[ZZ] * v[2]);
}


static inline real dot(const real u[3], const real v[3])
{
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}


// Lines 191-201.
static inline real solve_h(real Sxx, real Syy, real Sxy)
{
    real h2 = std::atan2(R(2) * Sxy, Syy - Sxx);
    if ((Sxx - Syy) * std::cos(h2) - R(2) * Sxy * std::sin(h2) < R(0))
        h2 += PI;                                   // landed on the maximum
    return h2 / R(2);
}


// Contrast (max - min) / (max + min) of the lateral-energy curve in h.
// Replaces the planarity ratio cc (line 217).
static inline real contrast(real Sxx, real Syy, real Sxy)
{
    const real tr = Sxx + Syy;
    return tr > R(0)
        ? std::sqrt((Syy - Sxx) * (Syy - Sxx) + R(4) * Sxy * Sxy) / tr : R(0);
}


// Replaces the single-sample pitch test (lines 161-164): of the two solutions
// pi apart, keep the one whose whale-frame x anti-correlates with depth rate
// over the whole segment. Q holds rows 0 and 1 of makeT([p0 r0 0]).
static inline real resolve_branch(real h0, const real Q[3][3], const real D[3], real *corr)
{
    *corr = std::cos(h0) * dot(Q[0], D) - std::sin(h0) * dot(Q[1], D);
    return *corr >= R(0) ? std::remainder(h0 - PI, R(2) * PI) : h0;
}


static inline void segment_reset(Segment *g)
{
    for (int i = 0; i < 6; i++) g->S[i] = R(0);
    for (int i = 0; i < 3; i++) g->D[i] = R(0);
    g->have_prev = 0;
    g->n = 0;
}


static inline void segment_add(Segment *g, const float a[3], float depth)
{
    const real x = R(a[0]), y = R(a[1]), z = R(a[2]);
    g->S[XX] += x * x; g->S[YY] += y * y; g->S[ZZ] += z * z;
    g->S[XY] += x * y; g->S[XZ] += x * z; g->S[YZ] += y * z;

    if (g->have_prev) {
        const real dz = R(depth) - R(g->prev_depth);
        g->D[0] += dz * x; g->D[1] += dz * y; g->D[2] += dz * z;
    }
    g->prev_depth = depth;
    g->have_prev = 1;
    g->n++;
}


#endif
