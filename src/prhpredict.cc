/*
 * src/prhpredict.cc
 * Tag orientation with no model: method 1 of Mark Johnson's prhpredictor.m
 * (WHOI), made causal and streaming. Line references below are to that file.
 *
 * Usage: prhpredict [--surface-depth M] [--breath-depth M] [--dive-depth M]
 *                   [--descent-samples N] [--min-breath N] [--min-aniso X]
 *
 * Compile-time switches, all off by default:
 *   -DSINGLE_PRECISION  all arithmetic in float instead of double
 *   -DVERBOSE           trace emissions and rejections to stderr
 *   -DSTATS             print per-state sample counts at EOF
 *
 * Created: 2026-09-02
 * Authors: Maxence Morel Dierckx, Claude Opus 5, Claude Opus 5.5
 */


#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>


#include "protocol.h"


#ifdef SINGLE_PRECISION
typedef float real;
#else
typedef double real;
#endif
#define R(x) static_cast<real>(x)


static const real PI = R(3.14159265358979323846);


#define N_INPUT   4
#define N_OUTPUT  3


#ifdef VERBOSE
#define TRACE(...) fprintf(stderr, __VA_ARGS__)
#else
#define TRACE(...) ((void)0)
#endif


#ifdef STATS
#define STAT(x) do { x; } while (0)
#else
#define STAT(x) ((void)0)
#endif


// Depth thresholds, metres.
static float SURFACE_DEPTH = 5.0f;
static float BREATH_DEPTH  = 3.0f;
static float DIVE_DEPTH    = 10.0f;


// Sample counts assume 5 Hz.
static int DESCENT_SAMPLES = 100;   // 20 s
static int MIN_BREATH      = 25;    // 5 s


// Minimum contrast (max - min) / (max + min) of the lateral-energy curve in h.
// Replaces the planarity ratio cc (line 217). 0 disables rejection.
static real MIN_ANISO = R(0.67);


enum { ST_DIVE = 0, ST_SHALLOW, ST_BREATH, ST_DESCENT };


// Scatter matrix storage order.
enum { XX = 0, YY, ZZ, XY, XZ, YZ };


typedef struct {
    int      st;
    size_t   sample_idx;

    // Logging segment Ak1 (line 95): A below BREATH_DEPTH.
    real     a_sum[3];
    uint32_t n_breath;

    // Diving segment Ak2 (line 96). Summed in the tag frame and rotated at solve
    // time, which equals rotating each sample (line 189) without knowing p0/r0.
    real     p0, r0;
    int      descent_n;
    real     S[6];          // sum of a a^T
    real     D[3];          // sum of dz a
    float    prev_depth;
    int      have_prev;

#ifdef STATS
    uint64_t n_state[4], n_descent_call, n_begin, n_solve, n_write;
#endif
} State;


// makeT.m: T = H*P*R, applied aw = T*a.
static void make_T(real p, real r, real h, real T[3][3])
{
    const real cp = std::cos(p), sp = std::sin(p);
    const real cr = std::cos(r), sr = std::sin(r);
    const real ch = std::cos(h), sh = std::sin(h);
    T[0][0] = ch * cp; T[0][1] = -sh * cr - ch * sp * sr; T[0][2] =  sh * sr - ch * sp * cr;
    T[1][0] = sh * cp; T[1][1] =  ch * cr - sh * sp * sr; T[1][2] = -ch * sr - sh * sp * cr;
    T[2][0] = sp;      T[2][1] =  cp * sr;                T[2][2] =  cp * cr;
}


// a2pr.m
static void a2pr(const real a[3], real *p, real *r)
{
    const real n = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    const real x = (n > R(0)) ? a[0] / n : R(0);
    *p = std::asin(x < R(-1) ? R(-1) : (x > R(1) ? R(1) : x));
    *r = std::atan2(a[1], a[2]);
}


// u^T S v for the packed symmetric S.
static real quad(const real S[6], const real u[3], const real v[3])
{
    return u[0] * (S[XX] * v[0] + S[XY] * v[1] + S[XZ] * v[2])
         + u[1] * (S[XY] * v[0] + S[YY] * v[1] + S[YZ] * v[2])
         + u[2] * (S[XZ] * v[0] + S[YZ] * v[1] + S[ZZ] * v[2]);
}


static real dot(const real u[3], const real v[3])
{
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}


// Lines 191-201.
static real solve_h(real Sxx, real Syy, real Sxy)
{
    real h2 = std::atan2(R(2) * Sxy, Syy - Sxx);
    if ((Sxx - Syy) * std::cos(h2) - R(2) * Sxy * std::sin(h2) < R(0))
        h2 += PI;                                   // landed on the maximum
    return h2 / R(2);
}


static void descent_reset(State *s)
{
    s->descent_n = 0;
    s->have_prev = 0;
    for (int i = 0; i < 6; i++) s->S[i] = R(0);
    for (int i = 0; i < 3; i++) s->D[i] = R(0);
}


static void logging_reset(State *s)
{
    s->a_sum[0] = s->a_sum[1] = s->a_sum[2] = R(0);
    s->n_breath = 0;
}


static void descent_sample(State *s, const float a[3], float depth)
{
    STAT(s->n_descent_call++);

    const real x = R(a[0]), y = R(a[1]), z = R(a[2]);
    s->S[XX] += x * x; s->S[YY] += y * y; s->S[ZZ] += z * z;
    s->S[XY] += x * y; s->S[XZ] += x * z; s->S[YZ] += y * z;

    if (s->have_prev) {
        const real dz = R(depth) - R(s->prev_depth);
        s->D[0] += dz * x; s->D[1] += dz * y; s->D[2] += dz * z;
    }
    s->prev_depth = depth;
    s->have_prev = 1;
}


// Solve, resolve the 180 deg ambiguity, emit. Returns 1 if a correction was
// written, 0 if the sample still owes a skip byte.
static int emit(State *s)
{
    STAT(s->n_solve++);

    // Rows 0 and 1 of makeT([p0 r0 0]) (line 188).
    real Q[3][3];
    make_T(s->p0, s->r0, R(0), Q);
    const real Sxx = quad(s->S, Q[0], Q[0]);
    const real Syy = quad(s->S, Q[1], Q[1]);
    const real Sxy = quad(s->S, Q[0], Q[1]);

    const real tr = Sxx + Syy;
    const real aniso = tr > R(0)
        ? std::sqrt((Syy - Sxx) * (Syy - Sxx) + R(4) * Sxy * Sxy) / tr : R(0);
    if (aniso < MIN_ANISO) {
        TRACE("reject idx=%zu aniso=%.4f\n", s->sample_idx, (double)aniso);
        return 0;
    }

    real h0 = solve_h(Sxx, Syy, Sxy);

    // Replaces the single-sample pitch test (lines 161-164): keep the solution
    // whose whale-frame x anti-correlates with depth rate over the whole segment.
    const real corr = std::cos(h0) * dot(Q[0], s->D) - std::sin(h0) * dot(Q[1], s->D);
    if (corr >= R(0))
        h0 = std::remainder(h0 - PI, R(2) * PI);

    const float out[N_OUTPUT] = { (float)s->p0, (float)s->r0, (float)h0 };
    write_output(out, N_OUTPUT);
    STAT(s->n_write++);
    TRACE("emit idx=%zu p=%.1f r=%.1f h=%.1f aniso=%.4f n_log=%u dir=%+.3f\n",
          s->sample_idx, (double)(s->p0 * R(180) / PI),
          (double)(s->r0 * R(180) / PI), (double)(h0 * R(180) / PI),
          (double)aniso, s->n_breath, (double)corr);
    return 1;
}


// Fix p0/r0 from the logging mean and start the descent. Returns 0 if the
// surfacing had too few logging samples.
static int begin_descent(State *s)
{
    if (s->n_breath < (uint32_t)MIN_BREATH)
        return 0;
    STAT(s->n_begin++);
    a2pr(s->a_sum, &s->p0, &s->r0);
    descent_reset(s);
    return 1;
}


static char defaults[128];


static void usage(void)
{
    fprintf(stderr,
        "Usage: prhpredict [--surface-depth M] [--breath-depth M] [--dive-depth M]\n"
        "                  [--descent-samples N] [--min-breath N] [--min-aniso X]\n"
        "Defaults: %s\n", defaults);
}


int main(int argc, const char *argv[])
{
    snprintf(defaults, sizeof(defaults),
             "surface %g m, breath %g m, dive %g m, descent %d, min-breath %d, min-aniso %g",
             (double)SURFACE_DEPTH, (double)BREATH_DEPTH, (double)DIVE_DEPTH,
             DESCENT_SAMPLES, MIN_BREATH, (double)MIN_ANISO);

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has = i + 1 < argc;
        if      (has && !strcmp(a, "--surface-depth"))   SURFACE_DEPTH   = atof(argv[++i]);
        else if (has && !strcmp(a, "--breath-depth"))    BREATH_DEPTH    = atof(argv[++i]);
        else if (has && !strcmp(a, "--dive-depth"))      DIVE_DEPTH      = atof(argv[++i]);
        else if (has && !strcmp(a, "--descent-samples")) DESCENT_SAMPLES = atoi(argv[++i]);
        else if (has && !strcmp(a, "--min-breath"))      MIN_BREATH      = atoi(argv[++i]);
        else if (has && !strcmp(a, "--min-aniso"))       MIN_ANISO       = R(atof(argv[++i]));
        else { usage(); return 1; }
    }
    if (DESCENT_SAMPLES < 2 || MIN_BREATH < 1 || MIN_ANISO < R(0) || MIN_ANISO > R(1) ||
        !(BREATH_DEPTH < SURFACE_DEPTH && SURFACE_DEPTH < DIVE_DEPTH)) {
        usage();
        fprintf(stderr, "Need descent >= 2, min-breath >= 1, 0 <= min-aniso <= 1, "
                        "breath < surface < dive.\n");
        return 1;
    }

    State s;
    memset(&s, 0, sizeof(s));
    logging_reset(&s);
    s.st = ST_DIVE;

    float sample[N_INPUT];
    while (read_sample(sample, N_INPUT)) {
        const float depth = sample[N_INPUT - 1];
        const float *a = sample;
        int wrote = 0;

        STAT(s.n_state[s.st]++);

        // Transition first, so the sample that crosses a threshold is
        // processed by the state it enters.
        switch (s.st) {
        case ST_DIVE:
            if (depth < BREATH_DEPTH)       s.st = ST_BREATH;
            else if (depth < SURFACE_DEPTH) s.st = ST_SHALLOW;
            break;
        case ST_SHALLOW:
        case ST_BREATH:
            if (depth > DIVE_DEPTH)
                s.st = begin_descent(&s) ? ST_DESCENT : (logging_reset(&s), ST_DIVE);
            else if (depth < BREATH_DEPTH)  s.st = ST_BREATH;
            else if (depth > BREATH_DEPTH)  s.st = ST_SHALLOW;
            break;
        case ST_DESCENT:                    // aborted dive keeps the logging sum
            if (depth < BREATH_DEPTH)       s.st = ST_BREATH;
            else if (depth < SURFACE_DEPTH) s.st = ST_SHALLOW;
            break;
        }

        if (s.st == ST_BREATH) {
            s.a_sum[0] += R(a[0]); s.a_sum[1] += R(a[1]); s.a_sum[2] += R(a[2]);
            s.n_breath++;
        } else if (s.st == ST_DESCENT) {
            descent_sample(&s, a, depth);
            if (++s.descent_n >= DESCENT_SAMPLES) {
                wrote = emit(&s);
                logging_reset(&s);
                s.st = ST_DIVE;
            }
        }

        if (!wrote)
            write_skip();
        s.sample_idx++;
    }

#ifdef STATS
    fprintf(stderr,
            "census total=%zu dive=%llu shallow=%llu breath=%llu descentstate=%llu "
            "descentcall=%llu begin=%llu solve=%llu write=%llu\n",
            s.sample_idx,
            (unsigned long long)s.n_state[ST_DIVE],
            (unsigned long long)s.n_state[ST_SHALLOW],
            (unsigned long long)s.n_state[ST_BREATH],
            (unsigned long long)s.n_state[ST_DESCENT],
            (unsigned long long)s.n_descent_call,
            (unsigned long long)s.n_begin,
            (unsigned long long)s.n_solve,
            (unsigned long long)s.n_write);
#endif

    return 0;
}
