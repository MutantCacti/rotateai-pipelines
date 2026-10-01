/*
 * src/prhpredict.cc
 * Tag orientation with no model: method 1 of Mark Johnson's prhpredictor.m
 * (WHOI), made causal and streaming. Line references below are to that file.
 *
 * Usage: prhpredict [--surface-depth M] [--breath-depth M] [--dive-depth M]
 *                   [--descent-samples N] [--min-breath N] [--log-samples N]
 *                   [--ascent-samples N] [--ascent-blocks N] [--min-ascent N]
 *                   [--min-aniso X] [--min-aniso-up X] [--no-ascents]
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
#include "prh.h"


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
static int LOG_SAMPLES     = 450;   // 90 s, the logging segment [-100 -10] s (line 52)
static int ASCENT_SAMPLES  = 200;   // 40 s before the surface edge (lines 52, 84); 0: all
static int MIN_ASCENT      = 25;    // 5 s


// The ascent's start is only known once it reaches the surface, so it is kept
// as a ring of block sums and the last ASCENT_SAMPLES are approximated by
// whole blocks.
#define MAX_BLOCKS 8
static int ASCENT_BLOCKS = 4;
static int ASCENTS = 1;             // --no-ascents: never wake for one


// Minimum contrast (max - min) / (max + min) of the lateral-energy curve in h.
// Replaces the planarity ratio cc (line 217). 0 disables rejection.
static real MIN_ANISO = R(0.67);
static real MIN_ANISO_UP = R(0.85);  // ascents are less often planar


enum { ST_DIVE = 0, ST_SHALLOW, ST_BREATH, ST_DESCENT, ST_ASCENT };


typedef struct {
    int      st;
    size_t   sample_idx;

    // Logging segment Ak1 (line 95): A below BREATH_DEPTH.
    real     a_sum[3];
    uint32_t n_breath;

    Segment  descent;

    // An ascent pairs with the logging segment after it (lines 84-86), so it
    // waits for that before it can be solved.
    Segment  ascent[MAX_BLOCKS];
    int      ascent_cur;
    int      ascent_pending;
    float    wake_depth;    // where the descent window ended; 0 when unset

#ifdef STATS
    uint64_t n_state[5], n_segment_add, n_solve, n_write;
#endif
} State;


static void add_sample(State *s, Segment *g, const float a[3], float depth)
{
    STAT(s->n_segment_add++);
    (void)s;
    segment_add(g, a, depth);
}


static void ascent_reset(State *s)
{
    for (int i = 0; i < ASCENT_BLOCKS; i++) segment_reset(&s->ascent[i]);
    s->ascent_cur = 0;
}


static void ascent_add(State *s, const float a[3], float depth)
{
    Segment *g = &s->ascent[s->ascent_cur];
    if (ASCENT_SAMPLES && g->n == ASCENT_SAMPLES / ASCENT_BLOCKS) {
        const float prev = g->prev_depth;
        s->ascent_cur = (s->ascent_cur + 1) % ASCENT_BLOCKS;
        g = &s->ascent[s->ascent_cur];
        segment_reset(g);
        g->prev_depth = prev;
        g->have_prev = 1;
    }
    add_sample(s, g, a, depth);
}


static void ascent_total(const State *s, Segment *t)
{
    segment_reset(t);
    for (int b = 0; b < ASCENT_BLOCKS; b++) {
        for (int i = 0; i < 6; i++) t->S[i] += s->ascent[b].S[i];
        for (int i = 0; i < 3; i++) t->D[i] += s->ascent[b].D[i];
        t->n += s->ascent[b].n;
    }
}


static void logging_reset(State *s)
{
    s->a_sum[0] = s->a_sum[1] = s->a_sum[2] = R(0);
    s->n_breath = 0;
}


// Solve a segment against the current logging mean and emit. Returns 1 if a
// correction was written, 0 if the sample still owes a skip byte.
static int emit(State *s, const Segment *g, const char *dir, real min_aniso)
{
    STAT(s->n_solve++);
    (void)dir;

    real p0, r0;
    a2pr(s->a_sum, &p0, &r0);

    // Rows 0 and 1 of makeT([p0 r0 0]) (line 188).
    real Q[3][3];
    make_T(p0, r0, R(0), Q);
    const real Sxx = quad(g->S, Q[0], Q[0]);
    const real Syy = quad(g->S, Q[1], Q[1]);
    const real Sxy = quad(g->S, Q[0], Q[1]);

    const real aniso = contrast(Sxx, Syy, Sxy);
    if (aniso < min_aniso) {
        TRACE("reject idx=%zu %s aniso=%.4f\n", s->sample_idx, dir, (double)aniso);
        return 0;
    }

    // The branch rule holds for ascents too, where nose-up meets decreasing depth.
    real corr;
    const real h0 = resolve_branch(solve_h(Sxx, Syy, Sxy), Q, g->D, &corr);

    const float out[N_OUTPUT] = { (float)p0, (float)r0, (float)h0 };
    write_output(out, N_OUTPUT);
    STAT(s->n_write++);
    TRACE("emit idx=%zu %s p=%.1f r=%.1f h=%.1f aniso=%.4f n=%d n_log=%u dir=%+.3f\n",
          s->sample_idx, dir, (double)(p0 * R(180) / PI), (double)(r0 * R(180) / PI),
          (double)(h0 * R(180) / PI), (double)aniso, g->n, s->n_breath, (double)corr);
    return 1;
}


static int emit_ascent(State *s)
{
    Segment t;
    ascent_total(s, &t);
    s->ascent_pending = 0;
    return emit(s, &t, "up", MIN_ANISO_UP);
}


static char defaults[256];


static void usage(void)
{
    fprintf(stderr,
        "Usage: prhpredict [--surface-depth M] [--breath-depth M] [--dive-depth M]\n"
        "                  [--descent-samples N] [--min-breath N] [--log-samples N]\n"
        "                  [--ascent-samples N] [--ascent-blocks N] [--min-ascent N]\n"
        "                  [--min-aniso X] [--min-aniso-up X] [--no-ascents]\n"
        "Defaults: %s\n", defaults);
}


int main(int argc, const char *argv[])
{
    snprintf(defaults, sizeof(defaults),
             "surface %g m, breath %g m, dive %g m, descent %d, min-breath %d, "
             "log-samples %d, ascent-samples %d, ascent-blocks %d, min-ascent %d, "
             "min-aniso %g, min-aniso-up %g, ascents on",
             (double)SURFACE_DEPTH, (double)BREATH_DEPTH, (double)DIVE_DEPTH,
             DESCENT_SAMPLES, MIN_BREATH, LOG_SAMPLES, ASCENT_SAMPLES, ASCENT_BLOCKS,
             MIN_ASCENT, (double)MIN_ANISO, (double)MIN_ANISO_UP);

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has = i + 1 < argc;
        if      (has && !strcmp(a, "--surface-depth"))   SURFACE_DEPTH   = atof(argv[++i]);
        else if (has && !strcmp(a, "--breath-depth"))    BREATH_DEPTH    = atof(argv[++i]);
        else if (has && !strcmp(a, "--dive-depth"))      DIVE_DEPTH      = atof(argv[++i]);
        else if (has && !strcmp(a, "--descent-samples")) DESCENT_SAMPLES = atoi(argv[++i]);
        else if (has && !strcmp(a, "--min-breath"))      MIN_BREATH      = atoi(argv[++i]);
        else if (has && !strcmp(a, "--log-samples"))     LOG_SAMPLES     = atoi(argv[++i]);
        else if (has && !strcmp(a, "--ascent-samples"))  ASCENT_SAMPLES  = atoi(argv[++i]);
        else if (has && !strcmp(a, "--ascent-blocks"))   ASCENT_BLOCKS   = atoi(argv[++i]);
        else if (has && !strcmp(a, "--min-ascent"))      MIN_ASCENT      = atoi(argv[++i]);
        else if (has && !strcmp(a, "--min-aniso-up"))    MIN_ANISO_UP    = R(atof(argv[++i]));
        else if (!strcmp(a, "--no-ascents"))             ASCENTS         = 0;
        else if (has && !strcmp(a, "--min-aniso"))       MIN_ANISO       = R(atof(argv[++i]));
        else { usage(); return 1; }
    }
    if (DESCENT_SAMPLES < 2 || MIN_BREATH < 1 || LOG_SAMPLES < MIN_BREATH ||
        MIN_ASCENT < 2 || ASCENT_BLOCKS < 1 || ASCENT_BLOCKS > MAX_BLOCKS ||
        (ASCENT_SAMPLES && ASCENT_SAMPLES < ASCENT_BLOCKS) || MIN_ANISO_UP > R(1) ||
        MIN_ANISO < R(0) || MIN_ANISO > R(1) || MIN_ANISO_UP < R(0) ||
        !(BREATH_DEPTH < SURFACE_DEPTH && SURFACE_DEPTH < DIVE_DEPTH)) {
        usage();
        fprintf(stderr, "Need descent >= 2, 1 <= min-breath <= log-samples, min-ascent >= 2, "
                        "1 <= ascent-blocks <= %d, ascent-samples 0 or >= ascent-blocks, "
                        "min-aniso and min-aniso-up in [0, 1], breath < surface < dive.\n", MAX_BLOCKS);
        return 1;
    }

    State s;
    memset(&s, 0, sizeof(s));
    logging_reset(&s);
    s.st = ST_DIVE;

    // Same key and format the TFLite pipelines use for their tensor arena
    // (pipeline.h), so the simulator reports memory for this pipeline too.
    // State is the whole working set: there is no heap allocation here, and
    // the ascent ring is a fixed MAX_BLOCKS array.
    fprintf(stderr, "arena_used_bytes:%zu\n", sizeof(State));

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
            if (s.wake_depth > 0.0f && depth < s.wake_depth && depth > SURFACE_DEPTH) {
                ascent_reset(&s);
                s.st = ST_ASCENT;
            }
            else if (depth < BREATH_DEPTH)  s.st = ST_BREATH;
            else if (depth < SURFACE_DEPTH) s.st = ST_SHALLOW;
            break;
        case ST_ASCENT:
            if (depth > s.wake_depth)
                s.st = ST_DIVE;             // dived again: restart on the next rise
            else if (depth < SURFACE_DEPTH) {   // the surface edge (line 84)
                Segment t;
                ascent_total(&s, &t);
                s.ascent_pending = t.n >= MIN_ASCENT;
                s.wake_depth = 0.0f;
                s.st = depth < BREATH_DEPTH ? ST_BREATH : ST_SHALLOW;
            }
            break;
        case ST_SHALLOW:
        case ST_BREATH:
            s.wake_depth = 0.0f;
            if (depth > DIVE_DEPTH) {
                if (s.n_breath >= (uint32_t)MIN_BREATH) {
                    if (s.ascent_pending)
                        wrote = emit_ascent(&s);
                    segment_reset(&s.descent);
                    s.st = ST_DESCENT;
                } else {
                    logging_reset(&s);
                    s.st = ST_DIVE;
                }
                s.ascent_pending = 0;
            }
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
            if (++s.n_breath == (uint32_t)LOG_SAMPLES && s.ascent_pending)
                wrote = emit_ascent(&s);
        } else if (s.st == ST_ASCENT) {
            ascent_add(&s, a, depth);
        } else if (s.st == ST_DESCENT) {
            add_sample(&s, &s.descent, a, depth);
            if (s.descent.n >= DESCENT_SAMPLES) {
                wrote = emit(&s, &s.descent, "down", MIN_ANISO);
                logging_reset(&s);
                // A wake depth of 0 leaves ST_ASCENT unreachable
                s.wake_depth = !ASCENTS ? 0.0f : depth > DIVE_DEPTH ? depth : DIVE_DEPTH;
                s.st = ST_DIVE;
            }
        }

        if (!wrote)
            write_skip();
        s.sample_idx++;
    }

#ifdef STATS
    fprintf(stderr,
            "census total=%zu dive=%llu shallow=%llu breath=%llu descent=%llu ascent=%llu "
            "segadd=%llu solve=%llu write=%llu\n",
            s.sample_idx,
            (unsigned long long)s.n_state[ST_DIVE],
            (unsigned long long)s.n_state[ST_SHALLOW],
            (unsigned long long)s.n_state[ST_BREATH],
            (unsigned long long)s.n_state[ST_DESCENT],
            (unsigned long long)s.n_state[ST_ASCENT],
            (unsigned long long)s.n_segment_add,
            (unsigned long long)s.n_solve,
            (unsigned long long)s.n_write);
#endif

    return 0;
}
