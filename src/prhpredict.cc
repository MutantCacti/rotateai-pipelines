/*
 * src/prhpredict.cc
 * Tag orientation with no model, from surfacings alone: p0 and r0 from the
 * logging mean (prhpredictor.m method 1, line 185), h0 from the surfacing's
 * plane of motion. With p0/r0 fixed, method 2's alignment of the plane normal
 * with y (lines 247-256) is method 1's lateral-energy solve (lines 191-201) on
 * the logging segment. Emits when the animal leaves the surface.
 *
 * Usage: prhpredict [--surface-depth M] [--breath-depth M] [--dive-depth M]
 *                   [--heading-depth M] [--emit-depth M] [--min-breath N]
 *                   [--block-samples N] [--log-blocks N] [--heading-blocks N]
 *                   [--refresh-samples N] [--min-aniso X]
 *
 * Compile-time switches, all off by default:
 *   -DSINGLE_PRECISION  all arithmetic in float instead of double
 *   -DVERBOSE           trace emissions and rejections to stderr
 *   -DSTATS             print per-state sample counts at EOF
 *
 * Created: 2026-10-01
 * Authors: Maxence Morel Dierckx, Claude Opus 5.5
 */


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


// Depth thresholds, metres. Heading and emit depths default to breath and dive.
static float SURFACE_DEPTH = 5.0f;
static float BREATH_DEPTH  = 3.0f;
static float DIVE_DEPTH    = 10.0f;
static float HEADING_DEPTH = -1.0f;
static float EMIT_DEPTH    = -1.0f;


// Sample counts assume 5 Hz.
static int MIN_BREATH      = 50;    // 10 s
static int BLOCK_SAMPLES   = 25;    // 5 s
// Emit every N samples of an unbroken surfacing, so an animal that never dives
// still gets corrections. 12 min only reaches surfacings that long; 0: off.
static int REFRESH_SAMPLES = 3600;


// Windows are the newest N blocks of the surfacing; 0 is the whole surfacing.
// A long surfacing is often milling, so the heading window may want to be
// shorter than the p0/r0 one.
#define MAX_BLOCKS 64
static int LOG_BLOCKS     = 0;
static int HEADING_BLOCKS = 0;


static real MIN_ANISO = R(0.6);


enum { ST_DIVE = 0, ST_SHALLOW, ST_BREATH };


// One block of the surfacing: the logging sum (Ak1, line 95) and the heading
// segment's scatter.
typedef struct {
    real     a_sum[3];
    uint32_t n_log;
    Segment  head;
    int      n;             // surfacing samples in the block
} Block;


typedef struct {
    int      st;
    size_t   sample_idx;

    Block    total;         // the whole surfacing
    Block    ring[MAX_BLOCKS];
    int      cur, filled;
    int      since_emit;

#ifdef STATS
    uint64_t n_state[3], n_log_add, n_head_add, n_solve, n_write;
#endif
} State;


static void block_reset(Block *b)
{
    b->a_sum[0] = b->a_sum[1] = b->a_sum[2] = R(0);
    b->n_log = 0;
    segment_reset(&b->head);
    b->n = 0;
}


static void surfacing_reset(State *s)
{
    block_reset(&s->total);
    for (int i = 0; i < MAX_BLOCKS; i++) block_reset(&s->ring[i]);
    s->cur = s->filled = 0;
    s->since_emit = 0;
}


static void add_to(Block *b, const float a[3], float depth, int log, int head)
{
    if (log) {
        b->a_sum[0] += R(a[0]); b->a_sum[1] += R(a[1]); b->a_sum[2] += R(a[2]);
        b->n_log++;
    }
    if (head)
        segment_add(&b->head, a, depth);
    b->n++;
}


// One surfacing sample, into the whole-surfacing total and the ring. `log` is
// ST_BREATH, as for prhpredict's logging segment; the heading segment takes the
// same samples unless --heading-depth differs from the breath depth.
static void surfacing_sample(State *s, const float a[3], float depth, int log)
{
    const int head = HEADING_DEPTH == BREATH_DEPTH ? log : depth < HEADING_DEPTH;
    STAT(if (log) s->n_log_add++; if (head) s->n_head_add++);

    Block *b = &s->ring[s->cur];
    if (s->filled == 0)
        s->filled = 1;
    else if (b->n == BLOCK_SAMPLES) {
        // Keep the depth rate continuous across the block boundary
        const float prev_depth = b->head.prev_depth;
        const int have_prev = b->head.have_prev;
        s->cur = (s->cur + 1) % MAX_BLOCKS;
        b = &s->ring[s->cur];
        block_reset(b);
        b->head.prev_depth = prev_depth;
        b->head.have_prev = have_prev;
        if (s->filled < MAX_BLOCKS) s->filled++;
    }
    add_to(b, a, depth, log, head);
    add_to(&s->total, a, depth, log, head);
    s->since_emit++;
}


// The newest k blocks (0: the whole surfacing) summed into w.
static void window(const State *s, int k, Block *w)
{
    if (k == 0 || k >= s->filled) {
        if (k == 0 || s->filled < MAX_BLOCKS) { *w = s->total; return; }
    }
    block_reset(w);
    for (int i = 0; i < k && i < s->filled; i++) {
        const Block *b = &s->ring[(s->cur - i + MAX_BLOCKS) % MAX_BLOCKS];
        for (int j = 0; j < 3; j++) w->a_sum[j] += b->a_sum[j];
        w->n_log += b->n_log;
        for (int j = 0; j < 6; j++) w->head.S[j] += b->head.S[j];
        for (int j = 0; j < 3; j++) w->head.D[j] += b->head.D[j];
        w->head.n += b->head.n;
        w->n += b->n;
    }
}


// Solve the current windows and emit. Returns 1 if a correction was written,
// 0 if the sample still owes a skip byte.
static int emit(State *s)
{
    Block lw, hw;
    window(s, LOG_BLOCKS, &lw);
    window(s, HEADING_BLOCKS, &hw);
    if (lw.n_log < (uint32_t)MIN_BREATH || hw.head.n < 2)
        return 0;
    STAT(s->n_solve++);

    real p0, r0;
    a2pr(lw.a_sum, &p0, &r0);

    // Rows 0 and 1 of makeT([p0 r0 0]) (line 188).
    real Q[3][3];
    make_T(p0, r0, R(0), Q);
    const real Sxx = quad(hw.head.S, Q[0], Q[0]);
    const real Syy = quad(hw.head.S, Q[1], Q[1]);
    const real Sxy = quad(hw.head.S, Q[0], Q[1]);

    const real aniso = contrast(Sxx, Syy, Sxy);
    if (aniso < MIN_ANISO) {
        TRACE("reject idx=%zu surface aniso=%.4f n_log=%u n_head=%d\n",
              s->sample_idx, (double)aniso, lw.n_log, hw.head.n);
        return 0;
    }

    real corr;
    const real h0 = resolve_branch(solve_h(Sxx, Syy, Sxy), Q, hw.head.D, &corr);

    const float out[N_OUTPUT] = { (float)p0, (float)r0, (float)h0 };
    write_output(out, N_OUTPUT);
    STAT(s->n_write++);
    TRACE("emit idx=%zu surface p=%.1f r=%.1f h=%.1f aniso=%.4f n_log=%u n_head=%d corr=%+.3f\n",
          s->sample_idx, (double)(p0 * R(180) / PI), (double)(r0 * R(180) / PI),
          (double)(h0 * R(180) / PI), (double)aniso, lw.n_log, hw.head.n, (double)corr);
    return 1;
}


static char defaults[256];


static void usage(void)
{
    fprintf(stderr,
        "Usage: prhpredict [--surface-depth M] [--breath-depth M] [--dive-depth M]\n"
        "                  [--heading-depth M] [--emit-depth M] [--min-breath N]\n"
        "                  [--block-samples N] [--log-blocks N] [--heading-blocks N]\n"
        "                  [--refresh-samples N] [--min-aniso X]\n"
        "Defaults: %s\n", defaults);
}


int main(int argc, const char *argv[])
{
    snprintf(defaults, sizeof(defaults),
             "surface %g m, breath %g m, dive %g m, heading = breath depth, "
             "emit = dive depth, min-breath %d, block-samples %d, log-blocks %d, "
             "heading-blocks %d (0: whole surfacing), refresh-samples %d, min-aniso %g",
             (double)SURFACE_DEPTH, (double)BREATH_DEPTH, (double)DIVE_DEPTH,
             MIN_BREATH, BLOCK_SAMPLES, LOG_BLOCKS, HEADING_BLOCKS, REFRESH_SAMPLES,
             (double)MIN_ANISO);

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has = i + 1 < argc;
        if      (has && !strcmp(a, "--surface-depth"))   SURFACE_DEPTH   = atof(argv[++i]);
        else if (has && !strcmp(a, "--breath-depth"))    BREATH_DEPTH    = atof(argv[++i]);
        else if (has && !strcmp(a, "--dive-depth"))      DIVE_DEPTH      = atof(argv[++i]);
        else if (has && !strcmp(a, "--heading-depth"))   HEADING_DEPTH   = atof(argv[++i]);
        else if (has && !strcmp(a, "--emit-depth"))      EMIT_DEPTH      = atof(argv[++i]);
        else if (has && !strcmp(a, "--min-breath"))      MIN_BREATH      = atoi(argv[++i]);
        else if (has && !strcmp(a, "--block-samples"))   BLOCK_SAMPLES   = atoi(argv[++i]);
        else if (has && !strcmp(a, "--log-blocks"))      LOG_BLOCKS      = atoi(argv[++i]);
        else if (has && !strcmp(a, "--heading-blocks"))  HEADING_BLOCKS  = atoi(argv[++i]);
        else if (has && !strcmp(a, "--refresh-samples")) REFRESH_SAMPLES = atoi(argv[++i]);
        else if (has && !strcmp(a, "--min-aniso"))       MIN_ANISO       = R(atof(argv[++i]));
        else { usage(); return 1; }
    }
    if (HEADING_DEPTH < 0.0f) HEADING_DEPTH = BREATH_DEPTH;
    if (EMIT_DEPTH < 0.0f) EMIT_DEPTH = DIVE_DEPTH;
    if (MIN_BREATH < 1 || BLOCK_SAMPLES < 1 || REFRESH_SAMPLES < 0 ||
        LOG_BLOCKS < 0 || LOG_BLOCKS > MAX_BLOCKS ||
        HEADING_BLOCKS < 0 || HEADING_BLOCKS > MAX_BLOCKS ||
        MIN_ANISO < R(0) || MIN_ANISO > R(1) ||
        !(BREATH_DEPTH < SURFACE_DEPTH && SURFACE_DEPTH < DIVE_DEPTH) ||
        HEADING_DEPTH <= 0.0f || EMIT_DEPTH < SURFACE_DEPTH) {
        usage();
        fprintf(stderr, "Need min-breath >= 1, block-samples >= 1, refresh-samples >= 0, "
                        "0 <= log-blocks and heading-blocks <= %d, 0 <= min-aniso <= 1, "
                        "breath < surface < dive, heading-depth > 0, "
                        "emit-depth >= surface.\n", MAX_BLOCKS);
        return 1;
    }

    State s;
    memset(&s, 0, sizeof(s));
    surfacing_reset(&s);
    s.st = ST_DIVE;

    float sample[N_INPUT];
    while (read_sample(sample, N_INPUT)) {
        const float depth = sample[N_INPUT - 1];
        const float *a = sample;
        int wrote = 0;

        STAT(s.n_state[s.st]++);

        // Transition first, so the sample that crosses a threshold is
        // processed by the state it enters. Leaving the surface ends the
        // surfacing: solve it, then start the next one empty.
        switch (s.st) {
        case ST_DIVE:
            if (depth < BREATH_DEPTH)       s.st = ST_BREATH;
            else if (depth < SURFACE_DEPTH) s.st = ST_SHALLOW;
            break;
        case ST_SHALLOW:
        case ST_BREATH:
            if (depth > EMIT_DEPTH) {
                wrote = emit(&s);
                surfacing_reset(&s);
                s.st = ST_DIVE;
            }
            else if (depth < BREATH_DEPTH)  s.st = ST_BREATH;
            else if (depth > BREATH_DEPTH)  s.st = ST_SHALLOW;
            break;
        }

        if (s.st != ST_DIVE) {
            surfacing_sample(&s, a, depth, s.st == ST_BREATH);
            if (REFRESH_SAMPLES && s.since_emit >= REFRESH_SAMPLES) {
                wrote = emit(&s);
                s.since_emit = 0;
            }
        }

        if (!wrote)
            write_skip();
        s.sample_idx++;
    }

#ifdef STATS
    fprintf(stderr,
            "census total=%zu dive=%llu shallow=%llu breath=%llu logadd=%llu headadd=%llu "
            "solve=%llu write=%llu\n",
            s.sample_idx,
            (unsigned long long)s.n_state[ST_DIVE],
            (unsigned long long)s.n_state[ST_SHALLOW],
            (unsigned long long)s.n_state[ST_BREATH],
            (unsigned long long)s.n_log_add,
            (unsigned long long)s.n_head_add,
            (unsigned long long)s.n_solve,
            (unsigned long long)s.n_write);
#endif

    return 0;
}
