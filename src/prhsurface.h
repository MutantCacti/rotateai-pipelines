/*
 * src/prhsurface.h
 * The surfacing solver behind prhsurface: p0 and r0 from the logging mean
 * (prhpredictor.m method 1, line 185), h0 from the surfacing's plane of motion.
 * Reads no input and writes no output; the pipeline feeds it samples.
 *
 * Created: 2026-11-02
 * Authors: Maxence Morel Dierckx, Claude Opus 5.5
 */

#ifndef PRHSURFACE_H
#define PRHSURFACE_H


#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>


#include "prh.h"


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
// |corr| per heading sample: how clearly depth rate picks the 180 deg branch.
static real MIN_BRANCH_CONF = R(0.003);


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
// ST_BREATH, as for prhedge's logging segment; the heading segment takes the
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


// Solve the current windows. Returns 1 if out holds a correction.
static int surfacing_solve(State *s, float out[3])
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
        TRACE("reject idx=%zu surface cause=aniso aniso=%.4f n_log=%u n_head=%d\n",
              s->sample_idx, (double)aniso, lw.n_log, hw.head.n);
        return 0;
    }

    real corr;
    const real h0 = resolve_branch(solve_h(Sxx, Syy, Sxy), Q, hw.head.D, &corr);

    // Normalised, so the gate does not depend on the segment's length.
    const real conf = std::fabs(corr) / R(hw.head.n);
    if (conf < MIN_BRANCH_CONF) {
        TRACE("reject idx=%zu surface cause=branch conf=%.5f aniso=%.4f n_log=%u n_head=%d\n",
              s->sample_idx, (double)conf, (double)aniso, lw.n_log, hw.head.n);
        return 0;
    }

    out[0] = (float)p0; out[1] = (float)r0; out[2] = (float)h0;
    STAT(s->n_write++);
    TRACE("emit idx=%zu surface p=%.1f r=%.1f h=%.1f aniso=%.4f n_log=%u n_head=%d corr=%+.3f "
          "conf=%.5f\n",
          s->sample_idx, (double)(p0 * R(180) / PI), (double)(r0 * R(180) / PI),
          (double)(h0 * R(180) / PI), (double)aniso, lw.n_log, hw.head.n, (double)corr,
          (double)conf);
    return 1;
}


// One option from argv[*i], taking its value. Returns 1 if it was one of ours.
static int surface_arg(int argc, const char *argv[], int *i)
{
    const char *a = argv[*i];
    if (*i + 1 >= argc) return 0;
    const char *v = argv[*i + 1];
    if      (!strcmp(a, "--surface-depth"))   SURFACE_DEPTH   = atof(v);
    else if (!strcmp(a, "--breath-depth"))    BREATH_DEPTH    = atof(v);
    else if (!strcmp(a, "--dive-depth"))      DIVE_DEPTH      = atof(v);
    else if (!strcmp(a, "--heading-depth"))   HEADING_DEPTH   = atof(v);
    else if (!strcmp(a, "--emit-depth"))      EMIT_DEPTH      = atof(v);
    else if (!strcmp(a, "--min-breath"))      MIN_BREATH      = atoi(v);
    else if (!strcmp(a, "--block-samples"))   BLOCK_SAMPLES   = atoi(v);
    else if (!strcmp(a, "--log-blocks"))      LOG_BLOCKS      = atoi(v);
    else if (!strcmp(a, "--heading-blocks"))  HEADING_BLOCKS  = atoi(v);
    else if (!strcmp(a, "--refresh-samples")) REFRESH_SAMPLES = atoi(v);
    else if (!strcmp(a, "--min-aniso"))       MIN_ANISO       = R(atof(v));
    else if (!strcmp(a, "--min-branch-conf")) MIN_BRANCH_CONF = R(atof(v));
    else return 0;
    (*i)++;
    return 1;
}


// Resolve the defaults that follow other options. Returns 1 if the options hold.
static int surface_args_ok(void)
{
    if (HEADING_DEPTH < 0.0f) HEADING_DEPTH = BREATH_DEPTH;
    if (EMIT_DEPTH < 0.0f) EMIT_DEPTH = DIVE_DEPTH;
    return MIN_BREATH >= 1 && BLOCK_SAMPLES >= 1 && REFRESH_SAMPLES >= 0 &&
           LOG_BLOCKS >= 0 && LOG_BLOCKS <= MAX_BLOCKS &&
           HEADING_BLOCKS >= 0 && HEADING_BLOCKS <= MAX_BLOCKS &&
           MIN_ANISO >= R(0) && MIN_ANISO <= R(1) && MIN_BRANCH_CONF >= R(0) &&
           BREATH_DEPTH < SURFACE_DEPTH && SURFACE_DEPTH < DIVE_DEPTH &&
           HEADING_DEPTH > 0.0f && EMIT_DEPTH >= SURFACE_DEPTH;
}


static void surface_init(State *s)
{
    memset(s, 0, sizeof(*s));
    surfacing_reset(s);
    s->st = ST_DIVE;
}


// One input sample. Returns 1 if out holds a correction.
static int surface_step(State *s, const float a[3], float depth, float out[3])
{
    int wrote = 0;

    STAT(s->n_state[s->st]++);

    // Transition first, so the sample that crosses a threshold is
    // processed by the state it enters. Leaving the surface ends the
    // surfacing: solve it, then start the next one empty.
    switch (s->st) {
    case ST_DIVE:
        if (depth < BREATH_DEPTH)       s->st = ST_BREATH;
        else if (depth < SURFACE_DEPTH) s->st = ST_SHALLOW;
        break;
    case ST_SHALLOW:
    case ST_BREATH:
        if (depth > EMIT_DEPTH) {
            wrote = surfacing_solve(s, out);
            surfacing_reset(s);
            s->st = ST_DIVE;
        }
        else if (depth < BREATH_DEPTH)  s->st = ST_BREATH;
        else if (depth > BREATH_DEPTH)  s->st = ST_SHALLOW;
        break;
    }

    if (s->st != ST_DIVE) {
        surfacing_sample(s, a, depth, s->st == ST_BREATH);
        if (REFRESH_SAMPLES && s->since_emit >= REFRESH_SAMPLES) {
            wrote = surfacing_solve(s, out);
            s->since_emit = 0;
        }
    }

    s->sample_idx++;
    return wrote;
}


#endif
