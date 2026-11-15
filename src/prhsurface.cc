/*
 * src/prhsurface.cc
 * Tag orientation with no model, from surfacings alone: p0 and r0 from the
 * logging mean (prhpredictor.m method 1, line 185), h0 from the surfacing's
 * plane of motion. With p0/r0 fixed, method 2's alignment of the plane normal
 * with y (lines 247-256) is method 1's lateral-energy solve (lines 191-201) on
 * the logging segment. Emits when the animal leaves the surface.
 *
 * Usage: prhsurface [--surface-depth M] [--breath-depth M] [--dive-depth M]
 *                   [--heading-depth M] [--emit-depth M] [--min-breath N]
 *                   [--block-samples N] [--log-blocks N] [--heading-blocks N]
 *                   [--refresh-samples N] [--min-aniso X] [--min-branch-conf X]
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
#include "prhsurface.h"


#define N_INPUT   4
#define N_OUTPUT  3


static char defaults[384];


static void usage(void)
{
    fprintf(stderr,
        "Usage: prhsurface [--surface-depth M] [--breath-depth M] [--dive-depth M]\n"
        "                  [--heading-depth M] [--emit-depth M] [--min-breath N]\n"
        "                  [--block-samples N] [--log-blocks N] [--heading-blocks N]\n"
        "                  [--refresh-samples N] [--min-aniso X] [--min-branch-conf X]\n"
        "Defaults: %s\n", defaults);
}


int main(int argc, const char *argv[])
{
    snprintf(defaults, sizeof(defaults),
             "surface %g m, breath %g m, dive %g m, heading = breath depth, "
             "emit = dive depth, min-breath %d, block-samples %d, log-blocks %d, "
             "heading-blocks %d (0: whole surfacing), refresh-samples %d, min-aniso %g, "
             "min-branch-conf %g",
             (double)SURFACE_DEPTH, (double)BREATH_DEPTH, (double)DIVE_DEPTH,
             MIN_BREATH, BLOCK_SAMPLES, LOG_BLOCKS, HEADING_BLOCKS, REFRESH_SAMPLES,
             (double)MIN_ANISO, (double)MIN_BRANCH_CONF);

    for (int i = 1; i < argc; i++) {
        if (!surface_arg(argc, argv, &i)) { usage(); return 1; }
    }
    if (!surface_args_ok()) {
        usage();
        fprintf(stderr, "Need min-breath >= 1, block-samples >= 1, refresh-samples >= 0, "
                        "0 <= log-blocks and heading-blocks <= %d, 0 <= min-aniso <= 1, "
                        "min-branch-conf >= 0, breath < surface < dive, heading-depth > 0, "
                        "emit-depth >= surface.\n", MAX_BLOCKS);
        return 1;
    }

    State s;
    surface_init(&s);

    float sample[N_INPUT];
    while (read_sample(sample, N_INPUT)) {
        float out[N_OUTPUT];
        if (surface_step(&s, sample, sample[N_INPUT - 1], out))
            write_output(out, N_OUTPUT);
        else
            write_skip();
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
