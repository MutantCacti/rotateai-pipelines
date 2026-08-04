/*
 * src/surface.cc
 * Detects surfacing periods and runs inference
 * on them with four possible strategies.
 *
 * Usage: surface --strategy start/end/bookend/average \
 *   --surface-depth 2 --dive-depth 10 --min-samples 600 --max-samples 3000
 *
 * --max-samples ends a surfacing period after N samples even without a dive,
 * giving a variable-style refresh on deployments that never pass dive-depth.
 * 0 disables it.
 *
 * Strategies:
 *   - start    single window beginning with first sample of the surfacing
 *              period, emitted as soon as that window fills
 *   - end      single window ending with last sample of the surfacing period
 *   - bookend  average of the start and end window (outputted at the end),
 *              collapsing to the end window alone when the two would overlap
 *   - average  average of N windows tiled to cover the surfacing period
 *              (anchored to the start)
 *
 * Created: 2026-03-10
 * Authors: Maxence Morel Dierckx, Claude Opus 4.6, Claude Opus 5
 */

#include "pipeline.h"

static void print_usage()
{
    fprintf(stderr, "Usage: surface --strategy STRAT --surface-depth DEPTH --dive-depth DEPTH [--min-samples INT] [--max-samples INT]\n");
}

// Run inference on window, return raw (cos, sin) pairs from the last prediction row.
// Returns 0 on success, 1 on failure.
static int infer(Pipeline* p, float* window, int kInputSize, float* raw)
{
    memcpy(p->input, window, sizeof(float) * kInputSize);
    if (pipeline_invoke(p)) {
        fprintf(stderr, "error: inference failed\n");
        return 1;
    }

    memcpy(raw, p->output + OUTPUT_LAST_ROW_OFFSET,
           sizeof(float) * OUTPUT_RAW_CHANNELS);
    return 0;
}

// Invoke model on current window, decode raw pairs to angles, and write output.
static int infer_and_write(Pipeline* p, float* window, int kInputSize)
{
    float raw[OUTPUT_RAW_CHANNELS];
    if (infer(p, window, kInputSize, raw))
        return 1;
    float angles[OUTPUT_CHANNELS];
    decode_angles(raw, angles, OUTPUT_CHANNELS);
    write_output(angles, OUTPUT_CHANNELS);
    return 0;
}

// Shift window left and insert sample at end.
static void advance_window(float* window, float* sample, int kInputSize)
{
    memmove(window, window + INPUT_CHANNELS,
            sizeof(float) * (kInputSize - INPUT_CHANNELS));
    memcpy(window + kInputSize - INPUT_CHANNELS, sample,
           sizeof(float) * INPUT_CHANNELS);
}

// Surfacing ends on a dive, or on a forced refresh once max_samples is reached.
static bool surfacing_ended(float depth, int dive_depth, int surface_count, int max_samples)
{
    return depth > dive_depth || (max_samples > 0 && surface_count >= max_samples);
}

// Strategy: start - emit once the first WINDOW_SIZE surface samples are in.
static int run_start(Pipeline* p, int surface_depth, int dive_depth, int max_samples)
{
    constexpr int kInputSize = WINDOW_SIZE * INPUT_CHANNELS;

    float window[kInputSize] = {0};
    float sample[INPUT_CHANNELS];

    bool surfacing = false;
    int surface_count = 0;

    while (read_sample(sample, INPUT_CHANNELS)) {
        float depth = sample[INPUT_CHANNELS - 1];

        normalize(sample, INPUT_MEANS, INPUT_STDS, INPUT_CHANNELS);

        if (!surfacing && depth <= surface_depth) {
            // Rising edge
            surfacing = true;
            surface_count = 0;
        }

        if (surfacing) {
            // Advance window only until full for start
            if (surface_count < WINDOW_SIZE) {
                advance_window(window, sample, kInputSize);
            }
            surface_count++;

            if (surfacing_ended(depth, dive_depth, surface_count, max_samples)) {
                // Falling edge
                surfacing = false;
            }

            if (surface_count == WINDOW_SIZE) {
                if (infer_and_write(p, window, kInputSize))
                    return 1;
                continue;
            }
        }

        write_skip();
    }

    return 0;
}

// Strategy: end - keep advancing window until falling edge.
static int run_end(Pipeline* p, int surface_depth, int dive_depth, int min_samples, int max_samples)
{
    constexpr int kInputSize = WINDOW_SIZE * INPUT_CHANNELS;

    float window[kInputSize] = {0};
    float sample[INPUT_CHANNELS];

    bool surfacing = false;
    int surface_count = 0;

    while (read_sample(sample, INPUT_CHANNELS)) {
        float depth = sample[INPUT_CHANNELS - 1];

        normalize(sample, INPUT_MEANS, INPUT_STDS, INPUT_CHANNELS);

        if (!surfacing && depth <= surface_depth) {
            // Rising edge
            surfacing = true;
            surface_count = 0;
        }

        if (surfacing) {
            // Advance window: always for end, only until full for start
            advance_window(window, sample, kInputSize);
            surface_count++;

            if (surfacing_ended(depth, dive_depth, surface_count, max_samples)) {
                // Falling edge
                surfacing = false;

                if (surface_count >= min_samples) {
                    if (infer_and_write(p, window, kInputSize))
                        return 1;
                    continue;
                }
            }
        }

        write_skip();
    }

    return 0;
}

// Strategy: bookend - average of start and end windows, or the end window
// alone when the surfacing period is too short to hold both.
// Average raw (cos, sin) pairs first, then atan2-decode (circular mean).
static int run_bookend(Pipeline* p,
                       int surface_depth, int dive_depth, int min_samples, int max_samples)
{
    constexpr int kInputSize = WINDOW_SIZE * INPUT_CHANNELS;

    float start_window[kInputSize] = {0};
    float end_window[kInputSize] = {0};
    float sample[INPUT_CHANNELS];

    bool surfacing = false;
    int surface_count = 0;

    while (read_sample(sample, INPUT_CHANNELS)) {
        float depth = sample[INPUT_CHANNELS - 1];

        normalize(sample, INPUT_MEANS, INPUT_STDS, INPUT_CHANNELS);

        if (!surfacing && depth <= surface_depth) {
            // Rising edge
            surfacing = true;
            surface_count = 0;
        }

        if (surfacing) {
            // Start window: advance only until full
            if (surface_count < WINDOW_SIZE) {
                advance_window(start_window, sample, kInputSize);
            }
            // End window: always advance
            advance_window(end_window, sample, kInputSize);
            surface_count++;

            if (surfacing_ended(depth, dive_depth, surface_count, max_samples)) {
                // Falling edge
                surfacing = false;

                if (surface_count >= min_samples) {
                    float raw[OUTPUT_RAW_CHANNELS];
                    if (infer(p, end_window, kInputSize, raw))
                        return 1;

                    // Below two windows' worth the two overlap, so the start
                    // window holds no samples the end window does not.
                    if (surface_count >= 2 * WINDOW_SIZE) {
                        float raw_start[OUTPUT_RAW_CHANNELS];
                        if (infer(p, start_window, kInputSize, raw_start))
                            return 1;
                        for (int i = 0; i < OUTPUT_RAW_CHANNELS; i++)
                            raw[i] = (raw[i] + raw_start[i]) * 0.5f;
                    }

                    float angles[OUTPUT_CHANNELS];
                    decode_angles(raw, angles, OUTPUT_CHANNELS);

                    write_output(angles, OUTPUT_CHANNELS);
                    continue;
                }
            }
        }

        write_skip();
    }

    return 0;
}

// Strategy: average - average of non-overlapping windows across surfacing period.
// Accumulate raw (cos, sin) pairs, then atan2-decode at the end (circular mean).
static int run_average(Pipeline* p,
                       int surface_depth, int dive_depth, int min_samples, int max_samples)
{
    constexpr int kInputSize = WINDOW_SIZE * INPUT_CHANNELS;

    float window[kInputSize] = {0};
    float accum[OUTPUT_RAW_CHANNELS] = {0};
    float sample[INPUT_CHANNELS];

    bool surfacing = false;
    int surface_count = 0;
    int window_pos = 0;   // samples in current window
    int window_count = 0; // completed windows

    while (read_sample(sample, INPUT_CHANNELS)) {
        float depth = sample[INPUT_CHANNELS - 1];

        normalize(sample, INPUT_MEANS, INPUT_STDS, INPUT_CHANNELS);

        if (!surfacing && depth <= surface_depth) {
            // Rising edge
            surfacing = true;
            surface_count = 0;
            window_pos = 0;
            window_count = 0;
            memset(accum, 0, sizeof(accum));
        }

        if (surfacing) {
            advance_window(window, sample, kInputSize);
            surface_count++;
            window_pos++;

            // Window full — run inference and accumulate raw pairs
            if (window_pos == WINDOW_SIZE) {
                float raw[OUTPUT_RAW_CHANNELS];
                if (infer(p, window, kInputSize, raw))
                    return 1;
                for (int i = 0; i < OUTPUT_RAW_CHANNELS; i++)
                    accum[i] += raw[i];
                window_count++;
                window_pos = 0;
            }

            if (surfacing_ended(depth, dive_depth, surface_count, max_samples)) {
                // Falling edge
                surfacing = false;

                if (window_count > 0 && surface_count >= min_samples) {
                    float averaged[OUTPUT_RAW_CHANNELS];
                    for (int i = 0; i < OUTPUT_RAW_CHANNELS; i++)
                        averaged[i] = accum[i] / window_count;

                    float angles[OUTPUT_CHANNELS];
                    decode_angles(averaged, angles, OUTPUT_CHANNELS);

                    write_output(angles, OUTPUT_CHANNELS);
                    continue;
                }
            }
        }

        write_skip();
    }

    return 0;
}

int main(int argc, const char *argv[])
{
    const char *strategy_str = NULL;
    int surface_depth = 0;
    int dive_depth = -1;
    int min_samples = -1;
    int max_samples = 0;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--strategy") == 0 || strcmp(argv[i], "-s") == 0) && i + 1 < argc) {
            strategy_str = argv[++i];
        } else if ((strcmp(argv[i], "--surface-depth") == 0 || strcmp(argv[i], "-u") == 0) && i + 1 < argc) {
            surface_depth = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "--dive-depth") == 0 || strcmp(argv[i], "-d") == 0) && i + 1 < argc) {
            dive_depth = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "--min-samples") == 0 || strcmp(argv[i], "-m") == 0) && i + 1 < argc) {
            min_samples = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "--max-samples") == 0 || strcmp(argv[i], "-x") == 0) && i + 1 < argc) {
            max_samples = atoi(argv[++i]);
        } else {
            print_usage();
            return 1;
        }
    }

    if (!strategy_str || (strcmp(strategy_str, "start") != 0 && strcmp(strategy_str, "end") != 0
    && strcmp(strategy_str, "bookend") != 0 && strcmp(strategy_str, "average") != 0)) {
        print_usage();
        fprintf(stderr, "Strategy must be one of: start, end, bookend, average.\n");
        return 1;
    }

    if (dive_depth < 0) {
        print_usage();
        fprintf(stderr, "Dive depth must be >= 0.\n");
        return 1;
    }

    if (strcmp(strategy_str, "start") == 0 && min_samples >= 0 && min_samples != WINDOW_SIZE) {
        print_usage();
        fprintf(stderr, "Strategy start emits when the window fills; min samples is fixed at %d.\n", WINDOW_SIZE);
        return 1;
    }

    if (min_samples < 0) {
        min_samples = WINDOW_SIZE;
    }

    if (max_samples != 0 && max_samples < WINDOW_SIZE) {
        print_usage();
        fprintf(stderr, "Max samples must be 0 (no maximum) or >= window size (%d).\n", WINDOW_SIZE);
        return 1;
    }

    Pipeline p = pipeline_init();

    if (strcmp(strategy_str, "start") == 0)
        return run_start(&p, surface_depth, dive_depth, max_samples);
    else if (strcmp(strategy_str, "end") == 0)
        return run_end(&p, surface_depth, dive_depth, min_samples, max_samples);
    else if (strcmp(strategy_str, "bookend") == 0)
        return run_bookend(&p, surface_depth, dive_depth, min_samples, max_samples);
    else
        return run_average(&p, surface_depth, dive_depth, min_samples, max_samples);
}
