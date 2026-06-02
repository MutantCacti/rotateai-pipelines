/*
 * src/variable.cc
 * Inference every X samples.
 * Usage: variable --offset X
 *
 * Created: 2026-03-10
 * Authors: Maxence Morel Dierckx, Claude Opus 4.6
 */

#include "pipeline.h"

static void print_usage()
{
    fprintf(stderr, "Usage: variable --offset X");
}

int main(int argc, const char *argv[]) 
{
    int offset = -1;
    
    // Parse offset
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--offset") == 0 || strcmp(argv[i], "-o") == 0) && i + 1 < argc) {
            offset = atoi(argv[++i]);
        } else {
            print_usage();
            return 1;
        }
    }

    if (offset == 0) {
        print_usage();
        fprintf(stderr, "Offset must be >= 1.");
        return 1;
    }
    
    // offset of 2 means a gap of 1
    offset--;

    // Run pipeline
    Pipeline p = pipeline_init();

    constexpr int kInputSize = WINDOW_SIZE * INPUT_CHANNELS;

    float window[kInputSize] = {0};
    float sample[INPUT_CHANNELS];

    int counter = 0;

    while (read_sample(sample, INPUT_CHANNELS))
    {
        // advance window
        normalize(sample, INPUT_MEANS, INPUT_STDS, INPUT_CHANNELS);

        memmove(window, window + INPUT_CHANNELS,
            sizeof(float) * (kInputSize - INPUT_CHANNELS));
        memcpy(window + kInputSize - INPUT_CHANNELS, sample,
            sizeof(float) * INPUT_CHANNELS);

        // run inference if offset is reached
        if (counter == 0)
        {
            counter = offset;

            memcpy(p.input, window, sizeof(float) * kInputSize);
            if (pipeline_invoke(&p)) {
                fprintf(stderr, "error: inference failed\n");
                return 1;
            }

            float raw[OUTPUT_RAW_CHANNELS];
            memcpy(raw, p.output + OUTPUT_LAST_ROW_OFFSET,
                sizeof(float) * OUTPUT_RAW_CHANNELS);

            float angles[OUTPUT_CHANNELS];
            decode_angles(raw, angles, OUTPUT_CHANNELS);

            write_output(angles, OUTPUT_CHANNELS);
        }
        else
        {
            counter--;
            write_skip();
        }
    }

    return 0;
}
