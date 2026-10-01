/*
 * src/pipeline.h
 * Shared interface for RotateAI inference pipelines.
 *
 * Provides model loading, op registration and z-score normalization,
 * plus the binary protocol I/O from protocol.h. Pipelines include this
 * and write their own main loop.
 *
 * Created: 2026-03-10
 * Authors: Maxence Morel Dierckx, Claude Opus 4.6
 */

#ifndef PIPELINE_H
#define PIPELINE_H


#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>


#include "protocol.h"


#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


#include "model_params.h"


// Model data embedded at compile time.
const unsigned char model_data[] = {
#include "model_data.inc"
};


// Arena. Sized for the new HART/Transformer model; bump if AllocateTensors fails.
constexpr int kArenaSize = 1024 * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];


// Pipeline state returned by pipeline_init.
struct Pipeline {
    tflite::MicroInterpreter* interpreter;
    float* input;
    float* output;
};


// Initialize model, register ops, allocate tensors.
// Returns Pipeline with pointers to input/output tensors.
// Exits on failure. No recovery on embedded.
inline Pipeline pipeline_init() {
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        fprintf(stderr, "error: model version %lu != %d\n",
                model->version(), TFLITE_SCHEMA_VERSION);
        exit(1);
    }

    static tflite::MicroMutableOpResolver<NUM_OPS> resolver;
    REGISTER_OPS(resolver);

    static tflite::MicroInterpreter interp(model, resolver, tensor_arena, kArenaSize);
    if (interp.AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "error: AllocateTensors failed\n");
        exit(1);
    }

    fprintf(stderr, "arena_used_bytes:%zu\n", interp.arena_used_bytes());

    TfLiteTensor* input = interp.input(0);
    TfLiteTensor* output = interp.output(0);
    if (!input || !output || !input->data.f || !output->data.f) {
        fprintf(stderr, "error: tensor allocation failed\n");
        exit(1);
    }

    return {&interp, input->data.f, output->data.f};
}


// Run inference. Returns 0 on success, 1 on failure.
inline int pipeline_invoke(Pipeline* p) {
    return p->interpreter->Invoke() != kTfLiteOk;
}


// Z-score normalization. No-op when means=={0,...} and stds=={1,...}.
inline void normalize(float* sample, const float* means, const float* stds, int n) {
    for (int i = 0; i < n; i++)
        sample[i] = (sample[i] - means[i]) / stds[i];
}


// Decode N (cos, sin) pairs into N angles via atan2.
// raw layout: cos0, sin0, cos1, sin1, ...
// angles: radians in (-pi, pi].
inline void decode_angles(const float* raw, float* angles, int n_angles) {
    for (int i = 0; i < n_angles; i++)
        angles[i] = atan2f(raw[2*i + 1], raw[2*i]);
}


#endif
