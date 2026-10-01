/*
 * src/protocol.h
 * Binary protocol I/O shared by every pipeline.
 *
 * Created: 2026-03-10
 * Authors: Maxence Morel Dierckx, Claude Opus 4.6, Claude Opus 5.5
 */

#ifndef PROTOCOL_H
#define PROTOCOL_H


#include <cstdio>
#include <cstdint>


// Read one sample from stdin. Returns 1 on success, 0 on EOF.
inline int read_sample(float* sample, int n) {
    return fread(sample, sizeof(float), n, stdin) == (size_t)n;
}


// Write flag 0x01 followed by output floats, then flush.
inline void write_output(const float* sample, int n) {
    uint8_t flag = 0x01;
    fwrite(&flag, 1, 1, stdout);
    fwrite(sample, sizeof(float), n, stdout);
    fflush(stdout);
}


// Write flag 0x00 (no output), then flush.
inline void write_skip(void) {
    uint8_t flag = 0x00;
    fwrite(&flag, 1, 1, stdout);
    fflush(stdout);
}


#endif
