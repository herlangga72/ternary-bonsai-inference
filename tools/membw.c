// Minimal RAM bandwidth probe: repeatedly scan a large buffer summing bytes.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    size_t gb = argc > 1 ? (size_t)atol(argv[1]) : 3;
    size_t n = gb << 30;
    volatile unsigned char* buf = (volatile unsigned char*)aligned_alloc(4096, n);
    if (!buf) { perror("alloc"); return 1; }
    memset((void*)buf, 0xAB, n);

    // warm + fill (also puts pages in RAM)
    double t0 = now_s();
    unsigned long long acc = 0;
    for (int it = 0; it < 2; it++) {
        for (size_t i = 0; i < n; i += 64) acc += buf[i];
    }
    double t1 = now_s();
    printf("scan pass 1+2 (some cold): %.2f GB in %.3fs -> %.2f GB/s (acc %llu)\n",
           (double)n, t1 - t0, (double)n * 2 / (t1 - t0), acc);

    // fully hot passes
    t0 = now_s();
    acc = 0;
    for (int it = 0; it < 6; it++) {
        for (size_t i = 0; i < n; i += 64) acc += buf[i];
    }
    t1 = now_s();
    printf("scan hot: 6x%.2f GB in %.3fs -> %.2f GB/s\n",
           (double)n, t1 - t0, (double)n * 6 / (t1 - t0));

    // touch every byte (true streaming)
    t0 = now_s();
    for (int it = 0; it < 3; it++) {
        for (size_t i = 0; i < n; i += 4096) acc += buf[i];
    }
    t1 = now_s();
    printf("stream (4k stride): 3x%.2f GB in %.3fs -> %.2f GB/s (acc %llu)\n",
           (double)n, t1 - t0, (double)n * 3 / (t1 - t0), acc);
    (void)acc;
    return 0;
}
