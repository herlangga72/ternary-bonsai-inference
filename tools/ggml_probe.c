/* ggml_probe: tiny reference harness that runs one op with ggml on real data.
 *
 * Usage:
 *   ggml_probe rmsnorm <x.bin> <w.bin> <eps> <out.bin>
 *       x,w are f32 arrays of N floats; writes RMSNorm(x)*w to out.bin.
 *   ggml_probe pq2row <row.bin> <ne0> <x.bin>
 *       row.bin is one PQ2_0 row (ceil(ne0/128)*34 bytes); prints the dot
 *       product of the dequantized row with x to stdout.
 *
 * Built against the PrismML llama.cpp fork's static ggml libs.
 */
#include "ggml.h"
#include "ggml-cpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *load_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(n > 0 ? (size_t) n : 1);
    if (n > 0 && fread(buf, 1, (size_t) n, f) != (size_t) n) { exit(2); }
    fclose(f);
    if (out_len) *out_len = (size_t) n;
    return buf;
}

static void write_file(const char *path, const void *data, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(2); }
    if (len && fwrite(data, 1, len, f) != len) { exit(2); }
    fclose(f);
}

static struct ggml_tensor *load_f32(struct ggml_context *ctx, const char *path, int64_t n, int n_dims) {
    size_t len = 0;
    unsigned char *buf = load_file(path, &len);
    if (len < (size_t) n * sizeof(float)) { fprintf(stderr, "short file %s\n", path); exit(2); }
    struct ggml_tensor *t;
    if (n_dims == 1) {
        t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    } else {
        t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
    }
    memcpy(ggml_get_data(t), buf, (size_t) n * sizeof(float));
    free(buf);
    return t;
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    const char *op = argv[1];

    if (strcmp(op, "rmsnorm") == 0 && argc == 6) {
        const char *xpath = argv[2];
        const char *wpath = argv[3];
        float eps = (float) atof(argv[4]);
        const char *outpath = argv[5];

        // discover n from the x file
        size_t xlen = 0;
        unsigned char *xbuf = load_file(xpath, &xlen);
        int64_t n = (int64_t) (xlen / sizeof(float));
        free(xbuf);

        struct ggml_init_params ip = { .mem_size = 64 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
        struct ggml_context *ctx = ggml_init(ip);
        struct ggml_tensor *x = load_f32(ctx, xpath, n, 1);
        struct ggml_tensor *w = load_f32(ctx, wpath, n, 1);
        struct ggml_tensor *nrm = ggml_rms_norm(ctx, x, eps);
        struct ggml_tensor *y = ggml_mul(ctx, nrm, w);

        struct ggml_cgraph *gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        if (ggml_graph_compute_with_ctx(ctx, gf, 1) != 0) {
            fprintf(stderr, "ggml compute failed\n");
            return 3;
        }
        write_file(outpath, ggml_get_data(y), (size_t) n * sizeof(float));
        ggml_free(ctx);
        return 0;
    }

    if (strcmp(op, "pq2row") == 0 && argc == 5) {
        const char *rowpath = argv[2];
        int64_t ne0 = atoll(argv[3]);
        const char *xpath = argv[4];

        size_t rowlen = 0;
        unsigned char *rowbuf = load_file(rowpath, &rowlen);

        struct ggml_init_params ip = { .mem_size = 64 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
        struct ggml_context *ctx = ggml_init(ip);
        struct ggml_tensor *wq = ggml_new_tensor_2d(ctx, GGML_TYPE_PQ2_0, ne0, 1);
        memcpy(ggml_get_data(wq), rowbuf, rowlen);
        free(rowbuf);
        struct ggml_tensor *x = load_f32(ctx, xpath, ne0, 2);
        struct ggml_tensor *y = ggml_mul_mat(ctx, wq, x);

        struct ggml_cgraph *gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        if (ggml_graph_compute_with_ctx(ctx, gf, 1) != 0) {
            fprintf(stderr, "ggml compute failed\n");
            return 3;
        }
        float *res = (float *) ggml_get_data(y);
        printf("%.9g\n", res[0]);
        ggml_free(ctx);
        return 0;
    }

    if (strcmp(op, "pq2deq") == 0 && argc == 5) {
        const char *rowpath = argv[2];
        int64_t ne0 = atoll(argv[3]);
        const char *outpath = argv[4];

        size_t rowlen = 0;
        unsigned char *rowbuf = load_file(rowpath, &rowlen);

        struct ggml_init_params ip = { .mem_size = 64 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
        struct ggml_context *ctx = ggml_init(ip);
        struct ggml_tensor *wq = ggml_new_tensor_2d(ctx, GGML_TYPE_PQ2_0, ne0, 1);
        memcpy(ggml_get_data(wq), rowbuf, rowlen);
        free(rowbuf);

        const struct ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_PQ2_0);
        float * out = malloc((size_t) ne0 * sizeof(float));
        tt->to_float(ggml_get_data(wq), out, ne0);
        write_file(outpath, out, (size_t) ne0 * sizeof(float));
        free(out);
        ggml_free(ctx);
        return 0;
    }

    if (strcmp(op, "gdn") == 0 && argc == 10) {
        const char * outpath = argv[2];
        const char * qpath   = argv[3];
        const char * kpath   = argv[4];
        const char * vpath   = argv[5];
        const char * gpath   = argv[6];
        const char * betapath= argv[7];
        const char * statepath = argv[8];
        int64_t K = atoll(argv[9]);

        const int64_t S = 128, H_k = 16, H_v = 48, nt = 1, ns = 1;

        struct ggml_init_params ip = { .mem_size = 512 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
        struct ggml_context * ctx = ggml_init(ip);

        struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, nt, ns);
        struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, nt, ns);
        struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_v, nt, ns);
        struct ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H_v, nt, ns);
        struct ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H_v, nt, ns);
        struct ggml_tensor * st = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H_v, ns);

        size_t l = 0; unsigned char * b;
        b = load_file(qpath, &l); memcpy(ggml_get_data(q), b, (size_t) S*H_k*nt*ns*4); free(b);
        b = load_file(kpath, &l); memcpy(ggml_get_data(k), b, (size_t) S*H_k*nt*ns*4); free(b);
        b = load_file(vpath, &l); memcpy(ggml_get_data(v), b, (size_t) S*H_v*nt*ns*4); free(b);
        b = load_file(gpath, &l); memcpy(ggml_get_data(g), b, (size_t) H_v*4); free(b);
        b = load_file(betapath, &l); memcpy(ggml_get_data(beta), b, (size_t) H_v*4); free(b);
        b = load_file(statepath, &l); memcpy(ggml_get_data(st), b, (size_t) S*S*H_v*ns*4); free(b);

        struct ggml_tensor * y = ggml_gated_delta_net(ctx, q, k, v, g, beta, st, K);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        if (ggml_graph_compute_with_ctx(ctx, gf, 1) != 0) { return 3; }
        write_file(outpath, ggml_get_data(y), ggml_nbytes(y));
        ggml_free(ctx);
        return 0;
    }

    if (strcmp(op, "rope") == 0 && argc == 8) {
        const char * outpath = argv[2];
        const char * inpath  = argv[3];
        int pt = atoi(argv[4]);
        int ph = atoi(argv[5]);
        int pw = atoi(argv[6]);
        int pe = atoi(argv[7]);

        const int64_t head_dim = 256, n_head = 1, n_dims = 64;
        int sections[4] = { 11, 11, 10, 0 };

        size_t l = 0;
        unsigned char * inb = load_file(inpath, &l);

        struct ggml_init_params ip = { .mem_size = 64 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, n_head);
        memcpy(ggml_get_data(a), inb, (size_t) head_dim * n_head * 4);
        free(inb);
        int32_t pos[4] = { pt, ph, pw, pe };
        struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        memcpy(ggml_get_data(b), pos, sizeof(pos));

        struct ggml_tensor * y = ggml_rope_multi(ctx, a, b, NULL,
                n_dims, sections, GGML_ROPE_TYPE_IMROPE,
                /*n_ctx_orig=*/4096, /*freq_base=*/1e7f, /*freq_scale=*/1.0f,
                /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                /*beta_fast=*/32.0f, /*beta_slow=*/1.0f);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        if (ggml_graph_compute_with_ctx(ctx, gf, 1) != 0) { return 3; }
        write_file(outpath, ggml_get_data(y), ggml_nbytes(y));
        ggml_free(ctx);
        return 0;
    }

    fprintf(stderr, "unknown usage\n");
    return 1;
}
