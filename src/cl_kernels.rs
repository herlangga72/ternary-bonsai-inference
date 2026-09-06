//! OpenCL kernel sources (G0).
//!
//! Plain OpenCL C 1.2 strings, compiled at runtime by `Program::build`. The
//! numeric conventions mirror `kernels.rs` (which stays the reference), so the
//! CPU-vs-GPU row checks in bonsai-opencl compare like-for-like.

#![allow(dead_code)]

/// PQ2_0 (ternary, group-128) matrix-vector: y[gid] = W[base_row + gid] . x.
///
/// Payload layout is the GGUF tensor data: each row is `nblocks * 34` bytes
/// (fp16 scale + 32 code bytes per 128-weight block), rows back to back.
/// One work-item computes one row; `base_row` lets fused projections (q|gate,
/// q|k|v) address a row window inside a shared payload. `y` is indexed by the
/// local work-item id (0-based within the launch), matching how the CPU row
/// slice writes `y[0..n_rows]`.
///
/// This is the correct-first kernel. Perf tuning for RDNA3 (LDS tiling of x,
/// wavefront-wide block unpacking, 16-byte code loads) is a G2 task; the math
/// and launch structure stay the same.
pub const PQ2_MATVEC: &str = concat!(
    "static inline float half_to_float(ushort h) {\n",
    "    uint sign = (uint)(h >> 15) & 1u;\n",
    "    uint exp  = (uint)(h >> 10) & 0x1fu;\n",
    "    uint man  = (uint)h & 0x3ffu;\n",
    "    uint bits;\n",
    "    if (exp == 0u) {\n",
    "        if (man == 0u) {\n",
    "            bits = sign << 31;\n",
    "        } else {\n",
    "            int e = 127 - 15 + 1;\n",
    "            uint m = man;\n",
    "            while ((m & 0x400u) == 0u) { m <<= 1; e -= 1; }\n",
    "            m &= 0x3ffu;\n",
    "            bits = (sign << 31) | ((uint)e << 23) | (m << 13);\n",
    "        }\n",
    "    } else if (exp == 0x1fu) {\n",
    "        bits = (sign << 31) | 0x7f800000u | (man << 13);\n",
    "    } else {\n",
    "        bits = (sign << 31) | ((exp + 127u - 15u) << 23) | (man << 13);\n",
    "    }\n",
    "    return as_float(bits);\n",
    "}\n",
    "\n",
    "__kernel void pq2_matvec(__global const uchar* w,\n",
    "                         __global const float* x,\n",
    "                         __global float* y,\n",
    "                         uint ne0,\n",
    "                         uint base_row) {\n",
    "    uint gid = (uint)get_global_id(0);\n",
    "    uint r = gid + base_row;\n",
    "    uint nblocks = (ne0 + 127u) / 128u;\n",
    "    uint row_bytes = nblocks * 34u;\n",
    "    const __global uchar* raw = w + (size_t)r * row_bytes;\n",
    "\n",
    "    float acc = 0.0f;\n",
    "    for (uint b = 0u; b < nblocks; ++b) {\n",
    "        const __global uchar* blk = raw + (size_t)b * 34u;\n",
    "        float scale = half_to_float((ushort)(blk[0] | (blk[1] << 8)));\n",
    "        const __global uchar* qs = blk + 2;\n",
    "        uint base = b * 128u;\n",
    "        uint cnt = min(128u, ne0 - base);\n",
    "        float block_acc = 0.0f;\n",
    "        for (uint l = 0u; l < cnt; ++l) {\n",
    "            uchar code = (qs[l >> 2] >> ((l & 3u) << 1)) & 3u;\n",
    "            block_acc += x[base + l] * (float)((int)code - 1);\n",
    "        }\n",
    "        acc += scale * block_acc;\n",
    "    }\n",
    "    y[gid] = acc;\n",
    "}\n"
);

/// Trivial elementwise kernel for bring-up probing (bonsai-clprobe).
pub const ADD1: &str = r#"
__kernel void add1(__global float* y, __global const float* x) {
    uint i = get_global_id(0);
    y[i] = x[i] + 1.0f;
}
"#;
