#!/usr/bin/env python3
"""Retag legacy Prism ternary GGUFs for the current prism llama.cpp branch.

The July 2026 Ternary-Bonsai GGUF files stored group-128 ternary weights under
ggml type id 42 ("Q2_0"). The current PrismML fork reserves id 42 for the
official group-64 Q2_0 and moved the group-128 2-bit codec to id 142
("PQ2_0"). The on-disk payload is identical (fp16 scale + 2-bit packed quants
per 128 weights), so a header-only retag makes the legacy file loadable by the
current build without touching tensor data.

Usage: retag_gguf.py INPUT.gguf OUTPUT.gguf
"""
import struct
import sys

OLD = 42  # legacy Prism group-128 ternary under the Q2_0 id
NEW = 142  # PQ2_0, official Prism group-128 ternary id

GGUF_TYPE_U32 = 4


def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n)


def skip_val(f, t):
    if t == 8:  # string
        read_str(f)
    elif t == 9:  # array
        tt = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            skip_val(f, tt)
    elif t in (0, 1, 7):  # u8 i8 bool
        f.seek(1, 1)
    elif t in (2, 3):  # u16 i16
        f.seek(2, 1)
    elif t in (4, 5, 6):  # u32 i32 f32
        f.seek(4, 1)
    elif t in (10, 11, 12):  # u64 i64 f64
        f.seek(8, 1)
    else:
        raise ValueError(f"unknown kv type {t}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]

    with open(src, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit("not a GGUF file")
        ver = struct.unpack("<I", f.read(4))[0]
        n_t = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        # ---- pass 1: walk metadata, remember where general.file_type lives
        ftype_patch = None  # absolute file offset of the u32 value
        for _ in range(n_kv):
            key = read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            if key == b"general.file_type" and t == GGUF_TYPE_U32:
                ftype_patch = f.tell()
            skip_val(f, t)
        kv_end = f.tell()

        # ---- pass 2: walk tensor info, remember type field offsets
        type_offsets = []  # (absolute offset of the u32 type id)
        for _ in range(n_t):
            read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            f.seek(8 * nd, 1)  # dims
            type_offsets.append(f.tell())
            f.seek(4 + 8, 1)  # type id + data offset
        ti_end = f.tell()

        # ---- read the whole header (metadata + tensor info) into memory
        f.seek(0)
        header = bytearray(f.read(ti_end))

        # ---- patch tensor type ids
        n_patch = 0
        for off in type_offsets:
            val = struct.unpack_from("<I", header, off)[0]
            if val == OLD:
                struct.pack_into("<I", header, off, NEW)
                n_patch += 1

        # ---- patch general.file_type: MOSTLY_Q2_0(28) -> MOSTLY_PQ2_0(128)
        if ftype_patch is not None:
            val = struct.unpack_from("<I", header, ftype_patch)[0]
            if val == 28:
                struct.pack_into("<I", header, ftype_patch, 128)
                print(f"general.file_type patched 28 -> 128")
            else:
                print(f"general.file_type left as {val}")

        # ---- write out
        with open(dst, "wb") as out:
            out.write(header)
            f.seek(ti_end)
            while True:
                chunk = f.read(8 << 20)
                if not chunk:
                    break
                out.write(chunk)

    print(f"patched {n_patch}/{n_t} tensors {OLD} -> {NEW}; wrote {dst}")
    if n_patch == 0:
        print("warning: no tensors needed patching")


if __name__ == "__main__":
    main()
