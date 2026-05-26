'''LZ77 variant used by TXA v0 / PIC v0.

2-byte backreference layout:
  b1: [offset_high(4) | length(4)]
  b2: [offset_low(8)]
length = (b1 & 0x0F) + 3
offset = ((b1 >> 4) << 8) | b2
'''

import numpy as np
from numba import njit
from numba.typed import Dict
from numba.core import types


def decompress_v0(input_data: bytes) -> bytes:
    input_arr = np.frombuffer(input_data, dtype=np.uint8)
    result = _decompress_v0(input_arr)
    return bytes(result)


def compress_v0(input_bytes: bytes) -> bytes:
    input_arr = np.frombuffer(input_bytes, dtype=np.uint8)
    result = _compress_v0(input_arr)
    return bytes(result)


@njit
def _decompress_v0(input_arr):
    n = len(input_arr)
    out = np.zeros(max(n, 256), dtype=np.uint8)
    out_pos = 0
    in_pos = 0
    marker = 1

    while in_pos < n:
        if marker == 1:
            marker = 0x100 | int(input_arr[in_pos])
            in_pos += 1
        if in_pos >= n:
            break
        if marker & 1:
            b1 = input_arr[in_pos]
            b2 = input_arr[in_pos + 1]
            in_pos += 2
            count = (b1 & 0x0F) + 3
            offset = ((b1 & 0xF0) << 4) | b2

            while out_pos + count > len(out):
                new_out = np.zeros(len(out) * 2, dtype=np.uint8)
                new_out[:out_pos] = out[:out_pos]
                out = new_out

            if offset + 1 >= count:
                start = out_pos - (offset + 1)
                out[out_pos:out_pos + count] = out[start:start + count]
            else:
                for j in range(count):
                    out[out_pos + j] = out[out_pos + j - (offset + 1)]
            out_pos += count
        else:
            if out_pos >= len(out):
                new_out = np.zeros(len(out) * 2, dtype=np.uint8)
                new_out[:out_pos] = out[:out_pos]
                out = new_out
            out[out_pos] = input_arr[in_pos]
            out_pos += 1
            in_pos += 1
        marker >>= 1

    return out[:out_pos]


@njit
def _compress_v0(input_arr):
    n = len(input_arr)
    if n == 0:
        return np.zeros(0, dtype=np.uint8)

    max_offset = 4096
    max_count = 18
    hash_table = Dict.empty(key_type=types.int64, value_type=types.int64)

    max_inst = n
    inst_type = np.zeros(max_inst, dtype=np.int8)
    inst_val0 = np.zeros(max_inst, dtype=np.int64)
    inst_val1 = np.zeros(max_inst, dtype=np.int64)
    inst_count = 0

    pos = 0
    while pos < n:
        best_len = 0
        best_off = 0

        if pos + 2 < n:
            key = np.int64(int(input_arr[pos]) << 16 | int(input_arr[pos + 1]) << 8 | int(input_arr[pos + 2]))
            if key in hash_table:
                candidate = hash_table[key]
                if pos - candidate <= max_offset:
                    ml = 0
                    max_possible = max_count if max_count < n - pos else n - pos
                    while ml < max_possible and input_arr[candidate + ml] == input_arr[pos + ml]:
                        ml += 1
                    if ml >= 3:
                        best_len = ml
                        best_off = pos - candidate

        if best_len >= 3:
            inst_type[inst_count] = 1
            inst_val0[inst_count] = best_len - 3
            inst_val1[inst_count] = best_off - 1
            inst_count += 1

            for j in range(best_len):
                if pos + j + 2 < n:
                    key = np.int64(int(input_arr[pos + j]) << 16 | int(input_arr[pos + j + 1]) << 8 | int(input_arr[pos + j + 2]))
                    hash_table[key] = pos + j
            pos += best_len
        else:
            inst_type[inst_count] = 0
            inst_val0[inst_count] = int(input_arr[pos])
            inst_val1[inst_count] = 0
            inst_count += 1

            if pos + 2 < n:
                key = np.int64(int(input_arr[pos]) << 16 | int(input_arr[pos + 1]) << 8 | int(input_arr[pos + 2]))
                hash_table[key] = pos
            pos += 1

    output = np.zeros(inst_count * 2 + inst_count // 8 + 1, dtype=np.uint8)
    out_pos = 0

    i = 0
    while i < inst_count:
        block_end = i + 8
        if block_end > inst_count:
            block_end = inst_count
        block_size = block_end - i

        bitmap = 0
        bitmap_pos = out_pos
        out_pos += 1

        for j in range(block_size):
            idx = i + j
            if inst_type[idx] == 0:
                output[out_pos] = inst_val0[idx]
                out_pos += 1
            else:
                bitmap |= (1 << j)
                length_val = inst_val0[idx]
                offset_val = inst_val1[idx]
                b1 = ((offset_val >> 4) & 0xF0) | (length_val & 0x0F)
                b2 = offset_val & 0xFF
                output[out_pos] = b1
                output[out_pos + 1] = b2
                out_pos += 2

        output[bitmap_pos] = bitmap
        i += block_size

    return output[:out_pos]
