'''
Implement the LZ77 variant used in the game.
This variant of LZ77 precedes a block of data with an 8-bit bitmap specifying whether decompressor should read a literal byte or a reference.

The references are encoded as 16-bit big-endian (sic!) integers. It encodes offset and length of the reference,
but the amount of bits spent on each part is dependent on the format (const generics are used to specify this, in this fuction def as *offset_bits*).

Offset is specified as amount to seek back from the current position.
The minimum offset is 1, so the actual offset is offset + 1.
The minimum length is 3, so the actual length is length + 3.

see also,
https://github.com/DCNick3/shin/blob/master/shin-core/src/format/lz77.rs
https://github.com/DCNick3/shin-translation-tools/blob/master/shin-font/src/lib.rs
https://gitlab.com/Neurochitin/kaleido/-/tree/saku/fnt
'''

import numpy as np
from numba import njit
from numba.typed import Dict
from numba.core import types


def decompress(input_data: bytes, seek_bits: int, backseek_nbyte: int) -> bytes:
    input_arr = np.frombuffer(input_data, dtype=np.uint8)
    result = _decompress(input_arr, seek_bits, backseek_nbyte)
    return bytes(result)


def compress(input_bytes: bytes, offset_bits: int = 10) -> bytes:
    input_arr = np.frombuffer(input_bytes, dtype=np.uint8)
    result = _compress(input_arr, offset_bits)
    return bytes(result)


@njit
def _decompress(input_arr, seek_bits, backseek_nbyte):
    n = len(input_arr)
    out = np.zeros(max(n, 256), dtype=np.uint8)
    out_pos = 0
    in_pos = 0

    while in_pos < n:
        map_byte = input_arr[in_pos]
        in_pos += 1
        for i in range(8):
            if in_pos >= n:
                break
            if ((map_byte >> i) & 1) == 0:
                if out_pos >= len(out):
                    new_out = np.zeros(len(out) * 2, dtype=np.uint8)
                    new_out[:out_pos] = out[:out_pos]
                    out = new_out
                out[out_pos] = input_arr[in_pos]
                out_pos += 1
                in_pos += 1
            else:
                # back reference
                # FNT4 v1 (2-byte): [len(16-OFFSET_BITS) | offset(OFFSET_BITS)]
                # FNT4 v0 (1-byte): [offset(8-LEN_BITS) | len(LEN_BITS)]
                if backseek_nbyte == 2:  # FNT4 v1
                    backseek_spec = int(input_arr[in_pos]) << 8 | int(input_arr[in_pos + 1])
                    in_pos += 2
                    offset_bits = seek_bits
                    back_offset_mask = (1 << offset_bits) - 1
                    back_length = (backseek_spec >> offset_bits) + 3
                    back_offset = (backseek_spec & back_offset_mask) + 1
                else:
                    backseek_spec = input_arr[in_pos]
                    in_pos += 1
                    len_bits = seek_bits
                    back_len_mask = (1 << len_bits) - 1
                    back_length = (backseek_spec & back_len_mask) + 2
                    back_offset = (backseek_spec >> len_bits) + 1

                while out_pos + back_length > len(out):
                    new_out = np.zeros(len(out) * 2, dtype=np.uint8)
                    new_out[:out_pos] = out[:out_pos]
                    out = new_out

                if back_offset >= back_length:
                    start = out_pos - back_offset
                    out[out_pos:out_pos + back_length] = out[start:start + back_length]
                else:
                    for j in range(back_length):
                        out[out_pos + j] = out[out_pos + j - back_offset]
                out_pos += back_length

    return out[:out_pos]


@njit
def _compress(input_arr, offset_bits):
    n = len(input_arr)
    if n == 0:
        return np.zeros(0, dtype=np.uint8)

    count_bits = 16 - offset_bits
    max_count = (1 << count_bits) - 1 + 3

    hash_table = Dict.empty(key_type=types.int64, value_type=types.int64)

    max_inst = n
    inst_type = np.zeros(max_inst, dtype=np.int8)
    inst_val0 = np.zeros(max_inst, dtype=np.int64)
    inst_val1 = np.zeros(max_inst, dtype=np.int64)
    inst_count = 0

    pos = 0
    while pos < n:
        best_len = 1
        best_off = 0

        if pos + 2 < n:
            key = np.int64(int(input_arr[pos]) << 16 | int(input_arr[pos + 1]) << 8 | int(input_arr[pos + 2]))
            if key in hash_table:
                candidate = hash_table[key]
                if pos - candidate <= (1 << offset_bits):
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
                len_b = (inst_val0[idx] << (8 - count_bits)) | ((inst_val1[idx]) >> 8)
                offset_b = inst_val1[idx] & 0xFF
                output[out_pos] = len_b
                output[out_pos + 1] = offset_b
                out_pos += 2

        output[bitmap_pos] = bitmap
        i += block_size

    return output[:out_pos]
