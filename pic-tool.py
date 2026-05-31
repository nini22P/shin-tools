from __future__ import annotations

import math
import struct
import os
import binascii
import argparse
from io import BytesIO
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import pngquant_py

import lz77
import lz77_v0


GRID_W = 256
GRID_H = 128
TILE_W = 258
TILE_H = 130


def quantize_rgba(png_path: str, colors: int = 256) -> Image.Image:
    with open(png_path, "rb") as f:
        out = pngquant_py.quantize(f.read(), speed=1)
    return Image.open(BytesIO(out)).convert("RGBA")


def detect_pic_version(file_path: str) -> int:
    size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'PIC4':
            raise ValueError("Not a PIC4 file")
        buf = f.read(8)
        if len(buf) < 8:
            raise ValueError("File too small")
        val0, val1 = struct.unpack("<II", buf)
        if val1 == size:
            return val0
        return 0


def decode_dict_block(data: bytes, w: int, h: int, flags: int, do_swap: bool = True) -> Optional[bytearray]:
    stride = (w + 3) & ~3

    if not (flags & 2):
        return None

    if len(data) < 1024:
        return None

    need = 1024 + stride * h
    if need > len(data):
        return None

    pal = data[:1024]
    idx_data = data[1024:need]
    alpha = data[need:] if (flags & 1) == 0 and need < len(data) else None

    pixels = bytearray(w * h * 4)
    if do_swap:
        for i in range(w * h):
            row = i // w
            col = i % w
            pi = idx_data[row * stride + col]
            po = i * 4
            pixels[po] = pal[pi * 4 + 2]
            pixels[po + 1] = pal[pi * 4 + 1]
            pixels[po + 2] = pal[pi * 4]
            pixels[po + 3] = alpha[row * stride + col] if alpha is not None else pal[pi * 4 + 3]
    else:
        for i in range(w * h):
            row = i // w
            col = i % w
            pi = idx_data[row * stride + col]
            po = i * 4
            pixels[po] = pal[pi * 4]
            pixels[po + 1] = pal[pi * 4 + 1]
            pixels[po + 2] = pal[pi * 4 + 2]
            pixels[po + 3] = alpha[row * stride + col] if alpha is not None else pal[pi * 4 + 3]

    return pixels


def decode_differential_block(data: bytes, w: int, h: int) -> bytearray:
    stride = (w * 4 + 0xf) & ~0xf
    pixels = bytearray(w * h * 4)
    if h > 0:
        row_size = w * 4
        pixels[:row_size] = data[:row_size]
        off = stride
        for j in range(1, h):
            prev_start = (j - 1) * row_size
            cur_start = j * row_size
            for i in range(row_size):
                pixels[cur_start + i] = (pixels[prev_start + i] + data[off + i]) & 0xFF
            off += stride
    return pixels


def encode_differential_block(pixels: bytes, w: int, h: int) -> bytes:
    stride = (w * 4 + 0xf) & ~0xf
    result = bytearray(stride * h)
    if h > 0:
        row_size = w * 4
        result[:row_size] = pixels[:row_size]
        off = stride
        for j in range(1, h):
            prev_start = (j - 1) * row_size
            cur_start = j * row_size
            for i in range(row_size):
                result[off + i] = (pixels[cur_start + i] - pixels[prev_start + i]) & 0xFF
            off += stride
    return bytes(result)


def encode_dict_block(pixels: bytes, w: int, h: int, flags: int, do_swap: bool = True) -> bytes:
    stride = (w + 3) & ~3
    use_inline_alpha = (flags & 1) != 0

    palette_map: dict[tuple[int, int, int, int], int] = {}
    palette_order: list[tuple[int, int, int, int]] = []

    for i in range(w * h):
        pos = i * 4
        color = (pixels[pos], pixels[pos + 1], pixels[pos + 2], pixels[pos + 3])
        if color not in palette_map:
            palette_map[color] = len(palette_order)
            palette_order.append(color)

    if len(palette_order) > 256:
        raise ValueError(f"tile has {len(palette_order)} colors, exceeds 256")

    while len(palette_order) < 256:
        palette_order.append((0, 0, 0, 0))

    pal_bytes = bytearray(1024)
    for i, (r, g, b, a) in enumerate(palette_order):
        if do_swap:
            pal_bytes[i * 4] = b
            pal_bytes[i * 4 + 1] = g
            pal_bytes[i * 4 + 2] = r
        else:
            pal_bytes[i * 4] = r
            pal_bytes[i * 4 + 1] = g
            pal_bytes[i * 4 + 2] = b
        pal_bytes[i * 4 + 3] = a

    indices = bytearray(stride * h)
    for i in range(w * h):
        row = i // w
        col = i % w
        pos = i * 4
        color = (pixels[pos], pixels[pos + 1], pixels[pos + 2], pixels[pos + 3])
        indices[row * stride + col] = palette_map[color]

    result = bytes(pal_bytes) + bytes(indices)

    if not use_inline_alpha:
        alpha = bytearray(stride * h)
        for i in range(w * h):
            row = i // w
            col = i % w
            pos = i * 4
            alpha[row * stride + col] = pixels[pos + 3]
        result += bytes(alpha)

    return result


def content_bounds(img: Image.Image) -> Optional[tuple[int, int, int, int]]:
    pixels = img.load()
    w, h = img.size
    x0, y0, x1, y1 = w, h, 0, 0
    found = False
    for y in range(h):
        for x in range(w):
            px = pixels[x, y]
            if len(px) == 4 and px[3] == 0:
                continue
            if all(c == 0 for c in px):
                continue
            if x < x0: x0 = x
            if y < y0: y0 = y
            if x > x1: x1 = x
            if y > y1: y1 = y
            found = True
    return (x0, y0, x1 + 1, y1 + 1) if found else None


def tile_has_alpha(img: Image.Image, bx: int, by: int, w: int, h: int) -> bool:
    pixels = img.load()
    iw, ih = img.size
    for y in range(by, min(by + h, ih)):
        for x in range(bx, min(bx + w, iw)):
            px = pixels[x, y]
            if len(px) == 4 and 0 < px[3] < 255:
                return True
    return False


def slice_blocks(img: Image.Image) -> list[dict]:
    w, h = img.size
    bounds = content_bounds(img)

    if bounds:
        bw, bh = bounds[2] - bounds[0], bounds[3] - bounds[1]
        content_ratio = (bw * bh) / (w * h)
    else:
        content_ratio = 0

    blocks = []

    if bounds and content_ratio <= 0.5:
        bx_base = max(0, bounds[0] - 2)
        by_base = max(0, bounds[1] - 2)
        bw_adj = min(w - bx_base, bounds[2] - bounds[0] + 4)
        bh_adj = min(h - by_base, bounds[3] - bounds[1] + 4)

        if bw_adj > TILE_W or bh_adj > TILE_H:
            cols = math.ceil(bw_adj / GRID_W)
            rows = math.ceil(bh_adj / GRID_H)
            for row in range(rows):
                for col in range(cols):
                    bx = bx_base + col * GRID_W
                    by = by_base + row * GRID_H
                    tw = min(TILE_W, bx_base + bw_adj - bx + 2)
                    th = min(TILE_H, by_base + bh_adj - by + 2)
                    has_alpha = tile_has_alpha(img, bx, by, tw, th)
                    blocks.append({
                        'bx': bx, 'by': by, 'w': tw, 'h': th,
                        't_flags': 2 if has_alpha else 3,
                        'op_verts': 0, 'tr_verts': 1,
                        'off_x': 0, 'off_y': 0,
                    })
        else:
            has_alpha = tile_has_alpha(img, bx_base, by_base, bw_adj, bh_adj)
            blocks.append({
                'bx': bx_base, 'by': by_base, 'w': bw_adj, 'h': bh_adj,
                't_flags': 2 if has_alpha else 3,
                'op_verts': 0, 'tr_verts': 1,
                'off_x': 0, 'off_y': 0,
            })
    else:
        cols = math.ceil(w / GRID_W)
        rows = math.ceil(h / GRID_H)
        for row in range(rows):
            for col in range(cols):
                bx = col * GRID_W
                by = row * GRID_H
                tw = min(TILE_W, w - bx + 2)
                th = min(TILE_H, h - by + 2)
                has_alpha = tile_has_alpha(img, bx, by, tw, th)
                blocks.append({
                    'bx': bx, 'by': by, 'w': tw, 'h': th,
                    't_flags': 2 if has_alpha else 3,
                    'op_verts': 0, 'tr_verts': 1,
                    'off_x': 0, 'off_y': 0,
                })

    if not blocks:
        blocks.append({
            'bx': 0, 'by': 0, 'w': w, 'h': h,
            't_flags': 3, 'op_verts': 0, 'tr_verts': 1,
            'off_x': 0, 'off_y': 0,
        })
    return blocks


def _write_chunk_header(data: bytearray, info: dict, comp_size: int, tw: int, th: int) -> int:
    vert_count = info['op_verts'] + info['tr_verts']
    data_off_no_align = 20 + vert_count * 8
    data_align = (0x10 - data_off_no_align % 0x10) % 0x10
    alignment = data_align // 2

    data.extend(struct.pack("<HHHHHHHHI",
        info['t_flags'], info['op_verts'], info['tr_verts'], alignment,
        info['off_x'], info['off_y'], tw, th,
        comp_size))

    mask_rect = struct.pack("<HHHH", 0, 0, tw - 2, th - 2)
    data.extend(mask_rect * vert_count)
    data.extend(b'\x00' * data_align)
    return alignment


def _pack_v0(png_path: str, output_path: str) -> bool:
    img = Image.open(png_path).convert("RGBA")
    w, h = img.size

    colors = img.getcolors(maxcolors=257)
    if colors is None or len(colors) > 256:
        img = quantize_rgba(png_path, 256)

    blocks = slice_blocks(img)

    chunks_out = bytearray()
    chunk_writers: list[tuple[int, int, int]] = []
    header_size = 24 + len(blocks) * 8
    chunk_start = (header_size + 15) // 16 * 16

    for info in blocks:
        cur = chunk_start + len(chunks_out)
        aligned = (cur + 15) // 16 * 16
        if aligned > cur:
            chunks_out.extend(b'\x00' * (aligned - cur))

        tw, th = info['w'], info['h']
        tile = img.crop((info['bx'], info['by'], info['bx'] + tw, info['by'] + th))
        pixels = bytearray(tile.tobytes())

        enc_data = encode_dict_block(bytes(pixels), tw, th, info['t_flags'])

        compressed = lz77_v0.compress_v0(enc_data)
        comp_size = len(compressed)
        comp_size = comp_size if comp_size < len(enc_data) and comp_size <= 0xFFFF else 0

        chunk_writers.append((info['bx'], info['by'], chunk_start + len(chunks_out)))
        _write_chunk_header(chunks_out, info, comp_size, tw, th)
        if comp_size > 0:
            chunks_out.extend(compressed)
        else:
            chunks_out.extend(enc_data)

    file_size = chunk_start + len(chunks_out)

    out = bytearray()
    out.extend(b"PIC4")
    ox, oy = w // 2, h // 2
    out.extend(struct.pack("<IhhHHII", file_size, ox, oy, w, h, 1, len(blocks)))
    for bx, by, off in chunk_writers:
        out.extend(struct.pack("<HHI", bx, by, off))
    while len(out) % 16 != 0:
        out.extend(b'\x00')
    out.extend(chunks_out)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(out)

    print(f"{os.path.abspath(png_path)} -> {os.path.abspath(output_path)}")
    return True


def _pack_v1(png_path: str, output_path: str) -> bool:
    img = Image.open(png_path).convert("RGBA")
    w, h = img.size
    ox = w // 2
    oy = h // 2

    colors = img.getcolors(maxcolors=257)
    if colors is None or len(colors) > 256:
        img = quantize_rgba(png_path, 256)

    blocks = slice_blocks(img)

    chunks_out = bytearray()
    chunk_writers: list[tuple[int, int, int]] = []
    header_size = 32 + len(blocks) * 8
    chunk_start = (header_size + 15) // 16 * 16
    crc_data = bytearray()

    for info in blocks:
        cur = chunk_start + len(chunks_out)
        aligned = (cur + 15) // 16 * 16
        if aligned > cur:
            chunks_out.extend(b'\x00' * (aligned - cur))

        tw, th = info['w'], info['h']
        tile = img.crop((info['bx'], info['by'], info['bx'] + tw, info['by'] + th))
        pixels = tile.tobytes()

        enc_data = encode_dict_block(pixels, tw, th, info['t_flags'])

        compressed = lz77.compress(enc_data, offset_bits=12)
        comp_size = len(compressed)
        comp_size = comp_size if comp_size < len(enc_data) and comp_size <= 0xFFFF else 0

        block_data = bytearray()
        _write_chunk_header(block_data, info, comp_size, tw, th)
        if comp_size > 0:
            block_data.extend(compressed)
        else:
            block_data.extend(enc_data)

        chunk_writers.append((info['bx'], info['by'], chunk_start + len(chunks_out)))
        chunks_out.extend(block_data)
        crc_data.extend(block_data)

    file_size = chunk_start + len(chunks_out)

    crc = binascii.crc32(crc_data) & 0xFFFFFFFF
    if crc == 0:
        crc = 1

    out = bytearray()
    out.extend(b"PIC4")
    out.extend(struct.pack("<IIhhHHIII", 1, file_size, ox, oy, w, h, 1, len(blocks), crc))
    for bx, by, off in chunk_writers:
        out.extend(struct.pack("<HHI", bx, by, off))
    while len(out) % 16 != 0:
        out.extend(b'\x00')
    out.extend(chunks_out)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(out)

    print(f"{os.path.abspath(png_path)} -> {os.path.abspath(output_path)}")
    return True


def _pack_v2(png_path: str, output_path: str) -> bool:
    img = Image.open(png_path).convert("RGBA")
    w, h = img.size
    ox = w // 2
    oy = h // 2

    blocks = slice_blocks(img)

    encoded_fragments = []
    crc_data = bytearray()
    for info in blocks:
        tw, th = info['w'], info['h']
        tile = img.crop((info['bx'], info['by'], info['bx'] + tw, info['by'] + th))
        pixels = tile.tobytes()

        t_flags = info['t_flags']
        try:
            enc_data = encode_dict_block(pixels, tw, th, t_flags, do_swap=False)
        except ValueError:
            try:
                enc_data = encode_dict_block(pixels, tw, th, 2, do_swap=False)
                t_flags = 2
            except ValueError:
                enc_data = encode_differential_block(pixels, tw, th)
                t_flags = 1

        compressed = lz77.compress(enc_data, offset_bits=12)
        comp_size = len(compressed)
        comp_size = comp_size if comp_size < len(enc_data) and comp_size <= 0xFFFF else 0

        vert_count = info['op_verts'] + info['tr_verts']
        data_off_no_align = 20 + vert_count * 8
        data_align = (0x10 - data_off_no_align % 0x10) % 0x10
        alignment = data_align // 2

        frag = bytearray()
        frag.extend(struct.pack("<HHHHHHHHHH",
            t_flags, info['op_verts'], info['tr_verts'], alignment,
            info['off_x'], info['off_y'], tw, th,
            comp_size, 0))
        mask_rect = struct.pack("<HHHH", 0, 0, tw - 2, th - 2)
        frag.extend(mask_rect * vert_count)
        frag.extend(b'\x00' * data_align)
        if comp_size > 0:
            frag.extend(compressed)
        else:
            frag.extend(enc_data)

        encoded_fragments.append((info['bx'], info['by'], frag))
        crc_data.extend(frag)

    crc = binascii.crc32(crc_data) & 0xFFFFFFFF
    if crc == 0:
        crc = 1

    header_size = 32 + len(encoded_fragments) * 12
    chunk_start = (header_size + 15) // 16 * 16

    out = bytearray()
    out.extend(b"PIC4")
    out.extend(b'\x00' * 28)
    for _ in range(len(encoded_fragments)):
        out.extend(b'\x00' * 12)
    while len(out) < chunk_start:
        out.extend(b'\x00')

    for i, (bx, by, frag_data) in enumerate(encoded_fragments):
        offset = len(out)
        out.extend(frag_data)
        entry_off = 32 + i * 12
        out[entry_off:entry_off + 12] = struct.pack("<HHII", bx, by, offset, len(frag_data))

    file_size = len(out)
    out[4:32] = struct.pack("<IIHHHHIII", 2, file_size, ox, oy, w, h, 1, len(encoded_fragments), crc)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(out)

    print(f"{os.path.abspath(png_path)} -> {os.path.abspath(output_path)}")
    return True


def _pack_v3(png_path: str, output_path: str) -> bool:
    img = Image.open(png_path).convert("RGBA")
    w, h = img.size
    ox = w // 2
    oy = h // 2

    blocks = slice_blocks(img)

    encoded_fragments = []
    crc_data = bytearray()
    for info in blocks:
        tw, th = info['w'], info['h']
        tile = img.crop((info['bx'], info['by'], info['bx'] + tw, info['by'] + th))
        pixels = tile.tobytes()

        t_flags = info['t_flags']
        try:
            enc_data = encode_dict_block(pixels, tw, th, t_flags, do_swap=False)
        except ValueError:
            try:
                enc_data = encode_dict_block(pixels, tw, th, 2, do_swap=False)
                t_flags = 2
            except ValueError:
                enc_data = encode_differential_block(pixels, tw, th)
                t_flags = 1

        compressed = lz77.compress(enc_data, offset_bits=12)
        comp_size = len(compressed)
        comp_size = comp_size if comp_size < len(enc_data) and comp_size <= 0xFFFF else 0

        vert_count = info['op_verts'] + info['tr_verts']
        data_off_no_align = 20 + vert_count * 8
        data_align = (0x10 - data_off_no_align % 0x10) % 0x10
        alignment = data_align // 2

        frag = bytearray()
        frag.extend(struct.pack("<HHHHHHHHHH",
            t_flags, info['op_verts'], info['tr_verts'], alignment,
            info['off_x'], info['off_y'], tw, th,
            comp_size, 0))
        mask_rect = struct.pack("<HHHH", 0, 0, tw - 2, th - 2)
        frag.extend(mask_rect * vert_count)
        frag.extend(b'\x00' * data_align)
        if comp_size > 0:
            frag.extend(compressed)
        else:
            frag.extend(enc_data)

        encoded_fragments.append((info['bx'], info['by'], frag))
        crc_data.extend(frag)

    crc = binascii.crc32(crc_data) & 0xFFFFFFFF
    if crc == 0:
        crc = 1

    # PIC3 header layout:
    #   32 bytes base header (same as v2)
    #   N * 12 bytes entries (N = len(encoded_fragments))
    #   padding to 16 bytes
    #   fragment data (16-byte aligned start)
    entry_count = len(encoded_fragments)
    header_size = 32 + entry_count * 12
    chunk_start = (header_size + 15) // 16 * 16

    out = bytearray()
    out.extend(b"PIC4")
    out.extend(b'\x00' * 28)
    for _ in range(entry_count):
        out.extend(b'\x00' * 12)
    while len(out) < chunk_start:
        out.extend(b'\x00')

    for i, (bx, by, frag_data) in enumerate(encoded_fragments):
        offset = len(out)
        out.extend(frag_data)
        frag_size = len(frag_data)
        entry_off = 32 + i * 12
        out[entry_off:entry_off + 12] = struct.pack("<IHHI", frag_size, bx, by, offset)

    file_size = len(out)
    out[4:32] = struct.pack("<IIHHHHIII", 3, file_size, ox, oy, w, h, 1, entry_count, crc)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(out)

    print(f"{os.path.abspath(png_path)} -> {os.path.abspath(output_path)}")
    return True


def _unpack_v0(file_path: str, output_path: str) -> bool:
    file_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as f:
        if f.read(4) != b"PIC4":
            print(f"[skip] not a PIC4 file: {os.path.abspath(file_path)}")
            return False

        header = struct.unpack("<IhhHHII", f.read(20))
        _, ew, eh, width, height, flags, block_count = header

        blocks = []
        for _ in range(block_count):
            blocks.append(struct.unpack("<HHI", f.read(8)))

        img = Image.new("RGBA", (width, height))
        processed_blocks = 0

        for i, (bx, by, offset) in enumerate(blocks):
            if offset >= file_size:
                continue
            f.seek(offset)

            tile_data = f.read(20)
            if len(tile_data) < 20:
                continue

            flags, op_verts, tr_verts, alignment, off_x, off_y, w, h, comp_size = struct.unpack("<HHHHHHHHI", tile_data)

            if w > width or h > height:
                print(f"[warn] Block {i} ({bx},{by}) at offset {offset}: suspicious header (w={w} h={h} op={op_verts} tr={tr_verts} flags={flags:#06x} comp={comp_size}), file may be corrupted")
                continue

            skip = (op_verts + tr_verts) * 8 + (alignment * 2)
            f.seek(skip, 1)

            if comp_size > 0:
                data_len = comp_size
            else:
                next_offset = blocks[i + 1][2] if i < block_count - 1 else file_size
                data_len = next_offset - offset - 20 - skip

            if data_len <= 0:
                continue

            raw_data = f.read(data_len)
            if not raw_data:
                continue

            if comp_size > 0:
                try:
                    dec_data = lz77_v0.decompress_v0(raw_data)
                except Exception as e:
                    print(f"[warn] Block {i} ({bx},{by}): flags={flags} op_verts={op_verts} tr_verts={tr_verts} align={alignment} off=({off_x},{off_y}) size=({w}x{h}) comp={comp_size} decompress failed: {e}")
                    continue
            else:
                dec_data = raw_data

            if not dec_data:
                continue

            pixel_bytes = decode_dict_block(dec_data, w, h, flags)

            if pixel_bytes:
                try:
                    tile_img = Image.frombytes("RGBA", (w, h), bytes(pixel_bytes))
                    img.paste(tile_img, (bx, by))
                    processed_blocks += 1
                except ValueError as e:
                    print(f"[warn] Block {i} ({bx},{by}): image build failed: {e}")

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        img.save(output_path)
        if processed_blocks < block_count:
            print(f"[v0] {os.path.abspath(file_path)} -> {os.path.abspath(output_path)}  [{processed_blocks}/{block_count} blocks, some skipped]")
        else:
            print(f"[v0] {os.path.abspath(file_path)} -> {os.path.abspath(output_path)}")
        return processed_blocks > 0


def _unpack_v1(file_path: str, output_path: str) -> bool:
    file_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as f:
        if f.read(4) != b"PIC4":
            print(f"[skip] not a PIC4 file: {os.path.abspath(file_path)}")
            return False

        header = struct.unpack("<IIhhHHIII", f.read(28))
        version, _, origin_x, origin_y, effective_width, effective_height, flags, block_count, crc = header

        blocks = []
        for _ in range(block_count):
            x, y, offset = struct.unpack("<HHI", f.read(8))
            blocks.append((x, y, offset))

        img = Image.new("RGBA", (effective_width, effective_height))
        processed_blocks = 0

        for i, (bx, by, offset) in enumerate(blocks):
            if offset >= file_size:
                continue
            f.seek(offset)

            tile_data = f.read(20)
            if len(tile_data) < 20:
                continue

            flags, op_verts, tr_verts, alignment, off_x, off_y, w, h, comp_size = struct.unpack("<HHHHHHHHI", tile_data)

            if w > effective_width or h > effective_height:
                print(f"[warn] Block {i} ({bx},{by}) at offset {offset}: suspicious header (w={w} h={h} op={op_verts} tr={tr_verts} flags={flags:#06x} comp={comp_size}), file may be corrupted")
                continue

            skip = (op_verts + tr_verts) * 8 + (alignment * 2)
            f.seek(skip, 1)

            if comp_size > 0:
                data_len = comp_size
            else:
                next_offset = blocks[i + 1][2] if i < block_count - 1 else file_size
                data_len = next_offset - offset - 20 - skip

            if data_len <= 0:
                continue

            raw_data = f.read(data_len)
            if not raw_data:
                continue

            if comp_size > 0:
                try:
                    dec_data = lz77.decompress(raw_data, seek_bits=12, backseek_nbyte=2)
                except Exception as e:
                    print(f"[warn] Block {i} ({bx},{by}): flags={flags} op_verts={op_verts} tr_verts={tr_verts} align={alignment} off=({off_x},{off_y}) size=({w}x{h}) comp={comp_size} decompress failed: {e}")
                    continue
            else:
                dec_data = raw_data

            if not dec_data:
                continue

            pixel_bytes = decode_dict_block(dec_data, w, h, flags)

            if pixel_bytes:
                try:
                    tile_img = Image.frombytes("RGBA", (w, h), bytes(pixel_bytes))
                    img.paste(tile_img, (bx, by))
                    processed_blocks += 1
                except ValueError as e:
                    print(f"[warn] Block {i} ({bx},{by}): image build failed: {e}")

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        img.save(output_path)
        if processed_blocks < block_count:
            print(f"[v1] {os.path.abspath(file_path)} -> {os.path.abspath(output_path)}  [{processed_blocks}/{block_count} blocks, some skipped]")
        else:
            print(f"[v1] {os.path.abspath(file_path)} -> {os.path.abspath(output_path)}")
        return processed_blocks > 0


def _unpack_v2(file_path: str, output_path: str) -> bool:
    file_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as f:
        if f.read(4) != b"PIC4":
            print(f"[skip] not a PIC4 file: {os.path.abspath(file_path)}")
            return False

        buf = f.read(28)
        if len(buf) < 28:
            print(f"[skip] {os.path.abspath(file_path)}: file too small for v2 header")
            return False

        fields = struct.unpack("<IIHHHHIII", buf)
        version, file_size_h, origin_x, origin_y, ew, eh, field20, entry_count, picture_id = fields

        entries = []
        for _ in range(entry_count):
            buf = f.read(12)
            if len(buf) < 12:
                break
            x, y, data_off, data_size = struct.unpack("<HHII", buf)
            entries.append((x, y, data_off, data_size))

        img = Image.new("RGBA", (ew, eh))
        processed = 0

        for x, y, data_off, data_size in entries:
            if data_off >= file_size or data_size == 0:
                continue
            f.seek(data_off)
            fragment_data = f.read(data_size)
            if len(fragment_data) < 20:
                print(f"[warn] Fragment ({x},{y}) at offset {data_off}: too small ({len(fragment_data)} bytes), skipping")
                continue

            frag_hdr = struct.unpack("<HHHHHHHHHH", fragment_data[:20])
            flags, op_verts, tr_verts, alignment, off_x, off_y, w, h, comp_size, unknown = frag_hdr

            if w > ew or h > eh:
                print(f"[warn] Fragment ({x},{y}) at offset {data_off}: suspicious header (w={w} h={h} op={op_verts} tr={tr_verts} flags={flags:#06x} comp={comp_size}), file may be corrupted")
                continue

            skip = (op_verts + tr_verts) * 8 + alignment * 2
            data_start = 20 + skip

            if data_start >= len(fragment_data):
                continue

            if comp_size > 0:
                data_block = fragment_data[data_start:data_start + comp_size]
                try:
                    dec_data = lz77.decompress(data_block, seek_bits=12, backseek_nbyte=2)
                except Exception as e:
                    print(f"[warn] Fragment ({x},{y}): decompress failed: {e}")
                    continue
            else:
                dec_data = fragment_data[data_start:]

            if flags & 2 == 0:
                pixel_bytes = decode_differential_block(dec_data, w, h)
            else:
                pixel_bytes = decode_dict_block(dec_data, w, h, flags, do_swap=False)

            if pixel_bytes:
                try:
                    tile_img = Image.frombytes("RGBA", (w, h), bytes(pixel_bytes))
                    img.paste(tile_img, (x, y))
                    processed += 1
                except Exception as e:
                    print(f"[warn] Fragment ({x},{y}): image build failed: {e}")

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        img.save(output_path)
        if processed < len(entries):
            print(f"[v2] {os.path.abspath(file_path)} -> {os.path.abspath(output_path)}  [{processed}/{len(entries)} fragments, some skipped]")
        else:
            print(f"[v2] {os.path.abspath(file_path)} -> {os.path.abspath(output_path)}")
        return processed > 0


def _unpack_v3(file_path: str, output_path: str) -> bool:
    file_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as f:
        if f.read(4) != b"PIC4":
            print(f"[skip] not a PIC4 file: {os.path.abspath(file_path)}")
            return False

        buf = f.read(28)
        if len(buf) < 28:
            print(f"[skip] {os.path.abspath(file_path)}: file too small for v3 header")
            return False

        fields = struct.unpack("<IIHHHHIII", buf)
        version, file_size_h, origin_x, origin_y, ew, eh, field20, entry_count, picture_id = fields

        entries = []
        for _ in range(entry_count):
            buf = f.read(12)
            if len(buf) < 12:
                break
            frag_size, x, y, data_off = struct.unpack("<IHHI", buf)
            entries.append((x, y, data_off, frag_size))

        img = Image.new("RGBA", (ew, eh))
        processed = 0

        for x, y, data_off, _ in entries:
            if data_off >= file_size:
                continue
            f.seek(data_off)
            frag_hdr_bytes = f.read(20)
            if len(frag_hdr_bytes) < 20:
                continue

            frag_hdr = struct.unpack("<HHHHHHHHHH", frag_hdr_bytes)
            flags, op_verts, tr_verts, alignment, off_x, off_y, w, h, comp_size, unknown = frag_hdr

            if w > ew or h > eh:
                print(f"[warn] Fragment ({x},{y}) at offset {data_off}: suspicious header (w={w} h={h} op={op_verts} tr={tr_verts} flags={flags:#06x} comp={comp_size}), file may be corrupted")
                continue

            skip = (op_verts + tr_verts) * 8 + alignment * 2
            if comp_size > 0:
                frag_total = 20 + skip + comp_size
            else:
                dict_stride = (w + 3) & ~3
                diff_stride = (w * 4 + 0xf) & ~0xf
                if flags & 2:
                    frag_total = 20 + skip + 0x400 + dict_stride * h
                    if (flags & 1) == 0:
                        frag_total += dict_stride * h
                else:
                    frag_total = 20 + skip + diff_stride * h
            if frag_total > file_size - data_off:
                continue

            f.seek(data_off)
            fragment_data = f.read(frag_total)

            data_start = 20 + skip

            if data_start >= len(fragment_data):
                continue

            if comp_size > 0:
                data_block = fragment_data[data_start:data_start + comp_size]
                try:
                    dec_data = lz77.decompress(data_block, seek_bits=12, backseek_nbyte=2)
                except Exception as e:
                    print(f"[warn] Fragment ({x},{y}): decompress failed: {e}")
                    continue
            else:
                dec_data = fragment_data[data_start:]

            if flags & 2 == 0:
                pixel_bytes = decode_differential_block(dec_data, w, h)
            else:
                pixel_bytes = decode_dict_block(dec_data, w, h, flags, do_swap=False)

            if pixel_bytes:
                try:
                    tile_img = Image.frombytes("RGBA", (w, h), bytes(pixel_bytes))
                    img.paste(tile_img, (x, y))
                    processed += 1
                except Exception as e:
                    print(f"[warn] Fragment ({x},{y}): image build failed: {e}")

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        img.save(output_path)
        if processed < len(entries):
            print(f"[v3] {os.path.abspath(file_path)} -> {os.path.abspath(output_path)}  [{processed}/{len(entries)} fragments, some skipped]")
        else:
            print(f"[v3] {os.path.abspath(file_path)} -> {os.path.abspath(output_path)}")
        return processed > 0


def unpack_file(file_path: str, output_path: str) -> bool:
    try:
        version = detect_pic_version(file_path)
    except (ValueError, OSError) as e:
        print(f"[skip] {os.path.abspath(file_path)}: {e}")
        return False

    if version == 0:
        return _unpack_v0(file_path, output_path)
    elif version == 1:
        return _unpack_v1(file_path, output_path)
    elif version == 2:
        return _unpack_v2(file_path, output_path)
    elif version == 3:
        return _unpack_v3(file_path, output_path)
    else:
        print(f"[skip] unsupported PIC version: {version}")
        return False


def pack_file(png_path: str, output_path: str, pic_version: int) -> bool:
    if pic_version == 0:
        return _pack_v0(png_path, output_path)
    elif pic_version == 1:
        return _pack_v1(png_path, output_path)
    elif pic_version == 2:
        return _pack_v2(png_path, output_path)
    else:
        return _pack_v3(png_path, output_path)


def process_unpack(input_path: str, output_path: str) -> None:
    abs_input = os.path.abspath(input_path)

    if os.path.isfile(abs_input):
        out_path = output_path
        folder = os.path.dirname(abs_input)
        name = os.path.splitext(os.path.basename(abs_input))[0]
        if not output_path.endswith(".png"):
            out_path = os.path.join(output_path, name + ".png")
        unpack_file(abs_input, out_path)

    elif os.path.isdir(abs_input):
        output_dir = output_path

        print(f"output dir: {output_dir}")

        tasks = []
        for root, dirs, files in os.walk(abs_input):
            for file in files:
                if file.lower().endswith(".pic"):
                    src = os.path.join(root, file)
                    name = os.path.splitext(file)[0]
                    rel = os.path.relpath(root, abs_input)
                    if rel == '.':
                        dst = os.path.join(output_dir, name + ".png")
                    else:
                        dst = os.path.join(output_dir, rel, name + ".png")
                    tasks.append((src, dst))

        count = 0
        if len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                fut_to_src = {executor.submit(unpack_file, src, dst): src for src, dst in tasks}
                for fut in as_completed(fut_to_src):
                    if fut.result():
                        count += 1
        else:
            for src, dst in tasks:
                if unpack_file(src, dst):
                    count += 1

        print(f"processed: {count} file(s)")


def process_pack(input_path: str, output_path: str, pic_version: int = 1) -> None:
    abs_input = os.path.abspath(input_path)

    if os.path.isfile(abs_input):
        out_path = output_path
        folder = os.path.dirname(abs_input)
        name = os.path.splitext(os.path.basename(abs_input))[0]
        if not output_path.endswith(".pic"):
            out_path = os.path.join(output_path, name + ".pic")
        pack_file(abs_input, out_path, pic_version)

    elif os.path.isdir(abs_input):
        output_dir = output_path

        print(f"output dir: {output_dir}")

        tasks = []
        for root, dirs, files in os.walk(abs_input):
            for file in files:
                if file.lower().endswith(".png"):
                    src = os.path.join(root, file)
                    name = os.path.splitext(file)[0]
                    rel = os.path.relpath(root, abs_input)
                    if rel == '.':
                        dst = os.path.join(output_dir, name + ".pic")
                    else:
                        dst = os.path.join(output_dir, rel, name + ".pic")
                    tasks.append((src, dst))

        count = 0
        if len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                fut_to_name = {executor.submit(pack_file, png, dst, pic_version): png for png, dst in tasks}
                for fut in as_completed(fut_to_name):
                    if fut.result():
                        count += 1
        else:
            for png, dst in tasks:
                if pack_file(png, dst, pic_version):
                    count += 1

        print(f"processed: {count} file(s)")


def run_unpack(args: argparse.Namespace) -> None:
    process_unpack(args.input, args.output)


def run_pack(args: argparse.Namespace) -> None:
    process_pack(args.input, args.output, args.version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIC tool")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    unpack_p = sub.add_parser("unpack", help="Convert PIC to PNG")
    unpack_p.add_argument("-i", "--input", required=True, help="Input .pic file or directory")
    unpack_p.add_argument("-o", "--output", required=True, help="Output .png file or directory")

    pack_p = sub.add_parser("pack", help="Convert PNG to PIC")
    pack_p.add_argument("-i", "--input", required=True, help="Input .png file or directory")
    pack_p.add_argument("-o", "--output", required=True, help="Output .pic file or directory")
    pack_p.add_argument("-v", "--version", type=int, choices=[0, 1, 2, 3], required=True,
                        help="PIC version: 0, 1, 2, 3")

    args = parser.parse_args()

    if args.command == "unpack":
        run_unpack(args)
    elif args.command == "pack":
        run_pack(args)
    else:
        parser.print_help()
