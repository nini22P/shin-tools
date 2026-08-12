from __future__ import annotations

import math
import struct
import os
import binascii
import argparse
from io import BytesIO
from typing import Optional, Callable, NamedTuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from PIL import Image
import pngquant_py

import lz77
import lz77_v0


GRID_W = 256
GRID_H = 128
TILE_W = 258
TILE_H = 130


def quantize_rgba(png_path: str) -> Image.Image:
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

    palette_bytes = data[:1024]
    index_bytes = data[1024:need]
    alpha_bytes = data[need:] if (flags & 1) == 0 and need < len(data) else None

    pixels = bytearray(w * h * 4)
    if do_swap:
        for i in range(w * h):
            row = i // w
            col = i % w
            pi = index_bytes[row * stride + col]
            po = i * 4
            pixels[po] = palette_bytes[pi * 4 + 2]
            pixels[po + 1] = palette_bytes[pi * 4 + 1]
            pixels[po + 2] = palette_bytes[pi * 4]
            pixels[po + 3] = alpha_bytes[row * stride + col] if alpha_bytes is not None else palette_bytes[pi * 4 + 3]
            if pixels[po + 3] == 0:
                pixels[po] = pixels[po + 1] = pixels[po + 2] = 0
    else:
        for i in range(w * h):
            row = i // w
            col = i % w
            pi = index_bytes[row * stride + col]
            po = i * 4
            pixels[po] = palette_bytes[pi * 4]
            pixels[po + 1] = palette_bytes[pi * 4 + 1]
            pixels[po + 2] = palette_bytes[pi * 4 + 2]
            pixels[po + 3] = alpha_bytes[row * stride + col] if alpha_bytes is not None else palette_bytes[pi * 4 + 3]
            if pixels[po + 3] == 0:
                pixels[po] = pixels[po + 1] = pixels[po + 2] = 0

    return pixels


def decode_diff_block(data: bytes, w: int, h: int) -> bytearray:
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
    for i in range(w * h):
        if pixels[i * 4 + 3] == 0:
            pixels[i * 4] = 0
            pixels[i * 4 + 1] = 0
            pixels[i * 4 + 2] = 0
    return pixels


def encode_diff_block(pixels: bytes, w: int, h: int) -> bytes:
    buf = bytearray(pixels)
    for i in range(w * h):
        if buf[i * 4 + 3] == 0:
            buf[i * 4] = 0
            buf[i * 4 + 1] = 0
            buf[i * 4 + 2] = 0
    stride = (w * 4 + 0xf) & ~0xf
    result = bytearray(stride * h)
    if h > 0:
        row_size = w * 4
        result[:row_size] = buf[:row_size]
        off = stride
        for j in range(1, h):
            prev_start = (j - 1) * row_size
            cur_start = j * row_size
            for i in range(row_size):
                result[off + i] = (buf[cur_start + i] - buf[prev_start + i]) & 0xFF
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
        if color[3] == 0:
            color = (0, 0, 0, 0)
        if color not in palette_map:
            palette_map[color] = len(palette_order)
            palette_order.append(color)

    if len(palette_order) > 256:
        raise ValueError(f"tile has {len(palette_order)} colors, exceeds 256")

    while len(palette_order) < 256:
        palette_order.append((0, 0, 0, 0))

    palette_bytes = bytearray(1024)
    for i, (r, g, b, a) in enumerate(palette_order):
        if do_swap:
            palette_bytes[i * 4] = b
            palette_bytes[i * 4 + 1] = g
            palette_bytes[i * 4 + 2] = r
        else:
            palette_bytes[i * 4] = r
            palette_bytes[i * 4 + 1] = g
            palette_bytes[i * 4 + 2] = b
        palette_bytes[i * 4 + 3] = a

    index_bytes = bytearray(stride * h)
    for i in range(w * h):
        row = i // w
        col = i % w
        pos = i * 4
        color = (pixels[pos], pixels[pos + 1], pixels[pos + 2], pixels[pos + 3])
        if color[3] == 0:
            color = (0, 0, 0, 0)
        index_bytes[row * stride + col] = palette_map[color]

    result = bytes(palette_bytes) + bytes(index_bytes)

    if not use_inline_alpha:
        alpha_bytes = bytearray(stride * h)
        for i in range(w * h):
            row = i // w
            col = i % w
            pos = i * 4
            alpha_bytes[row * stride + col] = pixels[pos + 3]
        result += bytes(alpha_bytes)

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


def tile_has_alpha(img: Image.Image, x: int, y: int, w: int, h: int) -> bool:
    pixels = img.load()
    iw, ih = img.size
    for ry in range(y, min(y + h, ih)):
        for rx in range(x, min(x + w, iw)):
            px = pixels[rx, ry]
            if len(px) == 4 and 0 < px[3] < 255:
                return True
    return False


def slice_entries(
    img: Image.Image,
    crop: bool = True,
    canvas_w: Optional[int] = None,
    canvas_h: Optional[int] = None,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[dict]:
    w, h = img.size
    if canvas_w is None:
        canvas_w, canvas_h = w, h
    bounds = content_bounds(img) if crop else None

    if bounds:
        content_w, content_h = bounds[2] - bounds[0], bounds[3] - bounds[1]
        content_ratio = (content_w * content_h) / (w * h)
    else:
        content_ratio = 0

    entries = []

    if bounds and content_ratio <= 0.5:
        x_base = max(0, bounds[0] - 2)
        y_base = max(0, bounds[1] - 2)
        bw_adj = min(w - x_base, bounds[2] - bounds[0] + 4)
        bh_adj = min(h - y_base, bounds[3] - bounds[1] + 4)

        if bw_adj > TILE_W or bh_adj > TILE_H:
            cols = math.ceil(bw_adj / GRID_W)
            rows = math.ceil(bh_adj / GRID_H)
            for row in range(rows):
                for col in range(cols):
                    x = x_base + col * GRID_W
                    y = y_base + row * GRID_H
                    tw = min(TILE_W, x_base + bw_adj - x + 2, w - x)
                    th = min(TILE_H, y_base + bh_adj - y + 2, h - y)
                    has_alpha = tile_has_alpha(img, x, y, tw, th)
                    entries.append({
                        'x': x, 'y': y, 'w': tw, 'h': th,
                        'tile_flags': 2 if has_alpha else 3,
                        'op_verts': 0, 'tr_verts': 1,
                        'offset_x': 0, 'offset_y': 0,
                    })
        else:
            has_alpha = tile_has_alpha(img, x_base, y_base, bw_adj, bh_adj)
            entries.append({
                'x': x_base, 'y': y_base, 'w': bw_adj, 'h': bh_adj,
                'tile_flags': 2 if has_alpha else 3,
                'op_verts': 0, 'tr_verts': 1,
                'offset_x': 0, 'offset_y': 0,
            })
    else:
        cols = math.ceil(w / GRID_W)
        rows = math.ceil(h / GRID_H)
        for row in range(rows):
            for col in range(cols):
                x = col * GRID_W
                y = row * GRID_H
                tw = min(TILE_W, w - x + (2 if offset_x + w == canvas_w else 0))
                th = min(TILE_H, h - y + (2 if offset_y + h == canvas_h else 0))
                if tw < 2 or th < 2:
                    continue
                has_alpha = tile_has_alpha(img, x, y, tw, th)
                entries.append({
                    'x': x, 'y': y, 'w': tw, 'h': th,
                    'tile_flags': 2 if has_alpha else 3,
                    'op_verts': 0, 'tr_verts': 1,
                    'offset_x': 0, 'offset_y': 0,
                })

    if not entries:
        entries.append({
            'x': 0, 'y': 0, 'w': w, 'h': h,
            'tile_flags': 3, 'op_verts': 0, 'tr_verts': 1,
            'offset_x': 0, 'offset_y': 0,
        })
    return entries


@dataclass
class _PicFmt:
    version: int
    do_swap: bool
    allow_diff_fallback: bool
    compress: Callable[[bytes], bytes]
    decompress: Callable[[bytes], bytes]
    hdr_fmt: str
    entry_fmt: str
    entry_has_size: bool
    entry_size_first: bool
    frag_fmt: str
    has_pid: bool


_FMTS: dict[int, _PicFmt] = {
    0: _PicFmt(
        version=0, do_swap=True, allow_diff_fallback=False,
        compress=lz77_v0.compress, decompress=lz77_v0.decompress,
        hdr_fmt='<IhhHHII',
        entry_fmt='<HHI', entry_has_size=False, entry_size_first=False,
        frag_fmt='<HHHHHHHHI',
        has_pid=False,
    ),
    1: _PicFmt(
        version=1, do_swap=True, allow_diff_fallback=False,
        compress=partial(lz77.compress, offset_bits=12),
        decompress=partial(lz77.decompress, seek_bits=12, backseek_nbyte=2),
        hdr_fmt='<IIhhHHIII',
        entry_fmt='<HHI', entry_has_size=False, entry_size_first=False,
        frag_fmt='<HHHHHHHHI',
        has_pid=True,
    ),
    2: _PicFmt(
        version=2, do_swap=False, allow_diff_fallback=True,
        compress=partial(lz77.compress, offset_bits=12),
        decompress=partial(lz77.decompress, seek_bits=12, backseek_nbyte=2),
        hdr_fmt='<IIhhHHIII',
        entry_fmt='<HHII', entry_has_size=True, entry_size_first=False,
        frag_fmt='<HHHHHHHHI',
        has_pid=True,
    ),
    3: _PicFmt(
        version=3, do_swap=False, allow_diff_fallback=True,
        compress=partial(lz77.compress, offset_bits=12),
        decompress=partial(lz77.decompress, seek_bits=12, backseek_nbyte=2),
        hdr_fmt='<IIhhHHIII',
        entry_fmt='<IHHI', entry_has_size=True, entry_size_first=True,
        frag_fmt='<HHHHHHHHI',
        has_pid=True,
    ),
}


class _PicLayout(NamedTuple):
    canvas_w: int
    canvas_h: int
    origin_x: int
    origin_y: int
    left: int
    top: int
    width: int
    height: int
    pid: Optional[int] = None


def _read_pic_layout(file_path: str) -> Optional[_PicLayout]:
    try:
        version = detect_pic_version(file_path)
    except (ValueError, OSError):
        return None
    fmt = _FMTS.get(version)
    if fmt is None:
        return None
    try:
        with open(file_path, "rb") as f:
            f.read(4)
            hdr = f.read(struct.calcsize(fmt.hdr_fmt))
            if len(hdr) < struct.calcsize(fmt.hdr_fmt):
                return None
            fields = struct.unpack(fmt.hdr_fmt, hdr)
            if fmt.version == 0:
                _, origin_x, origin_y, canvas_w, canvas_h, _f20, entry_count = fields
                pid = None
            else:
                _version, _file_size, origin_x, origin_y, canvas_w, canvas_h, _f20, entry_count, pid = fields

            entry_size = struct.calcsize(fmt.entry_fmt)
            entries = []
            for _ in range(entry_count):
                raw = f.read(entry_size)
                if len(raw) < entry_size:
                    break
                entries.append(struct.unpack(fmt.entry_fmt, raw))
    except OSError:
        return None

    file_size = os.path.getsize(file_path)
    left = top = 10 ** 9
    right = bottom = -1
    found = False
    with open(file_path, "rb") as f:
        for e in entries:
            if not fmt.entry_has_size:
                x, y, offset = e
                frag_size = None
            elif fmt.entry_size_first:
                _sz, x, y, offset = e
                frag_size = _sz
            else:
                x, y, offset, frag_size = e
            if offset >= file_size:
                continue
            if fmt.version == 2 and (frag_size is None or frag_size == 0):
                continue
            try:
                f.seek(offset)
                frag_hdr = f.read(20)
                if len(frag_hdr) < 20:
                    continue
                h = struct.unpack(fmt.frag_fmt, frag_hdr)
                tw, th = h[6], h[7]
            except OSError:
                continue
            left = min(left, x)
            top = min(top, y)
            right = max(right, x + tw)
            bottom = max(bottom, y + th)
            found = True

    if not found:
        return None
    right = min(right, canvas_w)
    bottom = min(bottom, canvas_h)
    return _PicLayout(canvas_w, canvas_h, origin_x, origin_y, left, top, right - left, bottom - top, pid)


def build_layout_map(orig_paths: list[str]) -> dict[str, _PicLayout]:
    layout_map: dict[str, _PicLayout] = {}
    for p in orig_paths:
        if os.path.isfile(p):
            layout = _read_pic_layout(p)
            if layout is not None:
                layout_map[os.path.basename(p).replace("\\", "/")] = layout
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                for file in files:
                    if file.lower().endswith(".pic"):
                        full = os.path.join(root, file)
                        rel = os.path.relpath(full, p).replace("\\", "/")
                        layout = _read_pic_layout(full)
                        if layout is not None:
                            layout_map[rel] = layout
        else:
            print(f"[warn] --orig path not found: {p}")
    return layout_map


def _pack(
    png_path: str,
    output_path: str,
    fmt: _PicFmt,
    origin: Optional[tuple[int, int]] = None,
    layout: Optional[_PicLayout] = None,
) -> bool:
    img = Image.open(png_path).convert("RGBA")
    img_w, img_h = img.size

    offset_x = offset_y = 0
    canvas_w, canvas_h = img_w, img_h
    is_actual_size = False
    if layout is not None:
        if (img_w, img_h) == (layout.width, layout.height):
            # Input is the merged content region: place it back at the original position
            offset_x, offset_y = layout.left, layout.top
            canvas_w, canvas_h = layout.canvas_w, layout.canvas_h
            origin = (layout.origin_x, layout.origin_y)
            is_actual_size = True
        elif (img_w, img_h) == (layout.canvas_w, layout.canvas_h):
            # Full-canvas image: keep old behavior, restore origin from original header
            origin = (layout.origin_x, layout.origin_y)
        else:
            print(f"[skip] {os.path.abspath(png_path)}: image size {img_w}x{img_h} "
                  f"does not match original canvas {layout.canvas_w}x{layout.canvas_h} "
                  f"or actual size {layout.width}x{layout.height}")
            return False

    if not fmt.allow_diff_fallback:
        colors = img.getcolors(maxcolors=257)
        if colors is None or len(colors) > 256:
            img = quantize_rgba(png_path)

    entries = slice_entries(img, crop=not is_actual_size,
                            canvas_w=canvas_w, canvas_h=canvas_h, offset_x=offset_x, offset_y=offset_y)
    fragments: list[dict] = []
    pid_data = bytearray()
    if origin is None:
        origin_x, origin_y = img_w // 2, img_h // 2
    else:
        origin_x, origin_y = origin

    for info in entries:
        tw, th = info['w'], info['h']
        tile = img.crop((info['x'], info['y'], info['x'] + tw, info['y'] + th))
        pixels = tile.tobytes()

        tile_flags = info['tile_flags']
        if fmt.allow_diff_fallback:
            try:
                encoded_bytes = encode_dict_block(pixels, tw, th, tile_flags, do_swap=False)
            except ValueError:
                try:
                    encoded_bytes = encode_dict_block(pixels, tw, th, 2, do_swap=False)
                    tile_flags = 2
                except ValueError:
                    encoded_bytes = encode_diff_block(pixels, tw, th)
                    tile_flags = 1
        else:
            encoded_bytes = encode_dict_block(pixels, tw, th, tile_flags, do_swap=fmt.do_swap)

        compressed = fmt.compress(encoded_bytes)
        comp_size = len(compressed) if len(compressed) < len(encoded_bytes) else 0

        vert_count = info['op_verts'] + info['tr_verts']
        data_offset_no_align = 20 + vert_count * 8
        data_align = (0x10 - data_offset_no_align % 0x10) % 0x10
        alignment = data_align // 2

        frag = bytearray()
        frag.extend(struct.pack(fmt.frag_fmt,
            tile_flags, info['op_verts'], info['tr_verts'], alignment,
            info['offset_x'], info['offset_y'], tw, th, comp_size))
        mask_rect = struct.pack("<HHHH", 0, 0, tw - 2, th - 2)
        frag.extend(mask_rect * vert_count)
        frag.extend(b'\x00' * data_align)
        frag.extend(compressed if comp_size > 0 else encoded_bytes)

        if fmt.has_pid:
            pid_data.extend(frag)
        fragments.append({'x': info['x'] + offset_x, 'y': info['y'] + offset_y, 'data': frag})

    entry_size = struct.calcsize(fmt.entry_fmt)
    hdr_field_size = struct.calcsize(fmt.hdr_fmt)
    header_size = 4 + hdr_field_size + len(fragments) * entry_size
    chunk_start = (header_size + 15) // 16 * 16

    out = bytearray()
    out.extend(b"PIC4")
    out.extend(b'\x00' * hdr_field_size)
    for _ in range(len(fragments)):
        out.extend(b'\x00' * entry_size)
    while len(out) < chunk_start:
        out.extend(b'\x00')

    frag_offsets: list[int] = []
    for frag_info in fragments:
        if fmt.version in (0, 1):
            cur = len(out)
            aligned = (cur + 15) // 16 * 16
            if aligned > cur:
                out.extend(b'\x00' * (aligned - cur))
        frag_offsets.append(len(out))
        out.extend(frag_info['data'])

    file_size = len(out)

    # Write header fields
    if fmt.has_pid:
        if layout is not None and layout.pid is not None:
            pid = layout.pid
        else:
            pid = binascii.crc32(pid_data) & 0xFFFFFFFF
            if pid == 0:
                pid = 1
    else:
        pid = 0

    if fmt.version == 0:
        hdr_bytes = struct.pack(fmt.hdr_fmt, file_size, origin_x, origin_y, canvas_w, canvas_h, 1, len(fragments))
    else:
        hdr_bytes = struct.pack(fmt.hdr_fmt, fmt.version, file_size, origin_x, origin_y, canvas_w, canvas_h, 1, len(fragments), pid)
    out[4:4 + hdr_field_size] = hdr_bytes

    # Write entry table
    for i, (frag_info, offset) in enumerate(zip(fragments, frag_offsets)):
        entry_off = 4 + hdr_field_size + i * entry_size
        frag_size = len(frag_info['data'])
        if not fmt.entry_has_size:
            out[entry_off:entry_off + entry_size] = struct.pack(fmt.entry_fmt, frag_info['x'], frag_info['y'], offset)
        elif fmt.entry_size_first:
            out[entry_off:entry_off + entry_size] = struct.pack(fmt.entry_fmt, frag_size, frag_info['x'], frag_info['y'], offset)
        else:
            out[entry_off:entry_off + entry_size] = struct.pack(fmt.entry_fmt, frag_info['x'], frag_info['y'], offset, frag_size)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(out)

    print(f"{os.path.abspath(png_path)} -> {os.path.abspath(output_path)}")
    return True


def _unpack(file_path: str, output_path: str, fmt: _PicFmt) -> bool:
    file_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as f:
        if f.read(4) != b"PIC4":
            print(f"[skip] not a PIC4 file: {os.path.abspath(file_path)}")
            return False

        hdr_bytes = f.read(struct.calcsize(fmt.hdr_fmt))
        if len(hdr_bytes) < struct.calcsize(fmt.hdr_fmt):
            print(f"[skip] {os.path.abspath(file_path)}: file too small")
            return False

        hdr_fields = struct.unpack(fmt.hdr_fmt, hdr_bytes)
        if fmt.version == 0:
            _, origin_x, origin_y, img_w, img_h, hdr_flags, entry_count = hdr_fields
        else:
            hdr_version, _, origin_x, origin_y, img_w, img_h, hdr_flags, entry_count, hdr_pid = hdr_fields

        # Read entry table
        entry_size = struct.calcsize(fmt.entry_fmt)
        entries = []
        for _ in range(entry_count):
            raw = f.read(entry_size)
            if len(raw) < entry_size:
                break
            fields = struct.unpack(fmt.entry_fmt, raw)
            if not fmt.entry_has_size:
                x, y, offset = fields
                entries.append((x, y, offset, None))
            elif fmt.entry_size_first:
                frag_size, x, y, offset = fields
                entries.append((x, y, offset, frag_size))
            else:
                x, y, offset, frag_size = fields
                entries.append((x, y, offset, frag_size))

        left = top = 10 ** 9
        right = bottom = -1
        found = False
        for x, y, offset, frag_size in entries:
            if offset >= file_size:
                continue
            if fmt.version == 2 and (frag_size is None or frag_size == 0):
                continue
            f.seek(offset)
            frag_hdr = f.read(20)
            if len(frag_hdr) < 20:
                continue
            h = struct.unpack(fmt.frag_fmt, frag_hdr)
            tw, th = h[6], h[7]
            left = min(left, x)
            top = min(top, y)
            right = max(right, x + tw)
            bottom = max(bottom, y + th)
            found = True

        if found:
            right = min(right, img_w)
            bottom = min(bottom, img_h)
        if found and right > left and bottom > top:
            img = Image.new("RGBA", (right - left, bottom - top))
        else:
            img = Image.new("RGBA", (img_w, img_h))
            left = top = 0
        processed = 0

        for i, (x, y, offset, frag_size) in enumerate(entries):
            if offset >= file_size:
                continue

            if fmt.version in (0, 1):
                f.seek(offset)
                tile_data = f.read(20)
                if len(tile_data) < 20:
                    continue

                hdr = struct.unpack(fmt.frag_fmt, tile_data)
                tile_flags, op_verts, tr_verts, alignment, offset_x, offset_y, tw, th = hdr[:8]
                comp_size = hdr[8]

                if tw > img_w + 2 or th > img_h + 2:
                    print(f"[warn] {os.path.abspath(file_path)}: Fragment {i} ({x},{y}) at offset {offset}: suspicious header (w={tw} h={th} op={op_verts} tr={tr_verts} flags={tile_flags:#06x} comp={comp_size}), file may be corrupted")
                    continue

                vert_count = op_verts + tr_verts
                skip_masks = vert_count * 8 + alignment * 2
                f.seek(skip_masks, 1)

                if comp_size > 0:
                    raw_data = f.read(comp_size)
                else:
                    next_offset = entries[i + 1][2] if i < entry_count - 1 else file_size
                    data_len = next_offset - offset - 20 - skip_masks
                    if data_len <= 0:
                        continue
                    raw_data = f.read(data_len)

                if not raw_data:
                    continue

                try:
                    decoded_bytes = fmt.decompress(raw_data) if comp_size > 0 else raw_data
                except Exception as e:
                    print(f"[warn] {os.path.abspath(file_path)}: Fragment {i} ({x},{y}): decompress failed: {e}")
                    continue

                if not decoded_bytes:
                    continue

                try:
                    pixel_bytes = decode_dict_block(decoded_bytes, tw, th, tile_flags)
                except Exception as e:
                    print(f"[warn] {os.path.abspath(file_path)}: Fragment {i} ({x},{y}) w={tw} h={th} flags={tile_flags:#06x} comp={comp_size}: dict decode failed: {e}")
                    continue

            else:
                # v2/v3: pre-read full fragment
                if fmt.version == 2:
                    if frag_size is None or frag_size == 0:
                        continue
                    f.seek(offset)
                    fragment_data = f.read(frag_size)
                    if len(fragment_data) < 20:
                        print(f"[warn] {os.path.abspath(file_path)}: Fragment ({x},{y}) at offset {offset}: too small ({len(fragment_data)} bytes), skipping")
                        continue
                else:
                    # v3: recalculate total size from header fields
                    f.seek(offset)
                    frag_hdr_bytes = f.read(20)
                    if len(frag_hdr_bytes) < 20:
                        continue
                    frag_hdr = struct.unpack(fmt.frag_fmt, frag_hdr_bytes)
                    tile_flags, op_verts, tr_verts, alignment, offset_x, offset_y, tw, th = frag_hdr[:8]
                    comp_size = frag_hdr[8]

                    if tw > img_w + 2 or th > img_h + 2:
                        print(f"[warn] {os.path.abspath(file_path)}: Fragment ({x},{y}) at offset {offset}: suspicious header (w={tw} h={th} op={op_verts} tr={tr_verts} flags={tile_flags:#06x} comp={comp_size}), file may be corrupted")
                        continue

                    vert_count = op_verts + tr_verts
                    skip_masks = vert_count * 8 + alignment * 2

                    if comp_size > 0:
                        frag_total = 20 + skip_masks + comp_size
                    else:
                        if tile_flags & 2:
                            pal_stride = (tw + 3) & ~3
                            frag_total = 20 + skip_masks + 0x400 + pal_stride * th
                            if (tile_flags & 1) == 0:
                                frag_total += pal_stride * th
                        else:
                            diff_stride = (tw * 4 + 0xf) & ~0xf
                            frag_total = 20 + skip_masks + diff_stride * th

                    if frag_total > file_size - offset:
                        continue

                    f.seek(offset)
                    fragment_data = f.read(frag_total)

                hdr = struct.unpack(fmt.frag_fmt, fragment_data[:20])
                tile_flags, op_verts, tr_verts, alignment, offset_x, offset_y, tw, th = hdr[:8]
                comp_size = hdr[8]

                vert_count = op_verts + tr_verts
                skip_masks = vert_count * 8 + alignment * 2
                data_start = 20 + skip_masks

                if data_start >= len(fragment_data):
                    continue

                if comp_size > 0:
                    data_chunk = fragment_data[data_start:data_start + comp_size]
                    try:
                        decoded_bytes = fmt.decompress(data_chunk)
                    except Exception as e:
                        print(f"[warn] {os.path.abspath(file_path)}: Fragment ({x},{y}): decompress failed: {e}")
                        continue
                else:
                    decoded_bytes = fragment_data[data_start:]

                if not decoded_bytes:
                    continue

                try:
                    if tile_flags & 2:
                        pixel_bytes = decode_dict_block(decoded_bytes, tw, th, tile_flags, do_swap=False)
                    else:
                        pixel_bytes = decode_diff_block(decoded_bytes, tw, th)
                except Exception as e:
                    print(f"[warn] {os.path.abspath(file_path)}: Fragment ({x},{y}) w={tw} h={th} flags={tile_flags:#06x} comp={comp_size}: decode failed: {e}")
                    continue

            if pixel_bytes:
                try:
                    tile_img = Image.frombytes("RGBA", (tw, th), bytes(pixel_bytes))
                    img.paste(tile_img, (x - left, y - top))
                    processed += 1
                except Exception as e:
                    print(f"[warn] {os.path.abspath(file_path)}: Fragment ({x},{y}): image build failed: {e}")

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        img.save(output_path)

        version_tag = fmt.version
        total = len(entries)
        if processed < total:
            print(f"[v{version_tag}] {os.path.abspath(file_path)} -> {os.path.abspath(output_path)}  [{processed}/{total} fragments, some skipped]")
        else:
            print(f"[v{version_tag}] {os.path.abspath(file_path)} -> {os.path.abspath(output_path)}")
        return processed > 0


def unpack_pic(file_path: str, output_path: str) -> bool:
    try:
        version = detect_pic_version(file_path)
    except (ValueError, OSError) as e:
        print(f"[skip] {os.path.abspath(file_path)}: {e}")
        return False

    fmt = _FMTS.get(version)
    if fmt is None:
        print(f"[skip] unsupported PIC version: {version}")
        return False
    return _unpack(file_path, output_path, fmt)


def pack_pic(
    png_path: str,
    output_path: str,
    pic_version: int,
    origin: Optional[tuple[int, int]] = None,
    layout: Optional[_PicLayout] = None,
) -> bool:
    fmt = _FMTS.get(pic_version)
    if fmt is None:
        print(f"[skip] unsupported PIC version: {pic_version}")
        return False
    return _pack(png_path, output_path, fmt, origin, layout)


def unpack(input_path: str, output_path: str) -> None:
    abs_input = os.path.abspath(input_path)

    if os.path.isfile(abs_input):
        out_path = output_path
        name = os.path.splitext(os.path.basename(abs_input))[0]
        if not output_path.endswith(".png"):
            out_path = os.path.join(output_path, name + ".png")
        unpack_pic(abs_input, out_path)

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
                fut_to_src = {executor.submit(unpack_pic, src, dst): src for src, dst in tasks}
                for fut in as_completed(fut_to_src):
                    if fut.result():
                        count += 1
        else:
            for src, dst in tasks:
                if unpack_pic(src, dst):
                    count += 1

        print(f"processed: {count} file(s)")


def pack(
    input_path: str,
    output_path: str,
    pic_version: int = 1,
    orig: Optional[list[str]] = None,
) -> None:
    abs_input = os.path.abspath(input_path)
    layout_map = build_layout_map(orig or [])

    if os.path.isfile(abs_input):
        out_path = output_path
        name = os.path.splitext(os.path.basename(abs_input))[0]
        if not output_path.endswith(".pic"):
            out_path = os.path.join(output_path, name + ".pic")
        if len(orig) != 1 or not os.path.isfile(orig[0]):
            print(f"[skip] single file input requires --orig to be a single .pic file")
            return
        layout = _read_pic_layout(orig[0])
        if layout is None:
            print(f"[warn] failed to read layout from {orig[0]}; "
                  f"canvas will be the PNG size (merged images will be misplaced)")
        pack_pic(abs_input, out_path, pic_version, None, layout)

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
                        key = name + ".pic"
                    else:
                        dst = os.path.join(output_dir, rel, name + ".pic")
                        key = os.path.join(rel, name + ".pic").replace("\\", "/")
                    layout = layout_map.get(key) if layout_map else None
                    if layout is None:
                        print(f"[warn] no original layout found for {key}; "
                              f"canvas will be the PNG size (merged images will be misplaced), pass --orig to fix")
                    tasks.append((src, dst, layout))

        count = 0
        if len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                fut_to_name = {
                    executor.submit(pack_pic, png, dst, pic_version, None, layout): png
                    for png, dst, layout in tasks
                }
                for fut in as_completed(fut_to_name):
                    if fut.result():
                        count += 1
        else:
            for png, dst, layout in tasks:
                if pack_pic(png, dst, pic_version, None, layout):
                    count += 1

        print(f"processed: {count} file(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIC tool")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    unpack_p = sub.add_parser("unpack", help="Convert PIC to PNG")
    unpack_p.add_argument("-i", "--input", required=True, help="Input .pic file or directory")
    unpack_p.add_argument("-o", "--output", required=True, help="Output .png file or directory")

    pack_p = sub.add_parser("pack", help="Convert PNG to PIC")
    pack_p.add_argument("-i", "--input", required=True, help="Input .png file or directory")
    pack_p.add_argument("-o", "--output", required=True, help="Output .pic file or directory")
    pack_p.add_argument("-v", "--version", type=int, choices=[0, 1, 2, 3], required=True, help="PIC version: 0, 1, 2, 3")
    pack_p.add_argument("--orig", required=True, help="Original .pic file or directory for layout reference")

    args = parser.parse_args()

    if args.command == "unpack":
        unpack(args.input, args.output)
    elif args.command == "pack":
        orig_list = [s.strip() for s in args.orig.split(",") if s.strip()] if args.orig else None
        pack(args.input, args.output, args.version, orig_list)
    else:
        parser.print_help()
