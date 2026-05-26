from __future__ import annotations

import math
import struct
import os
import argparse
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

import lz77
import lz77_v0


GRID_W = 256
GRID_H = 128
TILE_W = 258
TILE_H = 130


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


def decode_dict_block(data: bytes, w: int, h: int, flags: int) -> Optional[bytearray]:
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
    for i in range(w * h):
        row = i // w
        col = i % w
        pi = idx_data[row * stride + col]
        po = i * 4
        pixels[po] = pal[pi * 4 + 2]
        pixels[po + 1] = pal[pi * 4 + 1]
        pixels[po + 2] = pal[pi * 4]
        pixels[po + 3] = alpha[row * stride + col] if alpha is not None else pal[pi * 4 + 3]

    return pixels


def encode_dict_block(pixels: bytes, w: int, h: int, flags: int) -> bytes:
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
        pal_bytes[i * 4] = b
        pal_bytes[i * 4 + 1] = g
        pal_bytes[i * 4 + 2] = r
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
        bx = max(0, bounds[0] - 2)
        by = max(0, bounds[1] - 2)
        tw = min(w - bx, bounds[2] - bounds[0] + 4)
        th = min(h - by, bounds[3] - bounds[1] + 4)
        has_alpha = tile_has_alpha(img, bx, by, tw, th)
        blocks.append({
            'bx': bx, 'by': by, 'w': tw, 'h': th,
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


def _pack_v1(png_path: str, output_path: str) -> bool:
    img = Image.open(png_path).convert("RGBA")
    w, h = img.size
    ox = w // 2
    oy = h // 2

    blocks = slice_blocks(img)

    chunks_out = bytearray()
    chunk_writers: list[tuple[int, int, int]] = []
    header_size = 32 + len(blocks) * 12
    chunk_start = (header_size + 15) // 16 * 16

    for info in blocks:
        cur = chunk_start + len(chunks_out)
        aligned = (cur + 15) // 16 * 16
        if aligned > cur:
            chunks_out.extend(b'\x00' * (aligned - cur))

        tw, th = info['w'], info['h']
        tile = img.crop((info['bx'], info['by'], info['bx'] + tw, info['by'] + th))
        pixels = tile.tobytes()

        try:
            enc_data = encode_dict_block(pixels, tw, th, info['t_flags'])
        except ValueError:
            tile = tile.quantize(256, method=Image.Quantize.FASTOCTREE).convert("RGBA")
            pixels = tile.tobytes()
            enc_data = encode_dict_block(pixels, tw, th, info['t_flags'])

        compressed = lz77.compress(enc_data, offset_bits=12)
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
    out.extend(struct.pack("<IIhhHHIII", 1, file_size, ox, oy, w, h, 1, len(blocks), 0))
    for bx, by, off in chunk_writers:
        out.extend(struct.pack("<HHII", bx, by, off, 0))
    while len(out) % 16 != 0:
        out.extend(b'\x00')
    out.extend(chunks_out)

    with open(output_path, 'wb') as f:
        f.write(out)

    print(f"{os.path.abspath(png_path)} -> {os.path.abspath(output_path)}")
    return True


def _pack_v0(png_path: str, output_path: str) -> bool:
    img = Image.open(png_path).convert("RGBA")
    w, h = img.size

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

        try:
            enc_data = encode_dict_block(bytes(pixels), tw, th, info['t_flags'])
        except ValueError:
            tile = tile.quantize(256, method=Image.Quantize.FASTOCTREE).convert("RGBA")
            pixels = tile.tobytes()
            enc_data = encode_dict_block(pixels, tw, th, info['t_flags'])

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

    with open(output_path, 'wb') as f:
        f.write(out)

    print(f"{os.path.abspath(png_path)} -> {os.path.abspath(output_path)}")
    return True


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
            x, y, offset, _ = struct.unpack("<HHII", f.read(12))
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

        img.save(output_path)
        print(f"{os.path.abspath(file_path)} -> {os.path.abspath(output_path)}")
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

        img.save(output_path)
        print(f"{os.path.abspath(file_path)} -> {os.path.abspath(output_path)}")
        return True


def convert_file(file_path: str, output_path: str) -> bool:
    try:
        version = detect_pic_version(file_path)
    except (ValueError, OSError) as e:
        print(f"[skip] {os.path.abspath(file_path)}: {e}")
        return False

    if version == 0:
        return _unpack_v0(file_path, output_path)
    else:
        return _unpack_v1(file_path, output_path)


def pack_file_auto(png_path: str, output_path: str, pic_version: int) -> bool:
    if pic_version == 0:
        return _pack_v0(png_path, output_path)
    else:
        return _pack_v1(png_path, output_path)


def process_unpack(input_path: str, output_path: str) -> None:
    abs_input = os.path.abspath(input_path)

    if os.path.isfile(abs_input):
        out_path = output_path
        folder = os.path.dirname(abs_input)
        name = os.path.splitext(os.path.basename(abs_input))[0]
        if not output_path.endswith(".png"):
            out_path = os.path.join(output_path, name + ".png")
        convert_file(abs_input, out_path)

    elif os.path.isdir(abs_input):
        output_dir = output_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"output dir: {output_dir}")

        tasks = []
        for root, dirs, files in os.walk(abs_input):
            for file in files:
                if file.lower().endswith(".pic"):
                    src = os.path.join(root, file)
                    name = os.path.splitext(file)[0]
                    dst = os.path.join(output_dir, name + ".png")
                    tasks.append((src, dst))

        count = 0
        if len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                fut_to_src = {executor.submit(convert_file, src, dst): src for src, dst in tasks}
                for fut in as_completed(fut_to_src):
                    if fut.result():
                        count += 1
        else:
            for src, dst in tasks:
                if convert_file(src, dst):
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
        pack_file_auto(abs_input, out_path, pic_version)

    elif os.path.isdir(abs_input):
        output_dir = output_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"output dir: {output_dir}")

        tasks = []
        for file in os.listdir(abs_input):
            if file.lower().endswith(".png"):
                png_path = os.path.join(abs_input, file)
                name = os.path.splitext(file)[0]
                dst = os.path.join(output_dir, name + ".pic")
                tasks.append((png_path, dst))

        count = 0
        if len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                fut_to_name = {executor.submit(pack_file_auto, png, dst, pic_version): png for png, dst in tasks}
                for fut in as_completed(fut_to_name):
                    if fut.result():
                        count += 1
        else:
            for png, dst in tasks:
                if pack_file_auto(png, dst, pic_version):
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
    pack_p.add_argument("-v", "--version", type=int, choices=[0, 1], required=True,
                        help="PIC version: 0, 1")

    args = parser.parse_args()

    if args.command == "unpack":
        run_unpack(args)
    elif args.command == "pack":
        run_pack(args)
    else:
        parser.print_help()
