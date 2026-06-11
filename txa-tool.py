from __future__ import annotations

import struct
import os
import argparse
import re
from PIL import Image

import lz77
import lz77_v0

_INDEX_RE = re.compile(r'(\d+)\s*->\s*(\d+)\s+"(.+)"')


def parse_index(index_path: str) -> list[tuple[int, int, str]]:
    entries: list[tuple[int, int, str]] = []
    if not os.path.isfile(index_path):
        return entries
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _INDEX_RE.match(line)
            if m:
                entries.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    return entries


def write_index(index_path: str, entries: list[tuple[int, int, str]]) -> None:
    with open(index_path, "w", encoding="utf-8") as f:
        for pos, vidx, name in entries:
            f.write(f"{pos:03d} -> {vidx:03d} \"{name}\"\n")


def _dir_has_index(d: str) -> bool:
    try:
        return os.path.isfile(os.path.join(d, "index.txt"))
    except OSError:
        return False


def _bgra_to_rgba(buf: bytearray) -> None:
    for i in range(0, len(buf), 4):
        buf[i], buf[i + 2] = buf[i + 2], buf[i]


def _rgba_to_bgra(buf: bytearray) -> None:
    for i in range(0, len(buf), 4):
        buf[i], buf[i + 2] = buf[i + 2], buf[i]


def decode_dict(data: bytes, w: int, h: int, do_swap: bool = True) -> bytearray:
    stride = (w + 3) & ~3

    palette_bytes = data[:1024]
    index_bytes = data[1024:1024 + stride * h]

    pixels = bytearray(w * h * 4)
    for row in range(h):
        row_offset = row * stride
        for col in range(w):
            idx = index_bytes[row_offset + col]
            pos = (row * w + col) * 4
            if do_swap:
                pixels[pos]     = palette_bytes[idx * 4 + 2]
                pixels[pos + 1] = palette_bytes[idx * 4 + 1]
                pixels[pos + 2] = palette_bytes[idx * 4]
                pixels[pos + 3] = palette_bytes[idx * 4 + 3]
            else:
                pixels[pos]     = palette_bytes[idx * 4]
                pixels[pos + 1] = palette_bytes[idx * 4 + 1]
                pixels[pos + 2] = palette_bytes[idx * 4 + 2]
                pixels[pos + 3] = palette_bytes[idx * 4 + 3]

    mask = data[1024 + stride * h:]
    if mask:
        for i in range(min(len(mask), w * h)):
            pixels[i * 4 + 3] = mask[i]

    return pixels


def decode_diff(data: bytes, w: int, h: int, do_swap: bool = True) -> bytearray:
    stride = (4 * w + 15) & ~15

    pixels = bytearray(w * h * 4)
    row_bytes = w * 4

    pixels[0:row_bytes] = data[0:row_bytes]

    for row in range(1, h):
        prev_start = (row - 1) * row_bytes
        cur_start = row * row_bytes
        data_start = row * stride
        for i in range(row_bytes):
            pixels[cur_start + i] = (pixels[prev_start + i] + data[data_start + i]) & 0xFF

    if do_swap:
        _bgra_to_rgba(pixels)
    return pixels


def detect_version(data: bytes) -> tuple[int, int, int]:
    file_size = len(data)
    val4 = struct.unpack_from('<I', data, 4)[0]
    val8 = struct.unpack_from('<I', data, 8)[0]

    if val8 == file_size:
        version = val4
        indexed = struct.unpack_from('<I', data, 12)[0]
        cnt = struct.unpack_from('<I', data, 16)[0]
        return version, indexed, cnt
    else:
        indexed = val8
        cnt = struct.unpack_from('<I', data, 12)[0]
        return 0, indexed, cnt


def eligible_for_dict(img: Image.Image) -> bool:
    colors = img.getcolors(maxcolors=257)
    return colors is not None


def encode_dict(img: Image.Image, do_swap: bool = True) -> bytes:
    w, h = img.size
    stride = (w + 3) & ~3

    pixels = img.tobytes()

    palette_map: dict[tuple[int, ...], int] = {}
    palette_order: list[tuple[int, ...]] = []
    for y in range(h):
        for x in range(w):
            pos = (y * w + x) * 4
            color = (pixels[pos], pixels[pos+1], pixels[pos+2], pixels[pos+3])
            if color not in palette_map:
                palette_map[color] = len(palette_order)
                palette_order.append(color)

    while len(palette_order) < 256:
        palette_order.append((0, 0, 0, 0))

    palette_bytes = bytearray(1024)
    for i, (r, g, b, a) in enumerate(palette_order):
        if do_swap:
            palette_bytes[i * 4]     = b
            palette_bytes[i * 4 + 1] = g
            palette_bytes[i * 4 + 2] = r
        else:
            palette_bytes[i * 4]     = r
            palette_bytes[i * 4 + 1] = g
            palette_bytes[i * 4 + 2] = b
        palette_bytes[i * 4 + 3] = a

    index_bytes = bytearray(stride * h)
    for y in range(h):
        for x in range(w):
            pos = (y * w + x) * 4
            color = (pixels[pos], pixels[pos+1], pixels[pos+2], pixels[pos+3])
            index_bytes[y * stride + x] = palette_map[color]

    return bytes(palette_bytes + index_bytes)


def encode_diff(img: Image.Image, do_swap: bool = True) -> bytes:
    w, h = img.size
    stride = (4 * w + 15) & ~15

    pixels = bytearray(img.tobytes())
    if do_swap:
        _rgba_to_bgra(pixels)

    encoded_bytes = bytearray(stride * h)
    row_bytes = w * 4

    encoded_bytes[0:row_bytes] = pixels[0:row_bytes]

    for row in range(1, h):
        prev_start = (row - 1) * row_bytes
        cur_start = row * row_bytes
        data_start = row * stride
        for i in range(row_bytes):
            encoded_bytes[data_start + i] = (pixels[cur_start + i] - pixels[prev_start + i]) & 0xFF

    return bytes(encoded_bytes)


def unpack_txa(file_path: str, output_dir: str) -> bool:
    with open(file_path, 'rb') as f:
        data = f.read()

    if data[:4] != b"TXA4":
        print(f"[skip] not a TXA4 file: {os.path.abspath(file_path)}")
        return False

    version, indexed, count = detect_version(data)

    if version == 0:
        entry_hdr_size = 16
    elif version in (1, 2):
        entry_hdr_size = 16 if version == 1 else 20
    else:
        print(f"[skip] unsupported TXA version: {version}")
        return False

    print(f"[v{version}] {os.path.abspath(file_path)} -> {os.path.abspath(output_dir)}")

    os.makedirs(output_dir, exist_ok=True)

    old_index_path = os.path.join(output_dir, "index.txt")
    for _, _, old_name in parse_index(old_index_path):
        old_png = os.path.join(output_dir, f"{old_name}.png")
        if os.path.isfile(old_png):
            os.remove(old_png)

    entries: list[dict] = []

    offset = 32
    for i in range(count):
        if version == 2:
            entry_len, virtual_idx, width, height, data_off, comp_size, decomp_size = \
                struct.unpack_from("<HHHHIII", data, offset)
        else:
            entry_len, virtual_idx, width, height, data_off, comp_size = \
                struct.unpack_from("<HHHHII", data, offset)
            if indexed:
                stride = (width + 3) & ~3
                decomp_size = 1024 + stride * height
            else:
                stride = (4 * width + 15) & ~15
                decomp_size = stride * height

        name_start = offset + entry_hdr_size
        name_end = name_start
        while data[name_end] != 0:
            name_end += 1
        name = data[name_start:name_end].decode('utf-8', errors='replace')

        entries.append({
            'idx': i, 'virtual_idx': virtual_idx, 'name': name,
            'width': width, 'height': height, 'data_offset': data_off,
            'comp_size': comp_size, 'decomp_size': decomp_size,
        })
        offset += entry_len

    write_index(os.path.join(output_dir, "index.txt"),
                [(e['idx'], e['virtual_idx'], e['name']) for e in entries])

    do_swap = version in (0, 1)
    for entry in entries:
        width, height, name = entry['width'], entry['height'], entry['name']
        comp_size, decomp_size = entry['comp_size'], entry['decomp_size']

        print(f"  [{entry['idx']:03}] {name} ({width}x{height})")

        raw_size = comp_size if comp_size > 0 else decomp_size
        raw_data = data[entry['data_offset']:entry['data_offset'] + raw_size]

        if version == 0:
            dec_data = lz77_v0.decompress(raw_data) if comp_size > 0 else raw_data
        elif comp_size > 0:
            dec_data = lz77.decompress(raw_data, seek_bits=12, backseek_nbyte=2)
        else:
            dec_data = raw_data

        if indexed:
            pixel_bytes = decode_dict(dec_data, width, height, do_swap)
        else:
            pixel_bytes = decode_diff(dec_data, width, height, do_swap)

        img = Image.frombytes("RGBA", (width, height), bytes(pixel_bytes))
        png_path = os.path.join(output_dir, f"{name}.png")
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        img.save(png_path)

    return True


def pack_txa(source_dir: str, output_path: str, version: int = 2) -> bool:
    index_entries = parse_index(os.path.join(source_dir, "index.txt"))
    if not index_entries:
        print(f"[error] index.txt not found or empty in {source_dir}")
        return False

    entries: list[tuple[int, int, str, Image.Image]] = []
    for pos, vidx, name in index_entries:
        png_path = os.path.join(source_dir, f"{name}.png")
        if not os.path.isfile(png_path):
            print(f"[error] PNG not found: {png_path}")
            return False
        img = Image.open(png_path).convert("RGBA")
        entries.append((pos, vidx, name, img))

    if not entries:
        print(f"[error] no entries in index.txt")
        return False

    entries.sort(key=lambda x: x[0])

    use_dict = all(eligible_for_dict(e[3]) for e in entries)
    mode_str = "palette" if use_dict else "diff"
    print(f"[v{version}] {os.path.abspath(source_dir)} -> {os.path.abspath(output_path)}  [{mode_str}, {len(entries)} textures]")

    entry_hdr_size = 16 if version in (0, 1) else 20

    head_size = 32
    for _, _, name, _ in entries:
        name_bytes = name.encode('utf-8')
        entry_size = entry_hdr_size + len(name_bytes) + 1
        entry_size = (entry_size + 3) & ~3
        head_size += entry_size
    head_size = (head_size + 15) & ~15

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(b'\x00' * head_size)

        entry_infos: list[dict[str, int | bytes]] = []
        max_decomp_size = 0

        do_swap = version in (0, 1)
        for i, (idx, virtual_idx, name, img) in enumerate(entries):
            pos = f.tell()
            aligned = (pos + 15) & ~15
            if aligned > pos:
                f.write(b'\x00' * (aligned - pos))
            data_offset = f.tell()

            if use_dict:
                enc_data = encode_dict(img, do_swap)
            else:
                enc_data = encode_diff(img, do_swap)

            decomp_size = len(enc_data)

            if version == 0:
                compressed = lz77_v0.compress(enc_data)
            else:
                compressed = lz77.compress(enc_data, offset_bits=12)
            comp_size = len(compressed)

            use_comp = comp_size < decomp_size

            entry_infos.append({
                'name_bytes': name.encode('utf-8'),
                'virtual_idx': virtual_idx,
                'width': img.width,
                'height': img.height,
                'data_offset': data_offset,
                'comp_size': comp_size if use_comp else 0,
                'decomp_size': decomp_size,
            })

            if use_comp:
                f.write(compressed)
            else:
                f.write(enc_data)

            max_decomp_size = max(max_decomp_size, decomp_size)

            status = f"compressed ({decomp_size}->{comp_size})" if use_comp else f"uncompressed ({decomp_size})"
            print(f"  [{i:03}] {name} ({img.width}x{img.height}) {status}")

        file_size = f.tell()
        aligned_size = (file_size + 15) & ~15
        if aligned_size > file_size:
            f.write(b'\x00' * (aligned_size - file_size))
            file_size = aligned_size
        index_size = head_size - 32

        f.seek(0)
        if version == 0:
            header = struct.pack("<IIIIIIII",
                0x34415854, file_size,
                1 if use_dict else 0, len(entries),
                max_decomp_size, 0, 0, 0,
            )
        else:
            header = struct.pack("<IIIIIIII",
                0x34415854, version, file_size,
                1 if use_dict else 0, len(entries),
                max_decomp_size,
                index_size if version == 2 else 0, 0,
            )
        f.write(header)

        for info in entry_infos:
            name_bytes = info['name_bytes']
            total_size = entry_hdr_size + len(name_bytes) + 1
            aligned_size = (total_size + 3) & ~3
            padding = aligned_size - total_size

            if version in (0, 1):
                hdr = struct.pack("<HHHHII",
                    aligned_size, info['virtual_idx'],
                    info['width'], info['height'],
                    info['data_offset'], info['comp_size'],
                )
            else:
                hdr = struct.pack("<HHHHIII",
                    aligned_size, info['virtual_idx'],
                    info['width'], info['height'],
                    info['data_offset'], info['comp_size'],
                    info['decomp_size'],
                )
            f.write(hdr)
            f.write(name_bytes)
            f.write(b'\x00')
            if padding > 0:
                f.write(b'\x00' * padding)

    return True


def unpack(args: argparse.Namespace) -> None:
    if not os.path.exists(args.input):
        print("path does not exist")
        return

    abs_input = os.path.abspath(args.input)

    if os.path.isfile(abs_input):
        name = os.path.splitext(os.path.basename(abs_input))[0]
        out_dir = args.output or os.path.join(os.path.dirname(abs_input), name)
        unpack_txa(abs_input, out_dir)

    elif os.path.isdir(abs_input):
        for root, dirs, files in os.walk(abs_input):
            for file in files:
                if file.lower().endswith(".txa"):
                    src = os.path.join(root, file)
                    name = os.path.splitext(file)[0]
                    if args.output:
                        rel = os.path.relpath(root, abs_input)
                        out_dir = os.path.join(args.output, rel, name) if rel != '.' else os.path.join(args.output, name)
                    else:
                        out_dir = os.path.join(root, name)
                    unpack_txa(src, out_dir)


def pack(args: argparse.Namespace) -> None:
    if not os.path.exists(args.input):
        print("path does not exist")
        return
    abs_input = os.path.abspath(args.input)
    if not os.path.isdir(abs_input):
        print("[error] input must be a directory (containing index.txt and PNG files)")
        return

    if _dir_has_index(abs_input):
        pack_txa(abs_input, args.output, version=args.version)
        return

    found = False
    for root, dirs, _ in os.walk(abs_input):
        if _dir_has_index(root):
            rel = os.path.relpath(root, abs_input)
            out_path = os.path.join(args.output, f"{rel}.txa")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pack_txa(root, out_path, version=args.version)
            found = True
    if not found:
        print("[error] no directories with index.txt found")


def main() -> None:
    parser = argparse.ArgumentParser(description="TXA tool")
    sub = parser.add_subparsers(dest="command", required=True)

    unpack_parser = sub.add_parser("unpack", help="Extract PNGs from a TXA file")
    unpack_parser.add_argument("-i", "--input", required=True, help="Input .txa file or directory")
    unpack_parser.add_argument("-o", "--output", required=True, help="Output directory")

    pack_parser = sub.add_parser("pack", help="Pack PNGs into a TXA file(s)")
    pack_parser.add_argument("-i", "--input", required=True, help="Directory with index.txt, or parent of subdirectories to batch")
    pack_parser.add_argument("-o", "--output", required=True, help="Output .txa path (single) or directory (batch)")
    pack_parser.add_argument("-v", "--version", type=int, choices=[0, 1, 2], required=True,
                             help="TXA version: 0, 1, 2")

    args = parser.parse_args()

    if args.command == "unpack":
        unpack(args)
    elif args.command == "pack":
        pack(args)


if __name__ == "__main__":
    main()
