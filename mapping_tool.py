import csv
import os
import json
import struct
import argparse
from typing import Generator, Dict, Set, Tuple, Optional


def _resolve(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(config_path))
    cfg['_base_dir'] = base_dir
    cfg['fnt_path'] = _resolve(base_dir, cfg['fnt_path'])
    cfg['mapping_output'] = _resolve(base_dir, cfg['mapping_output'])
    for csv_cfg in cfg['csv_configs']:
        csv_cfg['input'] = _resolve(base_dir, csv_cfg['input'])
        csv_cfg['output'] = _resolve(base_dir, csv_cfg['output'])
    return cfg

def sjis_generator() -> Generator[int, None, None]:
    for c in range(0x20, 0x7F + 1):
        yield c
    for c in range(0xA0, 0xDF + 1):
        yield c

    for high in range(0x81, 0xA0):
        for low in range(0x40, 0xFD):
            if low == 0x7F: continue
            yield (high << 8) | low

    for high in range(0xE0, 0xEF):
        for low in range(0x40, 0xFD):
            if low == 0x7F: continue
            yield (high << 8) | low

def parse_fnt_inventory(fnt_path: str) -> Tuple[Optional[str], Dict[str, int]]:
    if not os.path.exists(fnt_path):
        print(f"Font file not found: {fnt_path}")
        return None, {}

    with open(fnt_path, 'rb') as f:
        data = f.read()

    if data[0:4] != b'FNT4':
        print("Invalid FNT4 magic header.")
        return None, {}

    if data[0x4:0x8] == b"\x01\x00\x00\x00":
        version = 'v1'
    elif data[0xC:0x10] == b"\x00\x00\x00\x00":
        version = 'v0'
    else:
        print("Unknown FNT version.")
        return None, {}

    print(f"Detected FNT4 Version: {version}")

    first_glyph_offset = struct.unpack('<I', data[0x10:0x14])[0]
    num_chars = (first_glyph_offset - 0x10) // 4
    
    inventory: Dict[str, int] = {}
    seen_offsets: Set[int] = set()
    
    print(f"  Scanning {num_chars} entries in character table...")

    if version == 'v0':
        gen = sjis_generator()
        for i in range(num_chars):
            offset = struct.unpack('<I', data[0x10 + i*4 : 0x14 + i*4])[0]
            
            try:
                code = next(gen)
            except StopIteration:
                break
            
            if offset >= len(data) or offset < 0x10:
                continue

            if offset in seen_offsets:
                continue
            
            seen_offsets.add(offset)

            try:
                if code <= 0xFF:
                    char_obj = code.to_bytes(1, 'big').decode('shift_jis')
                else:
                    char_obj = code.to_bytes(2, 'big').decode('shift_jis')
                inventory[char_obj] = code
            except:
                continue

    else: # v1 (Unicode)
        for i in range(num_chars):
            offset = struct.unpack('<I', data[0x10 + i*4 : 0x14 + i*4])[0]

            if offset >= len(data) or offset < 0x10:
                continue

            if offset in seen_offsets:
                continue
            
            seen_offsets.add(offset)

            try:
                char_obj = chr(i)
                inventory[char_obj] = i
            except:
                continue

    print(f"  Unique Glyphs Found: {len(seen_offsets)}")
    print(f"  Inventory loaded: {len(inventory)} chars")
    
    return version, inventory

def is_cjk_ideograph(char: str) -> bool:
    code_int = ord(char)
    return 0x4E00 <= code_int <= 0x9FFF

def main() -> None:
    parser = argparse.ArgumentParser(description='Create font mapping from FNT4 and CSVs')
    parser.add_argument('config', nargs='?', default='mapping-config.json',
                        help='Path to config JSON file (default: mapping-config.json)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    _version, font_inventory = parse_fnt_inventory(cfg['fnt_path'])
    if not font_inventory:
        print("Failed to load font inventory.")
        return
        
    needed_chars: Set[str] = set() 
    chars_in_csv: Set[str] = set() 
    
    for config in cfg['csv_configs']:
        path = config['input']
        if not os.path.exists(path):
            print(f"CSV not found: {path}")
            continue
            
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in config['original_cols']:
                    val = row.get(col, '')
                    if val:
                        for c in val:
                            chars_in_csv.add(c)
                
                for col in config['translation_cols']:
                    val = row.get(col, '')
                    if val:
                        for c in val:
                            chars_in_csv.add(c)
                            if ord(c) >= 0x80 and c not in font_inventory:
                                needed_chars.add(c)

    potential_slots: list[str] = [
        c for c in font_inventory.keys()
        if is_cjk_ideograph(c) and c not in needed_chars
    ]

    unused_slots: list[str] = [c for c in potential_slots if c not in chars_in_csv]
    low_priority_slots: list[str] = [c for c in potential_slots if c in chars_in_csv]

    unused_slots.sort(key=lambda x: font_inventory[x])
    low_priority_slots.sort(key=lambda x: font_inventory[x])

    final_candidates: list[str] = unused_slots + low_priority_slots
    missing_chars: list[str] = sorted(list(needed_chars))

    print(f"\nMissing characters to map: {len(missing_chars)}")
    print(f"Available slots: Unused({len(unused_slots)}), Low priority({len(low_priority_slots)})")

    if len(missing_chars) > len(final_candidates):
        print(f"⚠️ Warning - Not enough slots available! Missing: {len(missing_chars)}, Candidates: {len(final_candidates)}")
        missing_chars = missing_chars[:len(final_candidates)]

    final_mapping: Dict[str, str] = {}
    trans_table: Dict[int, str] = {}
    
    for i, cn_char in enumerate(missing_chars):
        slot_jp_char = final_candidates[i]
        final_mapping[slot_jp_char] = cn_char
        trans_table[ord(cn_char)] = slot_jp_char

    with open(cfg['mapping_output'], 'w', encoding='utf-8') as f:
        f.write("# Generated Mapping Table for fnt4-tool\n[replace]\n")
        for jp_char, cn_char in final_mapping.items():
            k_s = json.dumps(jp_char, ensure_ascii=False)
            v_s = json.dumps(cn_char, ensure_ascii=False)
            f.write(f"{k_s} = {v_s}\n")

    for config in cfg['csv_configs']:
        path = config['input']
        out_path = config['output']
        if not os.path.exists(path):
            continue
            
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        if fieldnames is None:
            print(f"Error: Could not read fieldnames from {path}")
            continue

        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                for col in config['translation_cols']:
                    val = row.get(col, '')
                    if val:
                        row[col] = val.translate(trans_table)
                writer.writerow(row)
        print(f"Mapped CSV saved: {out_path}")

    print("\nCompleted！")

if __name__ == '__main__':
    main()