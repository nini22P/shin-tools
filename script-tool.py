from __future__ import annotations

import pandas as pd
import re
import argparse
import sys
from typing import Optional

SEP: str = "⭕"
RUBY_REGEX: str = r'@b([^@.]+)\.@<([^@>]+)@>'
ARG_REGEX: str = r'@[abcopsuvwxz][^@\n\r.]*\.'
NO_ARG_REGEX: str = r'@[+-/<>[\]ekrty{|}]'
CODE_REGEX: str = f'({ARG_REGEX}|{NO_ARG_REGEX})'

RUBY_PATTERN = re.compile(RUBY_REGEX)
CODE_PATTERN = re.compile(CODE_REGEX)


def to_human(text: str) -> str:
    return RUBY_PATTERN.sub( r'[\1|\2]', text)


def to_game(text: str) -> str:
    return re.sub(r'\[([^|\]]+)\|([^\]]+)\]', r'@b\1.@<\2@>', text)


def has_name_box(parts: list[str]) -> bool:
    if '@r' not in parts:
        return False
    idx = parts.index('@r')
    if idx == 0:
        return False
    prev_content = parts[idx - 1]
    return bool(prev_content and prev_content.strip())


def get_segments(text: str) -> list[str]:
    parts = CODE_PATTERN.split(to_human(str(text)))
    start_idx = parts.index('@r') + 1 if has_name_box(parts) else 0

    segments: list[str] = []
    for p in parts[start_idx:]:
        if not p:
            continue
        if not CODE_PATTERN.match(p) and p.strip():
            segments.append(p.strip())
    return segments


def extract_texts(df_main: pd.DataFrame) -> pd.DataFrame:
    names: set[str] = set()
    rows: list[dict[str, str]] = []

    for _index, row in df_main.iterrows():
        text = str(row.get('s', ''))
        if not text or text == 'nan':
            continue

        parts = CODE_PATTERN.split(to_human(text))
        name = ""

        if has_name_box(parts):
            name = parts[parts.index('@r') - 1].strip()
            if name:
                names.add(name)

        segs = get_segments(text)
        if segs:
            rows.append({
                'index': str(row.get('index', '')),
                'offset': str(row.get('offset', '')),
                'type': str(row.get('source', '')),
                'name': name,
                'text': SEP.join(segs),
                'translated': ""
            })

    name_rows: list[dict[str, str]] = [{
        'index': "",
        'offset': "",
        'type': "name",
        'name': "",
        'text': n,
        'translated': ""
    } for n in sorted(names)]

    cols = ['index', 'offset', 'type', 'name', 'text', 'translated']
    return pd.DataFrame(name_rows + rows)[cols]


def inject_texts(df_main: pd.DataFrame, df_text: pd.DataFrame) -> pd.DataFrame:
    name_dict: dict[str, str] = {}
    trans_dict: dict[tuple[int, str], Optional[list[str]]] = {}
    errors: list[str] = []

    for row_num, (_index, row) in enumerate(df_text.iterrows(), start=2):
        row_type = str(row.get('type', ''))
        txt = str(row.get('text', ''))
        trans = str(row.get('translated', ''))
        idx_val = str(row.get('index', '')).strip()
        offset_val = str(row.get('offset', '')).strip()

        if row_type == 'name':
            if trans and trans != 'nan' and trans.strip():
                name_dict[txt] = trans
            continue

        if not idx_val:
            continue

        try:
            idx_val = int(float(idx_val))
        except ValueError:
            continue

        orig_segs = txt.split(SEP)
        key = (idx_val, offset_val)

        if trans and trans != 'nan' and trans.strip():
            trans_segs = trans.split(SEP)
            if len(trans_segs) == len(orig_segs):
                trans_dict[key] = trans_segs
            else:
                errors.append(f"Row {row_num} (index {idx_val}, offset {offset_val}): segment mismatch ({len(orig_segs)} vs {len(trans_segs)})")
        else:
            trans_dict[key] = None

    if errors:
        raise ValueError("Translation alignment errors:\n" + "\n".join(errors[:10]))

    def process_row(orig_text: str, idx_val: str, offset_val: str) -> str:
        if not idx_val:
            return orig_text

        try:
            idx = int(float(idx_val))
        except ValueError:
            return orig_text

        key = (idx, offset_val)

        if key not in trans_dict:
            return ""

        segs = trans_dict[key]
        if segs is None:
            return ""

        seg_idx = 0
        parts = CODE_PATTERN.split(to_human(orig_text))
        result: list[str] = []
        start = 0

        if has_name_box(parts):
            r_idx = parts.index('@r')
            result.extend(parts[:r_idx - 1])

            orig_name = parts[r_idx - 1]
            name_stripped = orig_name.strip()
            translated_name = name_dict.get(name_stripped, name_stripped)
            result.append(orig_name.replace(name_stripped, to_game(translated_name)))

            result.append(parts[r_idx])
            start = r_idx + 1

        for p in parts[start:]:
            if not p:
                continue

            if CODE_PATTERN.match(p):
                result.append(p)
            elif p.strip():
                if seg_idx < len(segs) and segs[seg_idx].strip():
                    translated_text = to_game(segs[seg_idx].strip())
                    result.append(p.replace(p.strip(), translated_text))
                else:
                    result.append(to_game(p))
                seg_idx += 1
            else:
                result.append(p)

        return "".join(result)

    df_out = df_main.copy()
    translated: list[str] = []
    for row in df_main.itertuples(index=False):
        s_val = getattr(row, 's', None)
        idx_val = getattr(row, 'index', None)
        offset_val = getattr(row, 'offset', None)
        translated.append(process_row(
            str(s_val) if pd.notna(s_val) else '',
            str(idx_val) if pd.notna(idx_val) else '',
            str(offset_val) if pd.notna(offset_val) else '',
        ))
    df_out['translated'] = translated
    return df_out


def cmd_export(main_file: str, text_file: str) -> None:
    df_main = pd.read_csv(main_file, encoding='utf-8', low_memory=False)
    df_text = extract_texts(df_main)
    df_text.to_csv(text_file, index=False, encoding='utf-8')
    print(f"Exported to {text_file}")


def cmd_import(main_file: str, text_file: str) -> None:
    df_main = pd.read_csv(main_file, encoding='utf-8', low_memory=False)
    df_text = pd.read_csv(text_file, encoding='utf-8', dtype=str).fillna("")

    try:
        df_out = inject_texts(df_main, df_text)
        df_out.to_csv(main_file, index=False, encoding='utf-8')
        print(f"Updated {main_file}")
    except ValueError as e:
        print(f"Import failed:\n{e}")
        sys.exit(1)


def cmd_test(main_file: str) -> None:
    print("Running in-memory loop test...")
    df_main = pd.read_csv(main_file, encoding='utf-8', low_memory=False)

    df_text = extract_texts(df_main)
    df_text['translated'] = df_text['text']

    try:
        df_out = inject_texts(df_main, df_text)
    except ValueError as e:
        print(f"Test failed during injection:\n{e}")
        return

    mismatches: list[tuple[str, str, str]] = []
    for idx, row in df_out.iterrows():
        orig = str(row.get('s', ''))
        if not orig or orig == 'nan':
            continue

        trans = str(row.get('translated', ''))
        if orig != trans:
            mismatches.append((str(row.get('index', idx)), orig, trans))

    if not mismatches:
        print("Test passed. All texts match exactly.")
    else:
        print(f"Test failed. {len(mismatches)} mismatches found.")
        for idx_val, orig, trans in mismatches[:5]:
            print(f"Index: {idx_val}\nOriginal: {repr(orig)}\nInjected: {repr(trans)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd', required=True)
    
    parser_exp = subparsers.add_parser('export')
    parser_exp.add_argument('--main', required=True)
    parser_exp.add_argument('--text', required=True)
    
    parser_imp = subparsers.add_parser('import')
    parser_imp.add_argument('--main', required=True)
    parser_imp.add_argument('--text', required=True)
    
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--main', required=True)
    
    args = parser.parse_args()
    
    if args.cmd == 'export': 
        cmd_export(args.main, args.text)
    elif args.cmd == 'import': 
        cmd_import(args.main, args.text)
    elif args.cmd == 'test': 
        cmd_test(args.main)