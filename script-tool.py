from __future__ import annotations

import pandas as pd
import re
import argparse
import sys
from typing import Optional

SEP: str = "⭕"
RUBY_REGEX: str = r'@b([^@.]+)\.@<([^@>]+)@>'
ARG_REGEX: str = r'@[abcopsuvwxz][^@\n\r.]*\.'
NO_ARG_REGEX: str = r'@[+-/<>[\]ekrtyi{|}]'
CODE_REGEX: str = f'({ARG_REGEX}|{NO_ARG_REGEX})'

RUBY_PATTERN = re.compile(RUBY_REGEX)
CODE_PATTERN = re.compile(CODE_REGEX)

SKIP_SOURCES = {"saveinfo", "select_choice", "voiceplay"}


def unescaped_to_escaped(text: str) -> str:
    result: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        c = text[i]

        if c == '!':
            if i + 1 < n:
                result.append(text[i + 1])
                i += 2
            else:
                i += 1
            continue

        if c in "abcopsuvwxz":
            j = i + 1
            while j < n and text[j] != '.':
                j += 1
            if j < n:
                arg = text[i + 1:j]
                result.append(f"@{c}{arg}.")
                i = j + 1
                continue

        if c in "+-/<>[]ekrtyi{|}":
            result.append(f"@{c}")
            i += 1
            continue

        result.append(c)
        i += 1

    return "".join(result)


def escaped_to_unescaped(text: str) -> str:
    result: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        c = text[i]

        if c == '@' and i + 1 < n:
            cmd = text[i + 1]

            if cmd in "abcopsuvwxz":
                j = i + 2
                while j < n and text[j] != '.':
                    j += 1
                if j < n:
                    arg = text[i + 2:j]
                    result.append(f"{cmd}{arg}.")
                    i = j + 1
                    if i < n and text[i] == ' ':
                        result.append(' ')
                        i += 1
                    continue

            if cmd in "+-/<>[]ekrtyi{|}":
                result.append(cmd)
                i += 2
                if i < n and text[i] == ' ':
                    result.append(' ')
                    i += 1
                continue

        if c == '!' or (c.isascii() and c.isprintable()):
            result.append('!' + c)
        else:
            result.append(c)

        i += 1

    return "".join(result)


def to_human(text: str) -> str:
    return RUBY_PATTERN.sub(r'[\1|\2]', text)


def to_game(text: str) -> str:
    return re.sub(r'\[([^|\]]+)\|([^\]]+)\]', r'@b\1.@<\2@>', text)


def has_name(parts: list[str]) -> bool:
    if '@r' not in parts:
        return False
    idx = parts.index('@r')
    if idx == 0:
        return False
    prev_content = parts[idx - 1]
    return bool(prev_content and prev_content.strip())


def get_segments(text: str) -> list[str]:
    parts = CODE_PATTERN.split(to_human(str(text)))
    start_idx = parts.index('@r') + 1 if has_name(parts) else 0

    segments: list[str] = []
    for p in parts[start_idx:]:
        if not p:
            continue
        if not CODE_PATTERN.match(p) and p.strip():
            segments.append(p.strip())
    return segments


def extract_texts(df_main: pd.DataFrame, escaped: bool) -> pd.DataFrame:
    names: set[str] = set()
    rows: list[dict[str, str]] = []

    for _index, row in df_main.iterrows():
        text = str(row.get('s', ''))
        if not text or text == 'nan':
            continue
        
        souces = str(row.get('source', ''))
        
        if not escaped and souces not in SKIP_SOURCES:
            text = unescaped_to_escaped(text)

        parts = CODE_PATTERN.split(to_human(text))
        name = ""

        if has_name(parts):
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


def inject_row(row: pd.Series, trans_dict: dict[tuple[int, str], Optional[list[str]]], name_dict: dict[str, str], escaped: bool = True) -> str:
    orig_text = str(row.get('s', ''))
    idx_val = str(row.get('index', ''))
    offset_val = str(row.get('offset', ''))

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

    souces = str(row.get('source', ''))
    if not escaped and souces not in SKIP_SOURCES:
        orig_text = unescaped_to_escaped(orig_text)

    parts = CODE_PATTERN.split(to_human(orig_text))
    result: list[str] = []
    seg_idx = 0
    start = 0

    if has_name(parts):
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


def inject_texts(df_main: pd.DataFrame, df_text: pd.DataFrame, escaped: bool) -> pd.DataFrame:
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

    df_out = df_main.copy()
    translated: list[str] = []
    for _, row in df_main.iterrows():
        text = inject_row(row, trans_dict, name_dict, escaped)
        souces = str(row.get('source', ''))
        if not escaped and souces not in SKIP_SOURCES:
            text = escaped_to_unescaped(text)
        translated.append(text)
    df_out['translated'] = translated
    return df_out


def cmd_export(main_file: str, text_file: str, escaped: bool) -> None:
    df_main = pd.read_csv(main_file, encoding='utf-8', low_memory=False)
    df_text = extract_texts(df_main, escaped)
    df_text.to_csv(text_file, index=False, encoding='utf-8')
    print(f"Exported to {text_file}")


def cmd_import(main_file: str, text_file: str, escaped: bool) -> None:
    df_main = pd.read_csv(main_file, encoding='utf-8', low_memory=False)
    df_text = pd.read_csv(text_file, encoding='utf-8', dtype=str).fillna("")

    try:
        df_out = inject_texts(df_main, df_text, escaped)
        df_out.to_csv(main_file, index=False, encoding='utf-8')
        print(f"Updated {main_file}")
    except ValueError as e:
        print(f"Import failed:\n{e}")
        sys.exit(1)


def cmd_test(main_file: str, escaped: bool) -> None:
    print("Running in-memory loop test...")
    df_main = pd.read_csv(main_file, encoding='utf-8', low_memory=False)

    df_text = extract_texts(df_main, escaped)
    df_text['translated'] = df_text['text']

    try:
        df_out = inject_texts(df_main, df_text, escaped)
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
        for idx_val, orig, trans in mismatches[:10]:
            print(f"Index: {idx_val}\nOriginal: {repr(orig)}\nInjected: {repr(trans)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd', required=True)
    
    parser_export = subparsers.add_parser('export')
    parser_export.add_argument('--main', required=True)
    parser_export.add_argument('--text', required=True)
    parser_export.add_argument('--format', choices=['escaped', 'unescaped'], default='escaped')
    
    parser_import = subparsers.add_parser('import')
    parser_import.add_argument('--main', required=True)
    parser_import.add_argument('--text', required=True)
    parser_import.add_argument('--format', choices=['escaped', 'unescaped'], default='escaped')
    
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--main', required=True)
    parser_test.add_argument('--format', choices=['escaped', 'unescaped'], default='escaped')
    
    args = parser.parse_args()
    
    escaped: bool = (args.format == 'escaped')
    
    if args.cmd == 'export': 
        cmd_export(args.main, args.text, escaped)
    elif args.cmd == 'import': 
        cmd_import(args.main, args.text, escaped)
    elif args.cmd == 'test': 
        cmd_test(args.main, escaped)