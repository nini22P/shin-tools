from __future__ import annotations

import difflib
import os
import pandas as pd
import re
import argparse
import sys
from typing import Any, Optional

SEP: str = "⭕"
RUBY_REGEX: str = r'@b([^@.]+)\.@<([^@>]+)@>'
ARG_REGEX: str = r'@[abcopsuvwxz][^@\n\r.]*\.'
NO_ARG_REGEX: str = r'@[+-/<>[\]ekrtyi{|}]'
CODE_REGEX: str = f'({ARG_REGEX}|{NO_ARG_REGEX})'

RUBY_PATTERN = re.compile(RUBY_REGEX)
CODE_PATTERN = re.compile(CODE_REGEX)

SKIP_SOURCES = {"saveinfo", "voiceplay"}


def _unescaped_to_escaped(text: str) -> str:
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


def _escaped_to_unescaped(text: str) -> str:
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
                    continue

            if cmd in "+-/<>[]ekrtyi{|}":
                result.append(cmd)
                i += 2
                continue

        if c == '!' or (c.isascii() and c.isprintable()):
            result.append('!' + c)
        else:
            result.append(c)

        i += 1

    return "".join(result)


def _to_human(text: str) -> str:
    return RUBY_PATTERN.sub(r'[\1|\2]', text)


def _to_game(text: str) -> str:
    return re.sub(r'\[([^|\]]+)\|([^\]]+)\]', r'@b\1.@<\2@>', text)


def _has_name(parts: list[str]) -> bool:
    if '@r' not in parts:
        return False
    idx = parts.index('@r')
    if idx == 0:
        return False
    prev_content = parts[idx - 1]
    return bool(prev_content and prev_content.strip())


def _get_segments(text: str) -> list[str]:
    parts = CODE_PATTERN.split(_to_human(str(text)))
    start_idx = parts.index('@r') + 1 if _has_name(parts) else 0

    segments: list[str] = []
    for p in parts[start_idx:]:
        if not p:
            continue
        if not CODE_PATTERN.match(p) and p.strip():
            segments.append(p.strip())
    return segments



def _align_entries(df1: pd.DataFrame, df2: pd.DataFrame, s1: str, s2: str) -> pd.DataFrame:
    def _key(d: Any) -> str:
        return str(d['type']) + "\t" + str(d['text'])

    def _row(r: Any, side: str) -> dict[str, str]:
        return {
            f'index_{side}': str(r['index']) if r is not None else "",
            f'offset_{side}': str(r['offset']) if r is not None else "",
            f'name_{side}': str(r['name']) if r is not None else "",
        }

    entries1 = [(r.to_dict(), _key(r.to_dict())) for _, r in df1.iterrows()]
    entries2 = [(r.to_dict(), _key(r.to_dict())) for _, r in df2.iterrows()]
    keys1 = [k for _, k in entries1]
    keys2 = [k for _, k in entries2]
    matcher = difflib.SequenceMatcher(None, keys1, keys2, autojunk=False)

    rows: list[dict[str, str]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        chunk1 = entries1[i1:i2]
        chunk2 = entries2[j1:j2]

        if tag == "equal":
            for (d1, _), (d2, _) in zip(chunk1, chunk2):
                rows.append(_row(d1, s1) | _row(d2, s2) | {
                    'type': d1['type'], 'text': d1['text'],
                })
            continue

        if tag == "replace" and chunk1 and chunk2 and 0.5 <= len(chunk1) / len(chunk2) <= 2:
            paired = min(len(chunk1), len(chunk2))
            for idx in range(paired):
                d1, _ = chunk1[idx]
                d2, _ = chunk2[idx]
                if str(d1.get('text', '')) == str(d2.get('text', '')):
                    rows.append(_row(d1, s1) | _row(d2, s2) | {
                        'type': d1['type'], 'text': d1['text'],
                    })
                else:
                    rows.append(_row(d1, s1) | _row(None, s2) | {
                        'type': d1['type'], 'text': d1['text'],
                    })
                    rows.append(_row(None, s1) | _row(d2, s2) | {
                        'type': d2['type'], 'text': d2['text'],
                    })
            for d1, _ in chunk1[paired:]:
                rows.append(_row(d1, s1) | _row(None, s2) | {
                    'type': d1['type'], 'text': d1['text'],
                })
            for d2, _ in chunk2[paired:]:
                rows.append(_row(None, s1) | _row(d2, s2) | {
                    'type': d2['type'], 'text': d2['text'],
                })
        else:
            for d1, _ in chunk1:
                rows.append(_row(d1, s1) | _row(None, s2) | {
                    'type': d1['type'], 'text': d1['text'],
                })
            for d2, _ in chunk2:
                rows.append(_row(None, s1) | _row(d2, s2) | {
                    'type': d2['type'], 'text': d2['text'],
                })

    cols = [
        f'index_{s1}', f'offset_{s1}', f'index_{s2}', f'offset_{s2}',
        'type', f'name_{s1}', f'name_{s2}', 'text', 'translation',
    ]
    for row in rows:
        row.setdefault('translation', "")
    return pd.DataFrame(rows, columns=cols)


def _report_mismatches(df_out: pd.DataFrame, label: str) -> int:
    mismatches: list[tuple[str, str, str]] = []
    for idx, row in df_out.iterrows():
        orig = str(row.get('s', ''))
        if not orig or orig == 'nan':
            continue
        trans = str(row.get('translated', ''))
        if orig != trans:
            mismatches.append((str(row.get('index', idx)), orig, trans))
    if not mismatches:
        print(f"  [{label}] All texts match exactly.")
    else:
        print(f"  [{label}] {len(mismatches)} mismatches found.")
        for idx_val, orig, trans in mismatches[:5]:
            print(f"    Index {idx_val}: {repr(orig)} != {repr(trans)}")
    return len(mismatches)


def extract_texts(df_main: pd.DataFrame, escaped: bool) -> pd.DataFrame:
    names: set[str] = set()
    rows: list[dict[str, str]] = []

    for _index, row in df_main.iterrows():
        text = str(row.get('s', ''))
        if not text or text == 'nan':
            continue
        
        souces = str(row.get('source', ''))
        
        if not escaped and souces not in SKIP_SOURCES:
            text = _unescaped_to_escaped(text)

        parts = CODE_PATTERN.split(_to_human(text))
        name = ""

        if _has_name(parts):
            name = parts[parts.index('@r') - 1].strip()
            if name:
                names.add(name)

        segs = _get_segments(text)
        if segs:
            rows.append({
                'index': str(row.get('index', '')),
                'offset': str(row.get('offset', '')),
                'type': str(row.get('source', '')),
                'name': name,
                'text': SEP.join(segs),
                'translation': ""
            })

    name_rows: list[dict[str, str]] = [{
        'index': "",
        'offset': "",
        'type': "name",
        'name': "",
        'text': n,
        'translation': ""
    } for n in sorted(names)]

    cols = ['index', 'offset', 'type', 'name', 'text', 'translation']
    return pd.DataFrame(name_rows + rows, columns=cols)


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
        orig_text = _unescaped_to_escaped(orig_text)

    parts = CODE_PATTERN.split(_to_human(orig_text))
    result: list[str] = []
    seg_idx = 0
    start = 0

    if _has_name(parts):
        r_idx = parts.index('@r')
        result.extend(parts[:r_idx - 1])

        orig_name = parts[r_idx - 1]
        name_stripped = orig_name.strip()
        translated_name = name_dict.get(name_stripped, name_stripped)
        result.append(orig_name.replace(name_stripped, _to_game(translated_name)))

        result.append(parts[r_idx])
        start = r_idx + 1

    for p in parts[start:]:
        if not p:
            continue

        if CODE_PATTERN.match(p):
                result.append(p)
        elif p.strip():
            if seg_idx < len(segs) and segs[seg_idx].strip():
                translated_text = _to_game(segs[seg_idx].strip())
                result.append(p.replace(p.strip(), translated_text))
            else:
                result.append(_to_game(p))
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
        trans = str(row.get('translation', ''))
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
            text = _escaped_to_unescaped(text)
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
    print("Running round-trip test...")
    df_main = pd.read_csv(main_file, encoding='utf-8', low_memory=False)

    df_text = extract_texts(df_main, escaped)
    df_text['translation'] = df_text['text']

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


def cmd_duel_export(main_files: list[str], formats: list[str], text_file: str, suffixes: list[str]) -> None:
    s1, s2 = suffixes
    f1 = formats[0] == 'escaped'
    f2 = formats[1] == 'escaped'
    df1 = pd.read_csv(main_files[0], encoding='utf-8', low_memory=False)
    df2 = pd.read_csv(main_files[1], encoding='utf-8', low_memory=False)
    t1 = extract_texts(df1, f1)
    t2 = extract_texts(df2, f2)
    merged = _align_entries(t1, t2, s1, s2)
    merged.to_csv(text_file, index=False, encoding='utf-8')
    both = ((merged[f'index_{s1}'] != "") & (merged[f'index_{s2}'] != "")).sum()
    only1 = ((merged[f'index_{s1}'] != "") & (merged[f'index_{s2}'] == "")).sum()
    only2 = ((merged[f'index_{s2}'] != "") & (merged[f'index_{s1}'] == "")).sum()
    print(f"Exported {text_file}  ({both} both, {only1} only {s1}, {only2} only {s2})")


def cmd_duel_import(main_files: list[str], formats: list[str], text_file: str, suffixes: list[str]) -> None:
    sides: list[tuple[str, bool, str]] = []
    for i, (f, fmt) in enumerate(zip(main_files, formats)):
        if not f:
            continue
        sides.append((f, fmt == 'escaped', suffixes[i]))
    if not sides:
        print("No valid files to import")
        sys.exit(1)
    df_merged = pd.read_csv(text_file, encoding='utf-8', dtype=str).fillna("")
    cols = ['index', 'offset', 'type', 'name', 'text', 'translation']

    for main_file, fmt_esc, suffix in sides:
        idx_col = f'index_{suffix}'
        mask = (df_merged[idx_col] != "") | (df_merged['type'] == 'name')
        df_side = df_merged.loc[mask, :].copy()
        if df_side.empty:
            continue
        df_side = df_side.rename(columns={
            idx_col: 'index', f'offset_{suffix}': 'offset', f'name_{suffix}': 'name'
        }).loc[:, cols]
        df_main = pd.read_csv(main_file, encoding='utf-8', low_memory=False)
        df_out = inject_texts(df_main, df_side, fmt_esc)
        df_out.to_csv(main_file, index=False, encoding='utf-8')
        print(f"Updated {main_file}")


def cmd_duel_test(main_files: list[str], formats: list[str], suffixes: list[str]) -> None:
    s1, s2 = suffixes
    f1 = formats[0] == 'escaped'
    f2 = formats[1] == 'escaped'
    print(f"=== Dual test: {os.path.basename(main_files[0])} ({formats[0]}) + {os.path.basename(main_files[1])} ({formats[1]}) ===\n")

    df1 = pd.read_csv(main_files[0], encoding='utf-8', low_memory=False)
    df2 = pd.read_csv(main_files[1], encoding='utf-8', low_memory=False)

    t1 = extract_texts(df1, f1)
    t2 = extract_texts(df2, f2)
    merged = _align_entries(t1, t2, s1, s2)
    merged['translation'] = merged['text']

    both = ((merged[f'index_{s1}'] != "") & (merged[f'index_{s2}'] != "")).sum()
    only1 = ((merged[f'index_{s1}'] != "") & (merged[f'index_{s2}'] == "")).sum()
    only2 = ((merged[f'index_{s2}'] != "") & (merged[f'index_{s1}'] == "")).sum()
    print(f"Alignment: {both} both, {only1} only {s1}, {only2} only {s2}\n")

    cols = ['index', 'offset', 'type', 'name', 'text', 'translation']
    total_mismatches = 0
    for main_file, fmt_esc, suffix in [(main_files[0], f1, s1), (main_files[1], f2, s2)]:
        idx_col = f'index_{suffix}'
        mask = (merged[idx_col] != "") | (merged['type'] == 'name')
        df_side = merged.loc[mask, :].copy()
        if df_side.empty:
            continue
        df_side = df_side.rename(columns={
            idx_col: 'index', f'offset_{suffix}': 'offset', f'name_{suffix}': 'name'
        }).loc[:, cols]
        df_main = pd.read_csv(main_file, encoding='utf-8', low_memory=False)
        df_out = inject_texts(df_main, df_side, fmt_esc)
        total_mismatches += _report_mismatches(df_out, suffix)

    if total_mismatches == 0:
        print("\nAll round-trip tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd', required=True)
    
    parser_export = subparsers.add_parser('export')
    parser_export.add_argument('--main', required=True)
    parser_export.add_argument('--text', required=True)
    parser_export.add_argument('--format', default='escaped')
    parser_export.add_argument('--suffix', default='')
    
    parser_import = subparsers.add_parser('import')
    parser_import.add_argument('--main', required=True)
    parser_import.add_argument('--text', required=True)
    parser_import.add_argument('--format', default='escaped')
    parser_import.add_argument('--suffix', default='')
    
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--main', required=True)
    parser_test.add_argument('--format', default='escaped')
    parser_test.add_argument('--suffix', default='')
    
    args = parser.parse_args()
    
    mains = args.main.split(',')
    fmts = args.format.split(',')
    valid_fmts = {'escaped', 'unescaped'}
    for f in fmts:
        if f not in valid_fmts:
            print(f"Invalid format '{f}', must be escaped or unescaped")
            sys.exit(1)
    
    if len(mains) == 1:
        if len(fmts) != 1:
            print("Single file requires single format")
            sys.exit(1)
        escaped = (fmts[0] == 'escaped')
        if args.cmd == 'export': 
            cmd_export(mains[0], args.text, escaped)
        elif args.cmd == 'import': 
            cmd_import(mains[0], args.text, escaped)
        elif args.cmd == 'test': 
            cmd_test(mains[0], escaped)
    elif len(mains) == 2:
        if len(fmts) == 1:
            fmts = [fmts[0], fmts[0]]
        if len(fmts) != 2:
            print("Dual files require 1 or 2 format values")
            sys.exit(1)
        suffixes: list[str] = args.suffix.split(',') if args.suffix else []
        if len(suffixes) == 1:
            suffixes = [suffixes[0], suffixes[0]]
        if len(suffixes) != 2:
            print("Dual files require --suffix with 2 comma-separated values (e.g. --suffix hou,sui)")
            sys.exit(1)
        if args.cmd == 'export': 
            cmd_duel_export(mains, fmts, args.text, suffixes)
        elif args.cmd == 'import': 
            cmd_duel_import(mains, fmts, args.text, suffixes)
        elif args.cmd == 'test': 
            cmd_duel_test(mains, fmts, suffixes)
    else:
        print("--main supports 1 or 2 files (comma-separated)")
        sys.exit(1)