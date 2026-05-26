import csv
import argparse


def split_by_null(data: bytes):
    start = 0
    for i, b in enumerate(data):
        if b == 0:
            if i > start:
                yield start, data[start:i]
            start = i + 1

    if start < len(data):
        yield start, data[start:]

def split_utf16(data: bytes):
    start = 0
    i = 0
    n = len(data)

    while i < n - 1:
        if data[i] == 0 and data[i+1] == 0:
            if i > start:
                yield start, data[start:i]
            start = i + 2
            i += 2
        else:
            i += 2

    if start < n:
        yield start, data[start:]


def in_encoding_range(text: str, encoding: str):
    if encoding.lower() in ("shift_jis", "sjis"):
        return any(
            '\u3040' <= c <= '\u30ff' or   # 平/片假名
            '\u4e00' <= c <= '\u9fff'      # 汉字
            for c in text
        )

    elif encoding.lower() in ("gbk", "gb2312"):
        return any(
            '\u4e00' <= c <= '\u9fff'
            for c in text
        )

    elif encoding.lower() in ("ascii",):
        return any(ord(c) < 128 for c in text)

    else:
        return True


def scan(data: bytes, decode_enc: str, filter_enc: str):
    results = []

    if decode_enc.lower().startswith("utf-16"):
        chunks = split_utf16(data)
    else:
        chunks = split_by_null(data)

    for offset, chunk in chunks:
        try:
            text = chunk.decode(decode_enc)
        except:
            continue

        if not text.strip():
            continue

        if not in_encoding_range(text, filter_enc):
            continue

        results.append((offset, len(chunk), text))

    return results


def export_csv(results, out_file, encoding):
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["offset", "length", "text", "translation"])

        for offset, length, text in results:
            writer.writerow([hex(offset), length, text, ""])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=True, help="input binary file")
    parser.add_argument("-o", "--output", required=True, help="output csv file")
    parser.add_argument("-e", "--encoding", default="utf-8", help="decode encoding (utf-8, shift_jis, gbk...)")
    parser.add_argument("-f", "--filter", default="utf-8", help="filter character set")

    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = f.read()

    results = scan(data, args.encoding, args.filter)

    export_csv(results, args.output, args.encoding)

    print(f"Found {len(results)} strings")


if __name__ == "__main__":
    main()