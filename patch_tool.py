import csv
import argparse
import os
import sys
import shutil


def patch_binary(binary_path: str, output_path: str | None, csv_path: str, encoding: str):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found")
        return

    if not os.path.exists(binary_path):
        print(f"Error: Input binary {binary_path} not found")
        return

    target_path = binary_path if output_path is None else output_path

    if target_path != binary_path:
        print(f"Copying {binary_path} -> {target_path}...")
        shutil.copyfile(binary_path, target_path)
    else:
        print(f"Patching {binary_path} in-place...")

    print(f"Reading {csv_path}...")
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Patching {target_path}...")

    with open(target_path, 'r+b') as f_bin:
        for i, row in enumerate(rows):
            offset_str = row.get('offset', '').strip()
            length_str = row.get('length', '').strip()
            translation = row.get('translation', '')

            if not offset_str:
                continue

            try:
                offset = int(offset_str, 16)
                max_length = int(length_str)
            except ValueError:
                print(f"Warning: Format error at {offset_str}, skipping")
                continue

            original_text = row.get('text', '').strip()

            if not translation:
                continue

            try:
                f_bin.seek(offset)
                raw_original = f_bin.read(max_length)
            except Exception as e:
                print(f"Error: Failed to read original data at {offset_str}: {e}")
                continue

            try:
                decoded_original = raw_original.decode(encoding)
            except Exception as e:
                print(f"Error: Failed to decode original text at {offset_str}: {e}")
                continue

            if decoded_original.replace('\r\n', '\n') != original_text.replace('\r\n', '\n'):
                print(f"Error: Original text mismatch")
                print(f"  Expected (CSV):  '{original_text}'")
                print(f"  Actual (binary): '{decoded_original}'")
                print(f"  Offset: {offset_str}, Length: {max_length}")
                continue

            try:
                encoded_text = translation.replace('\r\n', '\n').encode(encoding)
            except Exception as e:
                print(f"Encoding error at {offset_str}: {e}")
                continue

            current_len = len(encoded_text)

            if current_len > max_length:
                print(f"Error: Translation too long")
                print(f"  Original: {row.get('text', '')}")
                print(f"  Translation: {translation}")
                print(f"  Offset: {offset_str}, Length: {current_len}, Max: {max_length}")
                continue

            data_to_write = encoded_text + b'\x00' * (max_length - current_len)

            f_bin.seek(offset)
            f_bin.write(data_to_write)

    print("Patching complete!")


def main():
    parser = argparse.ArgumentParser(description='Binary Patching Tool')

    parser.add_argument('-b', '--bin', required=True, help='Input binary file')
    parser.add_argument('-o', '--output', default=None, help='Output binary file (omit for in-place patching)')
    parser.add_argument('-c', '--csv', required=True, help='CSV file path')
    parser.add_argument('-e', '--encoding', default='utf-8', help='Text encoding (utf-8, shift_jis, gbk...)')

    args = parser.parse_args()

    patch_binary(args.bin, args.output, args.csv, args.encoding)


if __name__ == '__main__':
    main()