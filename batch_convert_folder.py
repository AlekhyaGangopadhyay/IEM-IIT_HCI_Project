"""
batch_convert_folder.py

Walks a full Subject/Direction/Type dataset tree and converts every bipolar
.xlsx trial to a 19-channel referential .txt file (via bipolar_to_referential.py),
preserving the folder structure.

Expected input layout (matches the dataset as described):

    Original Dataset/
      Subject 1/
        Backward/
          ARROW_Backward.xlsx
          LETTER_Backward.xlsx
          WORD_Backward.xlsx
        Forward/
          ARROW_Forward.xlsx
          LETTER_Forward.xlsx
          WORD_Forward.xlsx
        Left/  ...
        Right/ ...
      Subject 2/ ...
      Subject 3/ ...

Output mirrors this exactly, one level down, as .txt files:

    referential_output/
      Subject 1/
        Backward/
          ARROW_Backward.txt
          LETTER_Backward.txt
          WORD_Backward.txt
        ...

Usage (Windows example, quote paths with spaces):

    python batch_convert_folder.py "C:\\Users\\Students\\Desktop\\HCI\\Original Dataset" "C:\\Users\\Students\\Desktop\\HCI\\referential_output"
"""

import argparse
from pathlib import Path

from bipolar_to_referential import convert_file

VALID_EXTENSIONS = {".xlsx", ".xls"}


def batch_convert(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    xlsx_files = sorted(
        p for p in input_root.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS
    )

    if not xlsx_files:
        print(f"No .xlsx/.xls files found under {input_root}")
        return

    print(f"Found {len(xlsx_files)} files under {input_root}\n")

    n_ok, n_failed = 0, 0
    for xlsx_path in xlsx_files:
        rel = xlsx_path.relative_to(input_root)
        out_path = (output_root / rel).with_suffix(".txt")
        try:
            result = convert_file(xlsx_path, out_path)
            print(f"OK   {rel}  ({result.shape[0]} x {result.shape[1]}) -> {out_path}")
            n_ok += 1
        except Exception as exc:
            print(f"FAIL {rel}: {exc}")
            n_failed += 1

    print(f"\nDone. {n_ok} converted, {n_failed} failed, out of {len(xlsx_files)} total.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_root", help="Root of the raw bipolar dataset (e.g. '...\\Original Dataset')")
    parser.add_argument("output_root", help="Where to write the referential .txt tree")
    args = parser.parse_args()
    batch_convert(args.input_root, args.output_root)


if __name__ == "__main__":
    main()
