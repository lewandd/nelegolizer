import argparse

def replace_prefix_inplace(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("1 15 "):
                line = line.replace("1 15 ", "1 7 ", 1)  # tylko pierwsze wystąpienie
            f.write(line)
    print(f"✅ Zmodyfikowano: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="In-place zamiana '1 16 ' na początku linii na '1 7 ' w jednym lub wielu plikach."
    )
    parser.add_argument(
        "files",
        nargs="+",  # co najmniej jeden plik, może być wiele
        help="Ścieżki do plików do edycji (modyfikowane w miejscu)"
    )
    args = parser.parse_args()

    for file_path in args.files:
        replace_prefix_inplace(file_path)

