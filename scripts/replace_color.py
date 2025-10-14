import argparse

def replace_prefix_inplace(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("1 16 "):
                line = line.replace("1 16 ", "1 7 ", 1)
            if line.startswith("1 15 "):
                line = line.replace("1 15 ", "1 7 ", 1)
            f.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="In-place zamiana '1 16 ' i '1 15 ' na początku linii na '1 7 '")
    parser.add_argument("file", help="Ścieżka do pliku do edycji (modyfikowany w miejscu)")
    args = parser.parse_args()

    replace_prefix_inplace(args.file)

