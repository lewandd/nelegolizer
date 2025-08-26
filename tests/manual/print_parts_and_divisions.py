from nelegolizer.data import part_by_id, part_by_filename
from nelegolizer.model import division_by_id, shape_label_part_id_map, shape_label_division_id_map


print("Part by filename")
for filename in part_by_filename:
    print(f"filename: {filename}, part: {part_by_filename[filename]}")

print("Part by id")
for id in part_by_id:
    print(f"id: {id}, part: {part_by_id[id]}")

print("Labels for parts")
for shape in shape_label_part_id_map:
    for label in shape_label_part_id_map[shape]:
        print(f"shape: {shape}, label: {label}, part_id: {shape_label_part_id_map[shape][label]}")

print("Divisions id's")
for id in division_by_id:
    print(f"id: {id} division: {division_by_id[id]}")

print("Labels for divisions")
for shape in shape_label_division_id_map:
    for label in shape_label_division_id_map[shape]:
        print(f"shape: {shape}, label: {label}, division_id: {shape_label_division_id_map[shape][label]}")
