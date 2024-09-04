from nelegolizer.data import part_by_filename, part_by_size_label

for key in part_by_filename:
    print("filename key", key)

for size_key in part_by_size_label:
    print("size key", size_key, type(size_key))
    for label_key in part_by_size_label[size_key]:
        print(" label key", label_key, type(label_key))
