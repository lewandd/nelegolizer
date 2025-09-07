import argparse
from pathlib import Path
from nelegolizer.model.io import save_model_str
from nelegolizer.model.cnn import net_types

def main():
    parser = argparse.ArgumentParser(description="Script that takes a model " \
                                    "input file and dataset txt file.")
    parser.add_argument(
        "model",
        type=str,
        help="Path to the output model file"
    )
    parser.add_argument(
        "net_type",
        type=str,
        help='Network type'
    )

    args = parser.parse_args()
    model_path = Path(args.model)
    
    #shape = tuple(int(ch) for ch in args.shape)
    #shape = (shape[0]*5, shape[1]*2, shape[2]*5)
    
    net_type = args.net_type
    model = net_types[net_type](input_shape=(30, 15, 30), num_classes=10)

    #model = create_model(shape)
    save_model_str(model_path, model)

if __name__ == "__main__":
    main()