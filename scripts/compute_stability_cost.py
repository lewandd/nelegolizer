from nelegolizer.model.evaluation import compute_stability_cost
import argparse
from nelegolizer.data import initilize_parts, LDrawFile, LDrawModel

def main(model_path: str):
    # load config
    initilize_parts()

    ldf = LDrawFile.load(model_path)
    lbm = LDrawModel.merge_multiple_models(ldf.models)
    bricks = lbm.as_bricks()
    print(f"Stability cost {len(bricks)}: {compute_stability_cost(bricks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to MPD LDraw model")
    args = parser.parse_args()

    main(args.model)