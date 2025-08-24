
from nelegolizer.data import LDrawFile, LDrawModel

ldf = LDrawFile.load("fixtures/5935 - Island Hopper.mpd")
merged_model = LDrawModel.merge_multiple_models(ldf.models)
merged_ldf = LDrawFile()
merged_ldf.add_model(merged_model)
merged_ldf.save("fixtures/merged 5935 - Island Hopper.mpd")