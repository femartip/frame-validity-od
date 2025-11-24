import json
import os
import pandas as pd
import statistics

def combine_results(data: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    results = {}

    for id in predictions.keys():
        id_int = int(id)
        prediction = predictions[id]
        print(f"Processing {id_int}")
        try:
            instance = data.loc[id_int]
        except KeyError:
            print(f"No entry for {id_int}")
            continue

        instance_dict = instance.to_dict()
        instance_dict["conf"] = statistics.mean(prediction["confidence"]) if prediction["confidence"] != [] else 0.0
        instance_dict["iou"] = prediction["iou"] 
        instance_dict["lrp"] = prediction["lrp"]
        
        results[id_int] = instance_dict

    return pd.DataFrame.from_dict(results, orient="index")


if __name__ == '__main__':
    data_path = "./data/metafeatures.csv"
    predictions_path = "./results/yolo/detections.json"

    data_df = pd.read_csv(data_path, index_col=0)
    with open(predictions_path) as f:
        predictions_dict = json.load(f)

    final_df = combine_results(data_df, predictions_dict)
    print(final_df.head())

    final_df.to_csv("./data/data.csv", index=True)
