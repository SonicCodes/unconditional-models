import os
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import json
from PIL import Image
import click



def build_ds_index(dataset_path):
    dataset = []

    for shard in os.listdir(dataset_path):
        for file in os.listdir(dataset_path + "/" + shard):
            if file.endswith(".jpg"):
                dataset.append(dataset_path + "/" + shard + "/" + file)
    # save as json at dataset_path/index.json
    with open(dataset_path + "/index.json", "w") as f:
        json.dump(dataset, f)


if __name__ == "__main__":
    build_ds_index("./dataset/train")
    build_ds_index("./dataset/valid")