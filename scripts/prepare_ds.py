from datasets import load_dataset
import os
import tqdm
import requests
from PIL import Image
from io import BytesIO
import click

@click.command()
@click.option("--set", type=str, default="train")
def main(set):
    NUM_WORKERS = int(os.environ.get("WORLD_SIZE", 1))
    shard_id = int(os.environ.get("RANK", 0))
    ds = load_dataset("Spawning/PD3M")
    start_idx = 0 if set == "train" else (len(ds["train"]) - 50_000)
    end_idx = (len(ds["train"]) - 50_000) if set == "train" else len(ds["train"])


    range_start = start_idx + (shard_id * ((end_idx - start_idx) // NUM_WORKERS))
    range_end = start_idx + ((shard_id + 1) * ((end_idx - start_idx) // NUM_WORKERS))
    
    dataset = ds["train"]
    # shuffle with seed 
    dataset = dataset.shuffle(seed=0)




    dataset = dataset.skip(range_start).take(range_end - range_start)
    # create shard folder
    shard_folder = f"dataset/{set}/{shard_id}"
    os.makedirs(shard_folder, exist_ok=True)
    for sample in tqdm.tqdm(dataset, desc=f"Processing shard {shard_id}", position=shard_id):
        try:
            # download image from url
            image_bytes = requests.get(sample["url"]).content
            # open the image
            image = Image.open(BytesIO(image_bytes))
            # save the image
            # center crop to 256x256, lets check the width and height
            width, height = image.size
            if width > height:
                # resize smallest dimension to 256 first
                top, left, right, bottom = 0, (width - height) // 2, (width + height) // 2, height
            else:
                top, left, right, bottom = (height - width) // 2, 0, width, (height + width) // 2
            image = image.crop((left, top, right, bottom))
            image = image.resize((256, 256))

            # convert to RGB
            image = image.convert("RGB")

            image.save(f"{shard_folder}/{sample['id']}.jpg")

            # save the caption
            with open(f"{shard_folder}/{sample['id']}.txt", "w") as f:
                f.write(sample["caption"])
        except Exception as e:
            print(f"Error processing sample {sample['id']}: {e}")
            continue



if __name__ == "__main__":
    main()