import glob
import os
import json
import tqdm
from PIL import Image
from haishoku.haishoku import Haishoku
from datasets import Features, Value, ClassLabel, Sequence, Dataset
import datasets
import numpy as np
import pickle
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from functools import lru_cache
import concurrent.futures

@lru_cache(maxsize=None)
def load_image(image_path):
    return Image.open(image_path).convert("RGBA")

def convert_color(color):
    # print(color)
    if color.startswith("cmyka"):
        c, m, y, k, a = re.findall(r"\((.*)\)", color)[0].split(", ")
        c, m, y, k, a = int(c), int(m), int(y), int(k), float(a)
        r = 255 * (1 - c) * (1 - k)
        g = 255 * (1 - m) * (1 - k)
        b = 255 * (1 - y) * (1 - k)
        return [float(r), float(g), float(b)]
    else:
        hex_color = color.lstrip("#")
        # Handle short form (e.g., '#03f' needs to be converted to '#0033ff')
        if len(hex_color) == 3:
            hex_color = "".join([2 * c for c in hex_color])
        # Split the string and convert to decimal
        return [float(int(hex_color[i : i + 2], 16)) for i in (0, 2, 4)]

def process_dataset(dataset_flag, sample_num, src_path):
    widths, heights, fonts = [], [], [""]
    pre_sampled_dirs = glob.glob(f"{src_path}/vista_data/{dataset_flag}/*")[-sample_num:]
    for template_path in tqdm.tqdm(pre_sampled_dirs, desc="Collecting features"):
        if not os.path.isdir(template_path) or not os.path.isfile(
            template_path + os.sep + "retrieve.png"
        ):
            continue
        template_id = template_path.split(os.sep)[-1]
        with open(
            template_path + os.sep + template_id + ".json", "r", encoding="utf-8"
        ) as f:
            template_data = json.load(f)

        widths.append(template_data["pixelWidth"])
        heights.append(template_data["pixelHeight"])
        fonts.extend(
            [
                element["font"]
                for element in template_data["template"][0]["elements"]
                if "font" in element.keys()
            ]
        )
    widths = list(set(widths))
    heights = list(set(heights))
    fonts = list(set(fonts))
    types = [
        "svgElement",
        "textElement",
        "imageElement",
        "coloredBackground",
        "maskElement",
    ]
    features = Features(
        {
            "id": Value("string"),
            "length": Value("int64"),
            "canvas_width": ClassLabel(names=widths),
            "canvas_height": ClassLabel(names=heights),
            "type": Sequence(feature=ClassLabel(names=types)),
            "left": Sequence(feature=Value(dtype="float32")),
            "top": Sequence(feature=Value(dtype="float32")),
            "width": Sequence(feature=Value(dtype="float32")),
            "height": Sequence(feature=Value(dtype="float32")),
            "font": Sequence(feature=ClassLabel(names=fonts)),
            "font_size": Sequence(feature=Value(dtype="float32")),
            # "color": Sequence(
            #     feature=Sequence(
            #         feature=Value(dtype="float32", id=None), length=3, id=None
            #     ),
            #     length=-1,
            #     id=None,
            # ),
            # "image": Sequence(
            #     feature=datasets.Image(decode=True, id=None), length=-1, id=None
            # ),
            "preview": datasets.Image(),
            # "retrieve": datasets.Image()
        }
    )
    return features, pre_sampled_dirs

def merge_features(features_list):
    merged_features = {}
    for feature_set in features_list:
        for key, value in feature_set.items():
            if key not in merged_features:
                merged_features[key] = value
            else:
                if isinstance(value, dict) and 'names' in value:
                    merged_features[key]['names'] = list(set(merged_features[key]['names'] + value['names']))
                elif isinstance(value, dict) and 'feature' in value and 'names' in value['feature']:
                    merged_features[key]['feature']['names'] = list(set(merged_features[key]['feature']['names'] + value['feature']['names']))
    return merged_features

def process_template(template_dir, features):
    try:
        types = [
            "svgElement",
            "textElement",
            "imageElement",
            "coloredBackground",
            "maskElement",
        ]
        if not os.path.isdir(template_dir) or not os.path.isfile(
            template_dir + os.sep + "retrieve.png"
        ):
            return None

        template_id = template_dir.split(os.sep)[-1]
        with open(
            template_dir + os.sep + template_id + ".json", "r", encoding="utf-8"
        ) as f:
            template_data = json.load(f)

        valid_elements = [
            element
            for element in template_data["template"][0]["elements"]
            if element["type"] in types
        ]
        for element in valid_elements:
            if "position" not in element.keys():
                element["position"] = {"top": element["top"], "left": element["left"]}

        # Load images sequentially
        # image_paths = sorted(glob.glob(template_dir + os.sep + "out" + os.sep + "*.png"))
        # images = [load_image(path) for path in image_paths]
        
        # Process other data
        data = {
            "id": template_id,
            "length": len(valid_elements),
            "canvas_width": features["canvas_width"].str2int(
                str(template_data["pixelWidth"])
            ),
            "canvas_height": features["canvas_height"].str2int(
                str(template_data["pixelHeight"])
            ),
            "type": [
                features["type"].feature.str2int(element["type"])
                for element in valid_elements
            ],
            "left": [
                element["position"]["left"] / template_data["pixelWidth"]
                for element in valid_elements
            ],
            "top": [
                element["position"]["top"] / template_data["pixelHeight"]
                for element in valid_elements
            ],
            "width": [
                element["width"] / template_data["pixelWidth"] for element in valid_elements
            ],
            "height": [
                element["height"] / template_data["pixelHeight"]
                for element in valid_elements
            ],
            "font": [
                features["font"].feature.str2int(
                    element["font"] if "font" in element.keys() else ""
                )
                for element in valid_elements
            ],
            "font_size": [
                element["fontSize"] if "fontSize" in element.keys() else 0
                for element in valid_elements
            ],
        }
        
        # Process colors
        # colors = []
        # for idx, element in enumerate(valid_elements):
        #     if element["type"] == "textElement":
        #         colors.append(convert_color(element["colorMap"][0]["value"]))
        #     else:
        #         colors.append(list(Haishoku.loadHaishoku(image_paths[idx]).dominant))
        
        # data["color"] = colors
        # data["image"] = images

        # Load preview image
        data["preview"] = load_image(template_dir + os.sep + "preview.png")

        return data
    except Exception as e:
        print(f"Error processing template {template_dir}: {e}")
        return None

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--sample_num", type=int, default=20000)
    argparser.add_argument("--out_path", type=str, default="/storage/dataset/crello/full_phres/raw")
    argparser.add_argument("--src_path", type=str, default="/storage/dataset/crello/crello_hfd")
    args = argparser.parse_args()

    dataset_flags = ["train", "test", "validation"]
    all_features = []
    all_template_dirs = []
    all_data = []

    # Process each dataset and collect features
    for flag in dataset_flags:
        features, pre_sampled_dirs = process_dataset(flag, args.sample_num, args.src_path)
        all_features.append(features)
        all_template_dirs.extend(pre_sampled_dirs)
    # Merge features
    merged_features = merge_features(all_features)

    # Process templates using single-thread approach
    for template_dir in tqdm.tqdm(all_template_dirs, desc="Processing templates"):
        result = process_template(template_dir, merged_features)
        if result:
            all_data.append(result)

    os.makedirs(args.out_path, exist_ok=True)
    Dataset.from_list(all_data, features=Features(merged_features)).save_to_disk(args.out_path)