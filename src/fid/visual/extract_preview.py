import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import datasets
from model import MAE
from dreamsim import dreamsim
import torchvision.transforms as T
import argparse
from tool import get_mae_transforms, load_arrow_dataset
# Set up device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
root_path = "/storage/dataset/crello/filter/cluster-10k-corrv2"
dataset = load_arrow_dataset(os.path.join(root_path, "merge"))

# Load MAE model
mae_model = MAE(pretrained_path=root_path).to(DEVICE)
mae_model.load_model("/storage/dataset/crello/filter/cluster-10k-corrv2/fid_weights/visual_mae.pth")
mae_model.eval()

# Load DreamSim model
dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, cache_dir='/storage/cache/dreamsim/models')
dreamsim_model = dreamsim_model.to(DEVICE)
dreamsim_model.eval()

# Set up transforms
args = argparse.Namespace(mode="mae-pretrain", input_size=224, crello_no_aug=True)
_,mae_transform = get_mae_transforms(args)



# Function to extract embeddings
def extract_embeddings(image, mae_model, dreamsim_model, dreamsim_preprocess):
    # MAE embedding
    with torch.no_grad():
        mae_embedding = mae_model.extract_embedding(image.convert("RGB"), DEVICE)
    
    # DreamSim embedding
    dreamsim_input = dreamsim_preprocess(image.convert("RGB")).to(DEVICE)
    with torch.no_grad():
        dreamsim_embedding = dreamsim_model.embed(dreamsim_input)
    
    return mae_embedding, dreamsim_embedding

# Create output directories
output_dir = os.path.join(root_path, "fid_weights")
os.makedirs(os.path.join(output_dir, "preview_embedding_mae"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "preview_embedding_dreamsim"), exist_ok=True)


mae_embed_list = []
dreamsim_embed_list = []
# Process images and save embeddings
for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc="Processing images"):
    image = example['preview']
    mae_emb, dreamsim_emb = extract_embeddings(image, mae_model, dreamsim_model, dreamsim_preprocess)
    mae_embed_list.append(mae_emb)
    dreamsim_embed_list.append(dreamsim_emb)

mae_embeds = torch.stack(mae_embed_list).squeeze(1)
dreamsim_embeds = torch.stack(dreamsim_embed_list).squeeze(1)

with open(os.path.join(output_dir, "preview_embedding_mae.pt"), "wb") as f:
    torch.save(mae_embeds, f)

with open(os.path.join(output_dir, "preview_embedding_dreamsim.pt"), "wb") as f:
    torch.save(dreamsim_embeds, f)

print("Embedding extraction complete!")