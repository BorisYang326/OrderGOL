from tqdm import tqdm
import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
import torch
from torch import Tensor, BoolTensor
from typing import Any, Dict, Optional
from einops import rearrange
import sys
from torch.utils.data import random_split
import subprocess
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)
from src.sampling import sample


def set_input(
    seq: Tensor,
    patches: Any,
    mask: Tensor,
    canvas_: Dict[str, Tensor],
    order_dics: Optional[Dict[str, Tensor]],
    device: torch.device,
):
    B, S = seq.shape[:2]
    seq = seq.to(device)
    patches = patches.to(device)
    mask = mask.to(device)
    canvas = canvas_["attr"].to(device)
    if order_dics is not None:
        order_dics = {key: order_dics[key].to(device) for key in order_dics.keys()}
    return seq, patches, mask, canvas, order_dics


def tokenize_tok(
    dataset: Any,
    seq: Tensor,
    mask: BoolTensor,
    canvas: Tensor,
    is_autoreg: bool = False,
):
    B, S = seq.shape[:2]
    batch_seq = []
    batch_mask = []
    # +1 for bos token
    if is_autoreg:
        max_len = dataset.max_seq_length + 1
        bos = torch.tensor(dataset.bos_token, device=seq.device).unsqueeze(0)
        eos = torch.tensor(dataset.eos_token, device=seq.device).unsqueeze(0)
    else:
        max_len = dataset.max_seq_length
    for i in range(B):
        filter_seq = seq[i][mask[i]]
        canv = canvas[i]
        if is_autoreg:
            seq_with_specials = torch.cat([bos, canv, filter_seq, eos])
        else:
            seq_with_specials = torch.cat([canv, filter_seq])
        seq_packed = (
            torch.zeros(
                max_len,
                dtype=torch.long,
                device=filter_seq.device,
            )
            + dataset.pad_token
        )
        mask_packed = torch.zeros(
            max_len,
            dtype=torch.bool,
            device=filter_seq.device,
        )
        seq_packed[: len(seq_with_specials)] = seq_with_specials
        mask_packed[: len(seq_with_specials)] = True
        batch_seq.append(seq_packed)
        batch_mask.append(mask_packed)
    return torch.stack(batch_seq), torch.stack(batch_mask)


def get_weight_mask(dataset, device, args):
    weight_mask = torch.ones(
        1,
        1,
        dataset.vocab_size,
        device=device,
    )
    if args.is_visual_only:
        img_weight = max(args.visual_balance_factor, 1.0)
        weight_mask[
            :, :, dataset._offset_image[0] : dataset._offset_image[1]
        ] = img_weight
    else:
        img_weight = max(args.visual_balance_factor, 1.0)
        font_weight = max(img_weight / 3, 1.0)
        weight_mask[
            :, :, dataset._offset_image[0] : dataset._offset_image[1]
        ] = img_weight
        weight_mask[
            :, :, dataset._offset_font[0] : dataset._offset_font[1]
        ] = font_weight

        return weight_mask

    return weight_mask


def split_dataset(dataset, validation_ratio=0.0, test_ratio=0.2, random_seed=42):
    len_train = int(len(dataset) * (1 - validation_ratio - test_ratio))
    len_val = int(len(dataset) * validation_ratio)
    len_test = len(dataset) - len_train - len_val
    train_dataset, test_dataset, val_dataset = random_split(
        dataset,
        [len_train, len_test, len_val],
        generator=torch.Generator().manual_seed(random_seed),
    )
    return train_dataset, val_dataset, test_dataset


def reconstruct_and_save(dataloader, design_vae, device, trainer, dataset, sampling_cfg, curr_path, draw_results=True):
    decode_results = {}
    _pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for it, (seq, patches, mask, canvas_, order_dics) in _pbar:
        B = seq.size(0)
        design_vae.eval()
        design_vae.to(device)
        seq, patches, mask, canvas, order_dics = trainer._set_input(
            seq, patches, mask, canvas_, order_dics
        )
        curr_ids = canvas_["id"]

        if design_vae.decoder_type == "at":
            enc_tok, enc_mask = tokenize_tok(dataset, seq, mask, canvas, False)
            dec_tok, dec_mask = tokenize_tok(
                dataset,
                seq,
                mask,
                canvas,
                True if design_vae.decoder_type == "at" else False,
            )
            src = enc_tok
            src_mask = enc_mask
            tgt = dec_tok[:, :-1]
            tgt_mask = dec_mask[:, :-1]
            y = dec_tok[:, 1:]
            logits, mu, logvar = design_vae(
                src, src_mask, tgt, tgt_mask, is_sampling=True
            )
        else:
            enc_tok, enc_mask = tokenize_tok(dataset, seq, mask, canvas, False)
            src = enc_tok
            src_mask = enc_mask
            tgt = enc_tok
            tgt_mask = enc_mask
            y = enc_tok
            logits, mu, logvar = design_vae(src, src_mask, tgt, tgt_mask)

        decode_seq = sample(rearrange(logits, "B S V -> B V S"), sampling_cfg).squeeze(1)
        dec_unpad_seq, dec_canvas, dec_mask_dic = trainer.dataset._detokenize(
            decode_seq.cpu()
        )
        gt_unpad_seq, gt_canvas, gt_mask_dic = trainer.dataset._detokenize(tgt.cpu())
        
        for idx in range(B):
            curr_id = curr_ids[idx]
            decode_results[curr_id] = {
                "gen": {
                    "seq": dec_unpad_seq[idx],
                    "canvas": dec_canvas[idx],
                    "mask_dic": dec_mask_dic,
                },
                "gt": {
                    "seq": gt_unpad_seq[idx],
                    "canvas": gt_canvas[idx],
                    "mask_dic": gt_mask_dic,
                },
            }

    with open(f"{curr_path}/decode_results.pkl", "wb") as f:
        pickle.dump(decode_results, f)
        
    if draw_results:
        os.makedirs(f"{curr_path}/decode_results", exist_ok=True)
        cnt = 0
        for id in decode_results:
            if cnt > 10:
                break
            curr_gen = decode_results[id]["gen"]
            curr_gt = decode_results[id]["gt"]
            gt_svg = trainer.dataset.render(curr_gt['seq'], curr_gt['canvas'])
            gen_svg = trainer.dataset.render(curr_gen['seq'], curr_gen['canvas'])
            with open(f"{curr_path}/decode_results/{id}_gt.svg", "w") as f:
                f.write(gt_svg)
            with open(f"{curr_path}/decode_results/{id}_gen.svg", "w") as f:
                f.write(gen_svg)
            subprocess.run(["inkscape", f"{curr_path}/decode_results/{id}_gt.svg", f"-e={curr_path}/decode_results/{id}_gt.png"])
            subprocess.run(["inkscape", f"{curr_path}/decode_results/{id}_gen.svg", f"-e={curr_path}/decode_results/{id}_gen.png"])
            subprocess.run(
                f"rm {curr_path}/decode_results/{id}_gt.svg",
                shell=True,
            )
            subprocess.run(
                f"rm {curr_path}/decode_results/{id}_gen.svg",
                shell=True,
            )
            cnt += 1


def extract_embeddings_and_save(dataloader, design_vae, device, trainer, dataset, curr_path):
    id2embed = {}
    _pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for it, (seq, patches, mask, canvas_, order_dics) in _pbar:
        B = seq.size(0)
        design_vae.eval()
        design_vae.to(device)
        seq, patches, mask, canvas, order_dics = trainer._set_input(
            seq, patches, mask, canvas_, order_dics
        )
        curr_ids = canvas_["id"]

        if design_vae.decoder_type == "at":
            enc_tok, enc_mask = tokenize_tok(dataset, seq, mask, canvas, False)
            dec_tok, dec_mask = tokenize_tok(
                dataset,
                seq,
                mask,
                canvas,
                True if design_vae.decoder_type == "at" else False,
            )
            src = enc_tok
            src_mask = enc_mask
            tgt = dec_tok[:, :-1]
            tgt_mask = dec_mask[:, :-1]
            y = dec_tok[:, 1:]
        else:
            enc_tok, enc_mask = tokenize_tok(dataset, seq, mask, canvas, False)
            src = enc_tok
            src_mask = enc_mask
            tgt = enc_tok
            tgt_mask = enc_mask
            y = enc_tok

        embeds = design_vae.extract_embedding(src, src_mask)
        embeds_np = embeds.cpu().detach().numpy()

        for idx in range(B):
            curr_id = curr_ids[idx]
            id2embed[curr_id] = embeds_np[idx]

    with open(f"{curr_path}/id2embed.pkl", "wb") as f:
        pickle.dump(id2embed, f)
    


def retrieve_nearest_neighbors(curr_path, trainer, n_neighbors=5, n_cluster=3):
    # Load embeddings
    with open(f"{curr_path}/id2embed.pkl", "rb") as f:
        id2embed = pickle.load(f)

    # Get all IDs and embeddings
    ids = list(id2embed.keys())
    embeddings = np.array(list(id2embed.values()))

    # Normalize embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute similarity matrix
    similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
    
    # Randomly select n_cluster IDs
    random_ids = random.sample(ids, n_cluster)

    results = {}
    for input_id in random_ids:
        idx = ids.index(input_id)
        similarities = similarity_matrix[idx]
        sorted_indices = np.argsort(similarities)[::-1]
        nearest_indices = [i for i in sorted_indices if ids[i] != input_id][:n_neighbors]
        nearest_ids = [ids[i] for i in nearest_indices]
        nearest_distances = [similarities[i] for i in nearest_indices]
        results[input_id] = {'id': nearest_ids, 'distance': nearest_distances}

    # Save the results
    with open(f"{curr_path}/nearest_neighbors.pkl", "wb") as f:
        pickle.dump(results, f)

    with open(f"{curr_path}/decode_results.pkl", "rb") as f:
        decode_results = pickle.load(f)
    
    for i, (input_id, nearest_ids) in enumerate(results.items()):
        input_gt = decode_results[input_id]["gt"]
        input_svg_path = f"{curr_path}/nearest_neighbors/input_{i}/svg"
        input_png_path = f"{curr_path}/nearest_neighbors/input_{i}/png"
        os.makedirs(input_svg_path, exist_ok=True)
        os.makedirs(input_png_path, exist_ok=True)
        input_gt_svg = trainer.dataset.render(input_gt['seq'], input_gt['canvas'])
        
        # Calculate average distance to all other embeddings
        idx = ids.index(input_id)
        avg_distance = np.mean(similarity_matrix[idx])
        input_png_name = f"input_{i}_{avg_distance:.4f}.png"
        
        with open(f"{input_svg_path}/{input_png_name.replace('.png', '.svg')}", "w") as f:
            f.write(input_gt_svg)
        subprocess.run(["inkscape", f"{input_svg_path}/{input_png_name.replace('.png', '.svg')}", f"-e={input_png_path}/{input_png_name}", "-d", "200"])
        
        # Process nearest neighbors
        for j in range(len(nearest_ids['id'])):
            id = nearest_ids['id'][j]
            dis = nearest_ids['distance'][j]
            curr_neighbor_gt = decode_results[id]["gt"]
            curr_neighbor_gt_svg = trainer.dataset.render(curr_neighbor_gt['seq'], curr_neighbor_gt['canvas'])
            with open(f"{input_svg_path}/neighbor_{j}_{dis:.4f}.svg", "w") as f:
                f.write(curr_neighbor_gt_svg)
            subprocess.run(["inkscape", f"{input_svg_path}/neighbor_{j}_{dis:.4f}.svg", f"-e={input_png_path}/neighbor_{j}_{dis:.4f}.png", "-d", "50"])

        # Randomly select two embeddings for each input_id
        random_indices = random.sample([i for i in range(len(ids)) if ids[i] != input_id], 2)
        for k, random_idx in enumerate(random_indices):
            random_id = ids[random_idx]
            random_distance = similarity_matrix[idx][random_idx]
            random_gt = decode_results[random_id]["gt"]
            random_gt_svg = trainer.dataset.render(random_gt['seq'], random_gt['canvas'])
            with open(f"{input_svg_path}/random_{k}_{random_distance:.4f}.svg", "w") as f:
                f.write(random_gt_svg)
            subprocess.run(["inkscape", f"{input_svg_path}/random_{k}_{random_distance:.4f}.svg", f"-e={input_png_path}/random_{k}_{random_distance:.4f}.png", "-d", "50"])
    
