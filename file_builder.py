import argparse
import json

import torch
import pandas as pd
from tqdm import tqdm
import pickle
from bokeh.palettes import d3

from utils.reactions import reaction_fps

parser = argparse.ArgumentParser()
parser.add_argument("--settings", "-s", type=str, help="Path to model settings JSON file")
parser.add_argument("--model", "-m", type=str, help="Path to a saved pytorch model")
parser.add_argument("--test-data", "-t", type=str, help="Path to test data")
parser.add_argument("--output", "-o", type=str, help="Name of the output file")
parser.add_argument("--device", "-d", type=str, default="cpu", help="Device: cpu or cuda")
parser.add_argument("--transformer", default=False, action="store_true",
                    help="A flag whether to transform test SMILES to BERT fingerprints")
parser.add_argument("--remove-repeated", default=False, action="store_true",
                    help="A flag whether to remove repeating compounds from reaction before calculating fingerprints")
args = parser.parse_args()

dev = torch.device(args.device)
loaded_model = torch.load(args.model,
                          map_location=dev)
loaded_model.eval()


def extract_ag_reag_prod(reac_smi: str):
    reactants, agents, products = reac_smi.split(">")
    return reactants.rstrip("."), agents.rstrip("."), products


def count_ag_reag(reac_smi: str):
    reag_smi, ag_smi, _ = extract_ag_reag_prod(reac_smi)
    return len((reag_smi + "." + ag_smi).rstrip(".").split("."))


def remove_repeated_reagents(reac_smi: str):
    reag_smi, ag_smi, prod_smi = extract_ag_reag_prod(reac_smi)
    present_rgs = set()
    refined_reag = []
    refined_ag = []
    for rg in reag_smi.split("."):
        if rg not in present_rgs:
            present_rgs.add(rg)
            refined_reag.append(rg)
    for rg in ag_smi.split("."):
        if rg not in present_rgs:
            present_rgs.add(rg)
            refined_ag.append(rg)
    return f'{".".join(refined_reag)}>{".".join(refined_ag)}>{prod_smi}'


with open("data/visual_validation/rxnClasses.pickle", "rb") as f:
    classes = pickle.load(f)
    classes = {int(k): v for k, v in classes.items()}

data = pd.read_csv(args.test_data, sep=";", header=None)
data.columns = ["smiles", "label"]
all_embs = {"x": [], "y": []}

if args.remove_repeated:
    data["smiles"] = data["smiles"].map(remove_repeated_reagents)

if args.transformer:
    from rxnfp.transformer_fingerprints import (
        get_default_model_and_tokenizer, RXNBERTFingerprintGenerator
    )

    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

    for i in tqdm(range(data.shape[0])):
        smi = data.iloc[i]["smiles"]
        fp = torch.tensor(rxnfp_generator.convert(smi))
        fp.resize_((1, 256))
        with torch.no_grad():
            embs = loaded_model(fp)
            x, y = embs.tolist()[0]
            all_embs["x"].append(x)
            all_embs["y"].append(y)


else:
    with open(args.settings) as f:
        all_settings = json.load(f)
        settings = all_settings["settings"]
    fp_method = settings["fp_method"]
    params = {"n_bits": settings["n_bits"],
              "fp_type": settings["fp_type"],
              "include_agents": settings["include_agents"],
              "agent_weight": settings["agent_weight"],
              "non_agent_weight": settings["non_agent_weight"],
              "bit_ratio_agents": settings["bit_ratio_agents"]
              }
    for i in tqdm(range(data.shape[0])):
        smi = data.iloc[i]["smiles"]
        descriptors = reaction_fps(smi,
                                   fp_method=fp_method,
                                   **params)

        fp = torch.from_numpy(descriptors).float().to(dev)
        fp.resize_((1, settings["n_bits"]))
        with torch.no_grad():
            embs = loaded_model(fp)
            x, y = embs.tolist()[0]
            all_embs["x"].append(x)
            all_embs["y"].append(y)

all_embs = pd.DataFrame(all_embs)
res = pd.concat((data, all_embs), axis=1)
res["alpha"] = 1
res["sizes"] = 5
res["reaction_class"] = res["label"].map(classes)

factors = [v for v in classes.values()]
palette = d3['Category10'][len(factors)]
color_map = {k: v for k, v in zip(factors, palette)}
res["color_transform"] = res["reaction_class"].map(color_map)
res["num_reagents"] = res["smiles"].map(count_ag_reag)

res.to_csv(args.output, sep=",", header=True, index=False)
