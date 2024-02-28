import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list,
                        genSpoof_list_vops, customTrainMS, customDevNevalMS
                        )
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


# load experiment configurations
# with open(args.config, "r") as f_json:
#     config = json.loads(f_json.read())
    
config = {
    "database_path": "./LA/LA/",
    "asv_score_path": "ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt",
    "model_path": "./models/weights/AASIST-L.pth",
    "batch_size": 8,
    "num_epochs": 20,
    "loss": "CCE",
    "track": "LA",
    "eval_all_best": "True",
    "eval_output": "eval_scores_using_best_dev_model.txt",
    "cudnn_deterministic_toggle": "True",
    "cudnn_benchmark_toggle": "False",
    "model_config": {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
        "gat_dims": [24, 32],
        "pool_ratios": [0.4, 0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    },
    "optim_config": {
        "optimizer": "adam", 
        "amsgrad": "False",
        "base_lr": 0.0001,
        "lr_min": 0.000005,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0001,
        "scheduler": "cosine"
    }
}

model_config = config["model_config"]
optim_config = config["optim_config"]
optim_config["epochs"] = config["num_epochs"]
track = config["track"]
assert track in ["LA", "PA", "DF"], "Invalid track given"
if "eval_all_best" not in config:
    config["eval_all_best"] = "True"
if "freq_aug" not in config:
    config["freq_aug"] = "False"


# make experiment reproducible
seed = 1234
set_seed(seed, config)


# define database related paths
output_dir = Path('/kaggle/working/exp_result')
prefix_2019 = "ASVspoof2019.{}".format(track)
database_path = Path(config["database_path"])
dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))
# define model related paths
model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        "AASIST",
        config["num_epochs"], config["batch_size"])
# if args.comment:
#     model_tag = model_tag + "_{}".format(acomment)

model_tag = output_dir / model_tag
model_save_path = model_tag / "weights"
eval_score_path = model_tag / config["eval_output"]
# writer = SummaryWriter(model_tag)
os.makedirs(model_save_path, exist_ok=True)
# copy(config, model_tag / "config.conf")

from main import get_model, get_loader

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: {}".format(device))
# if device == "cpu":
#     raise ValueError("GPU not detected!")

# define model architecture
model = get_model(model_config, device)

# define dataloaders
trn_loader, dev_loader, eval_loader = get_loader(
        database_path, seed, config)


a, b = next(iter(trn_loader))

print(a.shape)
print(b.shape)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(a[0].numpy())


