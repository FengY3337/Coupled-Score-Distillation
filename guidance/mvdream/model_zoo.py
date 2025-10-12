''' Utiliy functions to load pre-trained models more easily '''
import os
import pkg_resources
from omegaconf import OmegaConf

import torch
from huggingface_hub import hf_hub_download

from guidance.mvdream.ldm.util import instantiate_from_config
current_dir = os.path.dirname(os.path.abspath(__file__))
pretrained_model_dir = os.path.join(current_dir, "..", "pretrained_model")

PRETRAINED_MODELS = {
    "sd-v2.1-base-4view": {
        "config": "sd-v2-base.yaml",
        "repo_id": os.path.join(pretrained_model_dir, "mvdream/sd-v2.1-base-4view"),
        "filename": "sd-v2.1-base-4view.pt"
    },
    "sd-v1.5-4view": {
        "config": "sd-v1.yaml",
        "repo_id": os.path.join(pretrained_model_dir, "mvdream/sd-v1.5-4view"),
        "filename": "sd-v1.5-4view.pt"
    }
}


def get_config_file(config_path):
    # cfg_file = pkg_resources.resource_filename(
    #     "imagedream", os.path.join("configs", config_path)
    # )
    py_file = os.path.abspath(__file__)
    cfg_file = os.path.join(os.path.dirname(py_file), "configs", config_path)
    print(cfg_file)
    if not os.path.exists(cfg_file):
        raise RuntimeError(f"Config {config_path} not available!")
    return cfg_file

def build_model(model_name, config_path=None, ckpt_path=None, cache_dir=None):
    if (config_path is not None) and (ckpt_path is not None):
        config = OmegaConf.load(config_path)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        return model

    if not model_name in PRETRAINED_MODELS:
        raise RuntimeError(
            f"Model name {model_name} is not a pre-trained model. Available models are:\n- " + \
            "\n- ".join(PRETRAINED_MODELS.keys())
        )
    model_info = PRETRAINED_MODELS[model_name]

    # Instiantiate the model
    print(f"Loading model from config: {model_info['config']}")
    config_file = get_config_file(model_info["config"])
    config = OmegaConf.load(config_file)
    model = instantiate_from_config(config.model)

    # Load pre-trained checkpoint from huggingface
    if not ckpt_path:
        ckpt_path = os.path.join(model_info["repo_id"], model_info["filename"])
        # ckpt_path = hf_hub_download(
        #     repo_id=model_info["repo_id"],
        #     filename=model_info["filename"],
        #     cache_dir=cache_dir
        # )
        print(f"Loading model from cache file: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model
