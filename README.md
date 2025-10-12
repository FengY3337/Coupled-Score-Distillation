# Coupled-Score-Distillation


The pipeline runs on a machine with **two NVIDIA RTX 3090 GPUs**, and includes three main stages: 3D-GS training, DMTet geometry optimization, and texture fine-tuning.

---

## 🧱 Environment Setup

### Step 1: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Modified Gaussian Splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# kiuikit
pip install git+https://github.com/ashawkey/kiuikit
```

---

## 💾 Step 2: Download Pretrained Models

Download pretrained diffusion models from [Hugging Face](https://huggingface.co/):

```bash
# MVDream
# Directory: ./guidance/pretrained_model/mvdream/
# Download:
# - sd-v2.1-base-4view.pt → ./guidance/pretrained_model/mvdream/sd-v2.1-base-4view/
#   Source: https://huggingface.co/MVDream/MVDream/tree/main
# - open_clip_pytorch_model.bin → ./guidance/pretrained_model/mvdream/laion2b_s32b_b79k/
#   Source: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main

# Stable Diffusion 2.1
# Download files from:
#   https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main
#   https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main
```

---

## 🧠 Step 3: Train 3D Gaussian Splatting

Train initial 3D-GS models using text prompts:

```bash
python main.py --text "A frog lying on a lily pad." \
    --text_short "frog" --iters 4000 \
    --use_lora True --use_Sklar True --Sklar_finial_coef 0.5 --K2 1

python main.py --text "A DSLR photo of a Spiderman dancing ballet, Marvel character HD, highly detailed 3D model." \
    --text_short "Spiderman" --iters 4000 \
    --use_lora True --use_Sklar True --Sklar_finial_coef 0.5 --K2 1

python main.py --text "A church with towering spires and intricate details, 3D." \
    --text_short "church" --iters 4000 \
    --use_lora True --use_Sklar True --Sklar_finial_coef 0.3 --K2 1

python main.py --text "A cute squirrel with a fluffy tail." \
    --text_short "squirrel" --iters 4000 \
    --use_lora True --use_Sklar True --Sklar_finial_coef 0.5 --K2 1

python main.py --text "A dual-edged scythe with fiery blade, weapon, 3D." \
    --text_short "scythe" --iters 4000 \
    --use_lora True --use_Sklar True --Sklar_finial_coef 0.3 --K2 1
```

---

## 🏗️ Step 4: Fine-tune Geometry with DMTet

Use DMTet to optimize mesh geometry:

```bash
# Copy optimized 3D-GS results from save_3D_GS → dmtet_initial/stage_1_results
# Rename as [text_short].ply

python main.py --text "A frog lying on a lily pad." \
    --text_short "frog" --GS_dmtet True \
    --dmtet_iters 15000 --dmtet_t5_iters 5000 \
    --dmtet_init_path "./dmtet_initial/stage_1_results/frog.ply"

python main.py --text "A DSLR photo of a Spiderman dancing ballet, Marvel character HD, highly detailed 3D model." \
    --text_short "Spiderman" --GS_dmtet True \
    --dmtet_iters 15000 --dmtet_t5_iters 5000 \
    --dmtet_init_path "./dmtet_initial/stage_1_results/Spiderman.ply"

python main.py --text "A church with towering spires and intricate details, 3D." \
    --text_short "church" --GS_dmtet True \
    --dmtet_iters 15000 --dmtet_t5_iters 5000 \
    --dmtet_init_path "./dmtet_initial/stage_1_results/church.ply"

python main.py --text "A cute squirrel with a fluffy tail." \
    --text_short "squirrel" --GS_dmtet True \
    --dmtet_iters 15000 --dmtet_t5_iters 5000 \
    --dmtet_init_path "./dmtet_initial/stage_1_results/squirrel.ply"

python main.py --text "A dual-edged scythe with fiery blade, weapon, 3D." \
    --text_short "scythe" --GS_dmtet True \
    --dmtet_iters 15000 --dmtet_t5_iters 5000 \
    --dmtet_init_path "./dmtet_initial/stage_1_results/scythe.ply"
```

---

## 🎨 Step 5: Fine-tune Texture

Refine texture and lighting appearance using fine-tuned DMTet meshes:

```bash
# Copy optimized DMTet meshes
# from: GS_dmtet → dmtet_initial/stage_2_results
# Rename as [text_short].pth

# Copy optimized 3D-GS meshes
# from: save_3D_GS → dmtet_initial/stage_2_results
# Rename as [text_short]_scaled.ply

python main.py --text "A frog lying on a lily pad." \
    --text_short "frog" --GS_dmtet True --dmtet_finetune True \
    --dmtet_iters 3000 --dmtet_t5_iters 1000 \
    --dmtet_init_path_scale "./dmtet_initial/stage_2_results/frog_scaled.ply" \
    --finetune_path "./dmtet_initial/stage_2_results/frog.pth" \
    --use_lora True --use_Sklar True --Sklar_finial_coef 0.1 --K2 10

python main.py --text "A DSLR photo of a Spiderman dancing ballet, Marvel character HD, highly detailed 3D model." \
    --text_short "Spiderman" --GS_dmtet True --dmtet_finetune True \
    --dmtet_iters 3000 --dmtet_t5_iters 1000 \
    --dmtet_init_path_scale "./dmtet_initial/stage_2_results/Spiderman_scaled.ply" \
    --finetune_path "./dmtet_initial/stage_2_results/Spiderman.pth" \
    --use_lora True --use_Sklar True --Sklar_finial_coef 0.1 --K2 10

python main.py --text "A church with towering spires and intricate details, 3D." \
    --text_short "church" --GS_dmtet True --dmtet_finetune True \
    --dmtet_iters 3000 --dmtet_t5_iters 1000 \
    --dmtet_init_path_scale "./dmtet_initial/stage_2_results/church_scaled.ply" \
    --finetune_path "./dmtet_initial/stage_2_results/church.pth" \
    --use_lora True --use_Sklar True --Sklar_finial_coef 0.1 --K2 10

python main.py --text "A cute squirrel with a fluffy tail." \
    --text_short "squirrel" --GS_dmtet True --dmtet_finetune True \
    --dmtet_iters 3000 --dmtet_t5_iters 1000 \
    --dmtet_init_path_scale "./dmtet_initial/stage_2_results/squirrel_scaled.ply" \
    --finetune_path "./dmtet_initial/stage_2_results/squirrel.pth" \
    --use_lora True --use_Sklar True --Sklar_finial_coef 0.1 --K2 10

python main.py --text "A dual-edged scythe with fiery blade, weapon, 3D." \
    --text_short "scythe" --GS_dmtet True --dmtet_finetune True \
    --dmtet_iters 3000 --dmtet_t5_iters 1000 \
    --dmtet_init_path_scale "./dmtet_initial/stage_2_results/scythe_scaled.ply" \
    --finetune_path "./dmtet_initial/stage_2_results/scythe.pth" \
    --use_lora True --use_Sklar True --Sklar_finial_coef 0.1 --K2 10
```

---

## ⚙️ System Requirements

* **Python:** 3.8+
* **CUDA:** 11.3+
* **GPU:** Dual NVIDIA RTX 3090

Ensure that all directory paths (`save_3D_GS`, `GS_dmtet`, `dmtet_initial`) are consistent with the above examples.

---

## 📜 Citation

If you use this repository in your research, please cite:

```bibtex
@article{yang2025sklarguidance,
  title={Bridging Geometry-Coherent Text-to-3D Generation with Multi-View Diffusion Priors and Gaussian Splatting},
  author={Feng Yang and Wenliang Qian and Wangmeng Zuo and Hui Li},
  journal={arXiv preprint arXiv:2505.04262},
  year={2025},
  url= {https://arxiv.org/abs/2505.04262}
}
```

---

## 🧠 Acknowledgements

This work builds upon the excellent open-source contributions from:

* [MVDream](https://huggingface.co/MVDream/MVDream)
* [Gaussian Splatting](https://github.com/ashawkey/diff-gaussian-rasterization)
* [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1)

---

**Author:** Feng Yang
**Affiliation:** Harbin Institute of Technology, China
**Contact:** [yangfengdr@outlook.com]
