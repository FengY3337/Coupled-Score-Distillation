import os
import cv2
import sys
import math
import tqdm
import time
import rembg
import torch
import random
import einops
import torchvision
import tensorboardX

import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from rich.console import Console
from diffusers.utils import deprecate
from diffusers.loaders import AttnProcsLayers
from typing import List, Optional, Tuple, Union
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from lora_unet import UNet2DConditionModel

import GaussianSplatting.Gaussian_utils as G_utils

sys.path.append("..")

def load_input(file, W, H):
    print(f'[INFO] load image from {file}...')
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 3:
        bg_remover = rembg.new_session()
        img = rembg.remove(img, session=bg_remover)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    input_mask = img[..., 3:]
    input_img = img[..., :3] * input_mask + (1 - input_mask)
    input_img = input_img[..., ::-1].copy()

    file_prompt = file.replace("_rgba.png", "_caption.txt")
    if os.path.exists(file_prompt):
        print(f'[INFO] load prompt from {file_prompt}...')
        with open(file_prompt, "r") as f:
            img_prompt = f.read().strip()
    else:
        img_prompt = None

    return input_img, input_mask, img_prompt

def write_video_cv2(file_path, images, fps=25):
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    for img in images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()


class DDIMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler, v_pred=False, x_pred=False):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.v_pred = v_pred
        self.x_pred = x_pred

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            use_clipped_model_output: Optional[bool] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            pose=None,
            shading=None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                if `True` or `False`, see documentation for `DDIMScheduler.step`. If `None`, nothing is passed
                downstream to the scheduler. So use `None` for schedulers which don't support this argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        if (
                generator is not None
                and isinstance(generator, torch.Generator)
                and generator.device.type != self.device.type
                and self.device.type != "mps"
        ):
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{self.device}`, so the `generator` will be ignored. "
                f'Please use `generator=torch.Generator(device="{self.device}")` instead.'
            )
            deprecate(
                "generator.device == 'cpu'",
                "0.12.0",
                message,
            )
            generator = None

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.sample_size, int):
            image_shape = (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size)
        else:
            image_shape = (batch_size, self.unet.in_channels, *self.unet.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        rand_device = "cpu" if self.device.type == "mps" else self.device
        if isinstance(generator, list):
            shape = (1,) + image_shape[1:]
            image = [
                torch.randn(shape, generator=generator[i], device=rand_device, dtype=self.unet.dtype)
                for i in range(batch_size)
            ]
            image = torch.cat(image, dim=0).to(self.device)
        else:
            image = torch.randn(image_shape, generator=generator, device=rand_device, dtype=self.unet.dtype)
            image = image.to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            if pose is None:
                if shading is None:
                    model_output = self.unet(image, t).sample
                else:
                    model_output = self.unet(image, t, shading=shading).sample
            else:
                if shading is None:
                    model_output = self.unet(image, t, c=pose).sample
                else:
                    model_output = self.unet(image, t, c=pose, shading=shading).sample

            if self.v_pred or self.x_pred:
                sqrt_alpha_prod = self.scheduler.alphas_cumprod.to(image.device)[t] ** 0.5
                sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod.to(image.device)[t]) ** 0.5
                if self.v_pred:
                    model_output = sqrt_alpha_prod * model_output + sqrt_one_minus_alpha_prod * image
                elif self.x_pred:
                    model_output = (image - sqrt_alpha_prod * model_output) / sqrt_one_minus_alpha_prod
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

        return image

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

def Sklar_coefficient(Sklar_iter, max_iter, initial_value, finial_value, method='linear'):

    Sklar_iter = min(max(Sklar_iter, 0), max_iter)

    if method == 'linear':
        coef = initial_value + (finial_value - initial_value) * (Sklar_iter / max_iter)
    elif method == 'cosine':
        coef = initial_value + (finial_value - initial_value) * (1 - math.cos(math.pi * Sklar_iter / (max_iter * 2)))
    elif method == 'exponential':
        coef = finial_value + (initial_value - finial_value) * math.exp(-5 * (Sklar_iter / max_iter) ** 2)
    elif method == 'constant':
        coef = finial_value
    else:
        raise ValueError("Unsupported method. Choose from 'linear', 'cosine', 'exponential'.")

    return coef


class Trainer(object):
    def __init__(self,
                 argv,
                 opt,
                 model,
                 guidance,
                 optimizer=None,
                 device=None,
                 val_interval=10,
                 workspace='workspace',
                 use_tensorboardX=True,
                 lora_scheduler_update_every_iter=True,
                 Sklar_model=None,
                 device_Sklar=None
                 ):

        self.writer = None
        self.argv = argv
        self.name = opt.name
        self.opt = opt
        self.workspace = workspace
        self.val_interval = val_interval
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_Sklar = device_Sklar
        self.lora_scheduler_update_every_iter = lora_scheduler_update_every_iter
        self.console = Console()
        self.model = model
        self.guidance = guidance
        self.Sklar_model = Sklar_model

        if self.guidance is not None:
            for p in self.guidance.parameters():
                p.requires_grad = False
            self.prepare_text_embeddings()
        self.optimizer = optimizer
        self.use_lora = opt.use_lora

        if self.use_lora and self.guidance is not None:
            if not opt.v_pred:
                _model_key = "./guidance/pretrained_model/stable-diffusion-2-1-base/"
            else:
                _model_key = "./guidance/pretrained_model/stable-diffusion-2-1/"
            _unet = UNet2DConditionModel.from_pretrained(_model_key + 'unet', low_cpu_mem_usage=False, device_map=None).to(device)
            _unet.requires_grad_(False)
            lora_attn_procs = {}
            for name in _unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else _unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = _unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(_unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = _unet.config.block_out_channels[block_id]
                lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            _unet.set_attn_processor(lora_attn_procs)
            lora_layers = AttnProcsLayers(_unet.attn_processors)

            text_input = self.guidance.tokenizer(opt.text, padding='max_length', max_length=self.guidance.tokenizer.model_max_length,
                                                 truncation=True, return_tensors='pt')
            with torch.no_grad():
                text_embeddings = self.guidance.text_encoder(text_input.input_ids.to(self.device))[0]

            class LoraUnet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.unet = _unet
                    self.sample_size = 64
                    self.in_channels = 4
                    self.device = device
                    self.dtype = torch.float32
                    self.text_embeddings = text_embeddings

                def forward(self, x, t, c=None, shading='albedo'):
                    textemb = einops.repeat(self.text_embeddings, '1 L D -> B L D', B=x.shape[0]).to(device)
                    return self.unet(x, t, encoder_hidden_states=textemb, c=c, shading=shading)

            self._unet = _unet
            self.lora_layers = lora_layers
            self.unet = LoraUnet().to(self.device)
            lora_params = [
                {'params': self.lora_layers.parameters()},
                {'params': self._unet.camera_emb.parameters()},
                {'params': self._unet.normal_emb},
            ]
            self.unet_optimizer = optim.AdamW(lora_params, lr=self.opt.unet_lr)
            warm_up_lr_unet = lambda iter: iter / (self.opt.warm_iters + 1) if iter <= (self.opt.warm_iters + 1) else 1
            self.unet_scheduler = optim.lr_scheduler.LambdaLR(self.unet_optimizer, warm_up_lr_unet)

        self.epoch = 0
        self.epoch_ratio = 0.0
        self.global_iter = 0
        self.local_iter = 0
        self.stats = {
            "epoch_loss": [],
            "epoch_Skalr_loss": [],
            "epoch_lora_loss": [],
            "checkpoints": [],
        }

        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

        self.log(f'[INFO] Opt: {self.opt}')
        self.log(f'[INFO] Cmdline: {self.argv}')
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.workspace}')
        self.log("[INFO] Start training ...")

    def prepare_text_embeddings(self):
        if self.opt.text is None:
            self.log(f"[WARN] text prompt is not provided.")
            self.text_z = None
            return

        if not self.opt.dir_text:
            self.text_z = self.guidance.get_text_embeds([self.opt.text], [self.opt.negative_text])
        else:
            self.text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                text = f"{self.opt.text}, {d} view"

                negative_text = f"{self.opt.negative_text}"
                text_z = self.guidance.get_text_embeds([text], [negative_text])
                self.text_z.append(text_z)

        if self.opt.use_Sklar:
            _ = self.Sklar_model.get_text_embeds([self.opt.text], [self.opt.negative_text])

    def log(self, *args, **kwargs):
        self.console.print(*args, **kwargs)
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush()

    def train_iter(self, data, pbar):
        data_total_pred_depth = []
        data_total_pred_rgb = []
        data_total_loss = torch.tensor(0.0, device=self.device)
        data_total_Skalr_loss = torch.tensor(0.0, device=self.device)
        data_total_lora_loss = 0.0

        poses = data['poses']
        mvps = data['mvps']
        for i in range(poses.shape[0]):
            loss_iter = torch.tensor(0.0, device=self.device)

            Sklar_images = []
            Sklar_poses = []

            if self.opt.GS_dmtet:
                lr_xyz = self.opt.albedo_lr
                shading = 'normal'
                if self.opt.dmtet_finetune:
                    shading = 'albedo'
            else:
                lr_xyz = self.model.gaussians.update_learning_rate(self.global_iter)
                shading = 'albedo'

            self.optimizer.zero_grad()

            pose = poses[i]
            cam = G_utils.MiniCam(pose, data['W'], data['H'], data['fovy'], data['fovx'], data['near'], data['far'], self.device)

            if self.opt.random_bg_color:
                bg_color = torch.rand((3), device=self.device)
            else:
                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0],
                                        dtype=torch.float32, device=self.device)

            if self.opt.GS_dmtet:
                mvp = mvps[i].to(self.device)
                output = self.model.dmtet_render(mvp, data['H'], data['W'], shading, bg_color=bg_color)
            else:
                output = self.model.render(cam, bg_color=bg_color)

            pred_depth = output['depth'].unsqueeze(0)
            data_total_pred_depth.append(pred_depth)
            pred_rgb = output['image'].unsqueeze(0)
            data_total_pred_rgb.append(pred_rgb)

            if self.use_lora:
                q_unet = self.unet
                if self.opt.q_cond:
                    lora_pose = torch.tensor(pose, dtype=torch.float32).view(1, 16).to(self.device)
                else:
                    lora_pose = None
            else:
                q_unet = None
                lora_pose = None

            t5 = False
            if self.opt.t5_iters != -1 and self.global_iter >= self.opt.t5_iters:
                if self.global_iter == self.opt.t5_iters:
                    print("Change into tmax = 500 setting")
                t5 = True

            Sklar_loss = torch.tensor(0.0, device=self.device)
            if self.opt.use_Sklar:
                Sklar_images.append(pred_rgb.to(self.device_Sklar))
                Sklar_poses.append(pose)

                for view_i in range(1, 4):
                    pose_i = G_utils.certain_pose(data['elevations'][i], data['phis'][i] + np.deg2rad(90) * view_i, data['radius'][i])
                    Sklar_poses.append(pose_i)
                    if self.opt.GS_dmtet:
                        mvp_i = G_utils.certain_mvp(data['elevations'][i], data['phis'][i] + np.deg2rad(90) * view_i,
                                                    data['H'], data['W'], data['fovy'], data['near'], data['far'], data['radius'][i])
                        output_i = self.model.dmtet_render(mvp_i[0].to(self.device), data['H'], data['W'], shading, bg_color=bg_color)
                    else:
                        cam_i = G_utils.MiniCam(pose_i, data['W'], data['H'], data['fovy'], data['fovx'], data['near'],
                                                data['far'], self.device)
                        output_i = self.model.render(cam_i, bg_color=bg_color)
                    pred_rgb_i = output_i['image'].unsqueeze(0)
                    Sklar_images.append(pred_rgb_i.to(self.device_Sklar))

                Sklar_images = torch.cat(Sklar_images, dim=0)
                Sklar_poses = torch.from_numpy(np.stack(Sklar_poses, axis=0)).to(self.device_Sklar)
                Sklar_net_loss = self.Sklar_model.train_step(Sklar_images, Sklar_poses, t5)

                Sklar_coef = Sklar_coefficient(self.global_iter, self.opt.Sklar_max_iter, self.opt.Sklar_initial_coef,
                                               self.opt.Sklar_finial_coef, method=self.opt.Sklar_coef_method)
                Sklar_loss = Sklar_coef * Sklar_net_loss.to(self.device)

                loss_iter += Sklar_loss
                data_total_Skalr_loss += Sklar_loss
                self.writer.add_scalar('Sklar_loss_iter', Sklar_loss.item(), self.global_iter)

            if self.opt.dir_text:
                dirs = data['dir']  # [1,]
                text_z = self.text_z[int(dirs)]
            else:
                text_z = self.text_z

            loss, latent = self.guidance.train_step(text_z, pred_rgb, self.opt.scale, q_unet, lora_pose, shading=shading, t5=t5)

            loss_iter += loss
            data_total_loss += loss

            if self.opt.use_tensorboardX:
                self.writer.add_scalar('3D_GS_loss_iter', loss.item(), self.global_iter)

            if self.opt.GS_dmtet:
                if not self.opt.dmtet_finetune:
                    loss_iter = loss_iter + self.opt.lambda_normal * output['normal_loss']
                loss_iter = loss_iter + self.opt.lambda_lap * output['lap_loss']

            loss_iter.backward()
            self.optimizer.step()

            if not self.opt.GS_dmtet:
                if self.global_iter >= self.opt.density_start_iter and self.global_iter <= self.opt.density_end_iter:
                    viewspace_point_tensor, visibility_filter, radii = output["viewspace_points"], output["visibility_filter"], output["radii"]
                    self.model.gaussians.max_radii2D[visibility_filter] = torch.max(self.model.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.model.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            ###### update lora ######
            loss_unet_K = 0.0
            if (self.global_iter % self.opt.K2 == 0) and self.use_lora:
                for _ in range(self.opt.K):
                    self.unet_optimizer.zero_grad()

                    timesteps = torch.randint(0, 1000, (self.opt.unet_bs,), device=self.device).long()
                    with torch.no_grad():
                        latents_clean = latent.expand(self.opt.unet_bs, latent.shape[1], latent.shape[2], latent.shape[3]).contiguous()
                        if self.opt.q_cond:
                            lora_pose = torch.tensor(pose, dtype=torch.float32).view(1, 16).to(self.device)
                            lora_pose = lora_pose.expand(self.opt.unet_bs, 16).contiguous()
                            if random.random() < self.opt.uncond_p:
                                lora_pose = torch.zeros_like(lora_pose)
                        else:
                            lora_pose = None

                    noise = torch.randn(latents_clean.shape, device=self.device)
                    latents_noisy = self.guidance.scheduler.add_noise(latents_clean, noise, timesteps)
                    model_output = self.unet(latents_noisy, timesteps, c=lora_pose, shading=shading).sample

                    if self.opt.v_pred:
                        loss_unet = F.mse_loss(model_output, self.guidance.scheduler.get_velocity(latents_clean, noise, timesteps))
                    else:
                        loss_unet = F.mse_loss(model_output, noise)

                    loss_unet.backward()
                    self.unet_optimizer.step()
                    if self.lora_scheduler_update_every_iter:
                        self.unet_scheduler.step()
                    loss_unet_K += loss_unet.item()

            loss_unet_K_average = loss_unet_K / self.opt.K
            data_total_lora_loss += loss_unet_K_average
            if self.opt.use_tensorboardX and self.use_lora:
                self.writer.add_scalar('Lora_loss_iter', loss_unet_K_average, self.global_iter)

            self.local_iter += 1
            self.global_iter += 1

            pbar.set_description(f"loss_iter={loss_iter.item():.4f}, loss={loss.item():.4f}, Skalr_loss={Sklar_loss.item():.4f}, lr_position={lr_xyz:.6f}, lora_loss={loss_unet_K_average:.4f}")
            pbar.update(1)

        data_pred_depth = torch.cat(data_total_pred_depth, dim=0)
        data_pred_rgb = torch.cat(data_total_pred_rgb, dim=0)
        data_loss = data_total_loss
        data_Sklar_loss = data_total_Skalr_loss
        data_lora_loss = data_total_lora_loss

        return data_pred_rgb, data_pred_depth, data_loss, data_Sklar_loss, data_lora_loss

    def train_iter_initial(self, data, pbar):
        data_total_pred_depth = []
        data_total_pred_rgb = []
        data_total_loss = torch.tensor(0.0, device=self.device)
        data_total_Skalr_loss = torch.tensor(0.0, device=self.device)
        data_total_lora_loss = 0.0

        poses = data['poses']
        mvps = data['mvps']
        for i in range(poses.shape[0]):
            loss_iter = torch.tensor(0.0, device=self.device)
            Sklar_loss = torch.tensor(0.0, device=self.device)
            loss_unet_K_average = torch.tensor(0.0, device=self.device)
            lr_xyz = self.opt.albedo_lr

            shading = 'albedo'
            self.optimizer.zero_grad()

            pose = poses[i]
            cam = G_utils.MiniCam(pose, data['W'], data['H'], data['fovy'], data['fovx'], data['near'], data['far'], self.device)

            if self.opt.random_bg_color:
                bg_color = torch.rand((3), device=self.device)
            else:
                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0],
                                        dtype=torch.float32, device=self.device)

            mvp = mvps[i].to(self.device)
            output = self.model.dmtet_render(mvp, data['H'], data['W'], shading, bg_color=bg_color)
            with torch.no_grad():
                output_GS = self.model.render(cam, bg_color=bg_color)

            pred_depth = output['depth'].unsqueeze(0)
            data_total_pred_depth.append(pred_depth)
            pred_rgb = output['image'].unsqueeze(0)
            data_total_pred_rgb.append(pred_rgb)

            label_pred_rgb = output_GS['image'].detach().float()
            loss = F.mse_loss(label_pred_rgb, pred_rgb[0], reduction='sum')

            loss_iter += loss
            data_total_loss += loss

            loss_iter.backward()
            self.optimizer.step()

            self.local_iter += 1
            self.global_iter += 1

            pbar.set_description(f"loss_iter={loss_iter.item():.4f}, loss={loss.item():.4f}, Skalr_loss={Sklar_loss.item():.4f}, lr_position={lr_xyz:.6f}, lora_loss={loss_unet_K_average.item():.4f}")
            pbar.update(1)

        data_pred_depth = torch.cat(data_total_pred_depth, dim=0)
        data_pred_rgb = torch.cat(data_total_pred_rgb, dim=0)
        data_loss = data_total_loss
        data_Sklar_loss = data_total_Skalr_loss
        data_lora_loss = data_total_lora_loss

        return data_pred_rgb, data_pred_depth, data_loss, data_Sklar_loss, data_lora_loss

    def val_iter(self, data):

        pose = data['poses']
        cam = G_utils.MiniCam(pose, data['W'], data['H'], data['fovy'], data['fovx'], data['near'], data['far'], self.device)
        bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
        if self.opt.GS_dmtet:
            shading = 'normal'
            if self.opt.dmtet_finetune:
                shading = 'albedo'
        else:
            shading = 'albedo'

        if not self.opt.GS_dmtet:
            output = self.model.render(cam, bg_color=bg_color)
        else:
            mvp = data['mvps'].to(self.device)
            output = self.model.dmtet_render(mvp, data['H'], data['W'], shading, bg_color=bg_color)

        iter_pred_depth = output['depth']
        iter_pred_rgb = output['image']

        return iter_pred_rgb, iter_pred_depth

    def save_mesh(self, loader=None, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)
        self.model.export_mesh(save_path, resolution=self.opt.mcubes_resolution, decimate_target=self.opt.decimate_target)

        self.log(f"==> Finished saving mesh.")

    def train_one_epoch(self, train_dataset, max_epoch):

        if self.opt.GS_dmtet:
            self.model.GS_DMTET.train()
            lr_xyz = self.opt.albedo_lr
        else:
            lr_xyz = self.model.gaussians.update_learning_rate(self.global_iter)

        if self.opt.variable_resolution:
            self.epoch_ratio = min(1, self.epoch / max_epoch)
            if self.epoch_ratio < 0.1:
                render_resolution = 128
            elif self.epoch_ratio < 0.3:
                render_resolution = 256
            elif self.epoch_ratio < 0.5:
                render_resolution = 512
            else:
                render_resolution = 1024

            train_dataset.H = render_resolution
            train_dataset.W = render_resolution
        else:
            render_resolution = 512

        loader = train_dataset.dataloader()

        self.log(f"==> Start Training {self.workspace} Epoch {self.epoch}: render_resolution={render_resolution},lr_position={lr_xyz:.6f} ...")

        epoch_total_loss = 0.0
        epoch_total_Sklar_loss = 0.0
        epoch_total_lora_loss = 0.0

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_iter = 0
        for data in loader:
            if (self.opt.GS_dmtet and self.opt.dmtet_finetune) and (self.global_iter < self.opt.MLP_initial_iters):
                pred_rgbs, pred_depths, loss, Sklar_loss, lora_loss = self.train_iter_initial(data, pbar)
            else:
                pred_rgbs, pred_depths, loss, Sklar_loss, lora_loss = self.train_iter(data, pbar)

            epoch_total_loss += loss.item()
            epoch_total_Sklar_loss += Sklar_loss.item()
            epoch_total_lora_loss += lora_loss

        average_loss = epoch_total_loss / self.local_iter
        average_Sklar_loss = epoch_total_Sklar_loss / self.local_iter
        average_lora_loss = epoch_total_lora_loss / self.local_iter

        if self.opt.use_tensorboardX:
            self.writer.add_scalar('3D_GS_loss_epoch', average_loss, self.epoch)
            if self.use_lora:
                self.writer.add_scalar('Lora_loss_epoch', average_lora_loss, self.epoch)
            if self.opt.use_Sklar and (self.opt.Sklar_type == "mvdream"):
                self.writer.add_scalar('Sklar_mvdream_loss_epoch', average_Sklar_loss, self.epoch)

        pbar.set_description(f"epoch_loss_average={average_loss:.4f}, epoch_Sklar_loss_average={average_Sklar_loss:.4f}, epoch_lora_loss_average={average_lora_loss:.4f}")
        self.log(f"==> Epoch {self.epoch}: epoch_loss_average={average_loss:.4f}, epoch_Sklar_loss_average={average_Sklar_loss:.4f}, epoch_lora_loss_average={average_lora_loss:.4f}")
        self.stats["epoch_loss"].append(average_loss)
        self.stats["epoch_Skalr_loss"].append(average_Sklar_loss)
        self.stats["epoch_lora_loss"].append(average_lora_loss)
        pbar.close()

    def val_one_epoch(self, valid_dataset, max_epoch):
        if self.opt.variable_resolution:
            self.epoch_ratio = min(1, self.epoch / max_epoch)
            if self.epoch_ratio < 0.1:
                render_resolution = 128
            elif self.epoch_ratio < 0.3:
                render_resolution = 256
            elif self.epoch_ratio < 0.5:
                render_resolution = 512
            else:
                render_resolution = 1024
            valid_dataset.H = render_resolution
            valid_dataset.W = render_resolution
        else:
            render_resolution = 512

        loader = valid_dataset.dataloader()
        if self.opt.GS_dmtet:
            self.model.GS_DMTET.eval()
        self.log(f"++> Valid {self.workspace} at epoch {self.epoch}, the rendering resolution is {render_resolution}...")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_iter = 0
            pred_imgs_all = None
            pred_depths_all = None
            save_path = os.path.join(self.workspace, 'validation')
            os.makedirs(save_path, exist_ok=True)

            for data in loader:
                pred_rgbs, pred_depths = self.val_iter(data)

                if pred_imgs_all is None:
                    pred_imgs_all = pred_rgbs.unsqueeze(0)
                else:
                    pred_imgs_all = torch.cat([pred_imgs_all, pred_rgbs.unsqueeze(0)], dim=0)

                if pred_depths_all is None:
                    pred_depths_all = pred_depths.unsqueeze(0)
                else:
                    pred_depths_all = torch.cat([pred_depths_all, pred_depths.unsqueeze(0)], dim=0)

                pbar.set_description(f"The {self.local_iter:.4f} has finished!")
                pbar.update(loader.batch_size)
                self.local_iter += 1

            torchvision.utils.save_image(pred_imgs_all, os.path.join(save_path, f'{self.name}_rgb_epoch_{self.epoch:04d}.png'),
                                         nrow=self.opt.val_size)
            torchvision.utils.save_image(pred_depths_all, os.path.join(save_path, f'{self.name}_depth_epoch_{self.epoch:04d}.png'),
                                         nrow=self.opt.val_size)
        pbar.close()
        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def train(self, train_dataset, valid_dataset, test_dataset, max_epoch):

        assert self.text_z is not None, 'Training must provide a text prompt!'

        if self.use_tensorboardX:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "tensorboardX", self.name))

        start_t = time.time()
        for epoch in range(self.epoch, max_epoch):
            self.epoch = epoch
            self.train_one_epoch(train_dataset, max_epoch)

            if (self.epoch + 1) % self.val_interval == 0 or self.epoch == 0:
                self.val_one_epoch(valid_dataset, max_epoch)
                if not self.opt.GS_dmtet:
                    self.save_3D_GS(mode='model')
                    # self.save_3D_GS(mode='geo')
                else:
                    self.save_dmtet()
                    if self.epoch == 0:
                        self.save_3D_GS(mode='model')

            unet_bs = 2
            if ((self.epoch + 1) % self.val_interval == 0 or self.epoch == 0) and self.use_lora:
                pipeline = DDIMPipeline(unet=self.unet, scheduler=self.guidance.scheduler, v_pred=self.opt.v_pred)
                with torch.no_grad():
                    images = pipeline(batch_size=unet_bs, output_type="numpy", shading='albedo')
                    rgb = self.guidance.decode_latents(images)
                img = rgb.detach().permute(0, 2, 3, 1).cpu().numpy()
                img = torch.tensor(img.transpose(0, 3, 1, 2), dtype=torch.float32)
                torchvision.utils.save_image(img, os.path.join(self.workspace, 'validation', f'{self.name}_lora_epoch_{self.epoch:04d}' + ".png"), normalize=True, range=(0, 1))

                if self.opt.GS_dmtet and not self.opt.dmtet_finetune:
                    with torch.no_grad():
                        images = pipeline(batch_size=unet_bs, output_type="numpy", shading="normal")
                        rgb = self.guidance.decode_latents(images)
                    img = rgb.detach().permute(0, 2, 3, 1).cpu().numpy()
                    img = torch.tensor(img.transpose(0, 3, 1, 2), dtype=torch.float32)
                    torchvision.utils.save_image(img, os.path.join(self.workspace, 'validation', f'{self.name}_lora_epoch_{self.epoch:04d}_normal' + ".png"), normalize=True, range=(0, 1))

            if (self.epoch + 1) % self.opt.test_interval == 0 or self.epoch == 0:
                self.test(test_dataset)

            if not self.opt.GS_dmtet:
                if (self.global_iter >= self.opt.density_start_iter) and (self.global_iter <= self.opt.density_end_iter):
                    if self.global_iter % self.opt.densification_interval == 0:
                        self.model.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=self.opt.min_opacity,
                                                               extent=4, max_screen_size=1)
                    if self.global_iter % self.opt.opacity_reset_interval == 0:
                        self.model.gaussians.reset_opacity()

        end_t = time.time()
        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX:
            self.writer.close()

    def test(self, test_dataset, val_dataset=None, main_test=False):

        if main_test:
            save_path = os.path.join(self.workspace, 'testing_test')
            name = 'test_result'
        else:
            save_path = os.path.join(self.workspace, 'training_test')
            name = f'{self.name}_epoch_{self.epoch:03d}'
        os.makedirs(save_path, exist_ok=True)

        if self.opt.GS_dmtet:
            self.model.GS_DMTET.eval()

        if val_dataset is not None:
            val_loader = val_dataset.dataloader()

            self.log(f"==> Start Test, the rendering resolution is 512, and save the image to {save_path}")

            val_pbar = tqdm.tqdm(total=len(val_loader) * val_loader.batch_size,
                                 bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            with torch.no_grad():
                self.local_iter = 0
                pred_imgs_all = None
                pred_depths_all = None

                for data in val_loader:
                    pred_rgbs, pred_depths = self.val_iter(data)

                    if pred_imgs_all is None:
                        pred_imgs_all = pred_rgbs.unsqueeze(0)
                    else:
                        pred_imgs_all = torch.cat([pred_imgs_all, pred_rgbs.unsqueeze(0)], dim=0)

                    if pred_depths_all is None:
                        pred_depths_all = pred_depths.unsqueeze(0)
                    else:
                        pred_depths_all = torch.cat([pred_depths_all, pred_depths.unsqueeze(0)], dim=0)

                    val_pbar.set_description(f"The {self.local_iter:.4f} has finished!")
                    val_pbar.update(val_loader.batch_size)
                    self.local_iter += 1

                imgs_save_path = os.path.join(save_path, 'imgs')
                depths_save_path = os.path.join(save_path, 'depths')

                os.makedirs(imgs_save_path, exist_ok=True)
                os.makedirs(depths_save_path, exist_ok=True)

                torchvision.utils.save_image(pred_imgs_all, os.path.join(save_path, f'test_rgb.png'), nrow=self.opt.val_size)
                torchvision.utils.save_image(pred_depths_all, os.path.join(save_path, f'test_depth.png'), nrow=self.opt.val_size)

                for idx, img in enumerate(pred_imgs_all):
                    torchvision.utils.save_image(img, os.path.join(imgs_save_path, f'{idx}.png'))

                for idx, depth in enumerate(pred_depths_all):
                    torchvision.utils.save_image(depth, os.path.join(depths_save_path, f'{idx}.png'))
            val_pbar.close()

        test_loader = test_dataset.dataloader()
        self.log(f"==> Start Test, the rendering resolution is 512, and save the video to {save_path}")

        all_pred_rgbs = []
        all_pred_depths = []
        test_pbar = tqdm.tqdm(total=len(test_loader) * test_loader.batch_size,
                              bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        with torch.no_grad():

            for i, data in enumerate(test_loader):
                pred_rgbs, pred_depths = self.val_iter(data)

                pred_rgbs = pred_rgbs.detach().permute(1, 2, 0).cpu().numpy()
                pred_rgbs = (pred_rgbs * 255).astype(np.uint8)

                pred_depths = pred_depths.detach().permute(1, 2, 0).cpu().numpy()
                pred_depths = (pred_depths - pred_depths.min()) / (pred_depths.max() - pred_depths.min() + 1e-6)
                pred_depths = (pred_depths * 255).astype(np.uint8)
                pred_depths = cv2.applyColorMap(pred_depths, cv2.COLORMAP_JET)

                all_pred_rgbs.append(pred_rgbs)
                all_pred_depths.append(pred_depths)
                test_pbar.update(test_loader.batch_size)

        test_pbar.close()
        all_preds = np.stack(all_pred_rgbs, axis=0)
        all_preds_depth = np.stack(all_pred_depths, axis=0)

        write_video_cv2(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25)
        write_video_cv2(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25)

        self.log(f"==> Finished Test.")
        if not main_test:
            self.log(f"==> Finished Epoch {self.epoch}.")

    def save_3D_GS(self, mode='geo', texture_size=1024):
        output_dir = os.path.join(self.workspace, 'save_3D_GS')
        os.makedirs(output_dir, exist_ok=True)

        if mode == 'geo':
            path = os.path.join(output_dir, f'{self.name}_geo_mesh_epoch_{self.epoch:04d}.ply')
            mesh = self.model.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)
        else:
            path = os.path.join(output_dir, f'val_3D_GS_{self.name}_epoch_{self.epoch:03d}.ply')
            self.model.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    def save_dmtet(self):
        output_dir = os.path.join(self.workspace, 'GS_dmtet')
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'val_GS_dmtet_{self.name}_epoch_{self.epoch:03d}.pth')

        state = {
            'epoch': self.epoch,
            'global_iter': self.global_iter,
            'scale': self.model.gaussians.scale,
            'stats': self.stats,
            'GS_DMTET': self.model.GS_DMTET.state_dict()
        }
        torch.save(state, path)
