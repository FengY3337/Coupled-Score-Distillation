import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from guidance.mvdream.camera_utils import get_camera, convert_opengl_to_blender, normalize_camera
from guidance.mvdream.model_zoo import build_model
from guidance.mvdream.ldm.models.diffusion.ddim import DDIMSampler

from diffusers import DDIMScheduler

current_dir = os.path.dirname(os.path.abspath(__file__))

class MVDream(nn.Module):
    def __init__(
        self,
        device,
        opts=None,
        model_name='sd-v2.1-base-4view',
        ckpt_path=None,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.device = device
        self.opts = opts
        self.model_name = model_name
        self.ckpt_path = ckpt_path

        self.model = build_model(self.model_name, ckpt_path=self.ckpt_path).eval().to(self.device)
        self.model.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.dtype = torch.float32
        self.scheduler = DDIMScheduler.from_pretrained(os.path.join(current_dir, "pretrained_model/stable-diffusion-2-1-base/scheduler"), torch_dtype=self.dtype)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        self.embeddings = {}

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts).repeat(4, 1, 1)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts).repeat(4, 1, 1)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds
        return None
    
    def encode_text(self, prompt):
        # prompt: [str]
        embeddings = self.model.get_learned_conditioning(prompt).to(self.device)
        return embeddings
    
    @torch.no_grad()
    def refine(self, pred_rgb, camera,
               guidance_scale=100, steps=50, strength=0.8,
        ):

        batch_size = pred_rgb.shape[0]
        real_batch_size = batch_size // 4
        pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_256.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        camera = camera[:, [0, 2, 1, 3]]  # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)
        camera = camera.repeat(2, 1)

        embeddings = torch.cat([self.embeddings['neg'].repeat(real_batch_size, 1, 1), self.embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        context = {"context": embeddings, "camera": camera, "num_frames": 4}

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)
            
            tt = torch.cat([t.unsqueeze(0).repeat(batch_size)] * 2).to(self.device)

            noise_pred = self.model.apply_model(latent_model_input, tt, context)

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]
        return imgs

    def train_step(
        self,
        pred_rgb,   # [B, C, H, W], B is multiples of 4
        camera,     # [B, 4, 4]
        t5=False,
        guidance_scale=7.5,
    ):
        
        batch_size = pred_rgb.shape[0]
        real_batch_size = batch_size // 4
        pred_rgb = pred_rgb.to(self.dtype)

        # interp to 256x256 to be fed into vae.
        pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode="bilinear", align_corners=False)
        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(pred_rgb_256)

        if t5:
            t = torch.randint(self.min_step, 500 + 1, (real_batch_size,), dtype=torch.long, device=self.device).repeat(4)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (real_batch_size,), dtype=torch.long, device=self.device).repeat(4)
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        camera = camera[:, [0, 2, 1, 3]] # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)

        camera = camera.repeat(2, 1)
        embeddings = torch.cat([self.embeddings['neg'].repeat(real_batch_size, 1, 1), self.embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        context = {"context": embeddings, "camera": camera.float(), "num_frames": 4}

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            noise_pred = self.model.apply_model(latent_model_input, tt, context)

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        if self.opts.Sklar_object == 'latent':
            if self.opts.Sklar_single:
                loss = F.mse_loss(latents[0].float(), target[0], reduction='sum')
            else:
                loss = F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]
        else:
            target = self.decode_latents(target)
            if self.opts.Sklar_single:
                loss = F.mse_loss(pred_rgb_256[0].float(), target[0], reduction='sum')
            else:
                loss = F.mse_loss(pred_rgb_256.float(), target, reduction='sum') / pred_rgb_256.shape[0]

        return loss

    def decode_latents(self, latents):
        imgs = self.model.decode_first_stage(latents)
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256]
        imgs = 2 * imgs - 1
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents # [B, 4, 32, 32]

    @torch.no_grad()
    def prompt_to_img(self, prompts, negative_prompts="", images_condition=None, num_inference_steps=50,
                      custom_denoise=False, custom_max_step=500, guidance_scale=5.0, elevation=0, azimuth_start=0):
        text_embeddings = {}

        text_embeddings['pos'] = self.encode_text(prompts).repeat(4, 1, 1)
        text_embeddings['neg'] = self.encode_text(negative_prompts).repeat(4, 1, 1)

        batch_size = 4
        real_batch_size = batch_size // 4

        camera = get_camera(4, elevation=elevation, azimuth_start=azimuth_start)
        camera = camera.repeat(batch_size // 4, 1).to(self.device)

        camera = camera.repeat(2, 1)
        input_text_embeds = torch.cat([text_embeddings['neg'].repeat(real_batch_size, 1, 1),
                                    text_embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        context = {"context": input_text_embeds, "camera": camera, "num_frames": 4}

        if images_condition is not None:
            images_condition = images_condition.to(torch.float32)
            images_condition_256 = F.interpolate(images_condition, (256, 256), mode="bilinear", align_corners=False).to(self.device)
            latents = self.encode_imgs(images_condition_256)
            noise = torch.randn(latents.shape).to(self.device)
        else:
            latents_noisy = torch.randn((4, 4, 32, 32), device=self.device, dtype=self.dtype)

        if not custom_denoise:
            self.scheduler.set_timesteps(num_inference_steps)
            for i, t in enumerate(self.scheduler.timesteps):
                if i == 0 and (images_condition is not None):
                    latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 2)

                tt = torch.cat([t.unsqueeze(0).repeat(batch_size)] * 2).to(self.device)
                noise_pred = self.model.apply_model(latent_model_input, tt, context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy).prev_sample
        else:
            custom_set = torch.arange(1, custom_max_step, step=(custom_max_step // num_inference_steps), dtype=torch.int32)
            if images_condition is not None:
                noise_t = custom_set[-1].clone().detach().long()
                # noise_t = torch.tensor(50, dtype=torch.long, device=self.device)
                latents_noisy = self.scheduler.add_noise(latents, noise, noise_t)

            for i, t in enumerate(reversed(custom_set)):
                latent_model_input = torch.cat([latents_noisy] * 2)

                tt = torch.cat([t.unsqueeze(0).repeat(batch_size)] * 2).to(self.device)
                noise_pred = self.model.apply_model(latent_model_input, tt, context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                #######
                timestep = t
                prev_timestep = timestep - torch.tensor(custom_max_step // num_inference_steps, dtype=torch.long,
                                                        device=self.device)
                alpha_prod_t = self.alphas[timestep]
                alpha_prod_t_prev = self.alphas[prev_timestep] if prev_timestep >= 0 else self.alphas[0]

                beta_prod_t = 1 - alpha_prod_t

                pred_original_sample = (latents_noisy - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                pred_epsilon = noise_pred

                pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
                latents_noisy = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        imgs = self.decode_latents(latents_noisy)  # [4, 3, 256, 256]
        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    import os
    import argparse
    from PIL import Image
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A parrot perched on a stack of books.")
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument('--custom_denoise', type=bool, default=False)
    parser.add_argument('--custom_max_step', type=int, default=500)
    parser.add_argument('--imgs_input_as_latents', type=bool, default=False)
    parser.add_argument('--guidance_scale', type=float, default=100)
    parser.add_argument("--steps", type=int, default=20)
    opt = parser.parse_args()

    device = torch.device("cuda:1")
    with torch.no_grad():
        sd = MVDream(device)

        if opt.imgs_input_as_latents:
            images_path = ['mv_try_4.png', 'mv_try_1.png', 'mv_try_2.png', 'mv_try_3.png']
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            images = []
            for img_path in images_path:
                img = Image.open(os.path.join('./test_imagedream_data', img_path)).convert("RGB")
                image_tensor = transform(img)
                images.append(image_tensor)
            images_condition = torch.stack(images)
        else:
            images_condition = None

        output_imgs = sd.prompt_to_img(opt.prompt, opt.negative, images_condition=images_condition,
                                       num_inference_steps=opt.steps, custom_denoise=opt.custom_denoise,
                                       custom_max_step=opt.custom_max_step, guidance_scale=opt.guidance_scale)

        grid = np.concatenate([
            np.concatenate([output_imgs[0], output_imgs[1]], axis=1),
            np.concatenate([output_imgs[2], output_imgs[3]], axis=1),
        ], axis=0)

        # visualize image
        plt.imshow(grid)
        plt.show()
