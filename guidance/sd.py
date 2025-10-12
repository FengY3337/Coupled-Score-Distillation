import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

logging.set_verbosity_error()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


current_dir = os.path.dirname(os.path.abspath(__file__))

class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.1', opt=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.opt = opt
        self.add_z = None

        print(f'[INFO] loading guidance stable diffusion...')
        if self.sd_version == '2.1':
            model_key = os.path.join(current_dir, "pretrained_model", "stable-diffusion-2-1-base/")
        elif self.sd_version == '2.0':
            model_key = os.path.join(current_dir, "pretrained_model", "stable-diffusion-2-base/")
        elif self.sd_version == '1.5':
            model_key = os.path.join(current_dir, "pretrained_model", "stable-diffusion-v1-5/")
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        self.vae = AutoencoderKL.from_pretrained(model_key + 'vae').to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key + 'tokenizer',  clean_up_tokenization_spaces=False)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key + 'text_encoder').to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key + 'unet').to(self.device)
        
        self.scheduler = DDIMScheduler.from_pretrained(model_key + 'scheduler')

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * opt.t_range[0])
        self.max_step = int(self.num_train_timesteps * opt.t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        print(f'[INFO] loaded guidance stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):

        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=7.5, q_unet=None, pose=None, shading=None, t5=False):

        assert torch.isnan(pred_rgb).sum() == 0, print(pred_rgb)

        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512)

        with torch.no_grad():
            # Anneal time schedule
            if t5:
                t = torch.randint(self.min_step, 500 + 1, [1], dtype=torch.long, device=self.device)
            else:
                # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
                t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
            w = (1 - self.alphas[t])

            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if q_unet is not None:
                if pose is not None:
                    noise_pred_q = q_unet(latents_noisy, t, c=pose, shading=shading).sample
                else:
                    raise NotImplementedError()

                if self.opt.v_pred:
                    sqrt_alpha_prod = self.scheduler.alphas_cumprod.to(self.device)[t] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                    while len(sqrt_alpha_prod.shape) < len(latents_noisy.shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                    sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod.to(self.device)[t]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                    while len(sqrt_one_minus_alpha_prod.shape) < len(latents_noisy.shape):
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                    noise_pred_q = sqrt_alpha_prod * noise_pred_q + sqrt_one_minus_alpha_prod * latents_noisy
            else:
                noise_pred_q = noise
            grad = w * (noise_pred - noise_pred_q)
            grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()

        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        return loss, latents

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=30, guidance_scale=7.5,
                        latents=None, custom_denoise=False, custom_max_step=1000, custom_step=10):

        not_custom_latents = True
        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.config.in_channels, height // 8, width // 8),
                                  device=self.device)
            not_custom_latents = False
        else:
            noise = torch.randn_like(latents)

        if not custom_denoise:
            self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            if not custom_denoise:
                for i, t in enumerate(self.scheduler.timesteps):
                    if i == 0 and not_custom_latents:
                        latents = self.scheduler.add_noise(latents, noise, t)
                    latent_model_input = torch.cat([latents] * 2)

                    with torch.no_grad():
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
            else:
                custom_set = torch.arange(1, custom_max_step, step=(custom_max_step // custom_step), dtype=torch.int32)
                if latents is not None and not_custom_latents:
                    latents = self.scheduler.add_noise(latents, noise, custom_set[-1].clone().detach().long())
                for i, t in enumerate(reversed(custom_set)):
                    latent_model_input = torch.cat([latents] * 2)
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    timestep = t
                    prev_timestep = timestep - torch.tensor(custom_max_step // custom_step, dtype=torch.long, device=self.device)
                    alpha_prod_t = self.alphas[timestep]
                    alpha_prod_t_prev = self.alphas[prev_timestep] if prev_timestep >= 0 else self.alphas[0]

                    beta_prod_t = 1 - alpha_prod_t

                    pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                    pred_epsilon = noise_pred

                    pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
                    latents = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5,
                      latents=None, custom_denoise=False, custom_max_step=1000, custom_step=10):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds, size为[2, 77, 768]
        text_embeds = self.get_text_embeds(prompts, negative_prompts)

        # Text embeds -> img latents，size为[1, 4, 64, 64]
        latents = self.produce_latents(text_embeds, height, width, num_inference_steps, guidance_scale, latents,
                                       custom_denoise, custom_max_step, custom_step)
        
        # Img latents -> imgs，size为[1, 3, 512, 512]
        images = self.decode_latents(latents)

        # Img to Numpy
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype('uint8')

        return images


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='A Gothic church with towering spires and intricate stained glass, 3D, HD')
    parser.add_argument("--image", type=str, default="./test_imagedream_data/1_6.png")
    parser.add_argument('--img_input_as_latents', type=bool, default=False)
    parser.add_argument('--custom_denoise', type=bool, default=False)
    # 500
    parser.add_argument('--custom_max_step', type=int, default=900)
    parser.add_argument('--custom_step', type=int, default=10)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'])
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=512)
    parser.add_argument('--seed', default=None)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--t_range', type=float, default=[0.0, 1.0])
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    opts = parser.parse_args()

    if opts.seed is not None:
        seed_everything(opts.seed)

    try_device = torch.device('cuda')
    sd = StableDiffusion(try_device, opts.sd_version, opts)

    if opts.custom_max_step < opts.custom_step:
        opts.custom_max_step = opts.custom_step

    with torch.no_grad():
        if opts.img_input_as_latents:
            image = Image.open(opts.image).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            image = transform(image).unsqueeze(0).to(try_device)
            latents = sd.encode_imgs(image)
            noise = torch.randn_like(latents, device=try_device)
            noise_1 = torch.randn_like(latents, device=try_device)
            # 200
            latents = sd.scheduler.add_noise(latents, noise, torch.tensor(100, dtype=torch.long, device=try_device))

            image_noise = sd.decode_latents(latents)
            image_noise = image_noise.detach().cpu().permute(0, 2, 3, 1).numpy()
            image_noise = (image_noise * 255).round().astype('uint8')
            plt.imshow(image_noise[0])
            plt.show()

        else:
            latents = None

        try_imgs = sd.prompt_to_img(opts.prompt, opts.negative, opts.H, opts.W, opts.steps, opts.guidance_scale, latents,
                                    opts.custom_denoise, opts.custom_max_step, opts.custom_step)

    plt.imshow(try_imgs[0])
    plt.show()
