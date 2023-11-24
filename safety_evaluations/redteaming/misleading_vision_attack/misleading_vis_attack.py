import torch
import os
import numpy as np
import open_clip
from PIL import Image
from torch import optim
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop
from tqdm import tqdm
from torchvision import utils
import argparse
from transformers import CLIPProcessor, CLIPModel

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float,
                    nargs="+",
                    default=[1.0], )

parser.add_argument("--pics_range", type=int,
                    nargs="+",
                    default=[0, 200], )
parser.add_argument("--lr", type=float,
                    default=1e-3, )
parser.add_argument("--misleading_obj", type=str,
                    default='dog')
parser.add_argument("--noise_bound", type=int,
                    default=64)
parser.add_argument("--input_folder", type=str,
                    default="path/to/attack-bard/NIPS17")
parser.add_argument("--output_folder", type=str,
                    default="./adversarial_images")
parser.add_argument("--device", type=str,
                    default="cuda:0")

parser.add_argument("--mix_obj",
                    action="store_true")

# Normalization tools
def inverse_normalize(image, mean, std):
    image = image.clone()
    for i in range(3):
        image[i] = image[i] * std[i] + mean[i]
    return image

def normalize(images, mean, std):
    mean = list(mean)
    std = list(std)
    mean = torch.tensor(mean).cuda()
    std = torch.tensor(std).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images, mean, std):
    mean = list(mean)
    std = list(std)
    mean = torch.tensor(mean).cuda()
    std = torch.tensor(std).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

def inverse_transform(image_tensor, mean, std):
    inverse_transform = transforms.Compose([
    transforms.Lambda(lambda x: inverse_normalize(x, mean, std)),
    transforms.ToPILImage()
])
    image = inverse_transform(image_tensor)
    return image

def main(args):
    misleading_obj = args.misleading_obj if not args.mix_obj else "mixobj"
    noise_bound = args.noise_bound

    folder_path = args.input_folder
    folder_out = args.output_folder

    device = args.device
    os.makedirs(folder_out, exist_ok=True)
    file_list = os.listdir(folder_path)

    # CLIP_ViT_L_14
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    torch.autograd.set_detect_anomaly(True)
    model = model.to(device)

    image_mean = getattr(model.visual, 'image_mean', None)
    image_std = getattr(model.visual, 'image_std', None)

    model.eval()
    if not args.mix_obj:
        text = tokenizer([misleading_obj, misleading_obj, misleading_obj]).to(device)
    else:
        text = tokenizer(["dog", "coconut", "spaceship"]).to(device)
    logit_scale = model.logit_scale.exp().to(device)

    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)
    for param in model.parameters():
        param.requires_grad = False


    for num in range(args.pics_range[0], args.pics_range[1]):
        for alpha in args.alpha:
            image_path = os.path.join(folder_path, f"{num}.png")
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            
            named_parameters = list(model.named_parameters())
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
            
            image.requires_grad = False

            noise_param =  torch.randn_like(image).to(device)
            image_d = denormalize(image, image_mean, image_std).clone().to(device)
            noise_param.data = (noise_param.data + image_d).clamp(0, 1) - image_d.data
            noise_param.requires_grad_(True)
            noise_param.retain_grad()

            optimizer = optim.AdamW(
                [
                    {"params": noise_param , "weight_decay": 0.},
                ],
                lr=args.lr
            )
            
            file_name = num
            ima_p = os.path.join(folder_out, f"{num}.png")
            for epoch in tqdm(range(1000)):
                total_loss = 0
                optimizer.zero_grad() 
                noisy_image = normalize(image_d + noise_param * alpha, image_mean, image_std)
                image_features = model.encode_image(noisy_image)
                text_features = model.encode_text(text)
                image_features = torch.div(image_features.clone(), image_features.norm(dim=-1, keepdim=True))
                text_features /= text_features.norm(dim=-1, keepdim=True)
                labels = torch.arange(image_features.shape[0], device=device, dtype=torch.long)
                logits_per_image =  image_features @ text_features.T
                logits_per_text =  text_features @ image_features.T
                logits_per_image = F.softmax(logits_per_image, dim = -1)
                logits_per_text = F.softmax(logits_per_text.T, dim = -1)

                match_loss =  (
                    F.cross_entropy(logits_per_image, torch.tensor([1], device=device, dtype=torch.long)) +
                    F.cross_entropy(logits_per_text, torch.tensor([1], device=device, dtype=torch.long))
                ) / 2
                total_loss += match_loss 
                m_loss = nn.MSELoss()(noisy_image, image)

                total_loss += m_loss
                noise_loss = torch.mean(torch.square(noise_param)) 
                total_loss += noise_loss
                match_loss.backward(retain_graph=True)
                if epoch % 200 == 0:
                    print(f"""total_loss: {total_loss}; match_loss: {match_loss}; noise_loss: {noise_loss}; m_loss: {m_loss}""")
                noise_param.data = (noise_param.data - 10/255 * noise_param.grad.detach().sign()).clamp(-noise_bound / 255, noise_bound / 255)
                noise_param.data = (noise_param.data + image_d.data).clamp(0, 1) - image_d.data
                noise_param.grad.zero_()
                model.zero_grad()

            utils.save_image((image_d + noise_param * alpha).squeeze(0).detach().cpu(), ima_p)

if __name__=="__main__":
    args = parser.parse_args()
    main(args)