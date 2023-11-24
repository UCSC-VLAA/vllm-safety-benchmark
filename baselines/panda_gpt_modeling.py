"""
Refer to github.com/yxuansu/PandaGPT to prepare environments and checkpoints
"""

import torch

from panda_gpt_utils.openllama import OpenLLAMAPEFTModel
from base import VLLMBaseModel

class VLLMPandaGPT(VLLMBaseModel):
    def __init__(self, imagebind_ckpt_path, vicuna_ckpt_path, delta_ckpt_path, device):
        super().__init__(vicuna_ckpt_path, device)
        """
        Downloades and prepare chekcpoints according to github.com/yxuansu/PandaGPT
        """
        args = {
            'model': 'openllama_peft',
            'imagebind_ckpt_path': imagebind_ckpt_path,
            'vicuna_ckpt_path': vicuna_ckpt_path,
            'delta_ckpt_path': delta_ckpt_path,
            'stage': 2,
            'max_tgt_len': 400,
            'lora_r': 32,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
        }
        model = OpenLLAMAPEFTModel(**args)
        delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
        model.load_state_dict(delta_ckpt, strict=False)
        self.model = model.eval().half().to(self.device)

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        inputs = {
            'image_paths': images,    # modified image encoder to directly take urls (see ./pandas_gpt_utils/ImageBlind/data L100) 
            'prompt': instruction[0] if type(instruction)==list else instruction,
            'audio_paths': None,
            'video_paths': None,
            'thermal_paths': None,
            'modality_embeds': [],
            'modality_cache': '/data/haoqin_tu/workspace/img_cache',
            'top_p':0.5,
            'max_tgt_len':400,
            'temperature':0.9,
        }
        out = self.model.generate(inputs)
        return out
