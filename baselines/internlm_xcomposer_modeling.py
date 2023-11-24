from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

from base import VLLMBaseModel


class VLLMInternLM(VLLMBaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        # init model and tokenizer
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.tokenizer = self.tokenizer
    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        image_path = images[0]  # instruct_blip2 only supports single image
        instruction = instruction[0] if type(instruction)==list else instruction
        response = self.model.generate(instruction, image_path)
        return response