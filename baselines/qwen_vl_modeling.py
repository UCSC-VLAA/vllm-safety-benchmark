"""
Refer to https://huggingface.co/Qwen/Qwen-VL-Chat
"""

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from base import VLLMBaseModel


class VLLMQwenVL(VLLMBaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.model.eval()

    def generate_v0(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        query = self.tokenizer.from_list_format([
            {'image': images[0]},
            {'text': instruction},
        ])
        inputs = self.tokenizer(query, return_tensors='pt').to(self.device)
        pred = self.model.generate(**inputs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        return response

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        instruction = instruction[0] if type(instruction)==list else instruction
        query = self.tokenizer.from_list_format([
            {'image': images[0]},
            {'text': instruction},
        ])
        response, _ = self.model.chat(self.tokenizer, query=query, history=None)

        return response