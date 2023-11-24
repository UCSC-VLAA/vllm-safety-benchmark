"""
Refer to https://huggingface.co/Salesforce/instructblip-vicuna-13b
"""

from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
# from lavis.models import load_model_and_preprocess

from base import VLLMBaseModel


class VLLMInstructBLIP2(VLLMBaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        ## model_path="Salesforce/instructblip-vicuna-13b"
        self.model = InstructBlipForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
        self.processor = InstructBlipProcessor.from_pretrained(self.model_path)

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        image_path = images[0]  # instruct_blip2 only supports single image
        raw_image = Image.open(image_path).convert("RGB")
        instruction = instruction[0] if type(instruction)==list else instruction
        inputs = self.processor(images=raw_image, text=instruction, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )
        outputs[outputs == 0] = 2
        out = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return out