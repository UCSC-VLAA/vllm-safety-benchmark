import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import json
import torch
import time
import re
import argparse
from sat.model.mixins import CachedAutoregressiveMixin
from sat.mpu import get_model_parallel_world_size


from cogvlm_utils.utils.parser import parse_response
from cogvlm_utils.utils.chat import chat
from cogvlm_utils.models.cogvlm_model import CogVLMModel
from cogvlm_utils.utils.language import llama2_tokenizer, llama2_text_processor_inference
from cogvlm_utils.utils.vision import get_image_processor
from base import VLLMBaseModel

def process_image_without_resize(image_prompt):
    image = Image.open(image_prompt)
    # print(f"height:{image.height}, width:{image.width}")
    timestamp = int(time.time())
    file_ext = os.path.splitext(image_prompt)[1]
    filename_grounding = f"examples/{timestamp}_grounding{file_ext}"
    return image, filename_grounding

class VLLMCogVLM(VLLMBaseModel):
    def __init__(self, model_path, llm_path, device="cuda:0"):
        super().__init__(model_path, device)
        fp16, bf16 = False, False
        self.max_length = 32
        model, model_args = CogVLMModel.from_pretrained(
            model_path,
            args=argparse.Namespace(
            deepspeed=None,
            local_rank=0,
            rank=0,
            world_size=1,
            model_parallel_size=1,
            mode='inference',
            fp16=fp16,
            bf16=bf16,
            skip_init=True,
            use_gpu_initialization=True,
            device=device),
            overwrite_args={}
        )
        model = model.eval()
        # assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

        tokenizer = llama2_tokenizer(llm_path, signal_type="chat")
        image_processor = get_image_processor(model_args.eva_args["image_size"][0])

        model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

        text_processor_infer = llama2_text_processor_inference(tokenizer, self.max_length, model.image_length)

        self.model=model
        self.image_processor=image_processor
        self.text_processor_infer=text_processor_infer
        self.model.eval()
    
    def inference(
            self,
            input_text,
            temperature,
            top_p,
            top_k,
            image_prompt,
            result_previous,
            model, 
            image_processor, 
            text_processor_infer,
            ):
        result_text = [(ele[0], ele[1]) for ele in result_previous]
        for i in range(len(result_text)-1, -1, -1):
            if result_text[i][0] == "" or result_text[i][0] == None:
                del result_text[i]
        # print(f"history {result_text}")
        
        # global model, image_processor, text_processor_infer, is_grounding
        with torch.no_grad():
            pil_img, _ = process_image_without_resize(image_prompt)
            response, _, cache_image = chat(
                    image_path="", 
                    model=model, 
                    text_processor=text_processor_infer,
                    img_processor=image_processor,
                    query=input_text, 
                    history=result_text, 
                    image=pil_img, 
                    max_length=2048,
                    top_p=top_p, 
                    temperature=temperature,
                    top_k=top_k,
                    invalid_slices=text_processor_infer.invalid_slices if hasattr(text_processor_infer, "invalid_slices") else [],
                    no_prompt=False
            )
        return response

    def generate(self, instruction, images):
        instruction = instruction[0] if type(instruction)==list else instruction
        params = {
            "input_text": instruction,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 5,
            "image_prompt": images[0],
            "result_previous": [("", "Hi, What do you want to know about this image?")],
            "model": self.model,
            "image_processor": self.image_processor,
            "text_processor_infer": self.text_processor_infer,
        }
        response = self.inference(**params)
        return response