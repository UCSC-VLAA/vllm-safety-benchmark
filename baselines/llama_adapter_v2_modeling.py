"""
Refer to https://github.com/VegB/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal to prepare environments and checkpoints
"""

import torch
from PIL import Image
import requests
from io import BytesIO

from base import VLLMBaseModel

import llama_adapter_v2_utils as llama


LLAMA_DIR = 'REPLACE_WITH_LOCAL_LLAMA_7B_DIR'

def format_prompt(instruction, input=None):

    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }
    if input is None:
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})



class VLLMLlamaAdapterV2(VLLMBaseModel):
    def __init__(self, model_path, device="cuda:0", base_lm_path=None):
        super().__init__(model_path, device)
        ## model_path "path_to_BIAS-7B.pth"
        self.model, self.preprocess = llama.load(self.model_path, base_lm_path, self.device)

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image path (batch size is 1 by default)
        Return: (str) a string of generated response
        """

        if len(images) == 0:
            raise ValueError('No image is provided.')
        if len(images) > 1:
            return '[Skipped]: Currently only support single image.'

        img = Image.open(images[0]).convert("RGB")
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        instruction = instruction[0] if type(instruction)==list else instruction
        # predict
        out = self.model.generate(img, [format_prompt(instruction)])[0]

        return out
    

if __name__ == '__main__':
    # test
    from test_cases import TEST_CASES

    cache_dir = "/data/haoqin_tu/.cache/torch/transformers"
    model = VisITLlamaAdapterV2(f"{cache_dir}/llama-adapterv2/BIAS-7B.pth", "cuda:0", f"{cache_dir}/llama-7b")

    for test_case in TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=["../test_image.png"],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)
