import os
import torch
import requests

from PIL import Image
from io import BytesIO
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model import *
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


from base import VLLMBaseModel


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def get_model(args):
    """Model Provider with tokenizer and processor.
    """
    # in case of using a pretrained model with only a MLP projector weights
    disable_torch_init()
    model_path = os.path.expanduser(args['model_path'])
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name, args['device'])
    return model, tokenizer, image_processor

class VLLMLLaVA(VLLMBaseModel):
    """LLaVA model evaluation.
    """
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )
        self.model_path = model_args['model_path']

        self.model, self.tokenizer, self.image_processor = get_model(model_args)
        self.model.to(self.device)
        self.model.eval()
        self.mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)

        if "v0" in self.model_path.lower():
            conv_mode = "v0"
        elif "v1" in self.model_path.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_path.lower():
            conv_mode = "mpt_multimodal"
        elif "llama-2" in self.model_path.lower():
            conv_mode = "llava_llama_2"
        else:
            conv_mode = "v0"
        self.conv_mode = conv_mode

        ## Generation configs
        if "do_sample" in model_args:
            self.do_sample = model_args['do_sample']
        else:
            self.do_sample = True
        
        if "temperature" in model_args:
            self.temperature = model_args['temperature']
        else:
            self.temperature = 0.2

        if "max_new_tokens" in model_args:
            self.max_new_tokens = model_args['max_new_tokens']
        else:
            self.max_new_tokens = 1024
        self.max_new_tokens = 25

    def generate(self, instruction, images):

        assert len(images) == 1

        image_path = images[0]
        
        query = instruction[0] if type(instruction)==list else instruction
        if self.mm_use_im_start_end:
            query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query
        else:
            query = DEFAULT_IMAGE_TOKEN + '\n' + query

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

        image = Image.open(image_path)
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)


        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(self.device),
                do_sample=self.do_sample,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs