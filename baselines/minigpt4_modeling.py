"""
Refer to https://github.com/Vision-CAIR/MiniGPT-4 to prepare environments and checkpoints
"""

import torch
from PIL import Image
import requests
from io import BytesIO

from base import VLLMBaseModel

from minigpt4_utils.common.config import Config
from minigpt4_utils.common.dist_utils import get_rank
from minigpt4_utils.common.registry import registry
from minigpt4_utils.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

from minigpt4_utils.datasets.builders import *
from minigpt4_utils.models import *
from minigpt4_utils.processors import *
from minigpt4_utils.runners import *
from minigpt4_utils.tasks import *
from transformers import StoppingCriteriaList

conv_dict = {'pretrain_vicuna': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2,
             'pretrain_vicuna_13b': CONV_VISION_Vicuna0
             }


class VLLMMiniGPT4(VLLMBaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        ## './minigpt4_utils/minigpt4_eval.yaml'
        cfg = Config(cfg_path=self.model_path)
        model_config = cfg.model_cfg
        self.CONV_VISION = conv_dict[model_config.model_type]
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(self.device)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.stop_words_ids = [[835], [2277, 29937]]
        self.stop_words_ids = [torch.tensor(ids).to(self.device) for ids in self.stop_words_ids]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=self.stop_words_ids)])
        self.chat = Chat(model, self.vis_processor, device=device, stopping_criteria=self.stopping_criteria)

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """

        if len(images) == 0:
            raise ValueError('No image is provided.')
        if len(images) > 1:
            return '[Skipped]: Currently only support single image.'
        
        # Init chat state
        CONV_VISION = self.CONV_VISION
        chat_state = CONV_VISION.copy()
        img_list = []

        # download image image
        image_path = images[0]
        img = Image.open(image_path).convert("RGB")
        
        # upload Image
        self.chat.upload_img(img, chat_state, img_list)
        self.chat.encode_img(img_list)
        instruction = instruction[0] if type(instruction)==list else instruction
        # ask
        self.chat.ask(instruction, chat_state)

        # answer
        out = self.chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=5,
            temperature=1.0,
            max_new_tokens=100,
            max_length=2000
        )[0]

        return out


if __name__ == '__main__':
    # test
    from test_cases import TEST_CASES

    model = VisITMiniGPT4("./minigpt4_utils/minigpt4_eval.yaml")

    for test_case in TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)
