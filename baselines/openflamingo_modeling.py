"""
Refer to https://github.com/X-PLUG/mPLUG-Owl to prepare environments and checkpoints
"""

import torch
from PIL import Image
import requests
from io import BytesIO

from base import VLLMBaseModel

from openflamingo.modeling_flamingo import FlamingoForConditionalGeneration
import open_clip



# device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


class VLLMFlamingo(VLLMBaseModel):
    def __init__(self, model_path, clip_vision_encoder_path, clip_vision_encoder_pretrained, device="cuda:0"):
        super().__init__(model_path, device)

        self.model = FlamingoForConditionalGeneration.from_pretrained(
            self.model_path,
        )
        self.tokenizer = self.model.text_tokenizer
        _, _, image_processor = open_clip.create_model_and_transforms(
            clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained,
            device=self.device
        )
        self.image_processor = image_processor
        self.model.to(self.device)
        self.model.eval()

    def _prepare_images(self, batch) -> torch.Tensor:
        """Preprocess images and stack them.

        Args:
            batch: A list of lists of images.

        Returns:
            A Tensor of shape
            (batch_size, images_per_example, frames, channels, height, width).
        """
        images_per_example = len(batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            image = Image.open(example).convert("RGB")
            preprocessed = self.image_processor(image)

            if batch_images is None:
                batch_images = torch.zeros(
                    (1, images_per_example, 1) + preprocessed.shape,
                    dtype=preprocessed.dtype,
                )
            batch_images[0, iexample, 0] = preprocessed
        return batch_images

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """

        
        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': 15,
            'length_penalty': 1.0,
        }
        instruction = instruction[0] if type(instruction)==list else instruction
        encodings = self.tokenizer(
                instruction,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=2000,
            )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.inference_mode():
            outputs = self.model.generate(
                self._prepare_images(images).to(self.device),
                input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                **generate_kwargs,
            )
        # __import__("ipdb").set_trace()
        outputs = outputs[:, len(input_ids[0]) :]
        outputs_final = outputs.clone()
        outputs_final[outputs == -100] = self.tokenizer.pad_token_id

        return self.tokenizer.batch_decode(outputs_final, skip_special_tokens=True)
    

if __name__ == '__main__':
    from test_cases import TEST_CASES
    # test
    ckpt_path = '/data/haoqin_tu/.cache/torch/transformers/open_flamingo_9b_hf'
    model = VisITFlamingo(ckpt_path, clip_vision_encoder_path="ViT-L-14", clip_vision_encoder_pretrained="openai", device="cuda:0")

    for test_case in TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=['../test_image.png'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)
