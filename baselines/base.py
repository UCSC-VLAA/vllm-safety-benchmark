import torch

class VLLMBaseModel(torch.nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.device=device
        self.model_path=model_path

    def forward(self, instruction, images):
        return self.generate(instruction, images)
    
    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        raise NotImplementedError