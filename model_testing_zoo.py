import os, sys
import argparse
sys.path.append(os.getcwd())

image_path = "../assets/test_image.png"

instruction = "Create an English slogan for the tags on the hummer focusing on the meaning of the word."

cache_dir = "/data/haoqin_tu/.cache/torch/transformers"
cache_dir = "cache/path/to/all/your/model/weights"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str,
                    default="LlamaAdapterV2", 
                    choices=["LlamaAdapterV2", "MiniGPT4", "MiniGPT4v2",
                    "LLaVA", "mPLUGOwl", "mPLUGOwl2", "PandaGPT", "InstructBLIP2", "Flamingo", 
                    "LLaVAv1.5", "LLaVAv1.5-13B", "LLaVA_llama2", "LLaVA_llama2-13B", 
                    "MiniGPT4_llama2", "Qwen-VL-Chat", "InstructBLIP2-FlanT5-xl", 
                    "InstructBLIP2-FlanT5-xxl",  "InstructBLIP2-13B", "MiniGPT4_13B", "CogVLM", 
                    "Fuyu", "InternLM"])
args = parser.parse_args()
TESTING_MODEL=args.model_name


def load_model(TESTING_MODEL):
    if TESTING_MODEL == "Flamingo":
        from openflamingo_modeling import VLLMFlamingo
        ckpt_path = f'{cache_dir}/open_flamingo_9b_hf'
        model = VLLMFlamingo(ckpt_path, clip_vision_encoder_path="ViT-L-14", clip_vision_encoder_pretrained="openai", device="cuda:0")
    elif TESTING_MODEL == "MiniGPT4": ## mmqads
        from minigpt4_modeling import VLLMMiniGPT4
        model = VLLMMiniGPT4("./minigpt4_utils/minigpt4_eval.yaml")
    elif TESTING_MODEL == "MiniGPT4v2":
        from minigpt4_modeling import VLLMMiniGPT4
        model = VLLMMiniGPT4("./minigpt4_utils/minigptv2_eval.yaml")
    elif TESTING_MODEL == "MiniGPT4_llama2": ## mmqads
        from minigpt4_modeling import VLLMMiniGPT4
        model = VLLMMiniGPT4("./minigpt4_utils/minigpt4_eval_llama2.yaml")
    elif TESTING_MODEL == "MiniGPT4_13B": ## mmqads
        from minigpt4_modeling import VLLMMiniGPT4
        model = VLLMMiniGPT4("./minigpt4_utils/minigpt4_eval_13b.yaml")

    elif TESTING_MODEL == "LLaVA": ## mmqa
        from llava_modeling import VLLMLLaVA
        model = VLLMLLaVA({'model_path':f"{cache_dir}/llava-7b", "device":0, "do_sample": True})
    
    elif TESTING_MODEL == "LLaVA_llama2-13B": ## mmqa
        from llava_modeling import VLLMLLaVA
        model = VLLMLLaVA({'model_path':f"{cache_dir}/llava-llama-2-13b-chat", "device":0, "do_sample": True})

    elif TESTING_MODEL == "LLaVAv1.5":
        from llava_modeling import VLLMLLaVA
        model = VLLMLLaVA({'model_path':f"{cache_dir}/llava-v1.5-7b", "device":0, "do_sample": False})
    elif TESTING_MODEL == "LLaVAv1.5-13B":
        from llava_modeling import VLLMLLaVA
        model = VLLMLLaVA({'model_path':f"{cache_dir}/llava-v1.5-13b", "device":0, "do_sample": False})

    elif TESTING_MODEL == "LlamaAdapterV2": ## mmqa
        from llama_adapter_v2_modeling import VLLMLlamaAdapterV2
        model = VLLMLlamaAdapterV2(model_path=f"{cache_dir}/llama-adapterv2/BIAS-7B.pth", device="cuda:0", base_lm_path=f"{cache_dir}/LLaMA-7B")

    elif TESTING_MODEL == "mPLUGOwl": ## flamingo
        from mplug_owl_modeling import  VLLMmPLUGOwl
        model = VLLMmPLUGOwl(f"{cache_dir}/mplug-owl-7b-ft")
    elif TESTING_MODEL == "mPLUGOwl2": ## mmqa
        from mplug_owl2_modeling import  VLLMmPLUGOwl2
        model = VLLMmPLUGOwl2(f"{cache_dir}/mplug-owl2-llama2-7b")

    elif TESTING_MODEL == "PandaGPT": ## mmqa
        from panda_gpt_modeling import  VLLMPandaGPT
        model = VLLMPandaGPT(f"{cache_dir}/pandagpt", f"{cache_dir}/vicuna-7b-v0", f"{cache_dir}/pandagpt/pandagpt_7b_maxlength_1024.pt", device="cuda:0")

    elif TESTING_MODEL == "InstructBLIP2": ## mmqa
        from instruct_blip_modeling import VLLMInstructBLIP2
        model = VLLMInstructBLIP2(f"{cache_dir}/instructblip-vicuna-7b")
    elif TESTING_MODEL == "InstructBLIP2-13B": ## mmqa
        from instruct_blip_modeling import VLLMInstructBLIP2
        model = VLLMInstructBLIP2(f"{cache_dir}/instructblip-vicuna-13b")
    elif TESTING_MODEL == "InstructBLIP2-FlanT5-xl": ## mmqa
        from instruct_blip_modeling import VLLMInstructBLIP2
        model = VLLMInstructBLIP2(f"{cache_dir}/instructblip-flan-t5-xl")
    elif TESTING_MODEL == "InstructBLIP2-FlanT5-xxl": ## mmqa
        from instruct_blip_modeling import VLLMInstructBLIP2
        model = VLLMInstructBLIP2(f"{cache_dir}/instructblip-flan-t5-xxl")

    elif TESTING_MODEL == "Qwen-VL-Chat": ## mmqa
        from qwen_vl_modeling import VLLMQwenVL
        model = VLLMQwenVL(f"{cache_dir}/Qwen-VL-Chat")

    elif TESTING_MODEL == "CogVLM": ## llava
        from cogvlm_modeling import VLLMCogVLM
        model = VLLMCogVLM(model_path=f"{cache_dir}/cogvlm-chat", llm_path=f"{cache_dir}/llava-v1.5-7b")

    elif TESTING_MODEL == "Fuyu": ## mmqa
        from fuyu_modeling import VLLMFuyu
        model = VLLMFuyu(model_path=f"{cache_dir}/fuyu-8b")
    
    elif TESTING_MODEL == "InternLM": ## mmqa
        from internlm_xcomposer_modeling import VLLMInternLM
        model = VLLMInternLM(model_path=f"{cache_dir}/internlm-xcomposer-7b")
    
    return model

model = load_model(TESTING_MODEL)


pred = model.generate(
    instruction=[instruction],
    images=[image_path],
)
print(f'Instruction:\t{instruction}')
print(f'Answer:\t{pred}')
print('-'*20)
