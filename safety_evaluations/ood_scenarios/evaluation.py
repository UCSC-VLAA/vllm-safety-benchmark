import argparse
import json
import sys
import os
import numpy as np
import random
import uuid
from collections import defaultdict
from typing import Callable

import more_itertools
import torch

from eval_datasets import (
    OODCVDataset,
    PopeDataset,
    SketchyDataset
    )
from pope_metric import get_pope_results
from sketchy_metric import get_sketchy_results
from oodcv_vqa_metric import compute_oodcvqa_metrics
from tqdm import tqdm

sys.path.append(os.getcwd())

configs = json.load(open("./config.json"))

DATA_DIR = configs['DATA_DIR']
CACHE_DIR = configs['CACHE_DIR']

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str,
                    default="LlamaAdapterV2", 
                    choices=["LlamaAdapterV2", "MiniGPT4", "MiniGPT4v2",
                    "LLaVA", "mPLUGOwl", "mPLUGOwl2", "PandaGPT", "InstructBLIP2", "Flamingo", 
                    "LLaVAv1.5", "LLaVAv1.5-13B", "LLaVA_llama2", "LLaVA_llama2-13B", 
                    "MiniGPT4_llama2", "Qwen-VL-Chat", "MiniGPT4-13B", "InstructBLIP2-FlanT5-xl", 
                    "InstructBLIP2-FlanT5-xxl",  "InstructBLIP2-13B", "MiniGPT4_13B", "CogVLM", 
                    "Fuyu", "InternLM"])

parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)
parser.add_argument(
    "--precision", type=str, default='fp32', choices=['fp16', 'fp32']
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)

parser.add_argument(
    "--trial_seeds",
    nargs="+",
    default=[42],
    type=int,
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples", type=int, default=5000, help="Number of samples to evaluate on"
)

parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--device", type=int, default=0)

# Per-dataset evaluation flags
parser.add_argument(
    "--eval_oodcv",
    action="store_true",
    default=False,
    help="Whether to evaluate on COCO.",
)

parser.add_argument(
    "--eval_oodcv_cf",
    action="store_true",
    default=False,
    help="Whether to evaluate on COCO.",
)

parser.add_argument(
    "--eval_coco_pope",
    action="store_true",
    default=False,
    help="Whether to evaluate on COCO.",
)

parser.add_argument(
    "--eval_sketch",
    action="store_true",
    default=False,
    help="Whether to evaluate on SketchyVQA.",
)
parser.add_argument(
    "--eval_sketch_challenging",
    action="store_true",
    default=False,
    help="Whether to evaluate on SketchyVQA.",
)

## POPE Dataset
parser.add_argument(
    "--pope_annot_dir",
    type=str,
    default=os.path.join(DATA_DIR, "pope/output/coco"),
)
parser.add_argument(
    "--pope_image_dir_path",
    type=str,
    default="mscoco/val2014",
)

## SketchyVQA Dataset
parser.add_argument(
    "--sketch_annot_dir",
    type=str,
    default=os.path.join(DATA_DIR, "ood/sketchy-vqa/sketchy-vqa.json"),
)
parser.add_argument(
    "--sketch_annot_dir_challenging",
    type=str,
    default=os.path.join(DATA_DIR, "ood/sketchy-vqa/sketchy-challenging.json"),
)
parser.add_argument(
    "--sketch_image_dir_path",
    type=str,
    default=os.path.join(DATA_DIR, "ood/sketchy-vqa"),
)

parser.add_argument(
    "--oodcv_ques_dir",
    type=str,
    default=os.path.join(DATA_DIR, "ood/oodcv-vqa/"),
)
parser.add_argument(
    "--oodcv_image_dir_path",
    type=str,
    default=os.path.join(DATA_DIR, "ood/oodcv-vqa/"),
)

def setup_seed(seed=3407):
    os.environ["PYTHONHASHSEED"]=str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False

def load_model(TESTING_MODEL):
    if TESTING_MODEL == "Flamingo":
        from openflamingo_modeling import VLLMFlamingo
        ckpt_path = f'{CACHE_DIR}/open_flamingo_9b_hf'
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
        model = VLLMLLaVA({'model_path':f"{CACHE_DIR}/llava-7b", "device":0, "do_sample": True})
    
    elif TESTING_MODEL == "LLaVA_llama2-13B": ## mmqa
        from llava_modeling import VLLMLLaVA
        model = VLLMLLaVA({'model_path':f"{CACHE_DIR}/llava-llama-2-13b-chat", "device":0, "do_sample": True})

    elif TESTING_MODEL == "LLaVAv1.5":
        from llava_modeling import VLLMLLaVA
        model = VLLMLLaVA({'model_path':f"{CACHE_DIR}/llava-v1.5-7b", "device":0, "do_sample": False})
    elif TESTING_MODEL == "LLaVAv1.5-13B":
        from llava_modeling import VLLMLLaVA
        model = VLLMLLaVA({'model_path':f"{CACHE_DIR}/llava-v1.5-13b", "device":0, "do_sample": False})

    elif TESTING_MODEL == "LlamaAdapterV2": ## mmqa
        from llama_adapter_v2_modeling import VLLMLlamaAdapterV2
        model = VLLMLlamaAdapterV2(model_path=f"{CACHE_DIR}/llama-adapterv2/BIAS-7B.pth", device="cuda:0", base_lm_path=f"{CACHE_DIR}/LLaMA-7B")

    elif TESTING_MODEL == "mPLUGOwl": ## flamingo
        from mplug_owl_modeling import  VLLMmPLUGOwl
        model = VLLMmPLUGOwl(f"{CACHE_DIR}/mplug-owl-7b-ft")
    elif TESTING_MODEL == "mPLUGOwl2": ## mmqa
        from mplug_owl2_modeling import  VLLMmPLUGOwl2
        model = VLLMmPLUGOwl2(f"{CACHE_DIR}/mplug-owl2-llama2-7b")

    elif TESTING_MODEL == "PandaGPT": ## mmqa
        from panda_gpt_modeling import  VLLMPandaGPT
        model = VLLMPandaGPT(f"{CACHE_DIR}/pandagpt", f"{CACHE_DIR}/vicuna-7b-v0", f"{CACHE_DIR}/pandagpt/pandagpt_7b_maxlength_1024.pt", device="cuda:0")

    elif TESTING_MODEL == "InstructBLIP2": ## mmqa
        from instruct_blip_modeling import VLLMInstructBLIP2
        model = VLLMInstructBLIP2(f"{CACHE_DIR}/instructblip-vicuna-7b")
    elif TESTING_MODEL == "InstructBLIP2-13B": ## mmqa
        from instruct_blip_modeling import VLLMInstructBLIP2
        model = VLLMInstructBLIP2(f"{CACHE_DIR}/instructblip-vicuna-13b")
    elif TESTING_MODEL == "InstructBLIP2-FlanT5-xl": ## mmqa
        from instruct_blip_modeling import VLLMInstructBLIP2
        model = VLLMInstructBLIP2(f"{CACHE_DIR}/instructblip-flan-t5-xl")
    elif TESTING_MODEL == "InstructBLIP2-FlanT5-xxl": ## mmqa
        from instruct_blip_modeling import VLLMInstructBLIP2
        model = VLLMInstructBLIP2(f"{CACHE_DIR}/instructblip-flan-t5-xxl")

    elif TESTING_MODEL == "Qwen-VL-Chat": ## mmqa
        from qwen_vl_modeling import VLLMQwenVL
        model = VLLMQwenVL(f"{CACHE_DIR}/Qwen-VL-Chat")

    elif TESTING_MODEL == "CogVLM": ## llava
        from cogvlm_modeling import VLLMCogVLM
        model = VLLMCogVLM(model_path=f"{CACHE_DIR}/cogvlm-chat", llm_path=f"{CACHE_DIR}/llava-v1.5-7b")

    elif TESTING_MODEL == "Fuyu": ## mmqa
        from fuyu_modeling import VLLMFuyu
        model = VLLMFuyu(model_path=f"{CACHE_DIR}/fuyu-8b")
    
    elif TESTING_MODEL == "InternLM": ## mmqa
        from internlm_xcomposer_modeling import VLLMInternLM
        model = VLLMInternLM(model_path=f"{CACHE_DIR}/internlm-xcomposer-7b")
    
    return model


def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        print(num_samples + query_set, len(full_dataset))
        raise ValueError(
            f"num_samples + num_shots must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + query_set_size, replace=False
    )
    return random_indices


def prepare_eval_samples_and_dataset(full_dataset, random_indices, query_set_size):
    # get in context samples
    in_context_samples = [full_dataset[i] for i in random_indices[:query_set_size]]
    eval_dataset = torch.utils.data.Subset(
        full_dataset, random_indices[query_set_size:]
    )
    return in_context_samples, eval_dataset


def get_context_text(
    get_prompt: Callable[[dict], str],
    in_context_samples,
    effective_num_shots,
    num_shots,
) -> list:
    context_text = []
    if effective_num_shots > 0:
        for s in in_context_samples:
            context_text.append(get_prompt(s))
    return context_text

def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    return [random.sample(query_set, num_samples) for _ in range(batch_size)]

def main():
    args = parser.parse_args()
    model = load_model(args.model_name)

    if args.eval_oodcv:
        eval_task = "oodcv_vqa"
    elif args.eval_coco_pope:
        eval_task = "coco_pope"
    elif args.eval_oodcv_cf:
        eval_task = "oodcv_vqa_cf"
    elif args.eval_sketch:
        eval_task = "sketchyvqa"
    elif args.eval_sketch_challenging:
        eval_task = "sketchyvqa_challenging"

    save_path = f"../test_results/{eval_task}_{args.model_name}.json"

    final_results = {}
    if args.eval_oodcv or args.eval_oodcv_cf:
        acc_list, yes_no_acc_list, digits_acc_list, all_digits_acc_list = [], [], [], []
        test_file_name = "oodcv-vqa.json" if args.eval_oodcv else "oodcv-counterfactual.json"
        for seed in args.trial_seeds:
            acc, yes_no_acc, digits_acc, all_digits_acc, all_category_results = test_ood_tasks(
                model, 
                args.batch_size,
                args.oodcv_image_dir_path,
                os.path.join(args.oodcv_ques_dir, test_file_name),
                seed=seed,
                model_name=args.model_name,
                task_name=eval_task,
            )
            acc_list.append(acc)
            yes_no_acc_list.append(yes_no_acc)
            digits_acc_list.append(digits_acc)
            all_digits_acc_list.append(all_digits_acc)
            print("Acc.: {:.2f}; Yes/No Acc.: {:.2f}; Digits Acc.: {:.2f}".format(acc, yes_no_acc, digits_acc))
        final_results[args.model_name] = {
            "Acc": acc_list,

            "DigitsAcc": digits_acc_list,

            "YesNoAcc": yes_no_acc_list,

            "AllDigits": all_digits_acc_list[0],
            "AllCategories": all_category_results,
        }
    elif args.eval_coco_pope:
        for questions_path in ["coco_pope_popular.json", "coco_pope_random.json", "coco_pope_adversarial.json"]:
            print(f"Evaluating {questions_path} task in POPE..")
            questions_json_path = os.path.join(args.pope_annot_dir, questions_path)
            scores = evaluate_pope(
                model,
                batch_size=args.batch_size,
                image_dir_path=os.path.join(DATA_DIR, args.pope_image_dir_path),
                questions_json_path=questions_json_path,
                seed=args.trial_seeds[0],
                
            )
        final_results[questions_path] = [{
                "acc": scores[0],
                "precision": scores[1],
                "recall": scores[2],
                "f1": scores[3],
                "yes_ratio": scores[4],
            }]
    elif args.eval_sketch or args.eval_sketch_challenging:
        ques_json_path = args.sketch_annot_dir if args.eval_sketch else args.sketch_annot_dir_challenging
        scores = evaluate_sketchyvqa(
            model,
            batch_size=args.batch_size,
            image_dir_path=args.sketch_image_dir_path,
            questions_json_path=ques_json_path,
            seed=args.trial_seeds[0],
            model_name=args.model_name,
            task_name=eval_task,
        )
        final_results[args.model_name] = [{
                "acc": scores[0],
                "precision": scores[1],
                "recall": scores[2],
                "f1": scores[3],
                "yes_ratio": scores[4],
            }]

    with open(save_path, "w") as f:
        json.dump(final_results, f, indent=4)
        

def test_ood_tasks(
    eval_model,
    batch_size,
    image_dir_path,
    questions_json_path_test,
    seed=42,
    num_shots=0,
    additional_shots=0,
    model_name=None,
    task_name=None,
):
    setup_seed(seed)
    eval_dataset = OODCVDataset(
        image_base_dir_path=image_dir_path,
        question_path=questions_json_path_test,
    )

    ## effective_num_shots is for text instruction only
    num_shots=0
    if additional_shots == 0:
        effective_num_shots = num_shots
    else:
        effective_num_shots = num_shots if num_shots > 0 else additional_shots
    
    def get_prompt_choose(sample, train=True):
        question = sample['question']
        if train:
            return f"{question}, choose from A or B\n{' '.join(sample['options'])}\nAnswer: {sample['answer']}"
        else:
            return f"{question}, choose from A or B\n{' '.join(sample['options'])}"
    
    def get_prompt_generate(sample, train=True):
        question = sample['question']
        if train:
            return f"{question} Short Answer: {sample['text_answer']}"
        else:
            return f"{question}"

    predictions = defaultdict()
    failed_case = 0
    for batch in more_itertools.chunked(tqdm(eval_dataset), batch_size):
        batch_images = []
        batch_text = []
        batch_answers = []
        situations = []
        for i in range(len(batch)):
            batch_images.append(batch[i]["image"])

            batch_text.append(get_prompt_generate(batch[i], train=False, ))
            batch_answers.append(batch[i]["text_answer"])
            situations.append(batch[i]["situation"])
        # __import__("ipdb").set_trace()

        outputs = eval_model.generate(
            images=batch_images,
            instruction=batch_text[0],
        )

        for i, sample in enumerate(batch):
            predictions[sample["image"]] = {
                "prediction": outputs,
                "answer": batch_answers[0],
                "situation": situations[0],
            }
    
    os.makedirs("../test_results/generated/", exist_ok=True)
    save_path = f"../test_results/generated/{task_name}_{model_name}_output.json"
    with open(save_path, "w") as f:
        json.dump(predictions, f, indent=4)
    counter_factual = True if task_name == "oodcv_vqa_cf" else False
    acc, yes_no_acc, digits_acc, all_digits_acc, all_category_results = compute_oodcvqa_metrics(predictions, counter_factual)
    print(f"Failed Case Number: {failed_case}..")
    return acc, yes_no_acc, digits_acc, all_digits_acc, all_category_results

def evaluate_pope(
    eval_model,
    batch_size,
    image_dir_path,
    questions_json_path,
    seed=42,
    num_shots=0,
    additional_shots=0,
    model_name=None,
):
    """
    Evaluate a model on POPE benchmark.
    """
    setup_seed(seed)
    full_dataset = PopeDataset(
        image_dir_path=os.path.join(DATA_DIR, image_dir_path),
        question_path=questions_json_path,
    )

    def get_prompt(sample, train=True,):
        if train:
            return_list = [
                sample['question'].strip(),
                sample['answers'][0].strip()
            ]
        else:
            return_list = [
                sample['question'].strip()
            ]
        return return_list

    predictions = []

    for batch in more_itertools.chunked(
        tqdm(full_dataset, desc="Running inference"), batch_size
    ):

        batch_images = []
        batch_text = []

        ## get example batch
        for i in range(len(batch)):
            batch_images.append([batch[i]["image"]])
            batch_text.append(get_prompt(batch[i], train=False, )[0])

        outputs = eval_model.generate(
            images=batch_images[0],
            instruction=batch_text[0],
        )


        predictions.extend(
            [
                {"answer": p, "question_id": sample["question_id"], "label": sample["label"]}
                for p, sample in zip(outputs, batch)
            ]
        )
    # save the predictions to a temporary file
    test_results_dir = "coco_pope_generated"
    os.makedirs(f"../test_results/{test_results_dir}/", exist_ok=True)
    out_file_path = f"../test_results/{test_results_dir}/pope_results_{model_name}.json"
    with open(out_file_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))

    _, metrics = get_pope_results(out_file_path)

    # delete the temporary file
    os.remove(out_file_path)

    return metrics

def evaluate_sketchyvqa(
    eval_model,
    batch_size,
    image_dir_path,
    questions_json_path,
    seed=42,
    num_shots=0,
    additional_shots=0,
    model_name=None,
    task_name=None,
):
    """
    Evaluate a model on POPE benchmark.
    """
    setup_seed(seed)
    full_dataset = SketchyDataset(
        image_base_dir_path=image_dir_path,
        question_path=questions_json_path,
    )

    predictions = []

    for batch in more_itertools.chunked(
        tqdm(full_dataset, desc="Running inference"), batch_size
    ):

        batch_images = []
        batch_text = []

        ## get example batch
        for i in range(len(batch)):
            batch_images.append([batch[i]["image"]])
            batch_text.append(batch[i]['question'])

        outputs = eval_model.generate(
            images=batch_images[0],
            instruction=batch_text[0],
        )


        predictions.append(
                {"answer": outputs, "label": batch[0]["answer"], "image_path": batch[0]["image"]}
        )
    # save the predictions to a temporary file
    test_results_dir = "sketchyvqa_generated"
    os.makedirs(f"../test_results/{test_results_dir}/", exist_ok=True)
    out_file_path = f"../test_results/{test_results_dir}/{task_name}_results_{model_name}.json"
    with open(out_file_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))

    _, metrics = get_sketchy_results(predictions)

    return metrics

if __name__ == "__main__":
    main()