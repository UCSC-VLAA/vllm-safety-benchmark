<!-- <p align="center">
  <img src="unicorn.png" width="80">
</p> -->

# How many <img src="assets/unicorn.png" width="36"> Are in This Image? A Safety Evaluation Benchmark for Vision LLMs


[Haoqin Tu*](https://www.haqtu.me/), [Chenhang Cui*](https://gzcch.github.io/), [Zijun Wang*](https://asillycat.github.io/), [Yiyang Zhou](https://yiyangzhou.github.io/), [Bingchen Zhao](https://bzhao.me), [Junlin Han](https://junlinhan.github.io/), [Wangchunshu Zhou](https://michaelzhouwang.github.io/), [Huaxiu Yao](https://www.huaxiuyao.io/), [Cihang Xie](https://cihangxie.github.io/) (*equal contribution)

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

Our paper is online now: https://arxiv.org/abs/2311.xxxx

<p align="center">
  <img src="assets/teaser.png" width="1080">
</p>

## Installation
For different VLLMs, please refer to their specific envirnments for installation.

- LLaVA: https://github.com/haotian-liu/LLaVA
- MiniGPT4: https://github.com/Vision-CAIR/MiniGPT-4
- InstructBLIP: https://huggingface.co/Salesforce/instructblip-vicuna-7b
- LLaMA-Adapter: https://github.com/OpenGVLab/LLaMA-Adapter
- mPLUG-Owl1&2: https://github.com/X-PLUG/mPLUG-Owl
- PandaGPT: https://github.com/yxuansu/PandaGPT
- Qwen-VL-Chat: https://huggingface.co/Qwen/Qwen-VL-Chat
- CogVLM: https://github.com/THUDM/CogVLM
- InternLM-Xcomposer: https://huggingface.co/internlm/internlm-xcomposer-7b
- Fuyu: https://huggingface.co/adept/fuyu-8b

## Datasets
We host our datasets [here](https://huggingface.co/datasets/PahaII/vllm_safety_evaluation), containing both OOD and redteaming attack datasets. The full dataset should looks like this:

```
.
├── ./safety_evaluation_benchmark_datasets//                    
    ├── gpt4v_challenging_set # Contains the challenging test data for GPT4V
        ├── attack_images
        ├── sketchy_images
        ├── oodcv_images
        ├── misleading-attack.json
        ├── sketchy-vqa-challenging.json
        └── oodcv-vqa-counterfactual.json
    ├── redteaming-mislead # Contains the test data for redteaming tasks
        ├── redteaming_attack
            ├── gaussian_noise
            ├── mixattack_eps32
            ├── mixattack_eps64
            ├── sinattack_eps64_dog
            ├── sinattack_eps64_coconut
            ├── sinattack_eps64_spaceship
            └── annotation.json
        └── jailbreak_llm # adversarial suffixes for jailbreaking VLLM through LLM
    └── ood # Contains the test data for OOD scenarios
        ├── sketchy-vqa
            ├── sketchy-vqa.json
            ├── sketchy-challenging.json
        └── oodcv-vqa
            ├── oodcv-vqa.json
            └── oodcv-counterfactual.json
```

### Out-of-Distribution Scenario
For $\texttt{OODCV-VQA}$ and its counterfactual version, please download images from [OODCV](https://drive.google.com/file/d/1jq43Q0cenISIq7acW0LS-Lqghgy8exsj/view?usp=share_link), and put all images in `ood/oodcv-vqa`.

For $\texttt{Sketchy-VQA}$ and its challenging version, please first download images from [here](https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip), put the zip file into `ood/sketchy-vqa/skechydata/`, then unzip it.

### Redteaming Attack
For the proposed misleading attack, the full datasets and all trained adversarial examples are in `redteaming-mislead`, including images with gaussian noise, Sin.Attack and MixAttack with two pertubation budgets $\epsilon=32/255$ (eps32) or $\epsilon=64/255$ (eps64).

For jailbreaking methods, please refer to their respective repositories for more dataset details: [Jailbreak through ViT](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models), [Jailbreak through LLM](https://github.com/llm-attacks/llm-attacks).

## Testing
You can first test all models using `model_testing_zoo.py`, note to specify the cache directory inside the file.

```bash
cd baselines
python ../model_testing_zoo.py --model_name LLaVA
```
Choose `--model_name` from ["LlamaAdapterV2", "MiniGPT4", "MiniGPT4v2", "LLaVA", "mPLUGOwl", "mPLUGOwl2", "PandaGPT", "InstructBLIP2", "Flamingo", "LLaVAv1.5", "LLaVAv1.5-13B", "LLaVA_llama2-13B", "MiniGPT4_llama2", "Qwen-VL-Chat", "MiniGPT4_13B", "InstructBLIP2-FlanT5-xl", "InstructBLIP2-FlanT5-xxl",  "InstructBLIP2-13B", "CogVLM", "Fuyu", "InternLM"].

Before you start, make sure you have modified the `CACHE_DIR` (where you store all your model weights) and `DATA_DIR` (where you store the benchmark data) in `baselines/config.json` according to your local envirnment.

### $\texttt{OODCV-VQA}$ and its Counterfactual Variant

For $\texttt{OODCV-VQA}$:
```bash
cd baselines
python ../safety_evaluations/ood_scenarios/evaluation.py --model_name LLaVA --eval_oodcv
```

For the counterfactual version:

```bash
cd baselines
python ../safety_evaluations/ood_scenarios/evaluation.py --model_name LLaVA --eval_oodcv_cf
```

### $\texttt{Sketchy-VQA}$ and its Challenging Variant

For $\texttt{Sketchy-VQA}$:
```bash
cd baselines
python ../safety_evaluations/ood_scenarios/evaluation.py --model_name LLaVA --eval_sketch
```

For the challenging version:

```bash
cd baselines
python ../safety_evaluations/ood_scenarios/evaluation.py --model_name LLaVA --eval_sketch_challenging
```

### Misleading Attack
For training the misleading adversarial images:

```bash
cd safety_evaluations/redteaming/misleading_vision_attack

python misleading_vis_attack.py --lr 1e-3 --misleading_obj dog --input_folder path/to/attack-bard/NIPS2017 --output_folder ./misleading_adversarial_attack
```
Change `--input_folder` to the path of adversarial examples you want to test. If you want to use the MixAttack, add `--mix_obj` argument to the command.

For testing the VLLMs:

```bash
cd baselines

python ../safety_evaluations/redteaming/misleading_vision_attack/test_misleading.py --image_folder redteaming/misleading_attack/mixattack_eps64 --output_name misleading_attack_eps64 --human_annot_path redteaming-mislead/annotation.json
```

### Jailbreaking Methods

Please refer to these two repositories for detailed attack settings: [Jailbreak through ViT](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models), [Jailbreak through LLM](https://github.com/llm-attacks/llm-attacks). We give our trained adversarial images and suffixes for jailbreaking ViTs and LLMs in `redteaming/jailbreak_vit` and `redteaming/jailbreak_llm` in the data folder.

## Usage and License Notices
The data, code and checkpoint is intended and licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

## Citation
If you find our work useful to your research and applications, please consider citing the paper and staring the repo :)

```bibtex
@article{tu2023how,
  title={How Many Unicorns Are In This Image? A Safety Evaluation Benchmark For Vision LLMs},
  author={Tu, Haoqin and Cui, Chenhang and Wang, Zijun and Zhou, Yiyang and Zhao, Bingchen and Han, Junlin and Zhou, Wangchunshu and Yao, Huaxiu and Xie, Cihang},
  journal={arXiv preprint arXiv:2311.xxxx},
  year={2023}
}
```

## Acknowledgement
We thank Center for AI Safety (CAIS) for supporting our computing needs. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the sponsors.