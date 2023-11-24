## LLaVA Model Evaluation

### Setup

```
1. git clone https://github.com/haotian-liu/LLaVA.git
2. Follow the steps in their README to install their dependencies and recover LLaVA weights
3. mv llava ../
4. cd ..
5. rm -rf LLaVA
```

## MiniGPT-4 Model Evaluation

### Setup

1. Prepare conda environment:
```bash
conda env create -f minigpt4_utils/environment.yml
conda activate minigpt4
```

2. Follow instructions [here](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/PrepareVicuna.md) and prepare Vicuna weights. The final weights would be in a single folder in a structure similar to the following:

```
vicuna_weights
├── config.json
├── generation_config.json
├── pytorch_model.bin.index.json
├── pytorch_model-00001-of-00003.bin
...   
```

Then, set the path to the vicuna weight in the model config file 
[here](./minigpt4_utils/configs/models/minigpt4.yaml#16) at Line 16.

3. Download the pretrained checkpoints according to the Vicuna model you prepare.

|                                Checkpoint Aligned with Vicuna 13B                                |                               Checkpoint Aligned with Vicuna 7B                                |
:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
 [Downlad](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) 


Then, set the path to the pretrained checkpoint in the evaluation config file [here](./minigpt4_utils/minigpt4_eval.yaml#11) at Line 11. 


#### Reference
https://github.com/Vision-CAIR/MiniGPT-4#installation


## mPLUG-Owl Model Evaluation

### Setup

```bash
# Create conda environment
conda create -n mplug_owl python=3.10
conda activate mplug_owl

# Install PyTorch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install other dependencies
pip install -r mplug_owl_utils/requirements.txt
```

#### Reference
https://github.com/X-PLUG/mPLUG-Owl#install-requirements


## Llama-Adapter-v2 Model Evaluation

### Setup
1. Prepare conda environment
```bash
conda create -n llama_adapter_v2 python=3.8 -y
pip install -r llama_adapter_v2_utils/requirements.txt
```
2. Prepare Llama 7B weights and update [this line](./llama_adapter_v2_modeling.py#15). Organize the downloaded file in the following structure:
```
/path/to/llama_model_weights
├── 7B
│   ├── checklist.chk
│   ├── consolidated.00.pth
│   └── params.json
└── tokenizer.model
```


#### Reference
https://github.com/VegB/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal#setup


## PandaGPT Model Evaluation

### Setup
1. Prepare the environment according to https://github.com/yxuansu/PandaGPT
2. Download Imagebind, Vicuna, PandaGPT Delta checkpoints according to
3. Pass `imagebind_ckpt_path`, `vicuna_ckpt_path`, `delta_ckpt_path` to the `VisITPandaGPT` class.

#### Reference
https://github.com/yxuansu/PandaGPT#2-running-pandagpt-demo-back-to-top

## VisualChatGPT Model Evaluation
1. Prepare the environment
```
# clone the repo
git clone https://github.com/microsoft/TaskMatrix.git

# Go to directory
cd visual-chatgpt

# create a new environment
conda create -n visgpt python=3.8

# activate the new environment
conda activate visgpt

#  prepare the basic environments
pip install -r requirements.txt
pip install  git+https://github.com/IDEA-Research/GroundingDINO.git
pip install  git+https://github.com/facebookresearch/segment-anything.git

# prepare your private OpenAI key (for Linux)
export OPENAI_API_KEY={Your_Private_Openai_Key}

# prepare your private OpenAI key (for Windows)
set OPENAI_API_KEY={Your_Private_Openai_Key}
```
2. (Optional) Set ChatGPT model name in [`./visual_chatgpt_utils/visual_chatgpt.py` L](https://github.com/mlfoundations/VisIT-Bench/blob/main/baselines/visual_chatgpt_utils/visual_chatgpt.py#L44). It is recomended to use `text-davinci-003` as by default.

#### Reference
https://github.com/microsoft/TaskMatrix

## InstructBLIP2 Model Evaluation

### Option 1: use the transformers library (default)

##### Regerence
https://huggingface.co/Salesforce/instructblip-vicuna-13b

### Option 2: use the lavis library

##### Regerence
https://github.com/salesforce/LAVIS/tree/main/projects/instructblip
