# Data Construction
The data construction pipeline contains 5 steps: caption and object extraction, mask generation, editing pair construction, instruction recaption, quality evaluation.

This directory contains the official implementation of these 5 steps, each steps are in the corresponding directories.
## Caption & object extraction
### Caption
```
python data_processing/1_caption_object_extraction/1_1_recaption.py
```
The output json file example is stored in: ../data_processing/assets/1_caption_object_extraction
### Object Extraction
We use vLLM as a LLM server, and then to call the LLM to return the object json list. In this construction pipeline we choose [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct).
```
python data_processing/1_caption_object_extraction/1_2_object_extraction.py
```

### (optional)Object Filtering
Use LLM to help to filter the inappropriate object.
```
python data_processing/1_caption_object_extraction/1_3_1_remove_add_filter(optional).py
```
```
python data_processing/1_caption_object_extraction/1_3_2_replace_filter(optional).py
```
The output json file example is stored in: ../data_processing/assets/1_caption_object_extraction


## Mask Generation
The mask generation utilize the [GroundedSAM](https://github.com/IDEA-Research/Grounded-SAM-2) pipeline. The environment setup guideline is in the corresponding repository.
```
conda create --name sam python=3.10

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

export CUDA_HOME=/usr/local/cuda-11.8/

pip install -e .

pip install --no-build-isolation -e grounding_dino
```

The code to generate masks is:
```
bash data_processing/2_mask_generation/Grounded-SAM-2/run.sh
```

The output json file example is stored in: ../data_processing/assets/2_mask_generation/1543.json

## Editing Pair Construction
We choose [Powerpaintv2](https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1) as our mask-based editing model to generate target image. The environment setup guideline of [Powerpaintv2](https://github.com/open-mmlab/PowerPaint?tab=readme-ov-file) is in the original github repository.

The python script of addition and removal task is:
```
nohup python -u data_processing/3_editing_pair_construction/PowerPaint/3_1_remove_add_construction.py --share --version ppt-v2 --checkpoint_dir checkpoints/ppt-v2 > remove_add_log.log 2>&1 &
```

The replace task needs to involve a VLM to generate a replace object name. The python script of replace task is:
```
nohup python -u data_processing/3_editing_pair_construction/PowerPaint/3_2_replace_construction.py --share --version ppt-v2 --checkpoint_dir checkpoints/ppt-v2 > replace_log.log 2>&1 &
```

The output json file example is stored in:
../data_processing/assets/3_editing_pair_construction

## Instruction Recaption
To employ VLM to generate advanced instruction based on the templated instruction. The python scripts are below:
```
#For removal and addition tasks
python data_processing/4_instruction_recaption/recaption_instruction_remove.py

#For replacement tasks
python data_processing/4_instruction_recaption/recaption_instruction_replace.py
```

The output json file example is stored in: ../data_processing/assets/4_instruction_recaption
## Quality Evaluation
We employ the [VIEScore](https://github.com/TIGER-AI-Lab/VIEScore) to evaluate the constructed image editing pairs from senmantic consistency and perceptual quality.
```
python data_processing/5_quality_evaluation/5_evaluate.py
```
The output json file example is stored in:
../data_processing/assets/5_quality_evaluation