---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /home/zzj/pycharm_project/LLM4RGNN/src/LLM/models/mistral-7b/snapshots/7ad5799710574ba1c1d953eba3077af582f3a773
model-index:
- name: mistral-7b-lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# mistral-7b-lora

This model is a fine-tuned version of [/home/zzj/pycharm_project/LLM4RGNN/src/LLM/models/mistral-7b/snapshots/7ad5799710574ba1c1d953eba3077af582f3a773](https://huggingface.co//home/zzj/pycharm_project/LLM4RGNN/src/LLM/models/mistral-7b/snapshots/7ad5799710574ba1c1d953eba3077af582f3a773) on the train dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 64
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 4.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.10.0
- Transformers 4.38.2
- Pytorch 2.2.1+cu121
- Datasets 2.18.0
- Tokenizers 0.15.2