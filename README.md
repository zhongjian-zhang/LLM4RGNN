# LLM4RGNN

Source code for KDD 2025 paper "**Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**"
Paper link: https://arxiv.org/pdf/2408.08685

![QQ_1733562983626](https://img.dreamcodecity.cn/img/QQ_1733562983626.png)

## 1. Python Environment

- Python 3.8
- PyTorch 2.1.1
- torch_geometric 2.4.0
- OS: Linux ubuntu 5.15.0-102-generic.
- CPU: Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz.
- GPU: NVIDIA A800 80GB.

Create a conda (see Anaconda or Miniconda) environment with the required packages:
```sh
conda env create -f environment.yml
```

## 2. Code Structure

```
LLM4RGNN/
├── dataset
│   ├── arxiv
│   ├── citeseer
│   ├── cora
│   ├── product
│   └── pubmed
├── llm_response
│   └── mistral-7b-merge
│       ├── all
│       ├── clean
│       ├── global
│       └── negative
├── saved_model
│   ├── attack
│   ├── clean
│   ├── candidate_node
│   ├── llm
│   ├── negative_edge
│   ├── node_emb
│   └── purify
└── src
    ├── LLaMA-Factory
    │   ├── assets
    │   ├── data
    │   ├── evaluation
    │   ├── examples
    │   ├── scripts
    │   ├── src
    │   └── tests
    ├── model
    ├── script
    ├── util
    └── vllm
        ├── instruction
        └── output
```

## 3. Dataset

The data sources are as follows:

- Citeseer: [Graph-LLM Repository](https://github.com/CurryTang/Graph-LLM) (MIT license)
- OGBN-Products: [LLM-Structured-Data Repository](https://github.com/TRAIS-Lab/LLM-Structured-Data) (MIT license)
- Cora, Pubmed, OGBN-Arxiv, TAPE-Arxiv23: [TAPE Repository](https://github.com/XiaoxinHe/TAPE) (MIT license)

Notably, to conveniently load datasets, we integrate textual information and graph information into a single pt file, and you can download the pt file for big graph dataset from the links below:

https://drive.google.com/file/d/1GcZuuEIY8g4xgd6KWsglLjNnIveVLmLQ/view?usp=sharing

Additionally, you can download the cora and citeseer datasets from the links below:

https://drive.google.com/file/d/18byQN6O8FXOsUanbTGbqE0uJjzQeZcyf/view?usp=sharing

|        Dataset         | #Nodes  |  #Edges   | #Classes | #Features |  Method   |
| :--------------------: | :-----: | :-------: | :------: | :-------: | :-------: |
|          Cora          |  2,708  |   5,429   |    7     |   1,433   |    BoW    |
|        Citeseer        |  3,186  |   4,225   |    6     |   3,113   |    BoW    |
|         PubMed         | 19,717  |  44,338   |    3     |    500    |  TF-IDF   |
|       OGBN-Arxiv       | 169,343 | 1,166,243 |    40    |    128    | skip-gram |
|  OGBN-Arxiv (subset)   | 14,167  |  33,520   |    40    |    128    | skip-gram |
| OGBN-Products (subset) | 12,394  |  29,676   |    47    |    100    |    BoW    |

## 4. LLMs

LLM4RGNN is a general framework, suitable for different LLMs. As representative 7B-scale LLMs, Mistral-7B is selected as the local LLM in our experiment. You can download Mistral-7B from the link below:

Hugging Face: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1

We utilize GPT-4 to construct an instruction dataset, including GPT-4's maliciousness assessments and analyses of 26,518 edges. The dataset located in `LLM4RGNN/src/LLaMA-Factory/data/train.jsonl`, and you can tune any LLMs by running the following commend:

```bash
bash LLM4RGNN/src/LLaMA-Factory/instruct_tuning.sh
```

Then, you need to merge the Lora file and the LLM file to construct the tuned LLM by running the following commend:

```bash
bash LLM4RGNN/src/LLaMA-Factory/merge.sh
```

**Notably, you need to specify the original llm path by modifying the model_name_or_path in instruct_tuning.sh and merge.sh**

We also provide the mistral-7B lora file in `LLM4RGNN/saved_model/llm/mistral-7b-lora`, thus you can directly use it to skip tuning LLMs.

🎯We recently uploaded the well-tuned Mistral-7B at [https://huggingface.co/DreamCode/LLM4RGNN](https://huggingface.co/DreamCode/LLM4RGNN), and you can download it directly for use.

## 5. Experiment

First, for attacked graph structure and negative samples of each dataset, you need to create the inference file of LLMs:

```bash
python LLM4RGNN/src/script/create_instruction.py
```

Then, you need to add the inference file to `LLM4RGNN/src/LLaMA-Factory/data/dataset_info.json` and utilize the well-tuned LLMs to infer the edge relationships:

```bash
bash LLM4RGNN/src/LLaMA-Factory/inference.sh
```

Finally, you can purify the attacked graph structure and test the performance of GNNs:

```bash
python src/LLM/script/exp.py
```

## 6. vLLM for Improving Inference Efficiency

To extend LLM4RGNN to large scale graph, such as OGBN-Arxiv (with 169,343 nodes and 1,166,243 edges), we introduce the parallel inference framework vLLM and cache the edges inferred by the LLM. You can get inference result of LLM by running the following commend:

```bash
bash LLM4RGNN/src/vllm/vllm_inference.sh
```

## 7. Hyper-parameters

For local LLMs, when no purification occurs, the purification threshold 𝛽 is selected from {1, 2} to prevent deleting too many edges; otherwise, it is selected from {2, 3, 4}.

For LM-based edge predictor, the threshold 𝛾 is tuned from {0.91, 0.93, 0.95, 0.97, 0.99} and the number of edges 𝐾 is tuned from {1, 3, 5, 7, 9}.

## 8. 📝 Citation and Reference

If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:

```
@article{zhang2024llm4rgnn,
  title={Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?},
  author={Zhang, Zhongjian and Wang, Xiao and Zhou, Huichi and Yu, Yue and Zhang, Mengmei and Yang, Cheng and Shi, Chuan},
  journal={arXiv preprint arXiv:2408.08685},
  year={2024}
}
```
