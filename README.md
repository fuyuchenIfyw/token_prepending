# Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2412.11556)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)

**Official Repository for ACL 2025 Paper**

*A plug-and-play, training-free technique to enhance sentence embeddings from Large Language Models*

</div>

## 📖 Abstract

Extracting sentence embeddings from large language models (LLMs) is a promising direction, as LLMs have demonstrated stronger semantic understanding capabilities. Previous studies typically focus on prompt engineering to elicit sentence embeddings from LLMs by prompting the model to encode sentence information into the embedding of the last token. However, LLMs are mostly decoder-only models with causal attention and the earlier tokens in the sentence cannot attend to the latter tokens, resulting in biased encoding of sentence information and cascading effects on the final decoded token. 

To this end, we propose a novel **Token Prepending (TP)** technique that prepends each layer's decoded sentence embedding to the beginning of the sentence in the next layer's input, allowing earlier tokens to attend to the complete sentence information under the causal attention mechanism. The proposed TP technique is a plug-and-play and training-free technique, which means it can be seamlessly integrated with various prompt-based sentence embedding methods and autoregressive LLMs.

## 🚀 Key Features

- **🔧 Training-Free**: No additional training required, works out-of-the-box
- **🔌 Plug-and-Play**: Compatible with various LLMs and prompt methods
- **📈 Performance Boost**: Significant improvements on STS and Transfer tasks
- **⚡ Efficient**: Negligible additional inference cost
- **🎛️ Configurable**: YAML-based configuration system for easy experimentation

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (recommended for GPU acceleration)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/fuyuchenIfyw/token_prepending.git
cd token_prepending
```
 
### 2. Create Environment
```bash
conda create -n tp python=3.9
conda activate tp
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download SentEval Data
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
cd -
```

## ⚙️ Configuration System

We provide a flexible YAML-based configuration system for managing different model settings and hyperparameters.


### Example Configuration (`config.yaml`)

```yaml
# Default configuration
default_config: llama-2-7b

# GPU settings
gpu_config:
  cuda_visible_devices: "0,1"

# Model configurations
models:
  # Llama-2-7b with Token Prepending
  llama-2-7b-tp:
    model_name_or_path: "/path/to/Llama-2-7b-hf"
    use_which_plan: tp          # Enable Token Prepending
    output_layer: 27            # Use 27th layer
    tp_starting_index: 1        # TP starting layer
    tp_exiting_index: 7         # TP ending layer
    batch_size: 16
    mode: test
    task_set: sts
    prompt_method: prompteol
  
  # Llama-2-7b vanilla (baseline)
  llama-2-7b:
    model_name_or_path: "/path/to/Llama-2-7b-hf"
    use_which_plan: vanilla     # Standard approach
    output_layer: -1            # Use last layer
    tp_starting_index: 0        # Not used in vanilla mode
    tp_exiting_index: 0         # Not used in vanilla mode
    batch_size: 16
    mode: test
    task_set: sts
    prompt_method: prompteol
```

## 🎯 Usage

```bash
# Run with Token Prepending
bash run.sh llama-2-7b-tp

# Run vanilla baseline
bash run.sh llama-2-7b

# Use custom configuration file
bash run.sh qwen2-7b-tp config.yaml
```


## 📊 Supported Models

The following table lists all pretrained models used in our paper and their corresponding Hugging Face download paths:

<div align="center">

| 🤖 Model Family | 📏 Model Size | 🔗 Hugging Face Path | 📥 Direct Link |
|:---------------:|:-------------:|:---------------------|:---------------:|
| **Llama-2** | 7B | `meta-llama/Llama-2-7b-hf` | [![HF](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| **Llama-2** | 13B | `meta-llama/Llama-2-13b-hf` | [![HF](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/meta-llama/Llama-2-13b-hf) |
| **Llama-3** | 8B | `meta-llama/Meta-Llama-3-8B` | [![HF](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
| **Qwen2** | 7B | `Qwen/Qwen2-7B` | [![HF](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/Qwen/Qwen2-7B) |
| **Gemma2** | 9B | `google/gemma-2-9b` | [![HF](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/google/gemma-2-9b) |

</div>



## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{fu2024token,
  title={Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs},
  author={Fu, Yuchen and Cheng, Zifeng and Jiang, Zhiwei and Wang, Zhonghui and Yin, Yafeng and Li, Zhengliang and Gu, Qing},
  journal={arXiv preprint arXiv:2412.11556},
  year={2024}
}
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact the authors via email
- Check our [paper](https://arxiv.org/abs/2412.11556) for more details



---

<div align="center">

**[Paper](https://arxiv.org/abs/2412.11556) | [Code](https://github.com/your-username/token_prepending) | [Issues](https://github.com/your-username/token_prepending/issues)**

*Made with ❤️ for the NLP research community*

</div>