<p align="center" width="100%">
<a target="_blank"><img src="figs/CausalLens_logo_title.png" alt="CausalLens" style="width: 75%; min-width: 200px; display: block; margin: auto;"></a>
</p>

# CausalLens: Sensitivity-Guided Multi-Head Causal Intervention for Hallucination Mitigation in Large Vision-Language Models

This is the official repo for CausalLens, a simple, training-free method for mitigating hallucinations in Large Vision-Language Models (LVLMs) through sensitivity-guided multi-head causal intervention.

<div style='display:flex; gap: 0.25rem; '>
<a href='https://openaccess.thecvf.com/content/CVPR2026/papers/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
<a href='https://cvpr.thecvf.com/'><img src='https://img.shields.io/badge/CVPR-2026-blue'></a>
<a href='https://twitter.com/YourTwitter'><img src='https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40Us'></a>
</div>

## 🔥 Update
 🎉 CausalLens is accepted by CVPR 2026!


## 🎯 Overview

- We introduce **CausalLens**, a novel **training-free** method that mitigates object hallucinations in LVLMs through **Sensitivity-Guided Multi-Head Causal Intervention**.

- Different from existing contrastive decoding methods (e.g., VCD), CausalLens explicitly models and intervenes on the causal relationships between visual and textual representations in LVLMs.

- The proposed method effectively reduces hallucinations by:
  - **Sensitivity-guided intervention**: Identifying and adjusting attention heads that are most sensitive to visual hallucinations
  - **Multi-head causal intervention**: Applying targeted interventions across multiple attention layers
  - **Adaptive mixing strategy**: Balancing between original and intervened representations

- CausalLens achieves state-of-the-art performance on POPE benchmark and generalizes well to various LVLM architectures including LLaVA and Qwen2-VL.

## 🕹️ Usage

### Environment Setup

**For LLaVA experiments:**
```bash
conda create -yn causallens python=3.9
conda activate causallens
cd CausalLens
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r llavarequirements.txt
```

**For Qwen2-VL experiments:**
```bash
conda create -yn causallens_qwen python=3.9
conda activate causallens_qwen
cd CausalLens
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r qwen2vlrequirements.txt
```

### How to Use CausalLens in LVLMs

The core functions of CausalLens are located in the `causallens_utils` folder. The main components are:

1. **`sample_causallens.py`**: Modified sampling function with causal intervention
2. **`AttnAdapter.py`**: Attention adapter for multi-head causal intervention

#### Quick Start with LLaVA

Here's an example of how to use CausalLens with LLaVA:

**Step 1**: Import and initialize CausalLens sampling at the beginning of your script:

```python
from causallens_utils.sample_causallens import evolve_ours_sampling
evolve_ours_sampling()
```

The `evolve_ours_sampling` function replaces the sampling function in the transformers library with CausalLens's causal intervention mechanism.

**Step 2**: Set up the model with CausalLens attention adapters:

```python
from AttnAdapter import AttnAdapter

# Load your model
model_path = "path/to/llava-model"
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, ...)

# Apply CausalLens attention adapters to specified layers
for i, layer in enumerate(model.model.layers):
    if i >= args.layer_start and i <= args.layer_end:
        attn_adap = AttnAdapter(
            layer.self_attn.config,
            lambda_causal=args.lambda_causal,
            gamma_mix=args.gamma_mix,
            sys_len=args.sys_len,
            img_len=args.img_len
        )
        attn_adap.load_state_dict(layer.self_attn.state_dict())
        attn_adap = attn_adap.half().cuda()
        layer.self_attn = attn_adap
```

**Step 3**: Add noise to images (optional, for contrastive decoding):

```python
from causallens_utils.vcd_add_noise import add_diffusion_noise

image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
```

**Step 4**: Generate with CausalLens:

```python
output_ids = model.generate(
    input_ids,
    images=image_tensor.unsqueeze(0).half().cuda(),
    images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
    cd_alpha=args.cd_alpha,
    cd_beta=args.cd_beta,
    do_sample=True,
    temperature=args.temperature,
    top_p=args.top_p,
    top_k=args.top_k,
    max_new_tokens=1024,
    use_cache=False,
    output_attentions=True,
    output_hidden_states=True
)
```

### Key Hyperparameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|--------------|
| `lambda_causal` | Causal intervention strength | 0.15 | 0.1-0.3 |
| `gamma_mix` | Mixing ratio between residual and replacement | 0.15 | 0.1-0.2 |
| `sys_len` | System token count | 35 | 30-40 |
| `img_len` | Image token count | 576 | 576 (for LLaVA) |
| `layer_start` | Start layer index for intervention | 10 | 5-15 |
| `layer_end` | End layer index for intervention | 20 | 15-25 |

## 🏅 Experiments

**POPE Evaluation:**

```bash
python CausalLens/experiments_llava/cvpr/ours.py \
    --model-path /path/to/llava-v1.5-7b \
    --image-folder /path/to/coco/val2014 \
    --question-file /path/to/POPE/coco_pope_random.json \
    --answers-file /path/to/output.jsonl \
    --lambda_causal 0.15 \
    --gamma_mix 0.15 \
    --layer_start 10 \
    --layer_end 20 \
    --sys_len 35 \
    --img_len 576 \
    --use_cd \
    --noise_step 999

# Evaluate the generated answers
python eval_pope.py \
    --gt_files /path/to/POPE/coco_pope_random.json \
    --gen_files /path/to/output.jsonl
```

**For Qwen2-VL:**

```bash
python qwen2ours_pope.py \
    --model-path /path/to/qwen2-vl \
    --image-folder /path/to/images \
    --question-file /path/to/questions.json \
    --answers-file /path/to/output.jsonl \
    --lambda_causal 0.15 \
    --gamma_mix 0.15
```


## 📑 Citation

If you find our project useful, we hope you can star our repo and cite our paper as follows:

```bibtex
@InProceedings{Ji_2026_CVPR,
    author    = {Ji, Junyang and Liu, Qifan and Yang, Wenming and He, Zhihai},
    title     = {CausalLens: Sensitivity-Guided Multi-Head Causal Intervention for Hallucination Mitigation in Large Vision-Language Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {4199-4209}
}
```

## 📝 Related Projects

- [VCD](https://github.com/DAMO-NLP-SG/VCD): Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding (CVPR 2024)
- [VAF](https://github.com/ustc-hyin/ClearSight): A method for improving visual understanding in LVLMs (CVPR 2025)
- [Contrastive Decoding](https://github.com/XiangLi1999/ContrastiveDecoding): Open-ended Text Generation as Optimization
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL): A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
- [LLaVA 1.5](https://github.com/haotian-liu/LLaVA): Improved Baselines with Visual Instruction Tuning

## 📄 License

This project is licensed under the Apache 2.0 License.


