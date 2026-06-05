

import os
import json
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
from transformers.cache_utils import Cache
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import pdb

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match query heads (for GQA)
    (batch, num_key_value_heads, seq_len, head_dim) -> (batch, num_attention_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2VLAttnAdapter(nn.Module):
    """
    Custom Attention Adapter for Qwen2-VL
    
    Similar to AttnAdapter for LLaMA, this implements:
    - Head-level hybrid intervention
    - Enhancement of attention to image tokens
    - Suppression of attention to system/text tokens
    
    Key hyperparameters:
    - lambda_causal: intervention strength (higher = stronger modification)
    - gamma_mix: mixing ratio between residual and replacement
    - sys_len: number of system prompt tokens (before image tokens)
    - img_len: number of image tokens
    """
    
    def __init__(
        self,
        config,
        layer_idx: int = 0,
        lambda_causal: float = 0.15,
        gamma_mix: float = 0.2,
        sys_len: int = 31,
        img_len: int = 256,
    ):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        
        # Hyperparameters for attention modification
        self.lambda_causal = float(lambda_causal)
        self.gamma_mix = float(gamma_mix)
        self.sys_len = int(sys_len)
        self.img_len = int(img_len)
        
        # Model dimensions from config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_scaling = config.rope_scaling
        
        # Q/K/V/O projections (will be loaded from original attention)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Rotary embedding (will be copied from original)
        self.rotary_emb = None
        
        # Storage for attention weights (for analysis)
        self.last_attn_weights = None
        self.last_hidden_states = None
        self.prefill_attn_weights = None
    
    def update_token_range(self, sys_len: int, img_len: int):
        """Update token range for each sample"""
        self.sys_len = int(sys_len)
        self.img_len = int(img_len)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Custom forward with attention modification.
        """
        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device
        
        # Store hidden states for analysis
        self.last_hidden_states = hidden_states.detach().clone()
        
        # ========== Step 1: Q/K/V projections ==========
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to (bsz, num_heads, seq_len, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # ========== Step 2: Apply Rotary Position Embedding ==========
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )
        
        # ========== Step 3: Handle KV Cache ==========
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        
        # ========== Step 4: Repeat K/V for GQA ==========
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        kv_seq_len = key_states.shape[2]
        
        # ========== Step 5: Compute Attention Scores ==========
        attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :kv_seq_len]
            attn_scores = attn_scores + causal_mask
        
        # Fix precision issues in float16 inference
        if query_states.dtype == torch.float16:
            attn_scores = torch.where(torch.isinf(attn_scores), torch.zeros_like(attn_scores), attn_scores)
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Store attention weights for analysis
        self.last_attn_weights = attn_weights.detach().clone()
        if q_len > 1:
            self.prefill_attn_weights = attn_weights.detach().clone()
        
        # ========== Step 6: Path Decomposition & Causal Intervention ==========
        # Split into image tokens and non-image tokens (lang = sys + rest)
        SYS_LEN = min(self.sys_len, kv_seq_len)
        IMG_LEN = self.img_len
        vis_start = SYS_LEN
        vis_end = min(SYS_LEN + IMG_LEN, kv_seq_len)
        
        # Safe slices - combine sys tokens and rest tokens as non-image tokens
        A_sys = attn_weights[..., :vis_start]
        A_vis = attn_weights[..., vis_start:vis_end]
        A_rest = attn_weights[..., vis_end:]
        # A_lang = A_sys + A_rest (non-image attention)
        
        V_sys = value_states[..., :vis_start, :]
        V_vis = value_states[..., vis_start:vis_end, :]
        V_rest = value_states[..., vis_end:, :]
        
        # Compute per-path head outputs
        # h_lang = h_sys + h_rest (combined non-image tokens)
        h_sys = torch.zeros(bsz, self.num_heads, q_len, self.head_dim, device=device, dtype=query_states.dtype)
        if vis_start > 0:
            h_sys = torch.matmul(A_sys, V_sys)
        
        h_rest = torch.zeros(bsz, self.num_heads, q_len, self.head_dim, device=device, dtype=query_states.dtype)
        if kv_seq_len > vis_end:
            h_rest = torch.matmul(A_rest, V_rest)
        
        # Combine sys and rest as h_lang (non-image tokens)
        h_lang = h_sys + h_rest
        
        if vis_end > vis_start:
            h_vis = torch.matmul(A_vis, V_vis)
        else:
            h_vis = torch.zeros_like(h_lang)
        
        # Original full attention output
        h_orig = h_lang + h_vis
        
        # ========== Step 7: Compute Visual Causal Sensitivity Score ==========
        # Based on difference between image tokens and non-image tokens
        if vis_end > vis_start:
            var = A_vis.var(dim=-1, keepdim=True)
            mean = A_vis.mean(dim=-1, keepdim=True)
            s_score = var / (mean + 1e-6)
            s_score = s_score / (s_score.mean(dim=1, keepdim=True) + 1e-6)
        else:
            s_score = torch.zeros(bsz, self.num_heads, q_len, 1, device=device, dtype=query_states.dtype)
        
        # ========== Step 8: Dynamic Gamma Calculation ==========
        E_vis = (h_vis ** 2).mean()
        E_lang = (h_lang ** 2).mean()
        gamma_dynamic = E_lang / (E_vis + E_lang + 1e-6)
        gamma_dynamic = gamma_dynamic.view(1, 1, 1, 1)
        
        gamma = gamma_dynamic
        
        # ========== Step 9: Hybrid Intervention ==========
        # Compute difference between image tokens and non-image tokens
        # delta = h_vis - h_lang (image - non-image)
        h_head = (1.0 - gamma) * h_orig + gamma * (
            h_lang + self.lambda_causal * s_score * (h_vis - h_lang)
        )
        
        # ========== Step 10: Project Output ==========
        h_head_flat = h_head.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(h_head_flat)
        
        # ========== Step 11: Post O-proj Causal Residual ==========
        s_global = s_score.mean(dim=1, keepdim=False)
        delta_head = h_vis - h_lang
        delta_flat = delta_head.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        delta_proj = self.o_proj(delta_flat)
        
        # Final output with causal correction
        attn_output = attn_output + self.lambda_causal * s_global * delta_proj
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights


def find_vision_token_range(input_ids: torch.Tensor) -> Tuple[int, int]:
    """
    Find vision token range from input_ids
    
    Qwen2-VL special tokens:
        vision_start_token_id: 151652 (<|vision_start|>)
        vision_end_token_id: 151653 (<|vision_end|>)
    """
    vision_start_id = 151652
    vision_end_id = 151653
    
    input_ids_flat = input_ids[0].tolist()
    
    vision_start_idx = None
    vision_end_idx = None
    
    for i, token_id in enumerate(input_ids_flat):
        if token_id == vision_start_id and vision_start_idx is None:
            vision_start_idx = i
        if token_id == vision_end_id:
            vision_end_idx = i
            break
    
    if vision_start_idx is None:
        vision_start_idx = 31
    if vision_end_idx is None:
        vision_end_idx = vision_start_idx + 258
    
    return vision_start_idx, vision_end_idx


def replace_attention_with_adapter(
    model,
    target_layer_range: Tuple[int, int] = (10, 20),
    lambda_causal: float = 0.25,
    gamma_mix: float = 0.25,
    sys_len: int = 31,
    img_len: int = 256,
):
    """
    Replace self_attn in specified layers with Qwen2VLAttnAdapter
    """
    adapters = []
    
    qwen2vl_model = model.model
    
    if hasattr(qwen2vl_model, 'language_model'):
        lm_model = qwen2vl_model.language_model
        layers = lm_model.layers
    elif hasattr(qwen2vl_model, 'layers'):
        layers = qwen2vl_model.layers
    else:
        raise AttributeError(f"Cannot find layers in model. Available attributes: {[n for n, _ in qwen2vl_model.named_children()]}")
    
    for i, layer in enumerate(layers):
        if target_layer_range[0] < i < target_layer_range[1]:
            print(f"Replacing layer {i} self_attn with Qwen2VLAttnAdapter")
            
            attn_adapter = Qwen2VLAttnAdapter(
                config=model.config,
                layer_idx=i,
                lambda_causal=lambda_causal,
                gamma_mix=gamma_mix,
                sys_len=sys_len,
                img_len=img_len,
            )
            
            # Copy weights from original attention
            attn_adapter.q_proj.load_state_dict(layer.self_attn.q_proj.state_dict())
            attn_adapter.k_proj.load_state_dict(layer.self_attn.k_proj.state_dict())
            attn_adapter.v_proj.load_state_dict(layer.self_attn.v_proj.state_dict())
            attn_adapter.o_proj.load_state_dict(layer.self_attn.o_proj.state_dict())
            
            # Copy rotary embedding if exists
            if hasattr(layer.self_attn, 'rotary_emb'):
                attn_adapter.rotary_emb = layer.self_attn.rotary_emb
            
            # Convert to same dtype and device as model
            attn_adapter = attn_adapter.to(
                dtype=next(layer.self_attn.parameters()).dtype,
                device=next(layer.self_attn.parameters()).device
            )
            
            # Replace self_attn
            layer.self_attn = attn_adapter
            adapters.append(attn_adapter)
    
    print(f"\nReplaced {len(adapters)} layers with Qwen2VLAttnAdapter")
    return adapters


def update_adapters_token_range(adapters, sys_len: int, img_len: int):
    """Update token range for all adapters"""
    for adapter in adapters:
        adapter.update_token_range(sys_len, img_len)


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL Attention Adapter for POPE Evaluation")
    parser.add_argument("--model_path", type=str, 
                        default="/checkpoint/Qwen2-VL-7B-Instruct",
                        help="Path to Qwen2-VL model")
    parser.add_argument("--pope_path", type=str,
                        default="/CausalLens/experiments/data/POPE/coco/coco_pope_popular.json",
                        help="Path to POPE dataset json file")
    parser.add_argument("--image_dir", type=str,
                        default="/CausalLens/experiments/data/COCO/val2014",
                        help="Path to COCO val2014 images directory")
    parser.add_argument("--output_dir", type=str,
                        default="/CausalLens/qwenvloutput",
                        help="Output directory for results")
    parser.add_argument("--lambda_causal", type=float, default=0.15,
                        help="Intervention strength (higher = stronger modification)")
    parser.add_argument("--gamma_mix", type=float, default=0.15,
                        help="Mixing ratio between residual and replacement")
    parser.add_argument("--layer_start", type=int, default=10,
                        help="Start layer for replacement (exclusive)")
    parser.add_argument("--layer_end", type=int, default=20,
                        help="End layer for replacement (exclusive)")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Maximum new tokens to generate")
    
    args = parser.parse_args()
    
    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Qwen2-VL Attention Adapter - POPE Evaluation")
    print("=" * 70)
    print(f"lambda_causal: {args.lambda_causal} (intervention strength)")
    print(f"gamma_mix: {args.gamma_mix} (mixing ratio)")
    print(f"Layer range: ({args.layer_start}, {args.layer_end})")
    print(f"POPE dataset: {args.pope_path}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # Replace attention layers with initial token range (will be updated per sample)
    adapters = replace_attention_with_adapter(
        model,
        target_layer_range=(args.layer_start, args.layer_end),
        lambda_causal=args.lambda_causal,
        gamma_mix=args.gamma_mix,
        sys_len=31,
        img_len=256,
    )
    
    # Read POPE dataset
    print("\nReading POPE dataset...")
    pope_data = []
    with open(args.pope_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                pope_data.append(json.loads(line))
    
    print(f"Total samples: {len(pope_data)}")
    
    # Output file path
    output_filename = f"pope_ours_lambda_{args.lambda_causal}_gamma_{args.gamma_mix}.jsonl"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Process each sample
    print(f"\nProcessing samples and saving to {output_path}...")
    
    with open(output_path, 'w') as f_out:
        for item in tqdm(pope_data, desc="Evaluating"):
            question_id = item['question_id']
            image_name = item['image']
            question = item['text']
            label = item['label']
            
            # Construct image path
            image_path = os.path.join(args.image_dir, image_name)
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"\nWarning: Image not found: {image_path}")
                continue
            
            # # Prepare input
            # messages = [
            #     {
            #         "role": "user",
            #         "content": [
            #             {"type": "image", "image": image_path},
            #             {"type": "text", "text": question},
            #         ],
            #     }
            # ]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions."},
                        # {"type": "text", "text": "A chat between a curious human and an artificial intelligence assistant."},
                        # {"type": "text", "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. For visual tasks, the assistant explicitly analyzes image details and spatial relationships to provide answers grounded in strong logical reasoning and visual evidence."}, 
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question},
                    ],
                }
            ]
           
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # Find vision token range and update adapters
            vis_start, vis_end = find_vision_token_range(inputs['input_ids'])
            img_len = vis_end - vis_start
            sys_len = vis_start
            update_adapters_token_range(adapters, sys_len, img_len)
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Write result to jsonl
            result = {
                "question_id": question_id,
                "image": image_name,
                "question": question,
                "answer": output_text,
                "label": label
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()
    
    print("\n" + "=" * 70)
    print(f"Results saved to: {output_path}")
    print("=" * 70)
    
    # Calculate accuracy
    print("\nCalculating accuracy...")
    correct = 0
    total = 0
    
    with open(output_path, 'r') as f:
        for line in f:
            result = json.loads(line.strip())
            answer = result['answer'].lower().strip()
            label = result['label'].lower().strip()
            
            # Check if answer contains yes/no
            if 'yes' in answer:
                pred = 'yes'
            elif 'no' in answer:
                pred = 'no'
            else:
                pred = answer
            
            if pred == label:
                correct += 1
            total += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
