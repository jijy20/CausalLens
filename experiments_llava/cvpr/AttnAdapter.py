import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv, apply_rotary_pos_emb
import pdb


class AttnAdapter(LlamaAttention):
    """
    CausalLens-Hybrid Attention Adapter (single-pass)
    - Head-level hybrid intervention:
        h_head = (1-gamma) * h_orig + gamma * [h_lang + lambda * s * (h_vis - h_lang)]
    - Post o_proj projection residual:
        attn_output = o_proj(h_head_flat) + lambda * s_global * o_proj(delta_head_flat)
    Notes:
      - Single forward pass (no extra forward required).
      - Safe slicing for variable kv_seq_len (handles past_key_value).
      - All tensor shapes are annotated in comments below for clarity.
    """

    def __init__(self,
                 config: LlamaConfig,
                 lambda_causal: float = 0.15,
                 gamma_mix: float = 0.2,
                 sys_len: int = 35,
                 img_len: int = 576):
        super().__init__(config)
        # hyperparameters controlling intervention
        self.lambda_causal = float(lambda_causal)  # intervention strength
        self.gamma_mix = float(gamma_mix)          # mixing between residual & replacement (0..1)
        self.sys_len = int(sys_len)                # system/prompt token count
        self.img_len = int(img_len)                # image patch token count

    def forward(
        self,
        hidden_states: torch.Tensor,                      # (bsz, q_len, hidden_size)
        attention_mask: Optional[torch.Tensor] = None,    # (bsz, 1, q_len, kv_seq_len) or None
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        # --------------------------
        # 1) Q/K/V projections
        # --------------------------
        # Basic (non-sharded) Q/K/V. If you use tensor-parallelism you may need to include that logic.
        query_states = self.q_proj(hidden_states)  # (bsz, q_len, num_heads*head_dim)
        key_states = self.k_proj(hidden_states)    # (bsz, kv_seq_len, num_kv_heads*head_dim)
        value_states = self.v_proj(hidden_states)  # (bsz, kv_seq_len, num_kv_heads*head_dim)

        # reshape to multi-head: query: (bsz, num_heads, q_len, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # key/value: (bsz, num_kv_heads, kv_seq_len, head_dim)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # rotary embeddings (keeps shapes)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # concat past kv if provided (generation)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat kv heads if needed to match num_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # after repeat: key_states/value_states shape: (bsz, num_heads, kv_seq_len, head_dim)

        # --------------------------
        # 2) attention scores -> softmax
        #    attn_weights: (bsz, num_heads, q_len, kv_seq_len)
        # --------------------------
        attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            # basic size check
            if attention_mask.size() != (bsz, 1, q_len, attn_scores.shape[-1]):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, attn_scores.shape[-1])}, but is {attention_mask.size()}"
                )
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # shape: (bsz, num_heads, q_len, kv_seq_len)

        # --------------------------
        # 3) path decomposition: A_lang / A_vis / A_rest
        # --------------------------
        kv_seq_len = value_states.shape[-2]  # recalc in case of past concat
        SYS_LEN = min(self.sys_len, kv_seq_len)
        IMG_LEN = self.img_len
        vis_start = SYS_LEN
        vis_end = min(SYS_LEN + IMG_LEN, kv_seq_len)  # half-open: [vis_start, vis_end)

        # safe slices (some parts may be empty if kv_seq_len small)
        A_lang = attn_weights[..., :vis_start]        # (bsz, num_heads, q_len, vis_start)
        A_vis = attn_weights[..., vis_start:vis_end]  # (bsz, num_heads, q_len, vis_len)
        A_rest = attn_weights[..., vis_end:]          # (bsz, num_heads, q_len, kv_seq_len - vis_end)

        V_lang = value_states[..., :vis_start, :]     # (bsz, num_heads, vis_start, head_dim)
        V_vis = value_states[..., vis_start:vis_end, :]   # (bsz, num_heads, vis_len, head_dim)
        V_rest = value_states[..., vis_end:, :]       # (bsz, num_heads, rest_len, head_dim)

        # compute per-path head outputs (zeros if empty)
        if vis_start > 0:
            h_lang = torch.matmul(A_lang, V_lang)    # (bsz, num_heads, q_len, head_dim)
        else:
            h_lang = torch.zeros(bsz, self.num_heads, q_len, self.head_dim, device=device, dtype=query_states.dtype)

        if vis_end > vis_start:
            h_vis = torch.matmul(A_vis, V_vis)       # (bsz, num_heads, q_len, head_dim)
        else:
            h_vis = torch.zeros_like(h_lang)

        if kv_seq_len > vis_end:
            h_rest = torch.matmul(A_rest, V_rest)    # (bsz, num_heads, q_len, head_dim)
        else:
            h_rest = torch.zeros_like(h_lang)

        # original full attention head outputs (the conventional attn output)
        h_orig = h_lang + h_vis + h_rest            # (bsz, num_heads, q_len, head_dim)

        # --------------------------
        # 4) compute head-level visual causal sensitivity s_score
        #    s_score shape: (bsz, num_heads, q_len, 1)
        # --------------------------
        if (vis_end > vis_start):
            var = A_vis.var(dim=-1, keepdim=True)   # variance over vis_len
            mean = A_vis.mean(dim=-1, keepdim=True)
            s_score = var / (mean + 1e-6)
            # normalize across heads to make s comparable between heads
            s_score = s_score / (s_score.mean(dim=1, keepdim=True) + 1e-6)
        else:
            s_score = torch.zeros(bsz, self.num_heads, q_len, 1, device=device, dtype=query_states.dtype)

        # --------------------------
        # 5) head-level hybrid intervention:
        #    h_head = (1-gamma) * h_orig + gamma * [h_lang + lambda * s * (h_vis - h_lang)]
        #    shape: (bsz, num_heads, q_len, head_dim)
        # --------------------------
        E_vis  = (h_vis ** 2).mean(dim=-1, keepdim=True)  # (bsz, num_heads, q_len, 1)
        E_lang = (h_lang ** 2).mean(dim=-1, keepdim=True)
        # ----- Global causal gate (γ_dynamic) -----
        E_vis  = (h_vis ** 2).mean()
        E_lang = (h_lang ** 2).mean()
        gamma_dynamic = E_lang / (E_vis + E_lang + 1e-6)

# broadcast to shape (bsz, num_heads, q_len, 1)
        gamma_dynamic = gamma_dynamic.view(1, 1, 1, 1)

        self.gamma_mix = gamma_dynamic
        h_head = (1.0 - self.gamma_mix) * h_orig + self.gamma_mix * (
            h_lang + self.lambda_causal * s_score * (h_vis - h_lang)
        )

        # flatten head outputs to (bsz, q_len, hidden_size) for o_proj
        h_head_flat = h_head.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)  # (bsz, q_len, hidden_size)
        h_lang_flat = h_lang.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)  # for delta calc
        h_vis_flat = h_vis.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        # print(s_score)

        # --------------------------
        # 6) project via o_proj (multi-head fusion)
        #    attn_output: (bsz, q_len, hidden_size)
        # --------------------------
        attn_output = self.o_proj(h_head_flat)

        # --------------------------
        # 7) post-o_proj causal residual (ensure projection of head-delta via W_O)
        #    delta_head = h_vis - h_lang -> flatten -> o_proj -> add
        # --------------------------
        # s_global: (bsz, q_len, 1)
        s_global = s_score.mean(dim=1, keepdim=False)  # average over heads
        # print(s_global)
        # pdb.set_trace()

        # delta in head-space
        delta_head = (h_vis - h_lang)  # (bsz, num_heads, q_len, head_dim)
        # flatten and project to the same space as attn_output
        delta_flat = delta_head.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)  # (bsz, q_len, hidden_size)
        delta_proj = self.o_proj(delta_flat)  # (bsz, q_len, hidden_size)

        # final add (residual correction in projection space)
        attn_output = attn_output + self.lambda_causal * s_global * delta_proj

        # optionally zero out attn_weights if not requested
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
