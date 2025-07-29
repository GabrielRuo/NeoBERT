# From https://stackoverflow.com/a/23689767
# From https://github.com/pytorch/pytorch/issues/97899
# From https://github.com/facebookresearch/llama/blob/main/llama/model.py

import math
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import scaled_dot_product_attention

from typing import Any, Dict, List, Optional
from functools import partial

from xformers.ops import SwiGLU, memory_efficient_attention

from datasets import Dataset

from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizerFast, DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput

from tqdm import tqdm

from .rmsnorm import RMSNorm
from .rotary import precompute_freqs_cis, apply_rotary_emb


class NeoBERTConfig(PretrainedConfig):
    model_type = "neobert"

    # All config parameters must have a default value.
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        rms_norm: bool = True,
        rope: bool = True,
        norm_eps: float = 1e-06,
        hidden_act: str = "SwiGLU",
        vocab_size: int = 32064,
        pad_token_id: int = 0,
        max_length: int = 1024,
        flash_attention: bool = True,
        base_scale: float = 1.0 / (960.0**0.5),
        ngpt: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError("Hidden size must be divisible by the number of heads.")
        self.dim_head = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.rms_norm = rms_norm
        self.rope = rope
        self.norm_eps = norm_eps
        self.hidden_act = hidden_act
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.flash_attention = flash_attention
        self.base_scale = base_scale
        self.ngpt = ngpt
        self.kwargs = kwargs



class Expert(nn.Module):
# ATTENTION A LA CONFIG QUI OIT ETRE NEOBERTCONFIG
    """An FFN -based module that processes the input and returns an output."""
     """
        Initialize an expert. An expert is a FFN-based module that processes the input
        and returns an output. Architecture based on Mixtral

        :param: config: The configuration for the model.
        :param: expert_size The dimension of the hidden layer of the FFN.
        """
    def __init__(self, config: Config, expert_size: int,) -> None:
        super().__init__()
        self.identity = expert_size == 0

        if not self.identity:
            self.ffn = SwiGLU(config.hidden_size, expert_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
#ATTENTION AU SELF.IDENTITY
        if self.identity:
            return x
        x = self.ffn(x)
        return x


class MoEBlock(nn.Module):
    """
    A cost-based router, which uses a linear layer to route the input to the experts based on
    the active task and the previous layer's output.
    """

    def __init__(
        self,
        config: Config,
        expert_dims: list[int],
    ) -> None:
        """
        Initialize a cost-based router. A cost-based router is a router that uses a
        cost-based routing algorithm to route the input to the experts.

        :param: config: The configuration for the model.
        :param: expert_dims: The dimensions of the experts in this router.
        """

        super().__init__()

        self.config = config
        self.expert_dims = expert_dims
        self.num_experts = len(expert_dims)
        self.topk_training = config.num_experts_per_tok_training
        self.topk_inference = config.num_experts_per_tok_inference


        self.gate = nn.Linear(config.n_embd, len(expert_dims))
        
        self.experts = nn.ModuleList(
            [Expert(config, expert_dim) for expert_dim in expert_dims],
        )
        self.jitter_noise = config.router_jitter_noise

    #@torch.autocast(device_type="cuda", dtype=torch.float32) #probably useless given we use autocast in training loop


    def forward(
        self,
        hidden_states: torch.Tensor,
        #task_ids: torch.Tensor,
        *,
        inference: bool = False,
        inference_dropout_threshold: float | None = None,
        inference_disable_complex_experts: bool = False,
        output_activations: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of the cost-based router.

        :param: hidden_states: The output of the previous layer in the model.
        :param: inference: Whether to run the router in inference mode.
        :param: inference_dropout_threshold: If set when running in inference mode,
            experts with a routing weight below this threshold will be disabled.
        :param: inference_disable_complex_experts: If set to true when running in
            inference mode, the most complex expert in each layer will be disabled.
        :return: A tuple containing the raw router output, the router output, the
            indices of the experts to activate, and the mean expert usage loss.

        """
        #add jitter
        hidden_states_shape = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_states_shape[-1])

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        raw_routing_weights = routing_weights.clone()

        #compute expert usage loss 
        mean_expert_usage_loss = self._MoEBlockRoutingCost(raw_routing_weights)

        #if training, apply expert dropout and update routing weights
        if self.training:
            routing_weights = self._RoutingWeightsWithExpertDropout(router_logits,routing_weights)
        
        #Perform top k routing

        if self.training:
            top_k = self.topk_training
        else: 
            top_k = self.topk_inference
        
        final_hidden_states = self._top_k_routing(hidden_states, hidden_states_shape, routing_weights,top_k)


        return final_hidden_states, raw_routing_weights, mean_expert_usage_loss #will probably stop outputing raw_routing_weights in future


    def _RoutingWeightsWithExpertDropout(self, router_logits, routing_weights):

        if (self.config.dropout_max_prob is not None
                and self.config.dropout_router_weight_threshold is not None
        ):

            mask = (routing_weights < self.config.dropout_router_weight_threshold) & (
                    torch.rand_like(routing_weights)
                    < (
                        self.config.dropout_max_prob
                        - (
                            self.config.dropout_max_prob
                            / self.config.dropout_router_weight_threshold
                        )
                        * routing_weights
                    )
                )
        
            router_logits = torch.where(mask, float("-inf"), router_logits)
            routing_weights = F.softmax(router_logits, dim=-1)

        else:
            logger.warning(
                "Expert dropout was requested but `dropout_max_prob` or "
                "`dropout_router_weight_threshold` not set. "
                "Skipping expert dropout."
            )
    
        return routing_weights
    
    def _MoEBlockRoutingCost(self,raw_routing_weights):

        if len(self.expert_dims) > 1:

            routing_costs = torch.tensor(self.expert_dims,dtype=raw_routing_weights.dtype,
                device=raw_routing_weights.device )**self.config.expert_cost_exponent
            
            # Compute usage loss per token: (batch_size * seq_len)    
            expert_usage_loss = torch.einsum('ik,k->i', raw_routing_weights, routing_costs)

            # Average over all tokens in the batch
            mean_expert_usage_loss = expert_usage_loss.mean()
        else:
            mean_expert_usage_loss = None
        return mean_expert_usage_loss
    
    def _top_k_routing(self, hidden_states,hidden_states_shape,routing_weights,top_k):

            batch_size, sequence_length,hidden_dim = hidden_states_shape

            routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )

            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

            expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hitted:
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: NeoBERTConfig):
        super().__init__()

        self.config = config

        # Attention
        self.qkv = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 3, bias=False)
        self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Feedforward network
        match config.hidden_act.lower():
            case "swiglu":
                # To keep the number of parameters and the amount of computation constant, we reduce the number of
                # hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and make it a multiple of 8 to
                # avoid RuntimeError due to misaligned operand
                multiple_of = 8
                intermediate_size = int(2 * config.intermediate_size / 3)
                intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
                self.ffn = SwiGLU(config.hidden_size, intermediate_size, config.hidden_size, bias=False)
            case "gelu":
                self.ffn = nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                    nn.GELU(),
                    nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
                )

        self.attention_norm = (
            RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )
        self.expert_norm = (
            RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )

        self.ffn_dropout = nn.Dropout(config.dropout)

        self.block_moe = MoEBlock(
            config,
            expert_dims
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor):
        x = x + self._att_block(self.attention_norm(x), pad_mask, freqs_cis)

        moe_output = self.block_moe(x)

        x = x + self._ff_block(self.expert_norm(moe_output[0]))
        return (x,*moe_output[1:],)

    def _att_block(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = self.qkv(x).view(batch_size, seq_len, self.config.num_attention_heads, self.config.dim_head * 3).chunk(3, axis=-1)

        if self.config.rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        if self.config.flash_attention:
            attn = memory_efficient_attention(query=xq, key=xk, value=xv, attn_bias=pad_mask, p=0)
        else:
            # Input and output are of dimension (B, H, M, K)
            attn = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=pad_mask,
                dropout_p=self.config.dropout_prob if self.training else 0,
            ).transpose(1, 2)

        return self.resid_dropout(self.wo(attn.reshape(batch_size, seq_len, self.config.num_attention_heads * self.config.dim_head)))

    def _ff_block(self, x: torch.Tensor):
        return self.ffn_dropout(self.ffn(x))


class NormEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: NeoBERTConfig):
        super().__init__()

        self.config = config

        # Attention
        self.qkv = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 3, bias=False)
        self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.c_fc = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.ffn_dropout = nn.Dropout(config.dropout)

        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = config.base_scale
        self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling * torch.ones(config.hidden_size))

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = config.base_scale
        self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling * torch.ones(config.hidden_size))

        self.sqk_init_value = 1.0
        self.sqk_init_scaling = config.base_scale
        self.sqk = torch.nn.Parameter(self.sqk_init_scaling * torch.ones(config.hidden_size))

        self.suv_init_value = 1.0
        self.suv_init_scaling = 1.0
        self.suv = torch.nn.Parameter(self.suv_init_scaling * torch.ones(2 * config.intermediate_size))

    def justnorm(self, x):
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor):
        x_attn = self._att_block(x, pad_mask, freqs_cis)

        lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = self.justnorm(x)
        B_norm = self.justnorm(x_attn)
        x = self.justnorm(A_norm + lr * (B_norm - A_norm))

        x_ff = self._ff_block(x)

        lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = self.justnorm(x)
        B_norm = self.justnorm(x_ff)
        x = self.justnorm(A_norm + lr * (B_norm - A_norm))

        return x

    def _att_block(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = self.qkv(x).view(batch_size, seq_len, self.config.num_attention_heads, self.config.dim_head * 3).chunk(3, axis=-1)

        if self.config.rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
            1, 1, self.config.num_attention_heads, self.config.hidden_size // self.config.num_attention_heads
        )
        xq = sqk * self.justnorm(xq)
        xk = sqk * self.justnorm(xk)

        softmax_scale = (self.config.hidden_size / self.config.num_attention_heads) ** 0.5

        if self.config.flash_attention:
            attn = memory_efficient_attention(query=xq, key=xk, value=xv, attn_bias=pad_mask, p=0, scale=softmax_scale)
        else:
            # Input and output are of dimension (B, H, M, K)
            attn = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=pad_mask,
                dropout_p=self.config.dropout_prob if self.training else 0,
                scale=softmax_scale,
            ).transpose(1, 2)

        return self.resid_dropout(self.wo(attn.reshape(batch_size, seq_len, self.config.hidden_size)))

    def _ff_block(self, x: torch.Tensor):
        uv = self.c_fc(x)
        suv = self.suv * ((self.suv_init_value / self.suv_init_scaling) * (self.config.hidden_size**0.5))
        uv = suv * uv

        u, v = torch.chunk(uv, 2, dim=-1)
        x = u * self.silu(v)
        x = self.mlp_c_proj(x)

        return self.ffn_dropout(x)


class NeoBERTPreTrainedModel(PreTrainedModel):
    config_class = NeoBERTConfig
    _supports_cache_class = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.uniform_(-self.config.embedding_init_range, self.config.embedding_init_range)


class NeoBERT(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if self.config.rope:
            self.freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_length)
        else:
            self.positional_embedding = nn.Embedding(config.max_length + 1, config.hidden_size, padding_idx=config.pad_token_id)

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config))

        self.layer_norm = (
            RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, src, pad_mask=None):
        # Expand and repeat: (Batch, Length) -> (Batch, Heads, Length, Length)
        if pad_mask is not None:
            assert pad_mask.dtype != torch.bool and 1.0 not in pad_mask, "NeoBERT expects an additive pad_mask"
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.config.num_attention_heads, pad_mask.size(-1), 1)

        # RoPE
        freqs_cis = None
        if self.config.rope:
            self.freqs_cis = self.freqs_cis.to(src.device, non_blocking=True)
            freqs_cis = self.freqs_cis[: src.shape[1]]

        # Embedding
        x = self.encoder(src)

        # Positional embedding
        if not self.config.rope:
            mask = src.ne(self.config.pad_token_id).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask  #
            incremental_indices = incremental_indices.long() + self.config.pad_token_id
            x += self.positional_embedding(incremental_indices)
        
        total_mean_expert_usage_loss = torch.tensor(0.0, device=src.device)
        # Transformer encoder
        for layer in self.transformer_encoder:
            layer_output = layer(x, pad_mask, freqs_cis)
            x = layer_output[0]
            # router_logits = layer_output[1]
            mean_expert_usage_loss = layer_output[2]

            # if output_activations:
            #     block_activations = layer_output[3]

            if mean_expert_usage_loss is not None:
                total_mean_expert_usage_loss += mean_expert_usage_loss
                # for k in task_expert_usage_losses:
                #     total__expert_usage_loss[k] = (
                #         total_task_expert_usage_loss[k] + task_expert_usage_losses[k]
                #     )

            # if output_all_pathways:
            #     expert_usages.append(router_logits.cpu().detach().numpy())

            # if output_activations:
            #     expert_activations.append(block_activations)
            x = layer(x, pad_mask, freqs_cis)

        # Final normalization layer
        x = self.layer_norm(x)

        output = (x,total_mean_expert_usage_loss)

        # if output_all_pathways:
        #     output += (expert_usages,)

        # if output_activations:
        #     output += (expert_activations,)

        return output #ATTENTION ON A CHANGE LA FORME DE l'OUTPUT. MAINTENANT ON A AUSSI LA LOSS


class NormNeoBERT(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if self.config.rope:
            self.freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_length)
        else:
            self.positional_embedding = nn.Embedding(config.max_length + 1, config.hidden_size, padding_idx=config.pad_token_id)

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(NormEncoderBlock(config))

        self.layer_norm = (
            RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )

        # Initialize weights and apply final processing
        self.post_init()

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=config.base_scale / math.sqrt(2 * config.num_hidden_layers))

        self.sz_init_value = 1.00
        self.sz_init_scaling = config.base_scale
        self.sz = torch.nn.Parameter(self.sz_init_scaling * torch.ones(config.vocab_size, dtype=torch.float32))

    def forward(self, src, pad_mask=None):
        # Expand and repeat: (Batch, Length) -> (Batch, Heads, Length, Length)
        if pad_mask is not None:
            assert pad_mask.dtype != torch.bool and 1.0 not in pad_mask, "NeoBERT expects an additive pad_mask"
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.config.num_attention_heads, pad_mask.size(-1), 1)

        # RoPE
        freqs_cis = None
        if self.config.rope:
            self.freqs_cis = self.freqs_cis.to(src.device, non_blocking=True)
            freqs_cis = self.freqs_cis[: src.shape[1]]

        # Embedding
        x = self.encoder(src)

        # Positional embedding
        if not self.config.rope:
            mask = src.ne(self.config.pad_token_id).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask  #
            incremental_indices = incremental_indices.long() + self.config.pad_token_id
            x += self.positional_embedding(incremental_indices)

        # Transformer encoder
        for layer in self.transformer_encoder:
            x = layer(x, pad_mask, freqs_cis)

        # Return the output of the last hidden layer
        return x


class NeoBERTLMHead(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.model = NormNeoBERT(config) if self.config.ngpt else NeoBERT(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    def forward(self, src, pad_mask=None):
        hidden_representation,total_mean_expert_usage_loss = self.model.forward(src, pad_mask)
        logits = self.decoder(hidden_representation)

        return {"hidden_representation": hidden_representation, "logits": logits, "expert_usage_loss": total_mean_expert_usage_loss}

#ATTENTION: on a changé les outputs
class Expert(nn.Module):

    def __init__(
        self,
        config: Config,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.identity = hidden_dim == 0

        if not self.identity:
            self.swiglu= SwiGLU(config.n_embd, hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.identity:
            return x
        x = self.swiglu(x)
        return self.output_layer(x)



class NeoBERTForSequenceClassification(NeoBERTPreTrainedModel):

    def __init__(
        self,
        config: NeoBERTConfig,
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        classifier_init_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(config)

        self.config = config

        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        self.classifier_init_range = classifier_init_range

        self.model = NeoBERT(config)

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.classifier_init_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, src, pad_mask=None):
        hidden_representation = self.model.forward(src, pad_mask)

        x = hidden_representation[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        return {"hidden_representation": hidden_representation, "logits": logits}


class NeoBERTHFForSequenceClassification(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.num_labels = getattr(config, "num_labels", 2)
        self.classifier_dropout = getattr(config, "classifier_dropout", 0.1)
        self.classifier_init_range = getattr(config, "classifier_init_range", 0.02)

        self.model = NeoBERT(config)

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.classifier_init_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        hidden_representation = self.model.forward(input_ids, attention_mask)

        x = hidden_representation[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_representation,
            attentions=None,
        )


class NeoBERTForMTEB(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(
        self,
        config: NeoBERTConfig,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 1024,
        batch_size: int = 8,
        pooling: str = "avg",
        **kwargs,
    ):
        super().__init__(config)

        self.config = config
        self.model = NeoBERT(config)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling

    def encode_queries(self, queries: List[str], **kwargs):
        if "instructions" in kwargs:
            if kwargs["instructions"] is not None:
                queries = [(query + " " + kwargs["instructions"][query]).strip() for query in queries]
            new_kwargs = {k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]}
        else:
            new_kwargs = kwargs

        return self.encode(
            queries,
            **new_kwargs,
        )

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + " " + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            if isinstance(corpus[0], dict):
                sentences = [(doc["title"] + " " + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
            else:
                sentences = corpus

        if "instructions" in kwargs:  # not used on the doc side
            new_kwargs = {k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]}
        else:
            new_kwargs = kwargs

        return self.encode(
            sentences,
            **new_kwargs,
        )

    @torch.no_grad()
    def encode(self, sentences: list[str], **kwargs: Any) -> torch.Tensor:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _transform_func(tokenizer: PreTrainedTokenizerFast, x: Dict[str, List]):
            batch_dict = tokenizer(
                x["input_texts"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_token_type_ids=False,
            )

            return batch_dict

        dataset: Dataset = Dataset.from_dict({"input_texts": sentences})
        dataset.set_transform(partial(_transform_func, self.tokenizer))

        data_collator = data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        dataloader = DataLoader(
            dataset,
            collate_fn=data_collator,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
            pin_memory=True,
        )

        encodings = []
        for batch in tqdm(dataloader, desc="encoding", mininterval=10, disable=len(sentences) < 128):
            input_ids = batch["input_ids"].to(device)

            pad_mask = batch["attention_mask"].to(device)
            xformers_mask = torch.where(pad_mask == 1, float(0.0), float("-inf")).type(torch.float16)

            outputs = self.model(input_ids, xformers_mask)

            if self.pooling == "avg":
                outputs = outputs * pad_mask.unsqueeze(-1).expand(-1, -1, outputs.shape[-1])
                outputs = outputs.sum(dim=1) / pad_mask.to(device).sum(dim=1).unsqueeze(-1)
            else:
                outputs = outputs[:, 0, :]

            encodings.append(outputs.cpu().numpy())

        return np.concatenate(encodings, axis=0)
