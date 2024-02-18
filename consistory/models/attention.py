# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
from typing import Any, Dict, Optional


import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, USE_PEFT_BACKEND
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import  FeedForward, AdaLayerNorm, GatedSelfAttentionDense, _chunked_feed_forward

from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormZero # AdaLayerNormContinuous
from diffusers.models.unet_2d import UNet2DModel

from einops import rearrange, repeat
import random

from consistory.models.utils import expand_image, image_with_otsu, mask_with_otsu, mask_with_otsu_pytorch

class SelfSubjectReaderWriterMixin:
        
    def set_self_subject_mode(self, mode):
        self.mode = mode
        
    def capture(self, hidden_states):            
        self.last_hidden_state = hidden_states.clone()
        
    def copy_from(self, other):
        self.last_hidden_state = other.last_hidden_state
    
    def reset(self):
        self.last_hidden_state = None    


@maybe_allow_in_graph
class SETransformerBlock(nn.Module, SelfSubjectReaderWriterMixin):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"
        
        # self subject attention
        self.last_hidden_state = None
        self.mode = None # "writer" or "reader"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        # elif self.use_ada_layer_norm_continuous:
        #     self.norm1 = AdaLayerNormContinuous(
        #         dim,
        #         ada_norm_continous_conditioning_embedding_dim,
        #         norm_elementwise_affine,
        #         norm_eps,
        #         ada_norm_bias,
        #         "rms_norm",
        #     )
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)  

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if self.use_ada_layer_norm:
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            # elif self.use_ada_layer_norm_continuous:
            #     self.norm2 = AdaLayerNormContinuous(
            #         dim,
            #         ada_norm_continous_conditioning_embedding_dim,
            #         norm_elementwise_affine,
            #         norm_eps,
            #         ada_norm_bias,
            #         "rms_norm",
            #     )
            else:
                self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if self.use_ada_layer_norm_continuous:
            # self.norm3 = AdaLayerNormContinuous(
            #     dim,
            #     ada_norm_continous_conditioning_embedding_dim,
            #     norm_elementwise_affine,
            #     norm_eps,
            #     ada_norm_bias,
            #     "layer_norm",
            # )
            raise ValueError("not found AdaLayerNormContinuous")
        elif not self.use_ada_layer_norm_single:
            self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            # inner_dim=ff_inner_dim,
            # bias=ff_bias,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0
        
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        feature_map_layers = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        enable_sdsa: bool = True,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,        
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_layer_norm:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.use_ada_layer_norm_continuous:
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.use_ada_layer_norm_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)
               
        # mask を生成
        if enable_sdsa:
            sdsa_masks = []
            for batch_pos in range(batch_size):        
                attention_map = self.attn1.processor.attention_map
                tokenizer = self.attn1.processor.tokenizer
                            
                # TODO: support cfg
                prompts = self.attn1.processor.get_prompt_func()
                prompt = prompts[batch_pos]   
                keywords = self.attn1.processor.get_keywords_func()
                try:                 
                    global_heat_map = attention_map.compute_global_heat_map(tokenizer, prompt, batch_pos, factors=self.attn1.processor.factors)         
                except Exception as e:
                    print(f"Error: {e}")
                    sdsa_masks.append({})                
                    continue
                            
                if global_heat_map is not None:
                    
                    mask_size = (64, 64)
                    
                    heat_map = global_heat_map.compute_word_heat_map(keywords, mode="sum")
                    
                    heat_map_img = expand_image(heat_map, mask_size[1], mask_size[0])
                    tensor_otsu = mask_with_otsu_pytorch(heat_map_img).to(device=hidden_states.device, dtype=hidden_states.dtype)
                    
                    sdsa_masks.append(tensor_otsu)              
            
            if(len(sdsa_masks) > 0):  
                attention_mask = (attention_mask, sdsa_masks)               

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        
        # 2.1 Feature Injection
        
        alpha = 0.8
        if feature_map_layers is not None and len(sdsa_masks) > 0:
            
            batch, hw, c = attn_output.shape
            h = w = int(hw ** 0.5)
                        
            masks = torch.stack(sdsa_masks, dim=0)
            masks = masks.unsqueeze(1)
            masks = F.interpolate(masks.detach(), size=(h, w), mode='bicubic')
            masks = masks.squeeze(1)        
            
            feature_map = None
            if feature_map_layers[0].shape[1] == h:
                feature_map = feature_map_layers[0]
            elif feature_map_layers[1].shape[1] == h:
                feature_map = feature_map_layers[1]
            elif feature_map_layers[2].shape[1] == h:
                feature_map = feature_map_layers[2]
                
            # print("attn_output.shape", attn_output.shape)
            # print("feature_map.shape", feature_map.shape)
            
            if feature_map is not None:
                for b_index in range(batch):                    
                    for h_index in range(h):
                        for w_index in range(w):
                            q = feature_map[b_index, h_index, w_index]
                            mask = masks[b_index, h_index, w_index]
                            
                            if q[0] != -1 and mask > 0.5:
                                mask_q = masks[q[0], q[1], q[2]]
                                if mask_q > 0.5:
                                    d =  attn_output[q[0], q[1] * w + q[2]]
                                    attn_output[b_index, h_index * w + w_index] = (1 - alpha) * attn_output[b_index, h_index* w + w_index] + alpha * d        
        
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.use_ada_layer_norm_single:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
            
        

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"]) 
        
        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.use_ada_layer_norm_single:
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.use_ada_layer_norm_continuous:
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                norm_hidden_states = self.pos_embed(norm_hidden_states)    

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if self.use_ada_layer_norm_continuous:
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, lora_scale=lora_scale
            )
        else:
            ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states    
    
def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

    # print random module.parameters for debugging
def print_random_module_parameters(module):
    for name, param in module.named_parameters():
        if name.startswith("to_out") and torch.max(param) > 0.0:
            print(f"Name: {name}, Parameter:", param)
            break
