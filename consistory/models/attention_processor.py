from importlib import import_module
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import einsum, nn
from einops import rearrange   

from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import Attention

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None
    
    
class AttentionMappedAttnProcessor:
    r"""
    processor for attention map
    """
    
    def __init__(self, attention_map : "AttentionMap", tokenizer = None, factors = None, reference_index = None, get_keywords_func = None,  get_prompt_func = None, get_negative_prompt_func = None) -> None:
        self.attention_map = attention_map
        self.tokenizer = tokenizer
        self.factors = factors
        self.reference_index = reference_index
        self.get_keywords_func = get_keywords_func
        self.get_prompt_func = get_prompt_func
        self.get_negative_prompt_func = get_negative_prompt_func
        
        self.query_mode = None
        self.cached_queries = []
        self.vt = 0.9
    
    def set_reference_index(self, reference_index):
        self.reference_index = reference_index        
        
    def set_query_mode(self, mode : str, vt = 0.9):
        self.query_mode = mode
        self.vt = vt
        
    def clear_query_cache(self):
        self.cached_queries.clear()        

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:       
        
        is_cross = encoder_hidden_states is not None
        
        if not is_cross:
            is_sdsa = False
            if isinstance(attention_mask, tuple):
                attention_mask, self_subject_attention_mask = attention_mask
                is_sdsa = True # Subject Driven Self Attention
                    
            residual = hidden_states

            args = () if USE_PEFT_BACKEND else (scale,)
                
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, dim = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            
            if self.query_mode == "cache":
                query = attn.to_q(hidden_states, *args)
                self.cached_queries.append(query)
            elif self.query_mode == "interpolate":
                vanilla_query = self.cached_queries.pop(0)
                query = attn.to_q(hidden_states, *args)
                query = (1 - self.vt) * query +  self.vt * vanilla_query 
            else:
                query = attn.to_q(hidden_states, *args)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states, *args) # [batch_size, seq_len, dim]
            value = attn.to_v(encoder_hidden_states, *args) # [batch_size, seq_len, dim]
            
            # バッチ間のkeyとvalueを結合し、バッチごとにバッチ間を横断したattentionを計算する
            # 𝐾+ = [𝐾1 ⊕ 𝐾2 ⊕ . . . ⊕ 𝐾𝑁 ] 
            # 𝑉+ = [𝑉1 ⊕ 𝑉2 ⊕ . . . ⊕ 𝑉𝑁 ]
            
            if is_sdsa:
                key = key.reshape(batch_size * sequence_length, dim).unsqueeze(0).repeat(batch_size, 1, 1)
                value = value.reshape(batch_size * sequence_length, dim).unsqueeze(0).repeat(batch_size, 1, 1)
                                    
            query = attn.head_to_batch_dim(query) # [batch_size, seq_len, dim] -> [batch_size * heads, seq_len, dim // heads]
            key = attn.head_to_batch_dim(key) # [batch_size, seq_len, dim] -> [batch_size * heads, seq_len, dim // heads]
            value = attn.head_to_batch_dim(value) # [batch_size, seq_len, dim] -> [batch_size * heads, seq_len, dim // heads]
    
            head_size = attn.heads
    
            if is_sdsa:
                
                dropout = 0.5
               
                # key = rearrange(key, '(b h) s d -> h (b s) d', b=batch_size, h=head_size)
                # value = rearrange(value, '(b h) s d -> h (b s) d', b=batch_size, h=head_size)
                                
                # quety (batch * heads, seq_len, dim // heads)
                # key (batch * heads, seq_len * batch, dim // heads)
                # value (batch * heads, seq_len * batch, dim // heads)
                
                # # q * k^T (batch, seq_len, seq_len x 2)
                # # mask (batch, seq_len, seq_len x 2)
                
                # cache_key = f"{query.shape[0]}_{query.shape[1]}_{query.shape[1]}"
                # head_dim = query.shape[0] // batch_size
                
                # if cache_key in self.cached_j:
                #     j = self.cached_j[cache_key]
                # else:            
                #     j = torch.ones((query.shape[0], query.shape[1], query.shape[1]), device=query.device, dtype=query.dtype)
                #     self.cached_j[cache_key] = j
                    
                mask_size = int(sequence_length ** 0.5)
                batch_masks = []
                mask_one = torch.ones((query.shape[1], query.shape[1]), device=query.device, dtype=query.dtype)
                if self.reference_index is not None:
                    mask_zero = torch.zeros((query.shape[1], query.shape[1]), device=query.device, dtype=query.dtype)
                    
                for batch_index in range(batch_size):
                    masks = []
                    for idx, sdsa_mask in enumerate(self_subject_attention_mask):       
                        # print(sdsa_mask.shape, sdsa_mask)                 
                        if idx == batch_index:                            
                            sdsa_mask = mask_one
                        elif self.reference_index is not None and batch_index != 0 and idx != 0:
                            sdsa_mask = mask_zero                                       
                        else:
                            sdsa_mask = sdsa_mask.unsqueeze(0).unsqueeze(0)
                            mask_size = int(sequence_length ** 0.5)                            
                            sdsa_mask = F.interpolate(sdsa_mask, size=(mask_size, mask_size), mode='bilinear', align_corners=False)
                            sdsa_mask = sdsa_mask.view(-1, sequence_length)
                            sdsa_mask = sdsa_mask.unsqueeze(1).repeat(1, 1, sequence_length, 1)
                            sdsa_mask = sdsa_mask.squeeze(0).squeeze(0)
                        
                        # drop out mask(if pixel value is droped out, it is 0)
                        sdsa_mask = F.dropout(sdsa_mask, p=dropout, training=True)                        
                        
                        masks.append(sdsa_mask)
                        
                    masks = torch.cat(masks, dim=1)          
                    # print("masks", batch_index, masks.shape)
                    masks = masks.unsqueeze(0).repeat(head_size, 1, 1)
                    
                    batch_masks.append(masks)
                    
                attention_mask = torch.cat(batch_masks, dim=0)
                    
                # print("attention_mask", attention_mask.shape)
                # mask_size = int(sequence_length ** 0.5)
                # ss_mask = F.interpolate(self_subject_attention_mask, size=(mask_size, mask_size), mode='bilinear', align_corners=False)
                # ss_mask = ss_mask.view(-1, sequence_length)
                # # (batch, 1, sequence_length, sequence_length)
                # ss_mask = ss_mask.unsqueeze(1).repeat(1, 1, sequence_length, 1)
                # ss_mask = ss_mask.repeat(1, head_dim, 1, 1)
                # ss_mask = rearrange(ss_mask, 'b h s d -> (b h) s d')
                
                # mask_ref = self.omega_ref * ss_mask
                
                # attention mask (batch * heads, seq, dim * 2 )
                # j = torch.zeros((query.shape[0], query.shape[1], query.shape[1]), device=query.device, dtype=query.dtype)
                # j = torch.ones((query.shape[0], query.shape[1], query.shape[1]), device=query.device, dtype=query.dtype)
                # attention_mask = torch.cat([j] * batch_size, dim=2)
                            
                # # TODO: support reference mask
                # if attention_mask is not None:
                #     attention_mask = torch.cat([attention_mask, attention_mask], dim=1)         
                # print("j", j.shape) 
                # print("query", query.shape)
                # print("key", key.shape)
                # print("attention_mask", attention_mask.shape)
                attention_probs = self.get_subject_driven_attention_scores(query, key, attention_mask)
            else:
                attention_probs = attn.get_attention_scores(query, key, attention_mask)
                
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, *args)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
        
        else:
            # cross attention
            
            residual = hidden_states

            args = () if USE_PEFT_BACKEND else (scale,)

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states, *args)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states, *args)
            value = attn.to_v(encoder_hidden_states, *args)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            
            if is_cross and self.query_mode != "cache":
                self.attention_map.capture_attention_map(attention_probs, batch_size, attn.heads, query, key, value, self.factors)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, *args)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
        
    def get_subject_driven_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        
        alpha =  query.shape[2] ** (-0.5)

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            # self subject attention mask
            baddbmm_input = torch.log(attention_mask)
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=alpha,
        )
        del baddbmm_input

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs