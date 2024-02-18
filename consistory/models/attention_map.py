
import numpy as np
import torch
import torch.nn.functional as F
import math 


from collections import defaultdict
from copy import deepcopy

from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)

from .attention import SETransformerBlock, SelfSubjectReaderWriterMixin
from .attention_processor import AttentionMappedAttnProcessor
from .prompt_analyzer import PromptAnalyzer

class AttentionMap:
    
    def __init__(self) -> None:
        self.attention_maps = defaultdict(lambda: defaultdict(list)) 
        self.all_attention_maps = []
        # TODO: remove hardcoding
        self.img_height = 512
        self.img_width = 512
        self.weighted = False
        self.head_idx = 0
        
    def capture_attention_map(self, attention_probabilities, batch_size, heads, query, key, value, factors=None):
        batch_size_attention = query.shape[0]
        
        slice_size = batch_size_attention // batch_size
        
        def calc_factor_base(w, h):
            z = max(w/64, h/64)
            factor_b = min(w, h) * z
            return factor_b
        
        # factor_base = calc_factor_base(hk_self.img_width, hk_self.img_height)
        factor_base = calc_factor_base(512, 512)
        
        for batch_index in range(attention_probabilities.shape[0] // slice_size):
            start_idx = batch_index * slice_size
            end_idx = (batch_index + 1) * slice_size
            
            attention_slice = attention_probabilities[start_idx:end_idx]
            
            factor = int(math.sqrt(factor_base // attention_slice.shape[1]))
            
            if factor >= 1:
                factor //= 1                  
                
                if factors is not None and factor not in factors:
                    continue
                  
                maps = self._up_sample_attn(attention_slice, value, factor)
                self.attention_maps[batch_index][factor].append(maps)
                
    def store_attention_map(self):
        print(self.attention_maps.keys())
        for k, v in self.attention_maps.items():
            print(k, v.keys())
        self.all_attention_maps.append(deepcopy(self.attention_maps)) # time index
        self.attention_maps.clear()

    def prepare(self):
        self.attention_maps.clear()
        
    def clear_all(self):
        self.all_attention_maps.clear()
        self.attention_maps.clear()
    
        
    @torch.no_grad()
    def _up_sample_attn(self, x, value, factor, method='bicubic'):
        # type: (torch.Tensor, torch.Tensor, int, Literal['bicubic', 'conv']) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        """
        Up samples the attention map in x using interpolation to the maximum size of (64, 64), as assumed in the Stable
        Diffusion model.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.
            method (`str`): the method to use; one of `'bicubic'` or `'conv'`.

        Returns:
            `torch.Tensor`: the up-sampled attention map of shape (tokens, 1, height, width).
        """
        weight = torch.full((factor, factor), 1 / factor ** 2, device=x.device)
        weight = weight.view(1, 1, factor, factor)
        
        h = int(math.sqrt ( (self.img_height * x.size(1)) / self.img_width))
        w = int(self.img_width * h / self.img_height)
        
        h_fix = w_fix = 64
        if h >= w:
            w_fix = int((w * h_fix) / h)
        else:
            h_fix = int((h * w_fix) / w)
                
        maps = []
        x = x.permute(2, 0, 1)
        value = value.permute(1, 0, 2)
        weights = 1

        # with torch.cuda.amp.autocast(dtype=torch.float32):
        for map_ in x:
            map_ = map_.unsqueeze(1).view(map_.size(0), 1, h, w)

            if method == 'bicubic':
                map_ = F.interpolate(map_, size=(h_fix, w_fix), mode='bicubic')
                maps.append(map_.squeeze(1))
            else:
                maps.append(F.conv_transpose2d(map_, weight, stride=factor).squeeze(1))

        if self.weighted:
            weights = value.norm(p=1, dim=-1, keepdim=True).unsqueeze(-1)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)
        
        if self.head_idx:
            maps = maps[:, self.head_idx:self.head_idx+1, :, :]

        return (weights * maps).sum(1, keepdim=True) # .cpu()
    
    
    
    def compute_global_heat_map(self, tokenizer, prompt, batch_index, time_weights=None, time_idx=None, last_n=None, first_n=None, factors=None):
        # type: (Tokenizer, str, int, int, int, int, int, List[float]) -> HeatMap
        """
        Compute the global heat map for the given prompt, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            prompt: The prompt to compute the heat map for.
            time_weights: The weights to apply to each time step. If None, all time steps are weighted equally.
            time_idx: The time step to compute the heat map for. If None, the heat map is computed for all time steps.
                Mutually exclusive with `last_n` and `first_n`.
            last_n: The number of last n time steps to use. If None, the heat map is computed for all time steps.
                Mutually exclusive with `time_idx`.
            first_n: The number of first n time steps to use. If None, the heat map is computed for all time steps.
                Mutually exclusive with `time_idx`.
            factors: Restrict the application to heat maps with spatial factors in this set. If `None`, use all sizes.
        """
        if len(self.all_attention_maps) == 0:
            return None
        
        if time_weights is None:
            time_weights = [1.0] * len(self.all_attention_maps)

        time_weights = np.array(time_weights)
        time_weights /= time_weights.sum()
        all_heat_maps = self.all_attention_maps

        if time_idx is not None:
            heat_maps = [all_heat_maps[time_idx]]
        else:
            heat_maps = all_heat_maps[-last_n:] if last_n is not None else all_heat_maps
            heat_maps = heat_maps[:first_n] if first_n is not None else heat_maps
            

        if factors is None:
            factors = {1, 2, 4, 8, 16, 32}
        else:
            factors = set(factors)

        all_merges = []
        
        for batch_to_heat_maps in heat_maps:
            
            if not (batch_index in batch_to_heat_maps):
                continue    
            
            merge_list = []
                 
            factors_to_heat_maps = batch_to_heat_maps[batch_index]

            for k, heat_map in factors_to_heat_maps.items():
                # heat_map shape: (tokens, 1, height, width)
                # each v is a heat map tensor for a layer of factor size k across the tokens
                if k in factors:
                    merge_list.append(torch.stack(heat_map, 0).mean(0))

            if  len(merge_list) > 0:
               all_merges.append(merge_list)

        maps = torch.stack([torch.stack(x, 0) for x in all_merges], dim=0)
        maps = maps.sum(0).sum(2).sum(0)

        return HeatMap(tokenizer, prompt, maps)
    
class HeatMap:
    def __init__(self, tokenizer, prompt: str, heat_maps: torch.Tensor):
        self.prompt_analyzer = PromptAnalyzer(tokenizer, prompt)
        self.heat_maps = heat_maps
        self.prompt = prompt

    def compute_word_heat_map(self, word: str | list[str], mode: str = "mean",  word_idx: int = None) -> torch.Tensor:
        if isinstance(word, str):
            word = [word]
            
        merge_idxs = []
        for w in word:
            idxs, _ = self.prompt_analyzer.calc_word_indecies(w)
            merge_idxs.extend(idxs)
            
        # remove duplicated indexes
        merge_idxs = list(set(merge_idxs))        
            
        if len(merge_idxs) == 0:
            return None
        
        # print("self.heat_maps[merge_idxs].shape", self.heat_maps[merge_idxs].shape)
        
        if mode == "mean":
            return self.heat_maps[merge_idxs].mean(0)
        elif mode == "sum":
            return self.heat_maps[merge_idxs].sum(0)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
