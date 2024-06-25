import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutputWithPast


def prepare_attention_mask(seqlen, start_pos, device, type_as: torch.Tensor):
    mask = None
    if seqlen > 1:
        mask = torch.full(
            (1, 1, seqlen, seqlen), float("-inf"), device=device
        )
        mask = torch.triu(mask, diagonal=start_pos+ 1).type_as(type_as)
    return mask


def prepare_cache(blocks, seqlen: int) -> int:
    for block in blocks:
        start_pos = block.attn.start_pos
        will_cache_be_exceeded = start_pos + seqlen > block.attn.max_seq_len

        # Reset and avoid retaining state when processing context
        if seqlen > 1 and (will_cache_be_exceeded or start_pos > 0):
            block.attn.start_pos = block.attn.cache.roll_kv_n_steps(start_pos, n=start_pos)
        
        # Slowly roll out old tokens without performance hit if exceeded during decoding 
        elif seqlen == 1 and will_cache_be_exceeded:
            block.attn.start_pos = block.attn.cache.roll_kv_n_steps(start_pos, n=100)


def prepare_correct_devices(next_layer, hidden_states, mask):
    hidden_states = hidden_states.to(next_layer.device)
    if mask is not None:
        mask = mask.to(next_layer.device)
    return hidden_states, mask


def prepare_input_ids(input_ids: torch.Tensor, last_forward_num_tokens: int):
    # NOTE: from transformers 4.35.0, input_ids includes full context during decoding
    num_input_tokens = input_ids.shape[-1]
    num_new_tokens = num_input_tokens
    if num_input_tokens != 1:
        num_new_tokens = num_input_tokens - last_forward_num_tokens
        # after context is processed, slice to latest token
        if num_new_tokens == 1:
            input_ids = input_ids[:, -1:]
    return input_ids, last_forward_num_tokens + num_new_tokens


class LlamaLikeModel(nn.Module):
    """
    LlamaLikeModel is intended to be reused across models that have
    an architecture that closely resembles Llama, e.g. Mistral and Aquila.
    """
    def __init__(self, vocab_size, blocks, embedding, norm):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.blocks: List[LlamaLikeBlock] = nn.ModuleList(blocks)
        self.norm = norm
        self.last_forward_num_tokens = 0
    
    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, attn_bias=None, attention_mask=None, is_causal=None, *args, **kwargs):
        input_ids, self.last_forward_num_tokens = prepare_input_ids(
            input_ids,
            self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        prepare_cache(self.blocks, seqlen)

        h = self.embedding(input_ids)

        mask = prepare_attention_mask(
            seqlen=seqlen,
            start_pos=self.blocks[0].attn.start_pos,
            device=input_ids.device,
            type_as=h
        )

        for layer in self.blocks:
            h, mask = prepare_correct_devices(
                layer,
                h,
                mask,
            )
            h, _, past_key_value = layer(
                h,
                None,
                attention_mask=mask,
                is_causal=is_causal
            )
        h = self.norm(h)

        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=past_key_value, hidden_states=(), attentions=())
