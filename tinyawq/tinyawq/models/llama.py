from dataclasses import dataclass, field, fields
import gc
import json
import os
from typing import List, Optional, Tuple

from accelerate.big_modeling import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
)

from huggingface_hub import snapshot_download

import torch
import torch.nn as nn

from tqdm import tqdm

import transformers
from transformers import (
    AutoConfig,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as OldLlamaDecoderLayer,
    LlamaForCausalLM as OldLlamaForCausalLM,
)
from transformers.utils.hub import PushToHubMixin


from ..modules.fused.block import LlamaLikeBlock
from ..modules.fused.model import LlamaLikeModel
from ..modules.fused.norm import FasterTransformerRMSNorm
from ..modules.linear.quick import WQLinear_QUICK


def exclude_layers_to_not_quantize(linear_layers, modules_to_not_convert):
    if modules_to_not_convert is None:
        return linear_layers

    filtered_layers = {}
    for name, linear_layer in linear_layers.items():
        if not any(key in name for key in modules_to_not_convert):
            filtered_layers[name] = linear_layer
    return filtered_layers


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


@dataclass
class AwqConfig(PushToHubMixin):
    quant_method: str = field(default="awq")
    zero_point: bool = field(default=True)
    q_group_size: int = field(default=128)
    w_bit: int = field(default=4)
    version: str = field(default="GEMM")
    config_file_name = "quant_config.json"
    modules_to_not_convert: Optional[List] = None

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        commit_hash = kwargs.pop("_commit_hash", None)

        if os.path.isdir(save_dir):  # Local
            resolved_config_file = os.path.join(save_dir, cls.config_file_name)
        else: # Remote
            resolved_config_file = cached_file(
                save_dir,
                cls.config_file_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                use_auth_token=use_auth_token,
                revision=revision,
                local_files_only=local_files_only,
                subfolder=subfolder,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                _commit_hash=commit_hash,
            )
        
        if os.path.exists(resolved_config_file):
            with open(resolved_config_file, 'r', encoding="utf-8") as file:
                loaded_config = json.loads(file.read())
                quant_config = cls(**loaded_config)
        else:
            quant_config = cls()
        
        return quant_config


class BaseAWQForCausalLM(nn.Module):
    def __init__(
        self, model, model_type, is_quantized, config, quant_config, processor
    ):
        super().__init__()
        self.model: PreTrainedModel = model
        self.model_type: str = model_type
        self.is_quantized: bool = is_quantized
        self.search_result = None
        self.config: PretrainedConfig = config
        self.quant_config: AwqConfig = quant_config
        self.processor: CLIPImageProcessor = processor

    def generate(self, *args, **kwargs):
        with torch.inference_mode():
            return self.model.generate(*args, **kwargs)

    @classmethod
    def from_quantized(
        self,
        model_path,
        model_type,
        model_filename="",
        max_new_tokens=None,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        safetensors=True,
        is_quantized=True,
        fuse_layers=False,
        use_exllama=False,
        use_exllama_v2=False,
        version="GEMM",
        device_map="balanced",
        offload_folder=None,
        **config_kwargs,
    ):
        # [STEP 1-2] Load weights path and configs
        model_weights_path, config, quant_config = self._load_config(
            self,
            model_path,
            model_filename,
            safetensors,
            version,
            trust_remote_code,
            max_new_tokens=max_new_tokens,
            **config_kwargs,
        )

        target_cls_name = "AutoModelForCausalLM"
        target_cls = getattr(transformers, target_cls_name)

        # [STEP 3] Load model
        with init_empty_weights():
            model = target_cls.from_config(
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        
        # Prepare WQLinear layers, replace nn.Linear
        self._load_quantized_modules(
            self,
            model,
            quant_config,
            quant_config.version,
            use_exllama=use_exllama,
            use_exllama_v2=use_exllama_v2,
        )

        model.tie_weights()
        
        # if quant_config.version == "QUICK":
        #     for i in range(len(model.model.layers)): 
        #         del model.model.layers[i].self_attn.rope
        
        # loads the weights into modules and distributes
        # across available devices automatically
        load_checkpoint_and_dispatch(
            model,
            checkpoint=model_weights_path,
            device_map=device_map,
            no_split_module_classes=[self.layer_type],
            offload_folder=offload_folder,
            dtype=torch_dtype,
        )
        
        # Dispath to devices
        if fuse_layers:
            self.fuse_layers(model)
        
        if use_exllama:
            # creates q4 handle
            print('use_exllama')
            model = exllama_post_init(model)
        elif use_exllama_v2:
            # creates q4 handle and allocates scratch spaces wrt max_input_len and
            # max_batch_size, which are hardcoded for now but might be worth interfacing
            print('use_exllama_v2')
            model = exllamav2_post_init(
                model,
                max_input_len=max_new_tokens,
                max_batch_size=int(os.getenv("AWQ_BATCH_SIZE", 1))
            )

        return self(
            model,
            model_type,
            is_quantized=is_quantized,
            config=config,
            quant_config=quant_config,
            processor=None,
        )

    @staticmethod
    def fuse_layers(model):
        pass

    def _load_config(
        self,
        model_path,
        model_filename,
        safetensors=True,
        version="GEMM",
        trust_remote_code=True,
        max_new_tokens=4096,
        **config_kwargs,
    ):
        # [STEP 1] Download model if path is not a directory
        if not os.path.isdir(model_path):
            ignore_patterns = ["*msgpack*", "*h5*", "optimizer.pt"]
            if safetensors:
                ignore_patterns.extend(["*.pt*", "*.bin*", "consolidated*"])
            else:
                ignore_patterns.append("*.safetensors*")

            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)

        if model_filename != "":
            model_weights_path = model_path + f"/{model_filename}"
        else:
            model_weights_path = model_path

        # [STEP 2] Load config and set sequence length
        # TODO: Create BaseAWQConfig class
        quant_config = AwqConfig.from_pretrained(model_path)

        # Load model config and set max generation length
        if max_new_tokens is None and hasattr(self, "max_new_tokens_key"):
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_new_tokens = getattr(config, self.max_new_tokens_key, 2048)
            # To add the generate support for Multi-modal models as well
            if hasattr(config, "text_config"):
                config.text_config.max_new_tokens = getattr(
                    config, self.max_new_tokens_key, 2048
                )
        else:
            max_new_tokens = 2048 if max_new_tokens is None else max_new_tokens
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_new_tokens = max_new_tokens

        return model_weights_path, config, quant_config

    def _load_quantized_modules(
        self, model, quant_config, version, use_exllama, use_exllama_v2
    ):
        # Real quantization of weights
        assert quant_config.zero_point, "We only support zero_point quantization now."
        assert not (
            version == "GEMV" and (use_exllama or use_exllama_v2)
        ), "Exllama kernels only support GEMM version."
        print("Kernel Version: ", version)
        # Get blocks of model
        layers = self.get_model_layers(model)

        for i in tqdm(range(len(layers)), desc="Replacing layers..."):
            layer = layers[i]

            # Replace activation functions
            self._scale_activations(self, layer)
            
            # Get every linear layer in a block
            named_linears = get_named_linears(layer)

            # Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(
                named_linears, quant_config.modules_to_not_convert
            )
            gpu_A100 = False
            gpu_device = torch.cuda.get_device_name()
            if 'A100' in gpu_device and 'A1000' not in gpu_device:
                gpu_A100 = True
                
            # Replace nn.Linear with WQLinear
            for name, module in named_linears.items():
                if gpu_A100:
                    q_linear = WQLinear_QUICK.from_linear(
                        module, quant_config.w_bit, quant_config.q_group_size, True, k_split_1=16, k_split_2=16
                    )
                else:
                    q_linear = WQLinear_QUICK.from_linear(
                        module, quant_config.w_bit, quant_config.q_group_size, True
                    )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)

            torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def _scale_activations(self, layer):
        scale_dict = self.get_act_for_scaling(layer)

        if scale_dict["is_scalable"]:
            if not isinstance(scale_dict["scale_layer"], ScaledActivation):
                param = next(layer.parameters())

                # get activation scale
                scale_like = torch.ones(
                    scale_dict["scale_shape"], dtype=param.dtype, device=param.device
                )

                # scale activation
                scaled_act = ScaledActivation(scale_dict["scale_layer"], scale_like)
                set_op_by_name(layer, scale_dict["scale_name"], scaled_act)


class LlamaAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "LlamaDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: OldLlamaForCausalLM):
        fuser = LlamaFuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model: OldLlamaForCausalLM):
        return model.model.layers
    
    @staticmethod
    def get_act_for_scaling(module: OldLlamaDecoderLayer):
        return dict(is_scalable=False)


def fuse_qkv_quick(module, q_proj, k_proj, v_proj):
    qkv_layer = WQLinear_QUICK(
        q_proj.w_bit,
        q_proj.group_size,
        q_proj.in_features,
        q_proj.out_features + k_proj.out_features + v_proj.out_features,
        q_proj.bias is not None,
        next(iter(module.state_dict().values())).device
    )
    
    bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None

    qkv_layer.qweight = QUICK_cat(q_proj.qweight, k_proj.qweight, v_proj.qweight, options='qweight')
    qkv_layer.qzeros = QUICK_cat(q_proj.qzeros, k_proj.qzeros, v_proj.qzeros, options='qzeros')
    qkv_layer.scales = QUICK_cat(q_proj.scales, k_proj.scales, v_proj.scales, options='scales')
    
    qkv_layer.bias = bias

    return qkv_layer


def QUICK_cat(*input_layers: torch.Tensor, options: str, reshape_dims: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Concatenates multiple input layers after reshaping based on specified options for QUICK.

    Args:
        *input_layers: Variable number of tensor layers to concatenate.
        options: A string indicating how the layers should be reshaped ('qweight', 'qzeros', or 'scales').
        reshape_dims: Optional tuple indicating custom dimensions for reshaping. If None, default settings are used.

    Returns:
        torch.Tensor: The concatenated and reshaped layers.

    Raises:
        ValueError: If the options provided are invalid or if input layers have incompatible shapes.
    """
    # Check if there are at least two layers to concatenate
    if len(input_layers) < 2:
        raise ValueError("At least two input layers are required")

    # Check for shape compatibility
    H, W = input_layers[0].shape
    for layer in input_layers[1:]:
        if layer.shape != input_layers[0].shape:
            raise ValueError("All input layers must have the same shape")

    # Determine reshape dimensions based on options
    if not reshape_dims:
        reshape_dims = {
            'qweight': (H // 2, W * 2),
            'qzeros': (H * 4, W // 4),
            'scales': (H * 4, W // 4)
        }.get(options)

    if reshape_dims is None:
        raise ValueError("Unknown options provided or invalid reshape dimensions")

    # Reshape and concatenate the input layers
    layers_to_cat = [layer.reshape(*reshape_dims) for layer in input_layers]
    output_layer = torch.cat(layers_to_cat, dim=1).reshape(H, -1)

    return output_layer


class LlamaFuser:
    def __init__(self, model: OldLlamaForCausalLM):
        self.model = model

        self.llama_blocks: List[Tuple[str, OldLlamaDecoderLayer]] = [
            (name, module) for name, module in self.model.named_modules()
            if 'LlamaDecoderLayer'.lower() in module.__class__.__name__.lower()
        ]
    
    def fuse_transformer(self):
        blocks = []

        module: OldLlamaDecoderLayer
        for module in tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv_quick(
                module, module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj
            )
            norm_1 = FasterTransformerRMSNorm(
                module.input_layernorm.weight,
                module.input_layernorm.variance_epsilon
            )
            norm_2 = FasterTransformerRMSNorm(
                module.post_attention_layernorm.weight,
                module.post_attention_layernorm.variance_epsilon
            )
            blocks.append(LlamaLikeBlock(
                hidden_size=self.model.config.hidden_size,
                n_heads=self.model.config.num_attention_heads,
                n_kv_heads=self.model.config.num_key_value_heads,
                qkv_layer=qkv,
                o_proj=module.self_attn.o_proj,
                mlp=module.mlp,
                norm_1=norm_1,
                norm_2=norm_2,
                dev=device,
                max_seq_len=self.model.config.max_new_tokens,
                rope_theta=self.model.config.rope_theta
            ))
        
        self.model.model = LlamaLikeModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            self.model.model.norm,
        )
        setattr(self.model.model, "blocks", self.model.model.blocks)
