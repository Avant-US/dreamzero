from dataclasses import dataclass, field
import logging
import time
from typing import TypeAlias, cast
import os

from accelerate import load_checkpoint_and_dispatch

from einops import rearrange
from hydra.utils import instantiate
from peft import LoraConfig, get_peft_model
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from safetensors.torch import load_file
import json
from huggingface_hub import hf_hub_download

from groot.vla.model.dreamzero import perf_profile as _perf_profile


logger = logging.getLogger(__name__)

WAN_HF_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAN22_HF_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"


def hf_download(filename: str, repo_id: str = WAN_HF_REPO_ID) -> str:
    """Download a file from the specified HuggingFace repo to HF cache."""
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return path


def ensure_file(path: str | None, hf_filename: str, repo_id: str = WAN_HF_REPO_ID) -> str:
    """Return a valid local path: use `path` if it exists, otherwise download from HuggingFace."""
    if path is not None and os.path.exists(path):
        return path
    return hf_download(hf_filename, repo_id)

from torch.distributions import Beta
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torchvision.transforms import v2
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from groot.vla.model.n1_5.action_head.base_action_head import ActionHead
from groot.vla.model.dreamzero.modules.flow_match_scheduler import FlowMatchScheduler
from groot.vla.model.dreamzero.modules.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from groot.vla.model.dreamzero.modules.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler


KVCacheType: TypeAlias = torch.Tensor

@dataclass
class WANPolicyHeadConfig(PretrainedConfig):
    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )
    tiled: bool = field(default=True, metadata={"help": "Whether to use tiled input."})
    tile_size_height: int = field(default=34, metadata={"help": "Tile size height."})
    tile_size_width: int = field(default=34, metadata={"help": "Tile size width."})
    tile_stride_height: int = field(default=18, metadata={"help": "Tile stride height."})
    tile_stride_width: int = field(default=16, metadata={"help": "Tile stride width."})
    num_frame_per_block: int = field(default=1, metadata={"help": "Number of frames per block."})
    # Target video (H, W) for Wan22 resize. When set, videos are resized to this before VAE so latent
    # spatial size matches. Use height/width divisible by 32 for WanVideoVAE38 (16x) so latent H,W are even.
    target_video_height: int | None = field(default=None, metadata={"help": "Target video height for resize (e.g. 160 for even latent with VAE38)."})
    target_video_width: int | None = field(default=None, metadata={"help": "Target video width for resize (e.g. 320)."})

    lora_rank: int = field(default=4, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=4, metadata={"help": "LoRA alpha."})
    lora_target_modules: str = field(default="q,k,v,o,ffn.0,ffn.2")
    init_lora_weights: str = field(default="kaiming", metadata={"help": "LoRA initialization method."})
    train_architecture: str= field(default="lora", metadata={"help": "Train architecture."})
    skip_component_loading: bool = field(default=False, metadata={"help": "Skip loading individual component weights (used when loading from full pretrained model)."})

    use_gradient_checkpointing: bool = field(default=True, metadata={"help": "Whether to use gradient checkpointing."})
    qformer_cfg: dict = field(default=None, metadata={"help": "Qformer configuration."})
    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    # High noise emphasis for BASE (coupled) training - applies Beta distribution to BOTH video and action together
    use_high_noise_emphasis: bool = field(
        default=False, metadata={"help": "Use Beta distribution for noise sampling (biases BOTH video and action towards high noise levels together)."}
    )
    high_noise_beta_alpha: float = field(
        default=3.0, metadata={"help": "Beta alpha for high noise emphasis. Beta(3,1): mean=0.75, Beta(5,1): mean=0.83. Higher = more high noise bias."}
    )
    # Decoupled noise sampling config for training-inference alignment
    # When enabled: video uses Beta(alpha,beta) biased towards high noise, action uses independent uniform
    decouple_video_action_noise: bool = field(
        default=False, metadata={"help": "Decouple video/action noise: video uses Beta distribution (high noise bias), action uses independent uniform."}
    )
    video_noise_beta_alpha: float = field(
        default=3.0, metadata={"help": "Beta alpha for video noise. Beta(3,1): mean=0.75, Beta(5,1): mean=0.83. Higher alpha = more bias to high noise."}
    )
    video_noise_beta_beta: float = field(
        default=1.0, metadata={"help": "Beta beta for video noise. Keep at 1.0."}
    )
    # Decoupled inference config - allows video to stay noisy while action fully denoises
    decouple_inference_noise: bool = field(
        default=False, metadata={"help": "Use decoupled noise schedules during inference (video stays noisy, action fully denoises)."}
    )
    video_inference_final_noise: float = field(
        default=0.8, metadata={"help": "Final noise level for video during decoupled inference (0.0-1.0). E.g., 0.8 means video ends at 80% noise."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)
    defer_lora_injection: bool = field(default=False, metadata={"help": "Defer LoRA injection until after loading pretrained weights."})

    vl_self_attention_cfg: dict = field(default=None)
    text_encoder_cfg: dict = field(default=None)
    image_encoder_cfg: dict = field(default=None)
    vae_cfg: dict = field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class WANPolicyHead(ActionHead):
    config_class = WANPolicyHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: WANPolicyHeadConfig,
    ):
        super().__init__()
        self.tiled = config.tiled
        self.tile_size_height = config.tile_size_height
        self.tile_size_width = config.tile_size_width
        self.tile_stride_height = config.tile_stride_height
        self.tile_stride_width = config.tile_stride_width
        self.num_frame_per_block = config.num_frame_per_block
        self.hidden_size = config.hidden_size
        self.num_frames = config.num_frames
        self.text_encoder = instantiate(config.text_encoder_cfg)
        self.image_encoder = instantiate(config.image_encoder_cfg)
        self.vae = instantiate(config.vae_cfg)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.model_names = ['text_encoder']

        self.num_inference_steps = int(os.getenv("NUM_INFERENCE_STEPS", "16"))
        self.seed = 1140
        self.cfg_scale = float(os.environ.get("CFG_SCALE", "5.0"))
        self.denoising_strength = 1.0
        self.sigma_shift = 5.0
        self.kv_cache1: KVCacheType | None = None
        self.kv_cache_neg: KVCacheType | None = None
        self.crossattn_cache: KVCacheType | None = None
        self.crossattn_cache_neg: KVCacheType | None = None

        self.global_step = 0
        self.max_steps = 0
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_target_modules = config.lora_target_modules
        self.init_lora_weights = config.init_lora_weights
        self.train_architecture = config.train_architecture
        self.clip_feas = None
        self.ys = None
        self.current_start_frame = 0
        self._kv_cache_warm = False  # reset warm cache on new session
        self.language = None

        self.ip_rank = 0
        self.ip_size = 1
        self.ip_group = None
        self.sp_ctx = None
        
        self._device = "cuda"
        self.dynamic_cache_schedule = os.getenv("DYNAMIC_CACHE_SCHEDULE", "False").lower() == "true"


        num_dit_steps = 8
        if os.getenv("NUM_DIT_STEPS") is not None:
            num_dit_steps = int(os.getenv("NUM_DIT_STEPS"))
        if num_dit_steps == 4:
            # 4 steps: matches typical TeaCache adaptive behavior.
            # Steps 0, 5, 10, 15 (evenly spaced across the schedule).
            self.dit_step_mask = [True, False, False, False, False, True, False, False, False, False, True, False, False, False, False, True]
        elif num_dit_steps == 5:
            self.dit_step_mask = [True, True, True, False, False, False, False, True, False, False, False, False, True, False, False, False]
        elif num_dit_steps == 6:
            self.dit_step_mask = [True, True, False, False, False, True, False, False, False, False, True, False, False, False, True, True]
        elif num_dit_steps == 7:
            self.dit_step_mask = [True, True, True, False, False, False, True, False, False, False, True, False, False, False, True, True]
        elif num_dit_steps == 8:
            self.dit_step_mask = [True, True, True, False, False, False, True, False, False, False, True, False, False, True, True, True]
        else:
            self.dit_step_mask = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
        assert self.dit_step_mask[0] == True, "first step must be True"

        self.normalize_video = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        if self.training:
            self.scheduler.set_timesteps(1000, training=True)
        
        
        self.input_embedding_dim = config.input_embedding_dim

        self.cpu_offload = False

        self.model = instantiate(config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        
        text_enc_path = ensure_file(
            self.text_encoder.text_encoder_pretrained_path,
            "models_t5_umt5-xxl-enc-bf16.pth",
        )
        self.text_encoder.load_state_dict(torch.load(text_enc_path, map_location='cpu'))

        img_enc_path = ensure_file(
            self.image_encoder.image_encoder_pretrained_path,
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        )
        self.image_encoder.model.load_state_dict(torch.load(img_enc_path, map_location='cpu'), strict=False)

        # Wan2.2 (WanVideoVAE38, z_dim=48) uses Wan2.2_VAE.pth; Wan2.1 uses Wan2.1_VAE.pth
        vae_hf_filename = "Wan2.2_VAE.pth" if getattr(self.vae, "z_dim", 16) == 48 else "Wan2.1_VAE.pth"
        vae_repo_id = WAN22_HF_REPO_ID if getattr(self.vae, "z_dim", 16) == 48 else WAN_HF_REPO_ID
        vae_path = ensure_file(
            self.vae.vae_pretrained_path,
            vae_hf_filename,
            repo_id=vae_repo_id,
        )
        self.vae.model.load_state_dict(torch.load(vae_path, map_location='cpu'))

        if not config.skip_component_loading:
            dit_dir = self.model.diffusion_model_pretrained_path
            # Wan2.2 (in_dim=48) uses Wan2.2-TI2V-5B repo; Wan2.1 uses Wan2.1-I2V-14B-480P
            dit_repo_id = WAN22_HF_REPO_ID if getattr(self.model, "in_dim", 16) == 48 else WAN_HF_REPO_ID
            if dit_dir is None or not os.path.isdir(dit_dir):
                index_path = hf_hub_download(repo_id=dit_repo_id, filename="diffusion_pytorch_model.safetensors.index.json")
                dit_dir = os.path.dirname(index_path)
                with open(index_path, 'r') as f:
                    index = json.load(f)
                for shard_file in set(index["weight_map"].values()):
                    hf_hub_download(repo_id=dit_repo_id, filename=shard_file)

            if dit_dir is not None:
                safetensors_path = os.path.join(dit_dir, "diffusion_pytorch_model.safetensors")
                safetensors_index_path = os.path.join(dit_dir, "diffusion_pytorch_model.safetensors.index.json")
                state_dict = {}

                if os.path.exists(safetensors_index_path):
                    # Handle sharded safetensors
                    print(f"Loading sharded safetensors using index: {safetensors_index_path}")

                    with open(safetensors_index_path, 'r') as f:
                        index = json.load(f)

                    # Load each shard
                    for shard_file in set(index["weight_map"].values()):
                        shard_path = os.path.join(dit_dir, shard_file)
                        print(f"Loading shard: {shard_path}")
                        shard_state_dict = load_file(shard_path)
                        state_dict.update(shard_state_dict)

                elif os.path.exists(safetensors_path):
                    # Handle single safetensors file
                    print(f"Loading weights from safetensors: {safetensors_path}")
                    state_dict = load_file(safetensors_path)

                else:
                    raise ValueError(f"No safetensors file found at {safetensors_path} or {safetensors_index_path}")

                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

                if missing_keys:
                    print(f"Missing keys when loading pretrained weights: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")

                print("Successfully loaded pretrained weights")
        else:
            print("Skipping individual component loading (loading from full pretrained model)")
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        # Video noise Beta distribution (biased towards high noise levels when enabled)
        self.video_beta_dist = Beta(config.video_noise_beta_alpha, config.video_noise_beta_beta)
        # High noise emphasis Beta distribution for coupled training (applies to both video and action)
        self.high_noise_beta_dist = Beta(config.high_noise_beta_alpha, 1.0)
        # self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self._noise_logged = False
        self.defer_lora_injection = config.defer_lora_injection
        print("defer_lora_injection@@", self.defer_lora_injection)
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

        if self.train_architecture == "lora" and not self.defer_lora_injection:
            print("Adding LoRA to model")
            for p in self.parameters():
                p.requires_grad = False
            self.model = self.add_lora_to_model(
                self.model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                init_lora_weights=self.init_lora_weights,
            )
            self.model.state_encoder.requires_grad_(True)
            self.model.action_encoder.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)
        elif self.train_architecture == "lora" and self.defer_lora_injection:
            print("Deferring LoRA injection until after pretrained weights are loaded")
        else:
            self.print_trainable_params()

        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        if not self.defer_lora_injection:
            self.print_trainable_params()


    def print_trainable_params(self):
        """Print trainable parameters of the diffusion model."""
        trainable_params = []
        total_params = 0
        trainable_total = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params.append(name)
                trainable_total += param.numel()
                
        print(f"Total parameters in diffusion model: {total_params:,}")
        print(f"Trainable parameters in diffusion model: {trainable_total:,}")
        # print(trainable_params)


    def inject_lora_after_loading(self):
        """
        Inject LoRA adapters after pretrained weights have been loaded.
        This should be called when defer_lora_injection=True.
        """
        if self.train_architecture == "lora":
            print("Injecting LoRA after loading pretrained weights")
            for p in self.parameters():
                p.requires_grad = False
            self.model = self.add_lora_to_model(
                self.model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                init_lora_weights=self.init_lora_weights,
            )
            self.model.state_encoder.requires_grad_(True)
            self.model.action_encoder.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)
            # self.model.registers.requires_grad_(True)
            # self.model.time_modality_projection.requires_grad_(True)
            
            self.text_encoder.requires_grad_(False)
            self.image_encoder.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.print_trainable_params()
        else:
            print("LoRA injection not needed (train_architecture != 'lora')")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_diffusion_model:
                self.model.eval()
            self.text_encoder.eval()
            self.image_encoder.eval()
            self.vae.eval()
    
    
    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.dtype,
                computation_device='cuda',
            ),
        )

        self.cpu_offload = True

    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        print("offloadd")
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                # print("offload", module)
                                module.offload()
                    else:
                        print("tocpu")
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                    print("onload")
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            # print("onload", module)
                            module.onload()
                else:
                    print("togpu")
                    model.to(self._device)
        # fresh the cuda cache
        torch.cuda.empty_cache()

    def _create_kv_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        frame_seqlen: int,
    ) -> tuple[KVCacheType, KVCacheType]:
        """
        Initialize a Per-GPU KV cache for the Wan model.

        Default: allocate with seq_len=0 (grows on demand via torch.cat).

        Static mode (STATIC_KV_CACHE=true, vLLM/TRT-LLM style):
        allocate at max size up-front. Each self-attn call writes new tokens in
        place into a fixed-size buffer. Attention uses a key-side mask to ignore
        unused slots. This gives torch.compile a constant input-shape signature
        which is a prerequisite for CUDA graph capture.
        """
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads
        sp_size = self.sp_ctx.sp_size if self.sp_ctx is not None else 1
        effective_num_heads = num_heads // sp_size

        static_kv = os.environ.get("STATIC_KV_CACHE", "false").lower() == "true"
        if static_kv:
            # Use the same cap the self-attn uses for its rolling truncation.
            # `max_attention_size` lives on each CausalWanSelfAttention inside a block.
            max_attn = int(
                getattr(self.model.blocks[0].self_attn, "max_attention_size", 21 * frame_seqlen)
            )
            seq_cap = max_attn
            if self.ip_rank == 0:
                mb_per_cache = (
                    2 * batch_size * seq_cap * effective_num_heads * head_dim
                    * torch.tensor([], dtype=dtype).element_size()
                ) / (1024 ** 2)
                print(
                    f"[static_kv] Pre-allocating KV cache: max_seq={seq_cap}, "
                    f"layers={self.model.num_layers}, per-layer {mb_per_cache:.1f} MB "
                    f"(×2 groups = {2*self.model.num_layers*mb_per_cache:.1f} MB total per rank)"
                )
        else:
            seq_cap = 0

        kv_cache1: KVCacheType = []
        kv_cache_neg: KVCacheType = []
        for _ in range(self.model.num_layers):
            t1 = torch.zeros([2, batch_size, seq_cap, effective_num_heads, head_dim], dtype=dtype, device=device)
            t2 = torch.zeros([2, batch_size, seq_cap, effective_num_heads, head_dim], dtype=dtype, device=device)
            if static_kv:
                torch._dynamo.mark_static_address(t1)
                torch._dynamo.mark_static_address(t2)
            kv_cache1.append(t1)
            kv_cache_neg.append(t2)

        if static_kv:
            self._kv_fill_levels_pos = [0] * self.model.num_layers
            self._kv_fill_levels_neg = [0] * self.model.num_layers
            # Set each block's self_attn to static mode and attach persistent
            # GPU tensors for fill_level and index base. These are updated via
            # .fill_() OUTSIDE the compiled fn so dynamo treats them as data.
            for blk in self.model.blocks:
                blk.self_attn._use_static_kv = True
                fl_t = torch.zeros((), dtype=torch.int64, device=device)
                torch._dynamo.mark_static_address(fl_t)
                blk.self_attn._fill_level_t = fl_t
                idx_base = torch.arange(seq_cap, dtype=torch.int64, device=device)
                torch._dynamo.mark_static_address(idx_base)
                blk.self_attn._idx_base = idx_base

        return kv_cache1, kv_cache_neg

    def _create_crossattn_caches(
        self, batch_size: int, dtype: torch.dtype, device: torch.device,
    ) -> tuple[KVCacheType, KVCacheType]:
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        Use the model's num_heads and head_dim (5B has 24 heads, 14B has 40).
        """
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads
        crossattn_cache: KVCacheType = []
        crossattn_cache_neg: KVCacheType = []

        for _ in range(self.model.num_layers):
            crossattn_cache.append(
                torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            )
            crossattn_cache_neg.append(
                torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            )

        return crossattn_cache, crossattn_cache_neg
        
    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def preprocess_image(self, image):
        image = (image * (2 / 255) - 1).permute(0, 1, 4, 2, 3)
        return image

    def encode_prompt(self, input_ids, attention_mask):
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(input_ids, attention_mask)
        prompt_emb = prompt_emb.clone().to(dtype=torch.bfloat16)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def _ensure_vae_on_device(self, ref_tensor):
        """Lazily move the VAE to the correct device/dtype on first use."""
        if not getattr(self, '_vae_device_ready', False):
            self.vae.to(device=ref_tensor.device, dtype=torch.bfloat16)
            self.vae.eval()
            self._vae_device_ready = True

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        self._ensure_vae_on_device(input_video)
        with torch.no_grad():
            latents = self.vae.encode(input_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def encode_image(self, image, num_frames, height, width):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            batch_size = image.shape[0]
            clip_context = self.image_encoder.encode_image(image)
            image_input = image.transpose(1, 2)
            image_zeros = torch.zeros(batch_size, 3, num_frames-1, height, width, dtype=torch.bfloat16, device=self._device)
            self._ensure_vae_on_device(image_input)
            with torch.no_grad():
                y = self.vae.encode(torch.concat([image_input, image_zeros], dim=2))
            # Build mask to match VAE output shape (VAE may use different spatial downsampling, e.g. WanVideoVAE38 uses patch_size=2 -> height/16)
            # y shape is B * 16 * (1+(T-1)/4) * H_latent * W_latent
            num_t = y.shape[2]
            h_latent, w_latent = y.shape[3], y.shape[4]
            msk = torch.zeros(batch_size, 4, num_t, h_latent, w_latent, dtype=y.dtype, device=self._device)
            msk[:, :, 0:1, :, :] = 1
            new_image = y[:, :, 0:1]
            # concat: B * (4+16) * (1+(T-1)/4) * H_latent * W_latent
            y = torch.concat([msk, y], dim=1)
        return clip_context, y, new_image
    
    def prepare_extra_input(self, latents=None):
        return {}

    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming") -> nn.Module:
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = get_peft_model(model, lora_config)
        for param in model.parameters():
            param.data = param.to(torch.float32)
        return model

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        data = action_input 
        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id
        # print("embodiment_id", embodiment_id)
        has_real_action = action_input.has_real_action
        action_mask = action_input.action_mask

        state_features = action_input.state

        actions = action_input.action
        # assert the values of action is in between -1 and 1
        if actions.numel() > 0:
            assert actions.min() >= -1.0 and actions.max() <= 1.0, "actions must be in [-1,1] range"
        videos = data["images"]

        videos = rearrange(videos, "b t h w c -> b c t h w")
        print("videos", videos.shape)
        

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # back to [b, c, t, h, w]
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=self.dtype)
        
        # shape of B * max_length * dim
        prompt_embs = self.encode_prompt(data["text"], data["text_attention_mask"])

        # Wan 5B: resize to target resolution so latent tokens/frame matches DiT. Use config target when set
        # (e.g. 160x320 so latent is 10x20 with VAE38 16x → even H,W, no crop in dynamics loss); else 176x320.
        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)

        latents = self.encode_video(videos, self.tiled, (self.tile_size_height, self.tile_size_width), (self.tile_stride_height, self.tile_stride_width))

        # print("latents shape", latents.shape, self.dtype)
        _, _, num_frames, height, width = videos.shape
        image = videos[:, :, :1].transpose(1, 2)

        clip_feas, ys, _ = self.encode_image(image, num_frames, height, width)

        latents = latents.to(self._device)
        clip_feas = clip_feas.to(self._device)
        ys = ys.to(self._device)
        prompt_embs = prompt_embs.to(self._device)
       
        # Loss
        noise = torch.randn_like(latents)

        # specific to autoregressive 
        noise = noise.transpose(1, 2)
        latents = latents.transpose(1, 2)
        
        # ============ VIDEO TIMESTEP SAMPLING ============
        if self.config.decouple_video_action_noise:
            # Decoupled mode: sample video from Beta distribution biased towards HIGH noise
            video_noise_ratio = self.video_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - video_noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "DECOUPLED"
        elif self.config.use_high_noise_emphasis:
            # High noise emphasis mode (coupled): BOTH video and action use Beta distribution
            noise_ratio = self.high_noise_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "HIGH_NOISE_EMPHASIS"
        else:
            # Original: uniform sampling over full range
            timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (noise.shape[0], noise.shape[1]))
            noise_mode = "STANDARD"
        
        timestep_id_block = timestep_id[:, 1:].reshape(
                    timestep_id.shape[0], -1, self.num_frame_per_block)
        timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]
        
        if actions.numel() > 0:
            noise_action = torch.randn_like(actions)
            assert actions.shape[1] / (noise.shape[1]-1) == (self.model.num_action_per_block // self.num_frame_per_block), f"actions.shape, {actions.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"
            assert (noise.shape[1]-1) / state_features.shape[1] == (self.num_frame_per_block // self.model.num_state_per_block), f"state_features.shape, {state_features.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"
            
            # ============ ACTION TIMESTEP SAMPLING ============
            if self.config.decouple_video_action_noise:
                # Decoupled: sample action timestep independently with full range
                timestep_action_id = torch.randint(
                    0, 
                    self.scheduler.num_train_timesteps, 
                    (actions.shape[0], actions.shape[1])
                )
                action_mode = "INDEPENDENT"
            else:
                # Original coupled: action timestep derived from video timestep
                timestep_action_id = timestep_id_block.repeat(1, 1, actions.shape[1]//(noise.shape[1]-1))
                timestep_action_id = timestep_action_id.reshape(timestep_action_id.shape[0], -1)
                action_mode = "COUPLED"
            
            # Log noise mode once
            if not self._noise_logged:
                video_mean = timestep_id.float().mean().item()
                action_mean = timestep_action_id.float().mean().item()
                if noise_mode == "DECOUPLED":
                    print(f"[NOISE] Mode={noise_mode} | Video: Beta({self.config.video_noise_beta_alpha},1) mean_t={video_mean:.0f} | Action: {action_mode} Uniform mean_t={action_mean:.0f}")
                elif noise_mode == "HIGH_NOISE_EMPHASIS":
                    print(f"[NOISE] Mode={noise_mode} | Video+Action: Beta({self.config.high_noise_beta_alpha},1) mean_t={video_mean:.0f} | Action: {action_mode}")
                else:
                    print(f"[NOISE] Mode={noise_mode} | Video+Action: Uniform mean_t={video_mean:.0f} | Action: {action_mode}")
                self._noise_logged = True
        else:
            noise_action = None
            timestep_action_id = None
            
        timestep_id_block = timestep_id_block.reshape(timestep_id_block.shape[0], -1)
        timestep_id = torch.concat([timestep_id[:, :1], timestep_id_block], dim=1)
        _, num_frames, num_channels, height, width = noise.shape
        # DiT patch_embedding uses stride (1,2,2), so sequence length is num_frames * (H//2) * (W//2)
        tokens_per_frame = (height // 2) * (width // 2)
        seq_len = num_frames * tokens_per_frame

        timestep = self.scheduler.timesteps[timestep_id].to(self._device)
        noisy_latents = self.scheduler.add_noise(latents.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1]))
        training_target = self.scheduler.training_target(latents, noise, timestep).transpose(1, 2)
        
        if actions.numel() > 0:
            timestep_action = self.scheduler.timesteps[timestep_action_id].to(self._device)
            noisy_actions = self.scheduler.add_noise(
                actions.flatten(0, 1),
                noise_action.flatten(0, 1),
                timestep_action.flatten(0, 1),
            ).unflatten(0, (noise_action.shape[0], noise_action.shape[1]))
            training_target_action = self.scheduler.training_target(actions, noise_action, timestep_action)
        else:
            timestep_action = None
            noisy_actions = None
            training_target_action = None

        # Compute loss
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            if actions.numel() > 0:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2), timestep=timestep, clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                    state=state_features, embodiment_id=embodiment_id,
                    action=noisy_actions, timestep_action=timestep_action, 
                    clean_x=latents.transpose(1, 2),
                )
            else:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2), timestep=timestep, timestep_action=timestep_action, 
                    clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                    state=state_features, embodiment_id=embodiment_id,
                    clean_x=latents.transpose(1, 2),
                )

            # Per-sample dynamics loss
            # DiT patch_embedding uses stride (1,2,2), so output spatial size can be smaller than
            # latent when H or W is odd (e.g. latent 11x20 -> model output 10x20). Crop target to match.
            if training_target.shape != video_noise_pred.shape:
                training_target = training_target[
                    ..., : video_noise_pred.shape[3], : video_noise_pred.shape[4]
                ]
            dynamics_loss_per_sample = torch.nn.functional.mse_loss(
                video_noise_pred.float(), training_target.float(), reduction='none'
            ).mean(dim=(1,3,4))  # shape: [B, ...]

            weight_dynamics = dynamics_loss_per_sample * self.scheduler.training_weight(timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1])).to(self._device)
            weighted_dynamics_loss = weight_dynamics.mean()
            
            if actions.numel() > 0:
                action_loss_per_sample = torch.nn.functional.mse_loss(
                    action_noise_pred.float(), training_target_action.float(), reduction='none'
                ) * action_mask  # shape: [B, ...]
                action_loss_per_sample = has_real_action[:, None].float() * action_loss_per_sample  # apply has_real_action
                weight_action = action_loss_per_sample.mean(dim=2) * self.scheduler.training_weight(
                    timestep_action.flatten(0, 1),
                ).unflatten(0, (noise_action.shape[0], noise_action.shape[1])).to(self._device)
                weighted_action_loss = weight_action.mean()
                loss = weighted_dynamics_loss + weighted_action_loss
            else:
                weighted_action_loss = torch.tensor(0.0, device=self._device)
                loss = weighted_dynamics_loss
            # loss = dynamics_loss_per_sample.mean()

        # Record log
        output_dict = {
            "loss": loss,
            "dynamics_loss": weighted_dynamics_loss,
            "action_loss": weighted_action_loss,
        }

        return BatchFeature(data=output_dict)

    def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise
    
    def _get_caches(
        self, kv_caches_input: list[KVCacheType],
    ) -> list[KVCacheType]:
        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            assert len(kv_caches_input) == 2
            if self.ip_rank == 0:
                kv_caches = [kv_caches_input[0]]
            else:
                kv_caches = [kv_caches_input[1]]
        else:
            assert len(kv_caches_input) <= 2
            kv_caches = [kv_caches_input[0]]
            if self.cfg_scale != 1.0:
                kv_caches.append(kv_caches_input[1])
        return kv_caches

    def _prepare_text_inputs(self, data: BatchFeature) -> list[tuple[torch.Tensor, torch.Tensor]]:

        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            if self.ip_rank == 0:
                text_inputs = [(data["text"], data["text_attention_mask"])]
            else:
                text_inputs = [(data["text_negative"], data["text_attention_mask_negative"])]
        else:
            text_inputs = [(data["text"], data["text_attention_mask"])]
            if self.cfg_scale != 1.0:
                text_inputs.append((data["text_negative"], data["text_attention_mask_negative"]))
        return text_inputs


    def _run_diffusion_steps(
        self,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        y: torch.Tensor,
        clip_feature: torch.Tensor,
        kv_caches: list[KVCacheType],
        crossattn_caches: list[KVCacheType],
        kv_cache_metadata: dict[str, bool | int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        predictions = []
        for index, prompt_emb in enumerate(context):
            kv_cache = kv_caches[index]
            crossattn_cache = crossattn_caches[index]
            if not kv_cache_metadata["update_kv_cache"] and self.trt_engine is not None:
                obs_noise_pred, action_noise_pred = self.trt_engine(
                    noisy_input,
                    timestep,
                    action=action,
                    timestep_action=timestep_action,
                    state=state,
                    context=prompt_emb,
                    y=y,
                    clip_feature=clip_feature,
                    kv_cache=kv_cache,
                )
            else:
                # If TRT is loaded, PyTorch DiT lives on CPU — move to GPU for KV cache updates
                _model_offloaded = self.trt_engine is not None and not next(self.model.parameters()).is_cuda
                if _model_offloaded:
                    self.model.to(device=noisy_input.device, dtype=torch.bfloat16)

                # STATIC_KV_CACHE: tell the model which group's fill-levels to use
                # (pos=conditional, neg=unconditional) and whether this call
                # should advance the persistent fill_level (pre-pass) or just
                # use a scratch slot (main diffusion step).
                _static_kv = os.environ.get("STATIC_KV_CACHE", "false").lower() == "true"
                if _static_kv:
                    group = "pos" if index == 0 else "neg"
                    if not hasattr(self.model, f"_kv_fill_{group}"):
                        setattr(self.model, f"_kv_fill_{group}", [0] * self.model.num_layers)
                    self.model._current_static_kv_group = group
                    self.model._current_update_kv_cache = bool(kv_cache_metadata.get("update_kv_cache", True))

                # Save _fill_level_t before non-update calls so the CUDA graph's
                # in-place writes don't permanently advance the buffer position.
                _restore_fill = (
                    _static_kv
                    and not kv_cache_metadata.get("update_kv_cache", True)
                )
                if _restore_fill:
                    _saved_fills = [
                        blk.self_attn._fill_level_t.clone()
                        for blk in self.model.blocks
                    ]

                print(f'[DBG r={self.ip_rank}] model.forward START (group={getattr(self.model, "_current_static_kv_group", "?")}, update_kv={kv_cache_metadata.get("update_kv_cache", "?")})', flush=True)
                _fp8_ctx = self._get_fp8_context()
                with _fp8_ctx:
                    obs_noise_pred, action_noise_pred, updated_kv_caches = self.model(
                        noisy_input,
                        timestep,
                        action=action,
                        timestep_action=timestep_action,
                        state=state,
                        embodiment_id=embodiment_id,
                        context=prompt_emb,
                        seq_len=seq_len,
                        y=y,
                        clip_feature=clip_feature,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start_frame=kv_cache_metadata["start_frame"],
                    )

                print(f'[DBG r={self.ip_rank}] model.forward DONE', flush=True)
                if _static_kv:
                    self.model._current_static_kv_group = None

                # Restore fill levels after denoising calls so the circular
                # buffer write position stays correct for the next chunk.
                if _restore_fill:
                    for blk, sf in zip(self.model.blocks, _saved_fills):
                        blk.self_attn._fill_level_t.copy_(sf)

                if kv_cache_metadata["update_kv_cache"]:
                    if _static_kv:
                        # In static mode, the model already wrote into the
                        # preallocated buffer in-place. Nothing to copy.
                        # k_lens (fill_level) passed to FA kernel handles masking.
                        pass
                    else:
                        for block_index, updated_kv_cache in enumerate(updated_kv_caches):
                            kv_cache[block_index] = updated_kv_cache.clone()
                if _model_offloaded:
                    self.model.cpu()
                    torch.cuda.empty_cache()
            obs_noise_pred = obs_noise_pred.clone()
            if action_noise_pred is not None:
                action_noise_pred = action_noise_pred.clone()
            else:
                action_noise_pred = torch.tensor(0.0, device=obs_noise_pred.device) # dummy action noise prediction
            predictions.append((obs_noise_pred, action_noise_pred))
        return self._exchange_predictions(predictions)

    def _exchange_predictions(
        self,
        predictions: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.ip_size == 1:
            return predictions

        assert len(predictions) == 1
        my_predictions = list(predictions[0])

        other_predictions = [torch.empty_like(pred) for pred in my_predictions]

        send_ops = [
            dist.P2POp(op=dist.isend, tensor=pred, group_peer=(self.ip_rank + 1) % self.ip_size, group=self.ip_group)
            for pred in my_predictions
        ]
        recv_ops = [
            dist.P2POp(op=dist.irecv, tensor=other_pred, group_peer=(self.ip_rank + 1) % self.ip_size, group=self.ip_group)
            for other_pred in other_predictions
        ]
        ops = send_ops + recv_ops

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        output_predictions: list[tuple[torch.Tensor, torch.Tensor] | None] = [None for _ in range(self.ip_size)]
        output_predictions[self.ip_rank] = tuple(my_predictions)
        output_predictions[(self.ip_rank + 1) % self.ip_size] = tuple(other_predictions)
        assert all(isinstance(pred, tuple) for pred in output_predictions)
        return cast(list[tuple[torch.Tensor, torch.Tensor]], output_predictions)
    
    def should_run_model(self, index, current_timestep, prev_predictions):

        if not self.dynamic_cache_schedule:
            return self.dit_step_mask[index]

        # Always run first 2 steps to establish history
        if len(prev_predictions) < 2:
            return True

        if self.skip_countdown > 1:
            self.skip_countdown -= 1
            return False
        elif self.skip_countdown == 1:
            self.skip_countdown = 0 
            return True

        v_last = prev_predictions[-1][1].flatten(1).float()
        v_prev = prev_predictions[-2][1].flatten(1).float()
        sim = torch.nn.functional.cosine_similarity(v_last, v_prev, dim=1).mean()

        thresholds = [0.95, 0.93]
        countdowns = [4, 2]

        for threshold, countdown in zip(thresholds, countdowns):
            if sim > threshold:
                self.skip_countdown = countdown
                return False

        return True

    def lazy_joint_video_action(self, backbone_output: BatchFeature, action_input: BatchFeature, latent_video: torch.Tensor | None = None) -> BatchFeature:
        start_time = time.perf_counter()

        # Reset per-call profiler state (comm event accumulator). Cheap when
        # profiling is disabled.
        _perf_profile.reset_call()
        _perf_profile.maybe_start_trace(
            dist.get_rank() if dist.is_initialized() else 0
        )

        # Tracking time taken on GPU for various operations.
        start_text_encoder_event = torch.cuda.Event(enable_timing=True)
        end_text_encoder_event = torch.cuda.Event(enable_timing=True)
        start_image_encoder_event = torch.cuda.Event(enable_timing=True)
        end_image_encoder_event = torch.cuda.Event(enable_timing=True)
        start_vae_event = torch.cuda.Event(enable_timing=True)
        end_vae_event = torch.cuda.Event(enable_timing=True)
        start_kv_event = torch.cuda.Event(enable_timing=True)
        end_kv_event = torch.cuda.Event(enable_timing=True)
        start_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_inference_steps)]
        end_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_inference_steps)]

        self.set_frozen_modules_to_eval_mode()

        data = action_input

        videos = data["images"]

        embodiment_id = action_input.embodiment_id
        state_features = action_input.state

        videos = rearrange(videos, "b t h w c -> b c t h w")

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            videos = videos.to(dtype=self.dtype)
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # back to [b, c, t, h, w]
            # Removed: videos.min()/max() cause CPU-GPU sync (~0.1s penalty)
            videos = videos.to(dtype=self.dtype)

        state_features = state_features.to(dtype=torch.bfloat16)
        videos = videos.to(dtype=torch.bfloat16)

        # Wan 5B: same as training — resize to target resolution so latent matches DiT
        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)

        if self.language is None:
            print("language is None, reset current_start_frame to 0")
            self.language = data["text"]
            self.current_start_frame = 0
        elif not torch.equal(self.language, data["text"]):
            print("language changed, reset current_start_frame to 0")
            self.current_start_frame = 0
            self.language = data["text"]
        elif videos.shape[2] == 1:
            print("videos.shape[2] == 1, reset current_start_frame to 0")
            self.current_start_frame = 0
        elif self.current_start_frame >= self.model.local_attn_size:
            print("current_start_frame >= local_attn_size, reset current_start_frame to 0")
            self.current_start_frame = 0

        if self.ip_rank == 0:
            print("videos shape", videos.shape, self.num_frames, flush=True)

        import sys as _s
        print(f'[DBG r={self.ip_rank}] text_enc start', flush=True)
        _s.stdout.flush()
        start_text_encoder_event.record()

        text_inputs = self._prepare_text_inputs(data)
        # Cache text embeddings — prompt doesn't change between chunks
        _txt_cache = getattr(self, "_prompt_emb_cache", None)
        if _txt_cache is not None and _txt_cache.get("input_ids") is not None:
            # Check if input changed
            _same = all(torch.equal(ti[0], ci) for ti, ci in zip(text_inputs, _txt_cache["input_ids"]))
            if _same:
                prompt_embs = _txt_cache["embs"]
            else:
                prompt_embs = [self.encode_prompt(text, attn) for text, attn in text_inputs]
                self._prompt_emb_cache = {"input_ids": [ti[0].clone() for ti in text_inputs], "embs": prompt_embs}
        else:
            prompt_embs = [self.encode_prompt(text, attn) for text, attn in text_inputs]
            self._prompt_emb_cache = {"input_ids": [ti[0].clone() for ti in text_inputs], "embs": prompt_embs}

        end_text_encoder_event.record()
        print(f'[DBG r={self.ip_rank}] text_enc done', flush=True)

        start_image_encoder_event.record()
        print(f'[DBG r={self.ip_rank}] img_enc start', flush=True)

        _, _, num_frames, height, width = videos.shape
        if videos.shape[2] == 4 or videos.shape[2] == 9:
            # special case for real-world eval where language is updated
            image = videos[:, :, -1:].transpose(1, 2)
        else:
            image = videos[:, :, :1].transpose(1, 2)

        if self.current_start_frame == 0:
            clip_feas, ys, image = self.encode_image(image, self.num_frames, height, width)
            self.clip_feas = clip_feas.to(dtype=image.dtype)
            self.ys = ys.to(dtype=image.dtype)
        
        assert self.clip_feas is not None and self.ys is not None, "clip_feas and ys must be set"

        end_image_encoder_event.record()
        print(f'[DBG r={self.ip_rank}] img_enc done', flush=True)

        start_vae_event.record()
        print(f'[DBG r={self.ip_rank}] vae start', flush=True)

        # --- Pipeline optimization: overlap VAE+KV init with DiT ---
        # When OVERLAP_VAE_DIT=true on steady-state chunks:
        #   1. Run DiT FIRST using previous chunk's KV cache (stale by 1 chunk)
        #   2. Run VAE + KV init AFTER DiT to prepare fresh cache for next chunk
        # This overlaps VAE with the idle time between chunks.
        _kv_cache_thresh = float(os.environ.get("KV_INIT_CACHE_THRESH", "0"))
        _can_overlap = (_kv_cache_thresh > 0 and getattr(self, "_kv_cache_warm", False)
                         and self.current_start_frame > 1
                         and os.environ.get("OVERLAP_VAE_DIT", "false").lower() == "true")
        _vae_future = None

        if _can_overlap:
            # Defer VAE — create dummy image for now, run real VAE after DiT
            _lat_c = getattr(self.vae, 'z_dim', 16)
            _lat_h = videos.shape[3] // 8
            _lat_w = videos.shape[4] // 8
            image = torch.zeros(
                videos.shape[0], _lat_c, self.num_frame_per_block + 1, _lat_h, _lat_w,
                device=videos.device, dtype=torch.bfloat16
            )
            # Store VAE input for deferred execution after DiT
            if (videos.shape[2] - 1) // 4 == self.num_frame_per_block:
                self._deferred_vae_input = videos
            elif videos.shape[2] // 4 != self.num_frame_per_block:
                repeat_factor = self.num_frame_per_block // (videos.shape[2] // 4)
                vae_input = torch.repeat_interleave(videos, repeat_factor, dim=2)
                self._deferred_vae_input = torch.cat([vae_input[:, :, 0:1], vae_input], dim=2)
            else:
                self._deferred_vae_input = torch.cat([videos[:, :, 0:1], videos], dim=2)
            start_vae_event.record()
            end_vae_event.record()
        elif latent_video is not None and self.current_start_frame != 0:
            image = latent_video
        elif latent_video is None and self.current_start_frame != 0:
            # Prepare VAE input
            if (videos.shape[2] - 1) // 4 == self.num_frame_per_block:
                vae_input = videos
            elif videos.shape[2] // 4 != self.num_frame_per_block:
                repeat_factor = self.num_frame_per_block // (videos.shape[2] // 4)
                vae_input = torch.repeat_interleave(videos, repeat_factor, dim=2)
                first_frame = vae_input[:, :, 0:1]
                vae_input = torch.cat([first_frame, vae_input], dim=2)
            else:
                first_frame = videos[:, :, 0:1]
                vae_input = torch.cat([first_frame, videos], dim=2)

            _vae_stream = getattr(self, "_vae_stream", None)
            if os.environ.get("ASYNC_VAE", "false").lower() == "true":
                # Submit VAE to background stream
                if _vae_stream is None:
                    self._vae_stream = torch.cuda.Stream(device=videos.device)
                    _vae_stream = self._vae_stream
                _default = torch.cuda.current_stream(videos.device)
                _xfer = torch.cuda.Event()
                _xfer.record(_default)
                with torch.cuda.stream(_vae_stream):
                    _vae_stream.wait_event(_xfer)
                    image = self.vae.encode(
                        vae_input,
                        tiled=self.tiled,
                        tile_size=(self.tile_size_height, self.tile_size_width),
                        tile_stride=(self.tile_stride_height, self.tile_stride_width),
                    )
                _vae_future = torch.cuda.Event()
                _vae_future.record(_vae_stream)
                # image is on background stream — don't use until _vae_future is waited
            else:
                image = self.vae.encode(
                    vae_input,
                    tiled=self.tiled,
                    tile_size=(self.tile_size_height, self.tile_size_width),
                    tile_stride=(self.tile_stride_height, self.tile_stride_width),
                )

        end_vae_event.record()

        # Noise generation: use known shapes to avoid waiting for async VAE
        # image shape is [B, C, T, H, W] where C=16 (latent), H/W from VAE downsampling
        # For 14B: input 180x320 → latent 22x40. For 5B with target 176x320 → 22x40.
        # We can infer shape from the model config.
        _b = videos.shape[0]
        _lat_c = getattr(self.vae, 'z_dim', 16)  # 16 for Wan2.1, 48 for Wan2.2
        _lat_h = videos.shape[3] // 8  # VAE spatial downsampling 8x
        _lat_w = videos.shape[4] // 8
        noise_obs = self.generate_noise((_b, _lat_c, self.num_frame_per_block, _lat_h, _lat_w), seed=self.seed, device='cuda', dtype=torch.bfloat16)
        noise_action = self.generate_noise((_b, self.action_horizon, self.model.action_dim), seed=self.seed, device='cuda', dtype=torch.bfloat16)
        batch_size, num_channels, num_frames, height, width = noise_obs.shape
        ######### Generate video #########
        # DiT patch_embedding uses stride (1,2,2), so tokens per frame = (H//2)*(W//2)
        tokens_per_frame = (height // 2) * (width // 2)
        frame_seqlen = tokens_per_frame
        seq_len = num_frames * frame_seqlen

        # Wait for async VAE if needed (everything above ran in parallel with it)
        if _vae_future is not None:
            torch.cuda.current_stream(videos.device).wait_event(_vae_future)

        image = image.transpose(1, 2)
        noise_obs = noise_obs.transpose(1, 2)

        if self.current_start_frame == 0:
            _static_kv = os.environ.get("STATIC_KV_CACHE", "false").lower() == "true"
            if _static_kv and getattr(self, 'kv_cache1', None) is not None:
                # STATIC_KV_CACHE: REUSE existing buffers (same tensor identity
                # for CUDAGraph Trees cache hits). Just zero data + reset positions.
                for _layer_kv in self.kv_cache1:
                    _layer_kv.zero_()
                for _layer_kv in self.kv_cache_neg:
                    _layer_kv.zero_()
                for _blk in self.model.blocks:
                    _blk.self_attn._fill_level_t.zero_()
            else:
                # First call or dynamic mode: allocate fresh.
                self.kv_cache1, self.kv_cache_neg = self._create_kv_caches(
                    batch_size=batch_size,
                    dtype=noise_obs.dtype,
                    device=noise_obs.device,
                    frame_seqlen=frame_seqlen,
                )
            self.crossattn_cache, self.crossattn_cache_neg = self._create_crossattn_caches(
                batch_size=batch_size,
                dtype=noise_obs.dtype,
                device=noise_obs.device,
            )

        assert self.kv_cache1 is not None
        assert self.kv_cache_neg is not None
        assert self.crossattn_cache is not None
        assert self.crossattn_cache_neg is not None
        kv_caches = self._get_caches(
            [self.kv_cache1, self.kv_cache_neg],
        )
        crossattn_caches = self._get_caches(
            [self.crossattn_cache, self.crossattn_cache_neg],
        )

        start_kv_event.record()
        print(f'[DBG r={self.ip_rank}] kv_init start (frame={self.current_start_frame})', flush=True)

        if self.current_start_frame == 0:
            timestep = torch.ones([batch_size, 1], device=noise_obs.device, dtype=torch.int64) * 0
            self._run_diffusion_steps(
                noisy_input=image.transpose(1, 2),
                timestep=timestep * 0,
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=prompt_embs,
                seq_len=frame_seqlen,
                y=self.ys[:, :, 0:1],
                clip_feature=self.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(
                    start_frame=0,
                    update_kv_cache=True,
                ),
            )
            self.current_start_frame += 1
            
        timestep = torch.ones([batch_size, self.num_frame_per_block], device=noise_obs.device, dtype=torch.int64) * 0

        if self.current_start_frame != 1:
            current_ref_latents = image[:, -self.num_frame_per_block:]
            if self.current_start_frame <= self.ys.shape[2]:
                y = self.ys[:, :, self.current_start_frame - self.num_frame_per_block : self.current_start_frame]
            else:
                y = self.ys[:, :, -self.num_frame_per_block:]

            # --- Adaptive KV init skip based on latent similarity ---
            # When KV_INIT_CACHE_THRESH > 0, compare current VAE latents with
            # previous chunk's. If cosine similarity > threshold, the scene
            # hasn't changed much → skip KV init (reuse warm cache from previous
            # diffusion). If scene changed significantly → run KV init to refresh.
            # KV_INIT_CACHE_THRESH=0: never skip. KV_INIT_CACHE_THRESH=1: always skip.
            _skip_kv_init = False
            _kv_cache_thresh = float(os.environ.get("KV_INIT_CACHE_THRESH", "0"))
            if _kv_cache_thresh > 0:
                _has_warm_cache = getattr(self, "_kv_cache_warm", False)
                _prev_latents = getattr(self, "_prev_ref_latents", None)
                if _has_warm_cache and _prev_latents is not None and _kv_cache_thresh < 1.0:
                    # Compare current vs previous latents
                    _sim = torch.nn.functional.cosine_similarity(
                        current_ref_latents.flatten().unsqueeze(0).float(),
                        _prev_latents.flatten().unsqueeze(0).float(),
                    ).item()
                    _skip_kv_init = _sim > _kv_cache_thresh
                    if self.ip_rank == 0:
                        print(f"[KV skip] sim={_sim:.4f} thresh={_kv_cache_thresh} → {'SKIP' if _skip_kv_init else 'RUN'}")
                elif _has_warm_cache and _kv_cache_thresh >= 1.0:
                    # thresh=1.0: always skip (original behavior)
                    _skip_kv_init = True
            self._prev_ref_latents = current_ref_latents.detach()

            if not _skip_kv_init:
                self._run_diffusion_steps(
                    noisy_input=current_ref_latents.transpose(1, 2),
                    timestep=timestep * 0,
                    action=None,
                    timestep_action=None,
                    state=None,
                    embodiment_id=None,
                    context=prompt_embs,
                    seq_len=seq_len,
                    y=y,
                    clip_feature=self.clip_feas,
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                    kv_cache_metadata=dict(
                        start_frame=self.current_start_frame - self.num_frame_per_block,
                        update_kv_cache=True,
                    ),
                )

        end_kv_event.record()

        noisy_input = noise_obs
        noisy_input_action = noise_action

        # Step 3.1: Spatial denoising loop

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler_action = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(
            self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift)
        sample_scheduler_action.set_timesteps(
            self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift)

        # Decoupled inference: video sigmas end at video_final_noise instead of 0
        # This rescales the schedule so video still takes all denoising steps, 
        # but ends at a higher noise level (e.g., 1.0 → 0.9 → 0.8 instead of 1.0 → 0.5 → 0.0)
        if self.config.decouple_inference_noise:
            video_final_noise = self.config.video_inference_final_noise
            # Rescale video sigmas: map [sigma_max, 0] -> [sigma_max, video_final_noise]
            sigma_max = sample_scheduler.sigmas[0].item()
            sample_scheduler.sigmas = sample_scheduler.sigmas * (sigma_max - video_final_noise) / sigma_max + video_final_noise
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64)
            if self.ip_rank == 0:
                print(f"Decoupled inference: video sigmas {sigma_max:.3f} -> {sample_scheduler.sigmas[-1].item():.3f}")

        _profile = hasattr(self, '_perf_profile') and self._perf_profile
        if _profile:
            start_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps]
            end_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps]
        prev_predictions = [] 
        self.skip_countdown = 0
        dit_compute_steps = 0
        for index, current_timestep in enumerate(sample_scheduler.timesteps):
            if _profile:
                start_diffusion_events[index].record()

            # Get timesteps from respective schedulers
            action_timestep = sample_scheduler_action.timesteps[index]
            video_timestep = sample_scheduler.timesteps[index]  # Already rescaled if decoupled

            # set current timestep
            timestep = torch.ones(
                [batch_size, self.num_frame_per_block],
                device=noise_obs.device,
                dtype=torch.int64,
            ) * video_timestep
            timestep_action = torch.ones(
                [batch_size, self.action_horizon],
                device=noise_obs.device,
                dtype=torch.int64,
            ) * action_timestep

            # check if we need to run the DIT step
            should_run_model = self.should_run_model(index, current_timestep, prev_predictions)
            if should_run_model:
                dit_compute_steps += 1
                if self.current_start_frame + self.num_frame_per_block <= self.ys.shape[2]:
                    y = self.ys[:, :, self.current_start_frame : self.current_start_frame + self.num_frame_per_block]
                else:
                    y = self.ys[:, :, -self.num_frame_per_block:]
                predictions = self._run_diffusion_steps(
                    noisy_input=noisy_input.transpose(1, 2),
                    timestep=timestep,
                    action=noisy_input_action,
                    timestep_action=timestep_action,
                    state=state_features,
                    embodiment_id=embodiment_id,
                    context=prompt_embs,
                    seq_len=seq_len,
                    y=y,
                    clip_feature=self.clip_feas,
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                    kv_cache_metadata=dict(
                        start_frame=self.current_start_frame,
                        update_kv_cache=False,
                    ),
                )
                flow_pred_cond, flow_pred_cond_action = predictions[0]
                if len(predictions) > 1:
                    flow_pred_uncond, flow_pred_uncond_action = predictions[1]
                    flow_pred = flow_pred_uncond + self.cfg_scale * (flow_pred_cond - flow_pred_uncond)
                else:
                    # No CFG (cfg_scale=1.0): just use conditional prediction
                    flow_pred = flow_pred_cond
                prev_predictions.append((current_timestep, flow_pred, flow_pred_cond_action))
                max_cache_size = 2
                if len(prev_predictions) > max_cache_size:
                    prev_predictions.pop(0)

            else:
                assert len(prev_predictions) > 0, "prev_predictions must be set when skipping"
                _, flow_pred, flow_pred_cond_action = prev_predictions[-1]

            if _profile:
                end_diffusion_events[index].record()

            # Video: denoising step (uses rescaled schedule if decoupled)
            noisy_input = sample_scheduler.step(
                model_output=flow_pred.transpose(1, 2),
                timestep=video_timestep,
                sample=noisy_input,
                step_index=index,
                return_dict=False,
            )[0]
            
            # Action: always fully denoises with standard schedule (1000->0)
            noisy_input_action = sample_scheduler_action.step(
                model_output=flow_pred_cond_action,
                timestep=action_timestep,
                sample=noisy_input_action,
                step_index=index,
                return_dict=False,
            )[0]

        latents = noisy_input
        latents_action = noisy_input_action
        output = latents

        if self.current_start_frame == 1:
            output = torch.cat([image, output], dim=1)
        self.current_start_frame += self.num_frame_per_block

        # Mark KV cache as warm after diffusion (for KV init skip optimization)
        if float(os.environ.get("KV_INIT_CACHE_THRESH", "0")) > 0:
            self._kv_cache_warm = True

        # --- Deferred VAE + KV init: save as callable for post-action execution ---
        # The policy_server runs this AFTER sending the action to the client,
        # overlapping with the client's round-trip time.
        _deferred_vae = getattr(self, "_deferred_vae_input", None)
        if _deferred_vae is not None:
            self._deferred_vae_input = None
            # Capture variables for the deferred closure
            _dv_input = _deferred_vae
            _dv_prompts = prompt_embs
            _dv_kv = kv_caches
            _dv_ca = crossattn_caches
            _dv_seq = seq_len
            _dv_start = self.current_start_frame
            _dv_bs = batch_size
            _dv_device = noise_obs.device

            # Save deferred function — will be called by policy_server after sending action
            @torch.compiler.disable
            def _run_deferred_vae_kv():
                with torch.no_grad():
                    _img = self.vae.encode(
                        _dv_input,
                        tiled=self.tiled,
                        tile_size=(self.tile_size_height, self.tile_size_width),
                        tile_stride=(self.tile_stride_height, self.tile_stride_width),
                    )
                    _img = _img.transpose(1, 2)
                    _ref = _img[:, -self.num_frame_per_block:]
                    if _dv_start <= self.ys.shape[2]:
                        _y = self.ys[:, :, _dv_start - self.num_frame_per_block : _dv_start]
                    else:
                        _y = self.ys[:, :, -self.num_frame_per_block:]
                    _ts = torch.zeros([_dv_bs, self.num_frame_per_block],
                                       device=_dv_device, dtype=torch.int64)
                    self._run_diffusion_steps(
                        noisy_input=_ref.transpose(1, 2),
                        timestep=_ts,
                        action=None, timestep_action=None, state=None, embodiment_id=None,
                        context=_dv_prompts,
                        seq_len=_dv_seq,
                        y=_y,
                        clip_feature=self.clip_feas,
                        kv_caches=_dv_kv,
                        crossattn_caches=_dv_ca,
                        kv_cache_metadata=dict(
                            start_frame=_dv_start - self.num_frame_per_block,
                            update_kv_cache=True,
                        ),
                    )

        # Store deferred function if we have one
        if '_run_deferred_vae_kv' in dir():
            self._pending_deferred = _run_deferred_vae_kv
        else:
            self._pending_deferred = None

        # Do torch.cuda.synchronize() to ensure all operations are completed before timing.
        # This isn't expected to affect inference performance since it's at the end of an inference step.
        torch.cuda.synchronize()

        total_time = time.perf_counter() - start_time
        text_encoder_time = start_text_encoder_event.elapsed_time(end_text_encoder_event) / 1000
        image_encoder_time = start_image_encoder_event.elapsed_time(end_image_encoder_event) / 1000
        vae_time = start_vae_event.elapsed_time(end_vae_event) / 1000
        kv_creation_time = start_kv_event.elapsed_time(end_kv_event) / 1000
        try:
            diffusion_times = [s.elapsed_time(e) for s, e in zip(start_diffusion_events, end_diffusion_events)]
        except ValueError:
            diffusion_times = [0.0]
        diffusion_time = sum(diffusion_times) / 1000
        scheduler_time = total_time - kv_creation_time - diffusion_time - text_encoder_time - image_encoder_time - vae_time

        if self.ip_rank == 0:
            print(f"Time taken: Total {total_time:.2f} seconds, "
                  f"Text Encoder {text_encoder_time:.2f} seconds, "
                  f"Image Encoder {image_encoder_time:.2f} seconds, "
                  f"VAE {vae_time:.2f} seconds, "
                  f"KV Cache Creation {kv_creation_time:.2f} seconds, "
                  f"Diffusion {diffusion_time:.2f} seconds, "
                  f"DIT Compute Steps {dit_compute_steps} steps, "
                  f"Scheduler {scheduler_time:.2f} seconds")

        # Per-call profile dump (opt-in via PROFILE_INFERENCE=true). Only the
        # true global rank 0 writes — without this check, all 4 SP ranks within
        # CFG group 0 satisfy `ip_rank == 0` and duplicate the record.
        _global_rank_zero = (
            not dist.is_initialized() or dist.get_rank() == 0
        )
        if _perf_profile.enabled() and _global_rank_zero:
            try:
                dit_model = getattr(self, "model", None)
                model_cfg = {
                    "dim": int(getattr(dit_model, "dim", 5120)),
                    "num_heads": int(getattr(dit_model, "num_heads", 40)),
                    "ffn_dim": int(getattr(dit_model, "ffn_dim", 13824)),
                    "frame_seqlen": int(getattr(dit_model, "frame_seqlen", 880)),
                    "num_layers": int(getattr(dit_model, "num_layers", 40)),
                    # Best-effort actual sequence length during this call:
                    "seq_len": int(getattr(dit_model, "frame_seqlen", 880)),
                    "batch_size": 1,
                }
                sp_size = int(os.environ.get("SP_SIZE", "1"))
                num_gpus = int(os.environ.get("NUM_GPUS", "1"))
                cfg_size = max(num_gpus // max(sp_size, 1), 1)
                _perf_profile.finalize_and_dump(
                    phases_s={
                        "text_encoder_s": text_encoder_time,
                        "image_encoder_s": image_encoder_time,
                        "vae_s": vae_time,
                        "kv_creation_s": kv_creation_time,
                        "diffusion_s": diffusion_time,
                        "scheduler_s": scheduler_time,
                        "diffusion_per_step_s": [t / 1000.0 for t in diffusion_times],
                    },
                    total_s=total_time,
                    dit_compute_steps=int(dit_compute_steps),
                    num_dit_steps=int(self.num_inference_steps),
                    model_cfg=model_cfg,
                    sp_size=sp_size,
                    cfg_size=cfg_size,
                    rank=int(self.ip_rank) if self.ip_rank is not None else 0,
                )
            except Exception as _e:  # pragma: no cover
                print(f"[perf_profile] dump failed: {_e}")

        _global_rank = dist.get_rank() if dist.is_initialized() else 0
        _perf_profile.maybe_stop_trace(_global_rank)

        return BatchFeature(data={"action_pred": latents_action, "video_pred": output.transpose(1, 2)})
    
    def cache_predict_order1(self, current_timestep, timestep_1, f1, timestep_2, f2):
        h_curr = current_timestep - timestep_1
        h_past = timestep_1 - timestep_2

        v_prime = (f1 - f2) / h_past

        # Prediction 
        damping_factor = 0.25
        flow_pred = f1 + (v_prime * h_curr) * damping_factor
        return flow_pred

    def _get_fp8_context(self):
        """Return fp8_autocast context if FP8 inference is enabled, else a no-op context."""
        import contextlib
        if not self.fp8_inference:
            return contextlib.nullcontext()
        _fp8_mode = os.getenv("FP8_MODE", "static")
        if _fp8_mode == "static":
            # Static FP8: no TE context needed — weights are pre-quantized,
            # StaticFP8Linear handles activation quantization internally
            return contextlib.nullcontext()
        # TE dynamic FP8 path
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format
        _recipe = DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=16,
            amax_compute_algo="most_recent",
        )
        return te.fp8_autocast(enabled=True, fp8_recipe=_recipe, _graph=True)

    @staticmethod
    def _replace_linear_with_te(module: nn.Module) -> None:
        """Replace nn.Linear with te.Linear in-place for FP8 autocast support."""
        import transformer_engine.pytorch as te
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                te_linear = te.Linear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    params_dtype=child.weight.dtype,
                    device=child.weight.device,
                )
                te_linear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    te_linear.bias.data.copy_(child.bias.data)
                setattr(module, name, te_linear)
            elif isinstance(child, nn.Sequential):
                # Handle nn.Sequential (e.g. FFN layers)
                for i, subchild in enumerate(child):
                    if isinstance(subchild, nn.Linear):
                        te_linear = te.Linear(
                            subchild.in_features, subchild.out_features,
                            bias=subchild.bias is not None,
                            params_dtype=subchild.weight.dtype,
                            device=subchild.weight.device,
                        )
                        te_linear.weight.data.copy_(subchild.weight.data)
                        if subchild.bias is not None:
                            te_linear.bias.data.copy_(subchild.bias.data)
                        child[i] = te_linear
            else:
                WANPolicyHead._replace_linear_with_te(child)

    def post_initialize(self):
        # Move models to the cuda device and set the dtype to bfloat16.
        print("Moving models to the cuda device and setting the dtype to bfloat16.")
        self.model.to(device=self._device, dtype=torch.bfloat16)
        self.text_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.image_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.vae.to(device=self._device, dtype=torch.bfloat16)
        import os
        ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"
        LOAD_TRT_ENGINE = os.getenv("LOAD_TRT_ENGINE", None)

        self.fp8_inference = os.getenv("FP8_INFERENCE", "False").lower() == "true"
        _fp8_mode = os.getenv("FP8_MODE", "static")  # "static" (TRT-LLM) or "te" (TE dynamic)
        if self.fp8_inference:
            self.model.fp8_inference = True
            if _fp8_mode == "static":
                # Static FP8 (TRT-LLM pattern): pre-quantize weights, no per-call scaling overhead
                from groot.vla.model.dreamzero.modules.fp8_linear import replace_linear_with_fp8
                count = 0
                for block in self.model.blocks:
                    count += replace_linear_with_fp8(block.self_attn, target_modules=None)
                    count += replace_linear_with_fp8(block.ffn, target_modules=None)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"[StaticFP8] Replaced {count} nn.Linear → StaticFP8Linear (no TE)")
            else:
                # TE dynamic FP8 (original path)
                print("Enabling FP8 inference: replacing nn.Linear with te.Linear in DiT blocks.")
                for block in self.model.blocks:
                    self._replace_linear_with_te(block.self_attn)
                    self._replace_linear_with_te(block.ffn)

        DISABLE_TORCH_COMPILE = os.getenv("DISABLE_TORCH_COMPILE", "False").lower() == "true"
        COMPILE_DIT = os.getenv("COMPILE_DIT", "False").lower() == "true"
        if not ENABLE_TENSORRT and not DISABLE_TORCH_COMPILE:
            print("Torch compiling the TextEncoder, ImageEncoder, and VAE modules.")

            # Use mode=default for encoders to avoid cudagraph_trees
            # AssertionError bug in PyTorch 2.8 and graph-tree invalidation
            # on prompt length changes. Encoders run once per request so
            # CUDA graph replay saves negligible time.
            self.text_encoder.forward = torch.compile(
                mode="default", fullgraph=True, dynamic=False,
            )(self.text_encoder.forward)

            self.image_encoder.model.visual.forward = torch.compile(
                mode="default", fullgraph=True, dynamic=False,
            )(self.image_encoder.model.visual.forward)

            # Use mode=default for VAE: reduce-overhead (CUDA graphs) stores state
            # in thread-local storage and breaks when OVERLAP_VAE_DIT runs the
            # VAE in a background thread.
            self.vae.model.encode = torch.compile(
                mode="default", fullgraph=True, dynamic=False,
            )(self.vae.model.encode)

        if not ENABLE_TENSORRT and not DISABLE_TORCH_COMPILE and COMPILE_DIT:
            # Two knobs:
            #   COMPILE_DIT_MODE      (default "reduce-overhead")
            #   COMPILE_DIT_DYNAMIC   (default "false" — static shapes)
            #
            # Background: with dynamic=True, dynamo materializes symbolic ints
            # (e.g. KV-cache length) as 0-d CPU tensors in the traced FX graph.
            # `cudagraph_trees` sees those as "cpu device" inputs and skips
            # cudagraph capture — the whole point of reduce-overhead mode.
            # With dynamic=False we specialize shapes; there is a compile hit
            # on the first call with a new shape, but steady-state calls use
            # the captured cudagraph.
            _compile_mode = os.getenv("COMPILE_DIT_MODE", "reduce-overhead")
            _compile_dynamic = os.getenv("COMPILE_DIT_DYNAMIC", "false").lower() == "true"
            _compile_fullgraph = os.getenv("COMPILE_DIT_FULLGRAPH", "false").lower() == "true"
            print(f"Torch compiling DiT _forward_blocks (mode={_compile_mode}, "
                  f"dynamic={_compile_dynamic}, fullgraph={_compile_fullgraph}).")
            self.model._forward_blocks = torch.compile(
                mode=_compile_mode,
                dynamic=_compile_dynamic,
                fullgraph=_compile_fullgraph,
            )(self.model._forward_blocks)

        # Manual CUDA graph capture (vLLM/sglang pattern). Bypasses torch.compile
        # entirely — no dynamo, no inductor, no graph-break drama. Captures a raw
        # CUDA graph of the eager _forward_blocks, replays it on subsequent calls.
        CUDA_GRAPH_MANUAL = os.getenv("CUDA_GRAPH_DIT_MANUAL", "false").lower() == "true"
        if CUDA_GRAPH_MANUAL:
            from groot.vla.model.dreamzero.cuda_graph_runner import CudaGraphDiTRunner
            _dev = next(self.model.parameters()).device
            self._cuda_graph_runner = CudaGraphDiTRunner(self.model, _dev)
            self.model._cuda_graph_runner = self._cuda_graph_runner
            print(f"Manual CUDA graph runner created for DiT _forward_blocks (device={_dev}).")

        # Per-block torch.compile (FastVideo pattern): fuses small ops within
        # each block without CUDA graph complexity. Complementary to piecewise.
        COMPILE_BLOCKS = os.getenv("COMPILE_BLOCKS", "false").lower() == "true"
        if COMPILE_BLOCKS and not COMPILE_DIT:
            from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import compile_blocks_for_inference
            compile_blocks_for_inference(self.model)
        else:
            self._cuda_graph_runner = None

        self.trt_engine = None
        if LOAD_TRT_ENGINE is not None:
            print(f"Loading TRT engine from {LOAD_TRT_ENGINE}")
            # Offload PyTorch DiT to CPU to make room for TRT engine.
            # KV cache creation will move it back temporarily (slow on single GPU,
            # but TRT on H100 is only viable as a benchmark — use FA2+CFG for production).
            self.model.cpu()
            torch.cuda.empty_cache()
            import groot.control.tensorrt_utils as trt_utils
            model_path = LOAD_TRT_ENGINE
            self.trt_engine = trt_utils.load_tensorrt_engine(model_path, model_type="ar_14B")

    def parallelize(self, device_mesh: DeviceMesh) -> None:
        ip_mesh = device_mesh["ip"]
        self.ip_rank = ip_mesh.get_local_rank()
        self.ip_size = ip_mesh.size()
        self.ip_group = ip_mesh.get_group()

        assert self.ip_size == 1 or self.ip_size == 2, "ip_size must be 1 or 2"
        assert self.ip_rank >= 0 and self.ip_rank < self.ip_size, "ip_rank must be in [0, ip_size)"

        if "sp" in device_mesh.mesh_dim_names:
            from groot.vla.model.dreamzero.modules.sequence_parallel import SequenceParallelContext
            sp_mesh = device_mesh["sp"]
            self.sp_ctx = SequenceParallelContext(
                sp_group=sp_mesh.get_group(),
                sp_rank=sp_mesh.get_local_rank(),
                sp_size=sp_mesh.size(),
            )
            # Initialize pynccl all-to-all for graph-capturable SP (vLLM pattern).
            # Uses direct NCCL C calls on the CURRENT stream instead of
            # torch.distributed's side-stream NCCL. Enable via PYNCCL_ALLTOALL=true.
            if os.environ.get("PYNCCL_ALLTOALL", "false").lower() == "true" and sp_mesh.size() > 1:
                try:
                    from groot.vla.model.dreamzero.modules.pynccl_alltoall import PyNcclAllToAll
                    _dev = torch.device(f"cuda:{torch.cuda.current_device()}")
                    self.sp_ctx.pynccl_comm = PyNcclAllToAll(sp_mesh.get_group(), _dev)
                except Exception as e:
                    print(f"[pynccl] Failed to init: {e}. Falling back to dist.all_to_all")
                    self.sp_ctx.pynccl_comm = None
        else:
            self.sp_ctx = None
        self.model.set_sp_context(self.sp_ctx)

        # Initialize TeaCache for step-level block skipping
        if float(os.environ.get("TEACACHE_THRESH", "0")) > 0:
            self.model.init_teacache()
            if dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized():
                print(f"[TeaCache] Initialized with threshold={os.environ['TEACACHE_THRESH']}")

        # Fuse Q,K,V into single linear (1 cuBLAS instead of 3)
        # Skip when using FP8 — FP8 weights need special fusion logic
        if os.environ.get("FUSE_QKV", "false").lower() == "true":
            self.model.fuse_qkv_linears()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
