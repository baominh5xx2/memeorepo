from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from damp_es.clip_es.simple_tokenizer import SimpleTokenizer as _Tokenizer


_tokenizer = _Tokenizer()


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        bsz, q_len, dim = q.shape
        _, kv_len, _ = k.shape

        q = self.q_proj(q).reshape(bsz, q_len, self.num_heads, dim // self.num_heads)
        k = self.k_proj(k).reshape(bsz, kv_len, self.num_heads, dim // self.num_heads)
        v = self.v_proj(v).reshape(bsz, kv_len, self.num_heads, dim // self.num_heads)

        attn = torch.einsum("bnkc,bmkc->bknm", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum("bknm,bmkc->bnkc", attn, v).reshape(bsz, q_len, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ContextDecoder(nn.Module):
    def __init__(
        self,
        transformer_width: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 6,
        visual_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.decoder = nn.ModuleList(
            [TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)]
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if hasattr(nn.init, "trunc_normal_"):
                nn.init.trunc_normal_(module.weight, std=0.02)
            else:
                nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, text: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        visual = self.memory_proj(visual)
        x = self.text_proj(text)

        for layer in self.decoder:
            x = layer(x, visual)

        return self.out_proj(x)


class PromptLearner(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        class_names: list[str],
        n_ctx: int = 4,
        csc: bool = False,
        prompt_template: str = "a histopathology patch of {}.",
        naive_prompt_prefix: str = "a histopathology patch of a",
    ):
        super().__init__()

        from damp_es.clip_es import clip as clip_es

        n_cls = len(class_names)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if csc:
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        self.gamma_t = nn.Parameter(torch.ones(1) * 0.01)
        self.gamma_v = nn.Parameter(torch.ones(1) * 0.01)

        prompt_prefix = " ".join(["X"] * n_ctx)
        class_names_clean = [name.replace("_", " ") for name in class_names]

        prompts = [prompt_prefix + " " + name + "." for name in class_names_clean]
        naive_prompts = [naive_prompt_prefix + " " + name + "." for name in class_names_clean]

        tokenized_prompts = clip_es.tokenize(prompts)
        naive_tokenized_prompts = clip_es.tokenize(naive_prompts)

        device = clip_model.token_embedding.weight.device
        tokenized_prompts = tokenized_prompts.to(device)
        naive_tokenized_prompts = naive_tokenized_prompts.to(device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            naive_embedding = clip_model.token_embedding(naive_tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("naive_tokenized_prompts", naive_tokenized_prompts)
        self.register_buffer("naive_embedding", naive_embedding)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = csc
        self.prompt_template = prompt_template
        self.name_lens = [len(_tokenizer.encode(name)) for name in class_names_clean]

    def forward(self) -> torch.Tensor:
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts


class DAMPTextEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x_out = self.transformer(x)
        if isinstance(x_out, tuple):
            x = x_out[0]
        else:
            x = x_out
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        text_embedding_at_eos = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        text_embedding_all = torch.einsum("kld,dc->klc", x, self.text_projection)
        return text_embedding_at_eos, text_embedding_all


class DAMPWrapper:
    """Wrapper that supports full DAMP mutual prompting and CLIP-ES CAM hooks."""

    def __init__(
        self,
        backbone: str = "ViT-B/16",
        clip_weights: str | None = None,
        device: str = "cuda",
        feature_layer: int = -1,
        n_ctx: int = 4,
        class_names: Optional[list[str]] = None,
        prompt_template: str = "a histopathology patch of {}.",
        enable_mutual_prompting: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.backbone = backbone
        self.clip_model = self._load_clip_model(backbone, clip_weights)
        self.clip_model.eval().requires_grad_(False)

        self.feature_dim = int(getattr(self.clip_model.visual, "output_dim", 512))
        self.class_names = class_names or ["background", "tumor", "stroma", "tumor stroma"]

        self.prompt_learner = PromptLearner(
            clip_model=self.clip_model,
            class_names=self.class_names,
            n_ctx=int(n_ctx),
            csc=False,
            prompt_template=prompt_template,
            naive_prompt_prefix="a histopathology patch of a",
        ).to(self.device)
        self.context_decoder = ContextDecoder(
            transformer_width=256,
            transformer_heads=4,
            transformer_layers=6,
            visual_dim=self.feature_dim,
            dropout=0.1,
        ).to(self.device)
        self.text_encoder = DAMPTextEncoder(self.clip_model).to(self.device)

        self.feature_layer = feature_layer
        self.enable_mutual_prompting = bool(enable_mutual_prompting)

        self._feature_activation: Optional[torch.Tensor] = None
        self._feature_gradient: Optional[torch.Tensor] = None
        self._attn_history: list[torch.Tensor] = []

        self._register_feature_hooks()
        self._refresh_naive_text_embedding()

    def _refresh_naive_text_embedding(self) -> None:
        with torch.no_grad():
            naive_text, _ = self.text_encoder(
                self.prompt_learner.naive_embedding,
                self.prompt_learner.naive_tokenized_prompts,
            )
            self.naive_text_embedding = naive_text.detach()

    def stage1_parameters(self) -> Iterable[nn.Parameter]:
        for p in self.prompt_learner.parameters():
            yield p
        for p in self.context_decoder.parameters():
            yield p

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        # Backward-compatible alias used by existing training script.
        return self.stage1_parameters()

    def set_stage1_train(self, mode: bool) -> None:
        self.prompt_learner.train(mode)
        self.context_decoder.train(mode)
        self.text_encoder.eval()
        self.clip_model.eval()

    def set_adapter_train(self, mode: bool) -> None:
        # Backward-compatible alias.
        self.set_stage1_train(mode)

    def tokenize(self, prompts: list[str]) -> torch.Tensor:
        from damp_es.clip_es import clip as clip_es

        return clip_es.tokenize(prompts).to(self.device)

    def encode_text(self, tokenized_text: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokenized_text.to(self.device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward_stage1(self, image: torch.Tensor, ind: bool = False, pse: bool = False) -> tuple[torch.Tensor, ...]:
        global_feat, visual_embeddings, _ = self._extract_visual_features(image.to(self.device), for_cam=False)
        logits, visual, text = self._compute_mutual_prompt_logits(global_feat, visual_embeddings)

        outputs: list[torch.Tensor] = [logits]
        if ind:
            logits_ind = torch.einsum("ac,bkc->abk", visual, text).mean(dim=-1)
            logits_ind = logits_ind * self.clip_model.logit_scale.exp().float()
            outputs.append(logits_ind)

        if pse:
            image_features = F.normalize(global_feat.float(), dim=-1)
            text_features = F.normalize(self.naive_text_embedding.float(), dim=-1)
            pseudo_logits = self.clip_model.logit_scale.exp().float() * image_features @ text_features.t()
            outputs.append(pseudo_logits)

        return tuple(outputs)

    def forward_logits(
        self,
        image: torch.Tensor,
        text_features: torch.Tensor,
        replace_cls_with_avg: bool = False,
    ) -> torch.Tensor:
        global_feat, visual_embeddings, final_tokens = self._extract_visual_features(image.to(self.device), for_cam=True)

        if self.enable_mutual_prompting:
            image_embedding = self._apply_visual_prompt(global_feat, visual_embeddings, track_grad=True)
        else:
            image_embedding = self._tokens_to_embedding(
                final_tokens=final_tokens,
                replace_cls_with_avg=replace_cls_with_avg,
            )

        return self._compute_logits(image_embedding=image_embedding, text_features=text_features)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            global_feat, visual_embeddings, final_tokens = self._extract_visual_features(image.to(self.device), for_cam=False)
            if self.enable_mutual_prompting:
                image_embedding = self._apply_visual_prompt(global_feat, visual_embeddings, track_grad=False)
            else:
                image_embedding = self._tokens_to_embedding(final_tokens=final_tokens, replace_cls_with_avg=False)
            image_embedding = F.normalize(image_embedding.float(), dim=-1)
        return image_embedding

    def get_attention_affinity(self, num_layers: int = 8) -> torch.Tensor:
        if self._attn_history:
            sliced: list[torch.Tensor] = []
            for attn in self._attn_history[-num_layers:]:
                # MultiheadAttention can output [B, L, L] or [B, H, L, L].
                if attn.ndim == 4:
                    attn_2d = attn.mean(dim=1)[0]
                elif attn.ndim == 3:
                    attn_2d = attn[0]
                else:
                    continue
                if attn_2d.shape[0] > 1:
                    sliced.append(attn_2d[1:, 1:])

            if sliced:
                affinity = torch.stack(sliced, dim=0).mean(dim=0)
                return affinity.float().clamp(min=0.0)

        feature_map = self.get_feature_map()[0]
        c, h, w = feature_map.shape
        flattened = feature_map.reshape(c, h * w).permute(1, 0)
        flattened = torch.nn.functional.normalize(flattened, dim=-1)
        affinity = torch.matmul(flattened, flattened.t())
        return affinity.clamp(min=0.0)

    def zero_grad(self) -> None:
        self.clip_model.zero_grad(set_to_none=True)
        self.prompt_learner.zero_grad(set_to_none=True)
        self.context_decoder.zero_grad(set_to_none=True)

    def get_feature_map(self) -> torch.Tensor:
        if self._feature_activation is None:
            raise RuntimeError("No feature activation available. Run forward first.")
        return self._tokens_to_feature_map(self._feature_activation)

    def get_feature_gradient(self) -> torch.Tensor:
        if self._feature_gradient is None:
            raise RuntimeError("No feature gradient available. Run backward first.")
        return self._tokens_to_feature_map(self._feature_gradient)

    def load_damp_prompt_checkpoints(self, damp_ckpt_root: str | Path) -> None:
        """
        Loader for stage1 checkpoints saved inside damp_es.

        Preferred format:
            <root>/mutual_prompt/model-best.pth.tar

        Legacy compatibility:
            <root>/prompt_learner/model-best.pth.tar
            <root>/context_decoder/model-best.pth.tar
        """
        ckpt_root = Path(damp_ckpt_root)
        mutual_ckpt = ckpt_root / "mutual_prompt" / "model-best.pth.tar"

        if mutual_ckpt.exists():
            checkpoint = torch.load(mutual_ckpt, map_location=self.device)
            prompt_state = checkpoint.get("prompt_learner")
            context_state = checkpoint.get("context_decoder")
            if prompt_state is not None and context_state is not None:
                self.prompt_learner.load_state_dict(prompt_state, strict=True)
                self.context_decoder.load_state_dict(context_state, strict=True)
                self.enable_mutual_prompting = True
                self.set_stage1_train(False)
                self._refresh_naive_text_embedding()
                print(f"Loaded Stage1 mutual prompting checkpoint: {mutual_ckpt}")
                return

        prompt_ckpt = ckpt_root / "prompt_learner" / "model-best.pth.tar"
        context_ckpt = ckpt_root / "context_decoder" / "model-best.pth.tar"

        if prompt_ckpt.exists() and context_ckpt.exists():
            prompt_blob = torch.load(prompt_ckpt, map_location=self.device)
            context_blob = torch.load(context_ckpt, map_location=self.device)

            prompt_state = prompt_blob.get("state_dict", prompt_blob)
            context_state = context_blob.get("state_dict", context_blob)

            self.prompt_learner.load_state_dict(prompt_state, strict=False)
            self.context_decoder.load_state_dict(context_state, strict=False)
            self.enable_mutual_prompting = True
            self.set_stage1_train(False)
            self._refresh_naive_text_embedding()
            print(f"Loaded legacy Stage1 checkpoints: {prompt_ckpt} | {context_ckpt}")
            return

        adapter_ckpt = ckpt_root / "adapters" / "model-best.pth.tar"
        if adapter_ckpt.exists():
            raise RuntimeError(
                "Found legacy adapter checkpoint only, but full DAMP mutual prompting is required. "
                f"Please retrain Stage1. Checkpoint: {adapter_ckpt}"
            )

        raise FileNotFoundError(
            "DAMP checkpoints were not found. Expected one of: "
            f"{mutual_ckpt} or ({prompt_ckpt} and {context_ckpt})"
        )

    def save_stage1_checkpoint(self, output_root: str | Path, epoch: int, is_best: bool) -> None:
        out_root = Path(output_root)
        prompt_dir = out_root / "prompt_learner"
        context_dir = out_root / "context_decoder"
        mutual_dir = out_root / "mutual_prompt"

        prompt_dir.mkdir(parents=True, exist_ok=True)
        context_dir.mkdir(parents=True, exist_ok=True)
        mutual_dir.mkdir(parents=True, exist_ok=True)

        prompt_state = {
            "state_dict": self.prompt_learner.state_dict(),
            "epoch": int(epoch),
        }
        context_state = {
            "state_dict": self.context_decoder.state_dict(),
            "epoch": int(epoch),
        }
        mutual_state = {
            "epoch": int(epoch),
            "class_names": list(self.class_names),
            "n_ctx": int(self.prompt_learner.n_ctx),
            "prompt_learner": self.prompt_learner.state_dict(),
            "context_decoder": self.context_decoder.state_dict(),
        }

        torch.save(prompt_state, prompt_dir / "model-last.pth.tar")
        torch.save(context_state, context_dir / "model-last.pth.tar")
        torch.save(mutual_state, mutual_dir / "model-last.pth.tar")

        if is_best:
            torch.save(prompt_state, prompt_dir / "model-best.pth.tar")
            torch.save(context_state, context_dir / "model-best.pth.tar")
            torch.save(mutual_state, mutual_dir / "model-best.pth.tar")

    def _load_clip_model(self, backbone: str, clip_weights: str | None) -> nn.Module:
        from damp_es.clip_es import clip as clip_es

        model_name_or_path = backbone
        if clip_weights:
            weight_path = Path(clip_weights)
            if weight_path.exists():
                model_name_or_path = str(weight_path)
            else:
                print(
                    f"CLIP weights not found at {weight_path}; falling back to pretrained model '{backbone}'."
                )
        model, _ = clip_es.load(model_name_or_path, device=self.device, jit=False)
        return model

    def _register_feature_hooks(self) -> None:
        resblocks = self.clip_model.visual.transformer.resblocks
        n_layers = len(resblocks)
        if self.feature_layer < 0:
            layer_idx = n_layers + self.feature_layer
        else:
            layer_idx = self.feature_layer
        if layer_idx < 0 or layer_idx >= n_layers:
            raise IndexError(f"feature_layer={self.feature_layer} is out of range for {n_layers} layers")

        target = resblocks[layer_idx].ln_1

        def _fwd_hook(_: nn.Module, __, output: torch.Tensor) -> None:
            self._feature_activation = output

        def _bwd_hook(_: nn.Module, __, grad_output) -> None:
            self._feature_gradient = grad_output[0]

        target.register_forward_hook(_fwd_hook)
        target.register_full_backward_hook(_bwd_hook)

    def _encode_image_tokens(self, image: torch.Tensor) -> torch.Tensor:
        h, w = int(image.shape[-2]), int(image.shape[-1])
        tokens, attn_history = self.clip_model.encode_image(image, h, w)
        self._attn_history = list(attn_history) if isinstance(attn_history, list) else []
        return tokens

    def _run_last_block(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        last_idx = self.clip_model.visual.transformer.layers - 1
        last_block = self.clip_model.visual.transformer.resblocks[last_idx]
        out_tokens, attn_last = last_block(tokens)
        return out_tokens, attn_last

    def _extract_visual_features(
        self,
        image: torch.Tensor,
        for_cam: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self._encode_image_tokens(image)
        if for_cam:
            tokens = tokens.detach().requires_grad_(True)
        else:
            tokens = tokens.detach()

        final_tokens, attn_last = self._run_last_block(tokens)
        self._attn_history = list(self._attn_history) + [attn_last]

        x = final_tokens.permute(1, 0, 2)
        x = self.clip_model.visual.ln_post(x)
        if self.clip_model.visual.proj is not None:
            x = x @ self.clip_model.visual.proj

        global_feat = x[:, 0]
        patch_tokens = x[:, 1:]
        side = int(math.sqrt(patch_tokens.shape[1]))
        if side * side != patch_tokens.shape[1]:
            raise ValueError("Patch token count is not a perfect square")
        visual_embeddings = patch_tokens.reshape(x.shape[0], side, side, x.shape[2]).permute(0, 3, 1, 2)

        return global_feat, visual_embeddings, final_tokens

    def _tokens_to_embedding(self, final_tokens: torch.Tensor, replace_cls_with_avg: bool) -> torch.Tensor:
        x = final_tokens.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.visual.ln_post(x)
        if replace_cls_with_avg:
            x = torch.mean(x[:, 1:, :], dim=1)
        else:
            x = x[:, 0, :]

        if self.clip_model.visual.proj is not None:
            x = x @ self.clip_model.visual.proj
        return x

    def _compute_mutual_prompt_logits(
        self,
        global_feat: torch.Tensor,
        visual_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, channels, h, w = visual_embeddings.shape
        visual_contexts = torch.cat(
            [global_feat.reshape(bsz, channels, 1), visual_embeddings.reshape(bsz, channels, h * w)],
            dim=2,
        ).permute(0, 2, 1)

        raw_prompt = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_embeddings, text_contexts_all = self.text_encoder(raw_prompt, tokenized_prompts)

        text_embeddings = text_embeddings.expand(bsz, -1, -1)
        text_contexts = text_contexts_all.expand(bsz, -1, -1, -1)[:, 0, : self.prompt_learner.n_ctx, :]

        global_f = global_feat.float()
        visual_ctx_f = visual_contexts.float()
        text_embeddings_f = text_embeddings.float()
        text_contexts_f = text_contexts.float()

        vis_prompt_diff = self.context_decoder(global_f.reshape(bsz, 1, channels), text_contexts_f)
        vis_prompt_diff = vis_prompt_diff.reshape(bsz, channels)
        updated_vision = global_f + self.prompt_learner.gamma_v.float() * vis_prompt_diff

        text_diff = self.context_decoder(text_embeddings_f, visual_ctx_f)
        updated_text = text_embeddings_f + self.prompt_learner.gamma_t.float() * text_diff

        visual = F.normalize(updated_vision, dim=1, p=2)
        text = F.normalize(updated_text, dim=2, p=2)

        logits = torch.einsum("bc,bkc->bk", visual, text)
        logits = self.clip_model.logit_scale.exp().float() * logits
        return logits, visual, text

    def _apply_visual_prompt(
        self,
        global_feat: torch.Tensor,
        visual_embeddings: torch.Tensor,
        track_grad: bool,
    ) -> torch.Tensor:
        if not self.enable_mutual_prompting:
            return global_feat.float()

        bsz, channels, _, _ = visual_embeddings.shape

        with torch.no_grad():
            raw_prompt = self.prompt_learner()
            _, text_contexts_all = self.text_encoder(raw_prompt, self.prompt_learner.tokenized_prompts)
            text_contexts = text_contexts_all.expand(bsz, -1, -1, -1)[:, 0, : self.prompt_learner.n_ctx, :]

        global_f = global_feat.float()
        text_contexts_f = text_contexts.float().to(global_f.device)

        if track_grad:
            vis_prompt_diff = self.context_decoder(global_f.reshape(bsz, 1, channels), text_contexts_f)
        else:
            with torch.no_grad():
                vis_prompt_diff = self.context_decoder(global_f.reshape(bsz, 1, channels), text_contexts_f)

        vis_prompt_diff = vis_prompt_diff.reshape(bsz, channels)
        updated_vision = global_f + self.prompt_learner.gamma_v.float() * vis_prompt_diff
        return updated_vision

    def _compute_logits(self, image_embedding: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        image_features = F.normalize(image_embedding.float(), dim=-1)
        text_features = F.normalize(text_features.to(self.device).float(), dim=-1)
        logit_scale = self.clip_model.logit_scale.exp().float()
        return logit_scale * image_features @ text_features.t()

    @staticmethod
    def _tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
        # CLIP ViT token layout from internal transformer: [L, B, C].
        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor with 3 dims, got: {tokens.shape}")
        tokens = tokens.permute(1, 0, 2)  # [B, L, C]
        patch_tokens = tokens[:, 1:, :]
        side = int(math.sqrt(patch_tokens.shape[1]))
        if side * side != patch_tokens.shape[1]:
            raise ValueError(
                "Patch token count is not a perfect square, cannot reshape to feature map"
            )
        return patch_tokens.reshape(tokens.shape[0], side, side, tokens.shape[2]).permute(0, 3, 1, 2)
