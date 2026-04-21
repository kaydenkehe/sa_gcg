"""B1: GCG (Zou et al. 2023).

Faithful re-implementation rather than a wrapper around llm-attacks: gives us
a single source of truth for tokenizer handling, chat templating, and
suffix-span resolution that's identical across baselines and SA-GCG. The
core algorithm (gradient ranking → random sample → loss check) is small.
"""
from __future__ import annotations

import time

from ..data.registry import Behavior
from ..models.chat_template import build_input_ids
from ..utils.artifact import AttackResult
from ..utils.logging import get_logger
from ._common import init_suffix_ids
from .base import AttackBase
from .registry import register

LOG = get_logger(__name__)


@register("gcg")
class GCG(AttackBase):
    def run(self, behaviors):
        import torch

        cfg = self.config
        bundle = self.model_bundle
        model, tok = bundle.model, bundle.tokenizer
        device = next(model.parameters()).device

        suffix_ids = init_suffix_ids(bundle, cfg.init_suffix, cfg.suffix_length)
        suffix_ids_t = torch.tensor(suffix_ids, device=device, dtype=torch.long)

        loss_curve: list[float] = []
        t0 = time.time()
        for step in range(cfg.n_steps):
            # Pick a random behavior each step (universal) or the only one.
            beh = behaviors[step % len(behaviors)] if cfg.universal else behaviors[0]
            grad = self._token_gradients(beh, suffix_ids_t)
            top = grad.topk(cfg.top_k, dim=-1, largest=False).indices  # lower CE = better
            # Sample candidate replacements
            cand = self._sample_candidates(suffix_ids_t, top, cfg.batch_size)
            losses = self._batch_loss(beh, cand)
            best = int(losses.argmin().item())
            best_loss = float(losses[best].item())
            suffix_ids_t = cand[best]
            loss_curve.append(best_loss)
            if cfg.verbose and step % 20 == 0:
                LOG.info("[gcg %4d] loss=%.4f", step, best_loss)
            if best_loss < cfg.target_loss_threshold and not cfg.universal:
                break
            if cfg.wall_clock_budget_s and (time.time() - t0) > cfg.wall_clock_budget_s:
                break

        suffix_str = tok.decode(suffix_ids_t.tolist(), skip_special_tokens=False)
        return AttackResult(
            suffix_ids=suffix_ids_t.tolist(),
            suffix_str=suffix_str,
            wall_clock_s=time.time() - t0,
            n_steps=len(loss_curve),
            loss_curve=loss_curve,
            meta={"attack": self.name},
        )

    # --- internals --------------------------------------------------------

    def _token_gradients(self, behavior: Behavior, suffix_ids):
        """One-hot trick: differentiate loss w.r.t. one_hot(suffix_token)."""
        import torch
        import torch.nn.functional as F

        bundle = self.model_bundle
        model, tok = bundle.model, bundle.tokenizer
        device = next(model.parameters()).device
        embed_layer = model.get_input_embeddings()
        vocab_size = embed_layer.weight.shape[0]

        slot = build_input_ids(tok, bundle.spec, behavior.prompt, "x" * len(suffix_ids))  # placeholder length
        prefix_ids = torch.tensor(slot.prefix_ids, device=device, dtype=torch.long)
        postfix_ids = torch.tensor(slot.postfix_ids, device=device, dtype=torch.long)
        target_ids = torch.tensor(
            tok(behavior.target_prefix, add_special_tokens=False).input_ids,
            device=device,
            dtype=torch.long,
        )

        one_hot = F.one_hot(suffix_ids, num_classes=vocab_size).to(embed_layer.weight.dtype)
        one_hot.requires_grad_(True)

        emb_pre = embed_layer(prefix_ids)
        emb_post = embed_layer(postfix_ids)
        emb_tgt = embed_layer(target_ids)
        emb_suffix = one_hot @ embed_layer.weight
        emb_full = torch.cat([emb_pre, emb_suffix, emb_post, emb_tgt], dim=0).unsqueeze(0)

        prompt_len = emb_pre.shape[0] + emb_suffix.shape[0] + emb_post.shape[0]
        out = model(inputs_embeds=emb_full, use_cache=False)
        logits = out.logits[0, prompt_len - 1 : prompt_len - 1 + target_ids.shape[0]]
        loss = F.cross_entropy(logits, target_ids)
        (grad,) = torch.autograd.grad(loss, one_hot)
        return grad.detach()  # (L, V)

    def _sample_candidates(self, suffix_ids, top, batch_size):
        """For each of ``batch_size`` candidates, replace one position with a top-k pick."""
        import torch

        L, K = top.shape
        device = suffix_ids.device
        positions = torch.randint(0, L, (batch_size,), device=device)
        choices = torch.randint(0, K, (batch_size,), device=device)
        new_ids = top[positions, choices]
        cand = suffix_ids.unsqueeze(0).repeat(batch_size, 1)
        cand[torch.arange(batch_size, device=device), positions] = new_ids
        return cand  # (batch, L)

    def _batch_loss(self, behavior: Behavior, cand):
        """CE on target prefix for each row of ``cand``."""
        import torch
        import torch.nn.functional as F

        bundle = self.model_bundle
        model, tok = bundle.model, bundle.tokenizer
        device = next(model.parameters()).device
        embed_layer = model.get_input_embeddings()

        slot = build_input_ids(tok, bundle.spec, behavior.prompt, "x" * cand.shape[1])
        prefix_ids = torch.tensor(slot.prefix_ids, device=device, dtype=torch.long)
        postfix_ids = torch.tensor(slot.postfix_ids, device=device, dtype=torch.long)
        target_ids = torch.tensor(
            tok(behavior.target_prefix, add_special_tokens=False).input_ids,
            device=device,
            dtype=torch.long,
        )

        emb_pre = embed_layer(prefix_ids)
        emb_post = embed_layer(postfix_ids)
        emb_tgt = embed_layer(target_ids)

        losses = []
        # Process in chunks to avoid OOM
        chunk = max(1, min(cand.shape[0], 64))
        for i in range(0, cand.shape[0], chunk):
            ids = cand[i : i + chunk]  # (b, L)
            emb_suffix = embed_layer(ids)  # (b, L, d)
            b = ids.shape[0]
            emb_full = torch.cat(
                [
                    emb_pre.unsqueeze(0).expand(b, -1, -1),
                    emb_suffix,
                    emb_post.unsqueeze(0).expand(b, -1, -1),
                    emb_tgt.unsqueeze(0).expand(b, -1, -1),
                ],
                dim=1,
            )
            prompt_len = emb_pre.shape[0] + emb_suffix.shape[1] + emb_post.shape[0]
            with torch.no_grad():
                out = model(inputs_embeds=emb_full, use_cache=False)
            logits = out.logits[:, prompt_len - 1 : prompt_len - 1 + target_ids.shape[0]]
            per = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.unsqueeze(0).expand(b, -1).reshape(-1),
                reduction="none",
            ).reshape(b, -1).mean(dim=1)
            losses.append(per)
        return torch.cat(losses)  # (batch,)
