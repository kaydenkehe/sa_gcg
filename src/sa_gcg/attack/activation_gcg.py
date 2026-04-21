"""B3: AGCG — discrete GCG with activation-projection loss (no CE).

This is the prior Activation-Guided GCG paper's headline attack. We
implement it as plain GCG over the activation loss alone (no CE), so the
ablation against SA-GCG isolates the effect of (a) continuous optimization
and (b) keeping CE in the composite.
"""
from __future__ import annotations

import time

from ..losses.activation_loss import (
    ActivationScope,
    activation_projection_loss,
)
from ..models.chat_template import build_input_ids
from ..models.hooks import ResidualReader
from ..direction.extractor import load_direction
from ..utils.artifact import AttackResult
from ..utils.logging import get_logger
from ._common import init_suffix_ids
from .base import AttackBase
from .registry import register

LOG = get_logger(__name__)


@register("agcg")
class ActivationGCG(AttackBase):
    def run(self, behaviors):
        import torch

        cfg = self.config
        bundle = self.model_bundle
        model, tok = bundle.model, bundle.tokenizer
        device = next(model.parameters()).device

        if cfg.direction_path is None:
            raise ValueError("AGCG requires --direction-path")
        direction = load_direction(cfg.direction_path)
        r = direction.direction.to(device).to(model.dtype)
        ell_star = direction.ell_star
        layer_mod = model.model.layers if hasattr(model, "model") else model.layers
        target_layer = layer_mod[ell_star]

        suffix_ids = init_suffix_ids(bundle, cfg.init_suffix, cfg.suffix_length)
        suffix_ids_t = torch.tensor(suffix_ids, device=device, dtype=torch.long)
        loss_curve: list[float] = []
        t0 = time.time()

        embed_layer = model.get_input_embeddings()
        vocab_size = embed_layer.weight.shape[0]

        for step in range(cfg.n_steps):
            beh = behaviors[step % len(behaviors)] if cfg.universal else behaviors[0]
            slot = build_input_ids(tok, bundle.spec, beh.prompt, "x" * cfg.suffix_length)
            prefix_ids = torch.tensor(slot.prefix_ids, device=device, dtype=torch.long)
            postfix_ids = torch.tensor(slot.postfix_ids, device=device, dtype=torch.long)
            # Gradient via one-hot trick
            import torch.nn.functional as F

            one_hot = F.one_hot(suffix_ids_t, num_classes=vocab_size).to(embed_layer.weight.dtype)
            one_hot.requires_grad_(True)
            emb_suffix = one_hot @ embed_layer.weight
            emb_full = torch.cat(
                [embed_layer(prefix_ids), emb_suffix, embed_layer(postfix_ids)], dim=0
            ).unsqueeze(0)
            with ResidualReader(target_layer) as reader:
                model(inputs_embeds=emb_full, use_cache=False)
            h = reader.tensor
            loss = activation_projection_loss(
                residual_by_layer={ell_star: h},
                refusal_direction=r,
                scope=cfg.activation_scope,
                ell_star=ell_star,
                p_star=direction.p_star,
            )
            (grad,) = torch.autograd.grad(loss, one_hot)
            top = grad.detach().topk(cfg.top_k, dim=-1, largest=False).indices

            from .gcg import GCG

            cand = GCG._sample_candidates(self, suffix_ids_t, top, cfg.batch_size)
            losses = self._batch_loss_act(beh, cand, target_layer, r, ell_star, direction.p_star)
            best = int(losses.argmin().item())
            best_loss = float(losses[best].item())
            suffix_ids_t = cand[best]
            loss_curve.append(best_loss)
            if cfg.verbose and step % 20 == 0:
                LOG.info("[agcg %4d] act_loss=%.6f", step, best_loss)
            if cfg.wall_clock_budget_s and (time.time() - t0) > cfg.wall_clock_budget_s:
                break

        suffix_str = tok.decode(suffix_ids_t.tolist(), skip_special_tokens=False)
        return AttackResult(
            suffix_ids=suffix_ids_t.tolist(),
            suffix_str=suffix_str,
            wall_clock_s=time.time() - t0,
            n_steps=len(loss_curve),
            loss_curve=loss_curve,
            meta={"attack": self.name, "ell_star": ell_star, "scope": cfg.activation_scope},
        )

    def _batch_loss_act(self, behavior, cand, target_layer, r, ell_star, p_star):
        import torch

        bundle = self.model_bundle
        model, tok = bundle.model, bundle.tokenizer
        device = next(model.parameters()).device
        embed_layer = model.get_input_embeddings()
        slot = build_input_ids(tok, bundle.spec, behavior.prompt, "x" * cand.shape[1])
        prefix_ids = torch.tensor(slot.prefix_ids, device=device, dtype=torch.long)
        postfix_ids = torch.tensor(slot.postfix_ids, device=device, dtype=torch.long)

        emb_pre = embed_layer(prefix_ids)
        emb_post = embed_layer(postfix_ids)
        chunk = max(1, min(cand.shape[0], 32))
        all_losses = []
        for i in range(0, cand.shape[0], chunk):
            ids = cand[i : i + chunk]
            emb_suffix = embed_layer(ids)
            b = ids.shape[0]
            emb_full = torch.cat(
                [
                    emb_pre.unsqueeze(0).expand(b, -1, -1),
                    emb_suffix,
                    emb_post.unsqueeze(0).expand(b, -1, -1),
                ],
                dim=1,
            )
            with ResidualReader(target_layer) as reader, torch.no_grad():
                model(inputs_embeds=emb_full, use_cache=False)
            h = reader.tensor  # (b, T, d)
            from ..losses.activation_loss import activation_projection_loss

            losses = []
            for j in range(b):
                losses.append(
                    activation_projection_loss(
                        residual_by_layer={ell_star: h[j : j + 1]},
                        refusal_direction=r,
                        scope=self.config.activation_scope,
                        ell_star=ell_star,
                        p_star=p_star,
                    )
                )
            all_losses.append(torch.stack(losses))
        return torch.cat(all_losses)
