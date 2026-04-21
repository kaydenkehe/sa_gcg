"""SA-GCG: Soft Activation-Guided GCG (the headline attack of this paper).

Algorithm (Appendix E of EVAL_PLAN.md):

  φ ∈ R^{L × |V|}, init ~ N(0, 0.01)
  for t = 1..T_c:
      z = gumbel_softmax(φ, τ_t)               # (L, |V|)
      emb_suffix = z @ embed.weight             # (L, d)
      emb_full = [emb_pre, emb_suffix, emb_post, emb_target]
      with hook on layer ell*:
          forward(emb_full)
      L_CE  = CE on target span
      L_act = activation_projection_loss(scope, h[ell*][p*], r)
      if step==1:  freeze g_CE, g_act = ||grad_phi L_CE||, ||grad_phi L_act||
      L = L_CE / g_CE + λ * L_act / g_act
      Adam step on φ; anneal τ
  suffix_ids = argmax(φ)
  for t = 1..T_p: discrete GCG step on suffix_ids
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Sequence

from ..data.registry import Behavior
from ..direction.extractor import load_direction
from ..losses.activation_loss import (
    ActivationScope,
    AllLayerReaders,
    activation_projection_loss,
)
from ..losses.ce_loss import first_token_weighted_ce, target_ce_loss
from ..losses.gradnorm import GradNormSurrogates
from ..models.chat_template import build_input_ids
from ..models.hooks import ResidualReader
from ..schedule.slushy import make_schedule
from ..utils.artifact import AttackResult
from ..utils.logging import get_logger
from ._common import init_suffix_ids
from .base import AttackBase
from .registry import register

LOG = get_logger(__name__)


@register("sa_gcg")
class SAGCG(AttackBase):
    def run(self, behaviors: Sequence[Behavior]) -> AttackResult:
        import torch

        cfg = self.config
        bundle = self.model_bundle
        model, tok = bundle.model, bundle.tokenizer
        device = next(model.parameters()).device

        # ---- Direction ----------------------------------------------------
        r, ell_star, p_star = self._resolve_direction(model, device)

        layer_mod = model.model.layers if hasattr(model, "model") else model.layers
        embed_layer = model.get_input_embeddings()
        vocab_size = embed_layer.weight.shape[0]

        # ---- Phi (continuous parameterization) ----------------------------
        phi = torch.randn(
            cfg.suffix_length, vocab_size, device=device, dtype=torch.float32
        ) * 0.01
        phi.requires_grad_(True)
        opt = torch.optim.Adam([phi], lr=0.1)
        schedule = make_schedule(cfg.schedule, cfg.n_steps)

        # ---- Loss bookkeeping --------------------------------------------
        gradnorms = GradNormSurrogates()
        loss_curve: list[float] = []
        cosine_curve: list[float] = []
        t0 = time.time()

        scope = ActivationScope(cfg.activation_scope)

        # ---- Continuous phase --------------------------------------------
        for step in range(cfg.n_steps):
            tau = schedule(step)
            z = _gumbel_softmax(phi, tau)
            emb_suffix = z.to(embed_layer.weight.dtype) @ embed_layer.weight  # (L, d)

            # Pick a behavior (universal: round-robin; individual: fixed)
            beh = behaviors[step % len(behaviors)] if cfg.universal else behaviors[0]
            emb_full, prompt_len, target_ids = self._build_full_embeds(
                beh, emb_suffix, embed_layer, device
            )

            # Forward with appropriate readers
            if scope is ActivationScope.GLOBAL:
                with AllLayerReaders(layer_mod) as readers:
                    out = model(inputs_embeds=emb_full, use_cache=False)
                    h_by_layer = {i: r_.tensor for i, r_ in readers._readers.items()}
            else:
                with ResidualReader(layer_mod[ell_star]) as reader:
                    out = model(inputs_embeds=emb_full, use_cache=False)
                h_by_layer = {ell_star: reader.tensor}

            # CE loss on target span
            import torch.nn.functional as F

            logits = out.logits
            pred = logits[:, prompt_len - 1 : prompt_len - 1 + target_ids.shape[1]]
            if cfg.use_first_token_weight:
                per = F.cross_entropy(
                    pred.reshape(-1, pred.size(-1)),
                    target_ids.reshape(-1),
                    reduction="none",
                ).reshape(target_ids.shape)
                w = torch.ones_like(per)
                w[:, 0] = cfg.first_token_weight
                loss_ce = (per * w).sum() / w.sum()
            else:
                loss_ce = F.cross_entropy(
                    pred.reshape(-1, pred.size(-1)), target_ids.reshape(-1)
                )

            # Activation loss
            if cfg.composite_lambda > 0:
                loss_act = activation_projection_loss(
                    residual_by_layer={k: v.float() for k, v in h_by_layer.items()},
                    refusal_direction=r.float(),
                    scope=scope,
                    ell_star=ell_star,
                    p_star=p_star,
                )
            else:
                loss_act = torch.zeros((), device=device)

            # Optional perplexity penalty (§12 adaptive defense)
            if cfg.perplexity_penalty > 0:
                ppl_term = self._suffix_perplexity_proxy(z, embed_layer)
                loss_ce = loss_ce + cfg.perplexity_penalty * ppl_term

            # GradNorm freeze on first step
            if not gradnorms.frozen and cfg.composite_lambda > 0:
                gradnorms.capture_if_first(phi=phi, loss_ce=loss_ce, loss_act=loss_act)
                LOG.info(
                    "GradNorm captured: g_CE=%.4f g_act=%.4f", gradnorms.g_ce, gradnorms.g_act
                )

            loss = loss_ce / max(gradnorms.g_ce, 1e-8) + cfg.composite_lambda * loss_act / max(
                gradnorms.g_act, 1e-8
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_curve.append(float(loss.detach().item()))

            # Per-step cosine for diagnostics (cheap; one layer)
            with torch.no_grad():
                h_target = h_by_layer[ell_star][:, p_star, :].detach().to(torch.float32)
                d = r.to(torch.float32)
                cos = float(
                    (h_target @ d / (h_target.norm(dim=-1) * d.norm() + 1e-8)).mean().item()
                )
                cosine_curve.append(cos)

            if cfg.verbose and step % 20 == 0:
                LOG.info(
                    "[sa-gcg %4d] tau=%.3f L=%.4f L_CE=%.4f L_act=%.6f cos=%.3f",
                    step,
                    tau,
                    float(loss.item()),
                    float(loss_ce.item()),
                    float(loss_act.item()),
                    cos,
                )

            if cfg.wall_clock_budget_s and (time.time() - t0) > cfg.wall_clock_budget_s:
                LOG.info("Wall-clock budget reached at step %d", step)
                break

        # ---- Discretize ---------------------------------------------------
        with torch.no_grad():
            suffix_ids = phi.argmax(dim=-1).detach().cpu().tolist()

        # ---- Polish phase (discrete GCG steps) ----------------------------
        if cfg.polish_steps > 0:
            from .gcg import GCG

            polish_cfg = dataclasses.replace(
                cfg, n_steps=cfg.polish_steps, init_suffix=tok.decode(suffix_ids)
            )
            polish = GCG(self.model_bundle, polish_cfg)
            polish_result = polish.run(behaviors)
            suffix_ids = polish_result.suffix_ids
            loss_curve.extend(polish_result.loss_curve)

        suffix_str = tok.decode(suffix_ids, skip_special_tokens=False)
        return AttackResult(
            suffix_ids=list(suffix_ids),
            suffix_str=suffix_str,
            wall_clock_s=time.time() - t0,
            n_steps=len(loss_curve),
            loss_curve=loss_curve,
            cosine_curve=cosine_curve,
            meta={
                "attack": self.name,
                "scope": cfg.activation_scope,
                "lambda": cfg.composite_lambda,
                "polish": cfg.polish_steps,
                "ell_star": ell_star,
                "p_star": p_star,
                "g_CE": gradnorms.g_ce,
                "g_act": gradnorms.g_act,
                "schedule": cfg.schedule,
                "random_direction": cfg.random_direction,
            },
        )

    # ----------- internals -----------------------------------------------

    def _resolve_direction(self, model, device):
        import torch

        cfg = self.config
        if cfg.direction_path is None and cfg.composite_lambda > 0:
            raise ValueError("SA-GCG with composite_lambda > 0 needs --direction-path")
        if cfg.composite_lambda == 0:
            # Soft-GCG-CE: still need a direction for the cosine diagnostic;
            # use a unit vector in any direction.
            d_model = model.config.hidden_size
            r = torch.zeros(d_model, device=device)
            r[0] = 1.0
            return r, 0, -1
        rec = load_direction(cfg.direction_path)
        r = rec.direction.to(device)
        if cfg.random_direction:
            # §10.2 random-direction control
            torch.manual_seed(cfg.seed + 9999)
            r = torch.randn_like(r)
            r = r / r.norm().clamp_min(1e-8)
        return r, rec.ell_star, rec.p_star

    def _build_full_embeds(self, beh, emb_suffix, embed_layer, device):
        import torch

        bundle = self.model_bundle
        slot = build_input_ids(
            bundle.tokenizer, bundle.spec, beh.prompt, "x" * emb_suffix.shape[0]
        )
        prefix_ids = torch.tensor(slot.prefix_ids, device=device, dtype=torch.long)
        postfix_ids = torch.tensor(slot.postfix_ids, device=device, dtype=torch.long)
        target_ids = torch.tensor(
            [bundle.tokenizer(beh.target_prefix, add_special_tokens=False).input_ids],
            device=device,
            dtype=torch.long,
        )
        emb_pre = embed_layer(prefix_ids)
        emb_post = embed_layer(postfix_ids)
        emb_tgt = embed_layer(target_ids[0])
        emb_full = torch.cat([emb_pre, emb_suffix, emb_post, emb_tgt], dim=0).unsqueeze(0)
        prompt_len = emb_pre.shape[0] + emb_suffix.shape[0] + emb_post.shape[0]
        return emb_full, prompt_len, target_ids

    def _suffix_perplexity_proxy(self, z, embed_layer):
        """Cheap differentiable proxy for suffix perplexity under the model.

        We use the negative entropy of z averaged across positions: as
        ``phi`` becomes peaky around natural-language tokens that the base
        LM is comfortable predicting, this term drops. A more rigorous
        version forwards through a base LM and reads next-token CE; that
        requires loading a second model and is implemented in
        ``defenses.adaptive``.
        """
        import torch

        eps = 1e-8
        ent = -(z * torch.log(z + eps)).sum(dim=-1).mean()
        return ent


def _gumbel_softmax(logits, tau, *, hard: bool = False):
    """Standard Gumbel-softmax with explicit tau (no nn.functional dep)."""
    import torch
    import torch.nn.functional as F

    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
    y = (logits + g) / max(float(tau), 1e-6)
    y = F.softmax(y, dim=-1)
    if hard:
        idx = y.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
        y = (y_hard - y).detach() + y
    return y
