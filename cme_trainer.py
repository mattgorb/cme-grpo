"""Custom GRPOTrainer that uses per-token CME advantages.

Overrides advantage computation to use token-level rewards from the CME reward fn
rather than scalar sequence-level advantages.

For each position t, advantage_{i,t} = (r_{i,t} - mean_t) / (std_t + 1e-8)
where mean_t and std_t are computed across the G responses at that position.
Shorter responses are padded with zero advantage.
"""

from __future__ import annotations

from typing import Optional

import torch
from trl import GRPOTrainer


class CMETokenLevelGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that substitutes per-token advantages during loss computation.

    Assumes the reward_fn (first in reward_funcs) has `last_token_rewards`:
    a list of 1-D tensors, one per completion in the last reward_fn call,
    aligned to generator token positions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_reward_fn = None
        for rf in (self.reward_funcs if isinstance(self.reward_funcs, list) else [self.reward_funcs]):
            if callable(rf) and getattr(rf, "token_level", False):
                self._token_reward_fn = rf
                break

    def _compute_token_advantages(self, completion_ids: torch.Tensor, tokenizer, num_generations: int) -> Optional[torch.Tensor]:
        """Build [N, T] advantage tensor from the stashed token rewards.

        Groups of `num_generations` consecutive completions are normalized per-position.
        Looks up per-completion rewards by decoded text to handle grad accumulation.
        """
        rf = self._token_reward_fn
        if rf is None or not hasattr(rf, "completion_to_tokens"):
            return None
        cache = rf.completion_to_tokens
        if not cache:
            return None

        N, T = completion_ids.shape
        device = completion_ids.device

        decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        adv = torch.zeros(N, T, device=device, dtype=torch.float32)
        hits = 0
        for i, text in enumerate(decoded):
            r = cache.get(text)
            if r is None:
                # try stripping whitespace differences
                continue
            r = r.to(device=device, dtype=torch.float32)
            L = min(r.numel(), T)
            adv[i, :L] = r[:L]
            hits += 1

        if hits == 0:
            return None
        # Already normalized per-position inside reward_fn.
        return adv

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Drop-in replacement for GRPOTrainer.compute_loss using token-level advantages.

        Falls back to parent implementation if per-token rewards aren't available.
        """
        if self._token_reward_fn is None or self._token_reward_fn.last_token_rewards is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        num_generations = self.args.num_generations

        tokenizer = self.processing_class
        token_adv = self._compute_token_advantages(completion_ids, tokenizer, num_generations)
        if token_adv is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        # Always use manual policy-gradient path so we control the denominator.
        # With answer-only CME, most positions have advantage=0; averaging over all
        # completion tokens dilutes the gradient ~50x. Divide by the count of
        # non-zero-advantage tokens instead so the answer-token signal isn't washed out.
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]

        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attn_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        outputs = model(input_ids=full_ids, attention_mask=attn_mask)
        logits = outputs.logits[:, :-1, :]
        targets = full_ids[:, 1:]

        P = prompt_ids.shape[1]
        comp_logits = logits[:, P - 1 :, :]
        comp_targets = targets[:, P - 1 :]
        log_probs = torch.log_softmax(comp_logits, dim=-1)
        per_token_logp = log_probs.gather(-1, comp_targets.unsqueeze(-1)).squeeze(-1)

        mask = completion_mask.float()
        T = min(per_token_logp.shape[1], token_adv.shape[1], mask.shape[1])
        per_token_logp = per_token_logp[:, :T]
        adv = token_adv[:, :T]
        mask = mask[:, :T]

        active = ((adv != 0).float() * mask)
        denom = active.sum().clamp(min=1.0)
        loss = -(adv * per_token_logp * mask).sum() / denom
        return loss
