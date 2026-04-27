"""Cross-model perplexity (CME) reward.

Two modes:
- sequence-level: scalar reward per response (mean CE over response tokens)
- token-level: per-token reward tensor aligned to generator token positions

For token-level, we tokenize the response with both tokenizers and use character
offsets to map verifier per-token CME back onto generator token positions.
"""

from __future__ import annotations

import math
from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _find_boxed_span(text: str) -> Optional[tuple]:
    """Return (start, end) char offsets of the content inside the last \\boxed{...}.

    Handles nested braces. Returns None if not found or unbalanced.
    """
    key = "\\boxed{"
    idx = text.rfind(key)
    if idx == -1:
        return None
    start = idx + len(key)
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return (start, i)
        i += 1
    return None


def _per_token_metric(logits: torch.Tensor, shift_labels: torch.Tensor, use_pe: bool) -> torch.Tensor:
    """Return a [B, T] per-position scalar tensor.

    use_pe=False: cross-entropy of the actual label token -log p(label).
    use_pe=True : predictive entropy -sum_v p_v log p_v over the full vocab.
    Chunked over T to cap peak memory at [1, chunk, V] instead of [1, T, V].
    """
    if not use_pe:
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(shift_labels.shape)
    T = logits.shape[1]
    out = torch.empty(shift_labels.shape, device=logits.device, dtype=torch.float32)
    chunk = 256
    for s in range(0, T, chunk):
        e = s + chunk
        lp = torch.log_softmax(logits[:, s:e, :], dim=-1)
        out[:, s:e] = -(lp.exp() * lp).sum(dim=-1).float()
    return out


class CMERewardModel:
    def __init__(
        self,
        verifier_name: str,
        device: Optional[str] = None,
        max_length: int = 2048,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.max_length = max_length
        self.device = device or ("cuda:1" if torch.cuda.device_count() > 1 else
                                 ("cuda:0" if torch.cuda.is_available() else "cpu"))

        self.tokenizer = AutoTokenizer.from_pretrained(verifier_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            verifier_name,
            device_map={"": self.device},
            torch_dtype=dtype,
        )
        self.model.eval()

    @torch.no_grad()
    def score(
        self,
        prompts: List[str],
        responses: List[str],
        token_level: bool = False,
        gen_tokenizer=None,
        answer_only: bool = False,
        no_box_penalty: float = 5.0,
        reward_metric: str = "entropy",
        answer_weight: Optional[float] = None,
    ) -> Union[List[float], List[torch.Tensor]]:
        """Compute the CME reward.

        reward_metric:
          "entropy"            cross-entropy -log p(label) in nats (misnamed;
                               kept for backward compat).
          "perplexity"         exp(CE).
          "predictive_entropy" -sum_v p_v log p_v over vocab at each position
                               (does not depend on the actual label token).
        All are negated so lower verifier-surprise => higher reward.

        answer_weight: if not None, use the blended reward
            r = -[(1-w)*mean(CE over full response) + w*mean(CE over \\boxed{} span)]
          w=0.0 is equivalent to answer_only=False; w=1.0 to answer_only=True.
          Overrides `answer_only` when set. Sequence-level only.
        """
        if reward_metric not in ("entropy", "perplexity", "predictive_entropy"):
            raise ValueError(
                f"reward_metric must be 'entropy', 'perplexity', or 'predictive_entropy', "
                f"got {reward_metric!r}"
            )
        if answer_weight is not None and token_level:
            raise ValueError("answer_weight is not supported in token_level mode")
        if answer_weight is not None and not (0.0 <= answer_weight <= 1.0):
            raise ValueError(f"answer_weight must be in [0,1], got {answer_weight}")
        use_ppl = reward_metric == "perplexity"
        use_pe = reward_metric == "predictive_entropy"
        blended = answer_weight is not None
        rewards: List = []
        for prompt, response in zip(prompts, responses):
            if not response or not response.strip():
                rewards.append(-5.0 if not token_level else torch.tensor([-5.0]))
                continue

            # Need offsets whenever we have to restrict to the answer span.
            need_offsets = token_level or answer_only or blended
            prompt_enc = self.tokenizer(
                prompt, add_special_tokens=True, return_tensors="pt",
            )
            response_enc = self.tokenizer(
                response, add_special_tokens=False, return_tensors="pt",
                return_offsets_mapping=need_offsets,
            )
            response_offsets = (
                response_enc.offset_mapping[0].tolist() if need_offsets else None
            )

            answer_span = _find_boxed_span(response) if (answer_only or blended) else None

            prompt_ids = prompt_enc.input_ids[0]
            response_ids = response_enc.input_ids[0]

            if response_ids.numel() == 0:
                rewards.append(-5.0 if not token_level else torch.tensor([-5.0]))
                continue

            full_ids = torch.cat([prompt_ids, response_ids], dim=0)
            if full_ids.shape[0] > self.max_length:
                overflow = full_ids.shape[0] - self.max_length
                full_ids = full_ids[overflow:]
                prompt_len = max(1, prompt_ids.shape[0] - overflow)
                if token_level and response_offsets is not None:
                    # Response tokens not dropped; overflow comes from prompt side.
                    pass
            else:
                prompt_len = prompt_ids.shape[0]

            input_ids = full_ids.unsqueeze(0).to(self.device)
            labels = input_ids.clone()
            labels[0, :prompt_len] = -100

            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            if token_level:
                per_tok_ce = _per_token_metric(logits, shift_labels, use_pe)
                # Grab only response positions (where labels != -100).
                mask = shift_labels[0] != -100
                verifier_token_ce = per_tok_ce[0][mask].cpu()  # [response_len]

                if answer_only and answer_span is None:
                    # No \boxed{} — assign constant large CE to every response token
                    # so this completion gets a distinctly negative advantage after
                    # per-position normalization across the group.
                    verifier_token_ce = torch.full_like(
                        verifier_token_ce, float(no_box_penalty)
                    )
                elif not answer_only and _find_boxed_span(response) is None:
                    # Mirror sequence-level behavior: when scoring the full response
                    # and no \boxed{} is present, add the penalty to every token CE
                    # so the completion is consistently penalized for missing format.
                    verifier_token_ce = verifier_token_ce + float(no_box_penalty)
                elif answer_span is not None:
                    # Zero out CE for verifier tokens outside the \boxed{...} span.
                    a, b = answer_span
                    keep = torch.zeros_like(verifier_token_ce)
                    v_offsets = response_offsets[: verifier_token_ce.numel()]
                    for j, (va, vb) in enumerate(v_offsets):
                        if va == vb:
                            continue
                        if vb > a and va < b:
                            keep[j] = 1.0
                    verifier_token_ce = verifier_token_ce * keep

                # Align verifier tokens to generator tokens.
                if gen_tokenizer is not None:
                    gen_enc = gen_tokenizer(
                        response, add_special_tokens=False, return_tensors="pt",
                        return_offsets_mapping=True,
                    )
                    gen_offsets = gen_enc.offset_mapping[0].tolist()
                    aligned = _align_ce_to_generator(
                        verifier_token_ce, response_offsets, gen_offsets
                    )
                    if use_ppl:
                        aligned = torch.exp(aligned)
                    rewards.append(-aligned)
                else:
                    tok_rw = verifier_token_ce.exp() if use_ppl else verifier_token_ce
                    rewards.append(-tok_rw)
            else:
                per_tok_ce = _per_token_metric(logits, shift_labels, use_pe)
                ce_row = per_tok_ce[0]
                label_mask = shift_labels[0] != -100

                if blended:
                    full_val = ce_row[label_mask].mean().item() if label_mask.any() else float(no_box_penalty)
                    if answer_span is None:
                        ans_val = float(no_box_penalty)
                        full_val = full_val + float(no_box_penalty)
                    else:
                        a, b = answer_span
                        span_mask = torch.zeros_like(ce_row, dtype=torch.bool)
                        resp_start = prompt_len - 1
                        for j, (va, vb) in enumerate(response_offsets):
                            pos = resp_start + j
                            if pos < 0 or pos >= span_mask.shape[0]: continue
                            if va == vb: continue
                            if vb > a and va < b: span_mask[pos] = True
                        final_mask = label_mask & span_mask
                        ans_val = ce_row[final_mask].mean().item() if final_mask.any() else float(no_box_penalty)
                    w = float(answer_weight)
                    combined = (1.0 - w) * full_val + w * ans_val
                    val = math.exp(combined) if use_ppl else combined
                    rewards.append(-val)
                    continue

                if answer_only and answer_span is None:
                    penalty = math.exp(float(no_box_penalty)) if use_ppl else float(no_box_penalty)
                    rewards.append(-penalty)
                    continue
                if answer_span is not None:
                    a, b = answer_span
                    span_mask = torch.zeros_like(ce_row, dtype=torch.bool)
                    # shift_labels drops position 0; response tokens start at index prompt_len - 1
                    # in the shifted tensor. Walk response offsets and mark span overlaps.
                    resp_start = prompt_len - 1
                    v_offsets = response_offsets
                    for j, (va, vb) in enumerate(v_offsets):
                        pos = resp_start + j
                        if pos < 0 or pos >= span_mask.shape[0]:
                            continue
                        if va == vb:
                            continue
                        if vb > a and va < b:
                            span_mask[pos] = True
                    final_mask = label_mask & span_mask
                    if final_mask.any():
                        loss = ce_row[final_mask].mean()
                    else:
                        # No boxed answer found; fall back to full response CE.
                        loss = ce_row[label_mask].mean()
                else:
                    loss = ce_row[label_mask].mean()
                # When scoring the full response (answer_only=False), add an additive
                # penalty if the response has no \boxed{}. Without this the CME signal
                # doesn't consistently reward formatting, so a base model can drift
                # into producing coherent math without ever boxing the answer.
                if not answer_only and _find_boxed_span(response) is None:
                    loss = loss + float(no_box_penalty)
                val = math.exp(loss.item()) if use_ppl else loss.item()
                rewards.append(-val)

        return rewards


def _align_ce_to_generator(
    verifier_ce: torch.Tensor,
    verifier_offsets: list,
    gen_offsets: list,
) -> torch.Tensor:
    """Map verifier per-token CE onto generator token positions via character overlap.

    For each generator token covering [a, b], average verifier-token CE values for
    verifier tokens with any overlap with [a, b]. Fallback to nearest if no overlap.
    """
    n_gen = len(gen_offsets)
    out = torch.zeros(n_gen)
    # Skip special tokens whose offsets are (0,0).
    v_offsets = verifier_offsets[: len(verifier_ce)]
    for i, (ga, gb) in enumerate(gen_offsets):
        if ga == gb:  # empty gen token (special)
            out[i] = 0.0
            continue
        overlapping = []
        for j, (va, vb) in enumerate(v_offsets):
            if va == vb:
                continue
            if vb > ga and va < gb:  # overlap
                overlapping.append(verifier_ce[j].item())
        if overlapping:
            out[i] = sum(overlapping) / len(overlapping)
        else:
            # Find nearest verifier token by distance.
            best_j, best_d = 0, float("inf")
            mid = (ga + gb) / 2
            for j, (va, vb) in enumerate(v_offsets):
                if va == vb:
                    continue
                v_mid = (va + vb) / 2
                d = abs(v_mid - mid)
                if d < best_d:
                    best_d = d
                    best_j = j
            out[i] = verifier_ce[best_j].item()
    return out


def build_cme_reward_fn(
    reward_model: CMERewardModel,
    token_level: bool = False,
    gen_tokenizer=None,
    answer_only: bool = False,
    no_box_penalty: float = 5.0,
    reward_metric: str = "entropy",
    answer_weight: Optional[float] = None,
):
    """Return a TRL GRPO-compatible reward function.

    In token-level mode, stashes per-token rewards in reward_fn.last_token_rewards
    (list of tensors, one per completion) and returns mean reward per completion
    as the scalar GRPO expects.
    """

    mode = "token-level" if token_level else "sequence-level"
    if answer_weight is not None:
        mode += f" (blended w={answer_weight})"
    elif answer_only:
        mode += " (answer-only)"

    def reward_fn(prompts, completions, **kwargs) -> List[float]:
        prompt_texts: List[str] = []
        completion_texts: List[str] = []
        for p, c in zip(prompts, completions):
            if isinstance(p, list):
                p = "\n".join(m.get("content", "") for m in p)
            if isinstance(c, list):
                c = "\n".join(m.get("content", "") for m in c)
            prompt_texts.append(p)
            completion_texts.append(c)

        from eval import is_correct
        gold_answers = kwargs.get("gold_answer", [None] * len(completion_texts))
        unique_golds = list(dict.fromkeys(g for g in gold_answers if g))
        print(f"  gold: {unique_golds}")
        extracted = []
        for c, gold in zip(completion_texts, gold_answers):
            span = _find_boxed_span(c)
            ans = c[span[0]:span[1]] if span else "<NO_BOX>"
            if gold:
                pred = ans if ans != "<NO_BOX>" else None
                tag = "✓" if is_correct(pred, gold) else "✗"
                ans = f"{ans} {tag}"
            extracted.append(ans)
        print(f"  extracted \\boxed{{}}: {extracted}")

        raw = reward_model.score(
            prompt_texts, completion_texts,
            token_level=token_level, gen_tokenizer=gen_tokenizer,
            answer_only=answer_only, no_box_penalty=no_box_penalty,
            reward_metric=reward_metric,
            answer_weight=answer_weight,
        )

        if token_level:
            token_rewards = []
            scalar_rewards = []
            keys = []
            for c, r in zip(completion_texts, raw):
                if isinstance(r, float):
                    token_rewards.append(torch.tensor([r]))
                    scalar_rewards.append(r)
                else:
                    token_rewards.append(r)
                    if answer_only:
                        # Average over non-zero (answer-span) positions only, so the
                        # scalar isn't diluted by zeroed-out reasoning tokens.
                        nonzero = r[r != 0]
                        scalar_rewards.append(
                            float(nonzero.mean().item()) if nonzero.numel() > 0 else 0.0
                        )
                    else:
                        scalar_rewards.append(float(r.mean().item()))
                keys.append(c)

            # Pre-normalize per-position across the G responses so compute_loss
            # just looks them up. Handles variable-length responses via right-padding.
            max_len = max(t.numel() for t in token_rewards)
            padded = torch.stack([
                torch.nn.functional.pad(t, (0, max_len - t.numel()), value=float("nan"))
                for t in token_rewards
            ])  # [N, max_len] with NaN for missing positions
            # per-position mean/std ignoring NaN
            mask = ~torch.isnan(padded)
            counts = mask.sum(dim=0, keepdim=True).clamp(min=1).float()
            filled = torch.where(mask, padded, torch.zeros_like(padded))
            mean = filled.sum(dim=0, keepdim=True) / counts
            var = (torch.where(mask, (padded - mean) ** 2, torch.zeros_like(padded))).sum(dim=0, keepdim=True) / counts.clamp(min=1)
            std = var.sqrt()
            normalized = torch.where(mask, (padded - mean) / (std + 1e-4), torch.zeros_like(padded))

            # unpack back to original lengths
            normalized_rewards = []
            for i, t in enumerate(token_rewards):
                normalized_rewards.append(normalized[i, : t.numel()].clone())

            reward_fn.last_token_rewards = normalized_rewards
            reward_fn.completion_to_tokens = dict(zip(keys, normalized_rewards))
            print(f"  rewards: {scalar_rewards}")
            return scalar_rewards
        else:
            print(f"  rewards: {raw}")
            return raw

    reward_fn.last_token_rewards = None
    reward_fn.token_level = token_level
    return reward_fn
