# AlpacaEval 2.0 Evaluation

Publication-grade quality evaluation using [`eval_alpacaeval2.py`](eval_alpacaeval2.py).
Generates model responses on the 805 standard AlpacaEval 2.0 prompts and runs the
GPT-4-Turbo judge for length-controlled winrate against the GPT-4-Turbo reference.

## Setup (once)

```bash
pip install alpaca-eval
export OPENAI_API_KEY=sk-...    # needed for the judge
```

## Recommended workflow (two stages)

### Stage 1 — Generate all model responses (free, local, ~1.5–2 hr on A6000)

```bash
python eval_alpacaeval2.py \
    --config config_quality1_token_level.yaml \
    --checkpoint ./outputs/cme-grpo-quality-qwen0.5b-token-level/checkpoint-best \
    --also-baselines
```

Writes three JSON files into `outputs/.../alpaca_eval_2/`:

- `{checkpoint_basename}.json` — your finetuned model
- `base-Qwen_Qwen2.5-0.5B.json`
- `instruct-Qwen_Qwen2.5-0.5B-Instruct.json`

Each contains the 805 AlpacaEval instructions paired with your model's outputs in
the format AlpacaEval expects.

### Stage 2 — Run the judge (costs ~$30–45 total for all 3 models)

```bash
python eval_alpacaeval2.py \
    --config config_quality1_token_level.yaml \
    --checkpoint ./outputs/cme-grpo-quality-qwen0.5b-token-level/checkpoint-best \
    --also-baselines --run-eval
```

Because stage 1's outputs already exist, this **skips generation** and only
invokes `alpaca_eval evaluate` on each JSON. Leaderboards write to
`{model_name}_leaderboard/` subdirectories.

## What you get

For each model, a CSV and summary with:

- **`length_controlled_winrate`** — the headline number; winrate vs
  GPT-4-Turbo reference, corrected for length bias.
- **`win_rate`** — raw (length-biased) winrate.
- **`standard_error`** — 95% confidence interval based on 805 prompts.
- **`avg_length`** — sanity check that your model isn't just winning by being
  verbose.

Example output shape (numbers illustrative):

```
                                       win_rate  length_controlled_winrate  standard_error
cme-grpo-0.5b-token-best                 12.4                    11.8              1.2
base-Qwen_Qwen2.5-0.5B                    1.2                     0.9              0.3
instruct-Qwen_Qwen2.5-0.5B-Instruct       7.8                     7.5              0.9
```

(0.5B models vs GPT-4-Turbo will score low in absolute terms — the point is the
*relative* ordering. A paper-quality claim looks like *"CME-GRPO's LC-winrate
is 11.8 vs the base model's 0.9, a 13× improvement, p < 0.01."*)

## Gotchas

1. **OOM during generation** — lower `--batch-size` (default 4) or
   `--max-new-tokens` (default 2048). The script resumes automatically by
   skipping any JSON that already exists.
2. **Dataset download fails** — make sure `datasets >= 2.14`; `trust_remote_code=True`
   is already set in the script.
3. **Judge cost budget** — each model costs ~$12 in GPT-4-Turbo judgments
   (805 × ~1K tokens each). Three models ≈ $35–45. For a cheaper option, use
   the GPT-4o-mini annotator:
   ```bash
   alpaca_eval evaluate --model_outputs file.json \
       --annotators_config alpaca_eval_gpt4o_mini_fn
   ```
   Cheaper (~$2/model) but less standard and slightly lower agreement with
   the official leaderboard.
4. **Pairwise among your 3 models** is not what AlpacaEval 2.0 measures by
   default — it compares each model vs a fixed GPT-4-Turbo reference. For
   pairwise (yours vs base vs instruct directly), use
   [`eval_alpacaeval.py`](eval_alpacaeval.py) instead.

## Single-model shortcut

If you only want to evaluate your trained checkpoint (skip baselines):

```bash
# generate
python eval_alpacaeval2.py \
    --config config_quality1_token_level.yaml \
    --checkpoint ./outputs/.../checkpoint-best

# then judge
python eval_alpacaeval2.py \
    --config config_quality1_token_level.yaml \
    --checkpoint ./outputs/.../checkpoint-best --run-eval
```

## Manual invocation (if you prefer)

You don't have to pass `--run-eval` through the script. The JSON files are
standard AlpacaEval format, so you can always run the CLI yourself:

```bash
alpaca_eval evaluate --model_outputs outputs/.../alpaca_eval_2/checkpoint-best.json
```
