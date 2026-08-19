#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-numerical_rating/configs/kindness_full8_repaired.yaml}"
AUDITOR="${AUDITOR:-openrouter/openai/gpt-5.5}"
JUDGE="${JUDGE:-openrouter/anthropic/claude-opus-4.8}"
TRAJECTORIES="${TRAJECTORIES:-1}"
PARALLEL_SAMPLES="${PARALLEL_SAMPLES:-5}"

targets=(
  "openrouter/anthropic/claude-sonnet-4.6"
  "openrouter/openai/gpt-4.1"
  "openrouter/google/gemini-2.5-pro"
  "openrouter/x-ai/grok-4.3"
  "openrouter/deepseek/deepseek-chat-v3-0324"
  "openrouter/qwen/qwen3-235b-a22b-2507"
  "openrouter/moonshotai/kimi-k2-0905"
  "openrouter/meta-llama/llama-4-maverick"
)

for target in "${targets[@]}"; do
  ./petri_multiturn/run_constitution_petri.sh \
    -T "config=$CONFIG" \
    --trajectories "$TRAJECTORIES" \
    --parallel-samples "$PARALLEL_SAMPLES" \
    --model-role "auditor=$AUDITOR" \
    --model-role "target=$target" \
    --model-role "judge=$JUDGE" \
    --no-fail-on-error \
    --continue-on-fail \
    "$@"
done
