#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ZAI_API_KEY:-}" ]]; then
  echo "Missing ZAI_API_KEY. Run: export ZAI_API_KEY='your-key'" >&2
  exit 1
fi

export OPENAI_API_KEY="${ZAI_API_KEY}"
export GATEWAY_AUTH_MODE="${GATEWAY_AUTH_MODE:-shared}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://router.z.ai/api/v1}"
export BIG_MODEL="${BIG_MODEL:-claude-sonnet-4-6}"
export MIDDLE_MODEL="${MIDDLE_MODEL:-claude-sonnet-4-6}"
export SMALL_MODEL="${SMALL_MODEL:-claude-sonnet-4-6}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8082}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export MAX_TOKENS_LIMIT="${MAX_TOKENS_LIMIT:-8192}"
export MIN_TOKENS_LIMIT="${MIN_TOKENS_LIMIT:-1}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
export PROVIDER_COMPAT="${PROVIDER_COMPAT:-zai}"

exec uv run claude-code-proxy
