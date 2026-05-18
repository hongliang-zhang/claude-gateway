# Local ZAI Claude Code Proxy

This local proxy exposes an Anthropic-compatible endpoint for Claude Code:

```text
http://127.0.0.1:8082/v1/messages
```

It forwards requests to:

```text
https://router.z.ai/api/v1/chat/completions
```

Default model mapping:

```text
opus   -> claude-sonnet-4-6
sonnet -> claude-sonnet-4-6
haiku  -> claude-sonnet-4-6
```

## Start

```bash
cd /Users/zhanghongliang/Documents/claude-code-proxy-self-use/claude-code-proxy
export ZAI_API_KEY='your-zai-api-key'
./run-zai-local.sh
```

`run-zai-local.sh` sets `PROVIDER_COMPAT=zai` by default. This keeps Claude Desktop
text-model traffic compatible with Z.AI by flattening Claude multimodal content blocks
into text-only messages before forwarding them upstream.

## Optional PostHog Monitoring

Use a PostHog project token, not a personal API key:

```bash
export POSTHOG_API_KEY='phc_your_project_token'
export POSTHOG_HOST='https://us.i.posthog.com'
export POSTHOG_LOGS_ENABLED=true
export POSTHOG_SERVICE_NAME='claude-code-proxy'
```

The proxy sends traffic events such as `gateway request`, `gateway completion`,
and `gateway token count`. When logs are enabled, upstream API errors are also
sent to PostHog Logs using OTLP HTTP.

## Use With Claude Code

In another terminal:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8082 \
ANTHROPIC_API_KEY=local-only \
claude --model claude-sonnet-4-6
```

## Health Check

```bash
curl http://127.0.0.1:8082/health
```
