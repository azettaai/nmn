---
id: rt-codex-impl
kind: runtime
created: 2026-08-29T17:48:40Z
created_by: a-root
name: codex-impl
harness: codex
binary: /Applications/ChatGPT.app/Contents/Resources/codex
invoke_mode: stdin
invoke_args: "[exec, --json, --ephemeral, --color, never, --sandbox, workspace-write]"
global_args: "[--ask-for-approval, never]"
sandbox_ro_args: "[--sandbox, read-only]"
env_passthrough: "[HOME, PATH, USER, LOGNAME, TMPDIR, CODEX_HOME]"
model_flag: --model
usage_format: codex-jsonl
behavioral_preflight: codex-exec-json-v2
context_provenance: "[user-config=enumerated, repository-instructions=enumerated, global-skills=enumerated, plugins-extensions=enumerated, mcp-servers=enumerated, environment-config=isolated]"
---
# codex-impl
Flags here are assumptions until `dacli runtime doctor` verifies them against the installed binary.
