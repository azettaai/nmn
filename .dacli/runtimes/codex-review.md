---
id: rt-codex-review
kind: runtime
created: 2026-08-22T15:30:32Z
created_by: a-root
name: codex-review
binary: codex
invoke_mode: stdin
invoke_args: "[exec, --json, --ephemeral, --color, never, --sandbox, read-only]"
global_args: "[--ask-for-approval, never]"
sandbox_ro_args: "[--sandbox, read-only]"
env_passthrough: "[HOME, PATH, USER, LOGNAME, TMPDIR, CODEX_HOME]"
model_flag: --model
usage_format: codex-jsonl
context_provenance: "[user-config=enumerated, repository-instructions=enumerated, global-skills=enumerated, plugins-extensions=enumerated, mcp-servers=enumerated, environment-config=isolated]"
---
# codex-review
Flags here are assumptions until `dacli runtime doctor` verifies them against the installed binary.
