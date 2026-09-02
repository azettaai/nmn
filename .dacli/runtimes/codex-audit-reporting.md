---
id: rt-codex-audit-reporting
kind: runtime
created: 2026-08-22T15:38:12Z
created_by: a-root
name: codex-audit-reporting
binary: /Applications/ChatGPT.app/Contents/Resources/codex
invoke_mode: stdin
invoke_args: "[--ask-for-approval, never, exec, --json, --ephemeral, --color, never, --sandbox, workspace-write]"
env_passthrough: "[HOME, PATH, USER, LOGNAME, TMPDIR, CODEX_HOME]"
model_flag: --model
context_provenance: "[user-config=unsupported, repository-instructions=enumerated, global-skills=unsupported, plugins-extensions=unsupported, mcp-servers=unsupported, environment-config=unsupported]"
---
# codex-audit-reporting
Flags here are assumptions until `dacli runtime doctor` verifies them against the installed binary.
