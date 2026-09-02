---
id: f-nmn-doctor-aborts-the-process-while-probing-mlx-in-headless-environments
kind: note
note_kind: finding
created: 2026-08-22T15:55:04Z
created_by: a-nmn-auditor-55x57h
about: "[[t-01M0N1M6AFM02BZN9S5FKKNQEN]]"
source_event: 01M0N2V2GV2BJMH6JC5FXCH2PH
---
# nmn doctor aborts the process while probing MLX in headless environments
Evidence: src/nmn/cli.py:318-327 imports each probe in-process, including mlx.core configured at line 81. Running python3 -m nmn doctor on this headless Apple host with MLX 0.31.1 terminates via uncaught Objective-C NSRangeException in mlx Metal Device construction (NSArray objectAtIndex 0 beyond bounds), so Python exception handling never runs and no report is printed. Impact: contradicts the CLI and nmn.doctor contract that missing/unusable backends never raise and makes one optional backend take down diagnostics for all others. tests/test_cli.py:47,254-259 and 281-290 invoke doctor in-process, so the suite itself aborts under this condition. Dedup: full GitHub issue list unavailable and targeted public search found no match. Acceptance: isolate unsafe backend probes in subprocesses with timeout/signal/exit handling or use non-import metadata checks; doctor must return all six statuses and exit normally when an MLX probe aborts.
