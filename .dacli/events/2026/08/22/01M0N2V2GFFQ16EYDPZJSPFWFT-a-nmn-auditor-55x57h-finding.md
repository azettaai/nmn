---
id: 01M0N2V2GFFQ16EYDPZJSPFWFT
kind: event
schema_version: 1
event_kind: finding
created: 2026-08-22T15:53:42Z
created_by: a-nmn-auditor-55x57h
about: "[[t-01M0N1M6AFM02BZN9S5FKKNQEN]]"
origin: agent
applied: true
checksum: sha256:7d9db209c8d88d80959606f814d0d6d7bc4035cb1208ade74438083511e446b7
---
MLX fused YatNMN detaches learnable epsilon from autodiff

Decisive code path: src/nmn/mlx/nmn.py:243-248 converts nn.softplus(epsilon_param)[0] to a Python float before calling fused_yat_score. The fused API at src/nmn/mlx/fused.py:178-212 accepts epsilon as float and rebuilds an array, so epsilon_param is outside the fused autodiff graph even though the custom VJP computes an internal eps gradient. Impact: fused=True with learnable_epsilon=True either errors on tracer-to-float conversion or silently yields no epsilon_param gradient, violating advertised lazy/trainability semantics. Tests/test_mlx/test_fused.py covers module gradients only with constant epsilon; learnable epsilon tests use non-fused mode. MLX runtime aborts during Metal initialization in this headless host, so evidence is source-decisive. Dedup: full GitHub list unavailable and targeted public search found no match. Acceptance: pass epsilon as an MLX array through fused_yat_score and return its VJP to epsilon_param; fused/eager outputs and kernel/bias/alpha/epsilon gradients must match for learnable epsilon under value_and_grad and compile.
