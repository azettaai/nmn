---
id: d-use-the-pixi-python-interpreter-for-jax-keras-and-pytest-audit-commands
kind: note
note_kind: decision
created: 2026-08-22T21:00:56Z
created_by: a-root
about: "[[t-01M0N7AN7XAXWW4EN9R4DCFCRW]]"
---
# Use the pixi Python interpreter for JAX Keras and pytest audit commands
## Chose
Use the pixi Python interpreter for JAX Keras and pytest audit commands
## Rejected
Use python3 resolved from the spawned shell PATH
## Because
The spawned shell resolves python3 to Homebrew Python without JAX Flax or Keras, while /Users/tahabsn/.pixi/bin/python3 contains the repository test environment.
