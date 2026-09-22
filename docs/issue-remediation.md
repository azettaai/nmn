# Open-issue remediation evidence

This change addresses the open issue set inspected on 2026-09-21. Issue closure
must follow CI and merge; this document does not claim that a local check proves
a production publication or live mirror synchronization.

| Issue | Implementation / evidence | Remaining gate |
| --- | --- | --- |
| #170 | Compatible `image-size` 2.0.4 lock update; empty high-advisory allowance; clean npm audit, dependency tree, production build and links | Merge so Dependabot observes the patched default branch |
| #160 | Five automatic tier markers; required-backend CI checks; feature-based regression files with preserved test-name inventory; shared runners and tolerance lookup | Full platform CI |
| #158 | Existing merged reusable audited website build verified by workflow policy tests | Deployment uses the vetted artifact after merge |
| #157 | Existing merged backend path selector verified; new shared tier/tolerance files included; minimum Keras convolution coverage added | Clean minimum-version CI matrix |
| #156 | Exact stable tags, pinned build tools, matching wheel/sdist metadata, strict Twine, clean-wheel smoke, exact-commit CI gate, artifact digest checks, no skip-existing | First authorized release exercises OIDC/index publication |
| #155 | Correct multiplicative alpha, backend feature ownership, obsolete import/command/link fixes, local-link checks and executable curated examples | Website CI |
| #154 | Annotated exported functions/constructors, documented dynamic Keras boundaries, strict modules, export regression gate, generated API pages, installed-wheel positive/negative consumer check | Full CI type/environment matrix |
| #153 | Existing merged import-light root and explicit optional dependency boundaries verified in clean base-only subprocesses | None beyond CI |
| #152 | Backend-local Keras/TF/Linen rank-generic cores, unchanged constructor/default inventory, serialization regression suites, isolated private Keras adapter, legacy Torch deprecation, complexity budget | Native-device CI; minimum/current Keras checked locally |
| #150 | Existing merged manifest/oracle/adapters and generated capabilities verified with available eager/compiled modes and TF/Keras in a separate environment | Linux all-backend CI and Apple fixture/Metal CI |
| #146 | Existing merged scalar validation verified by shared/backend regression suites | Six-backend CI |
| #137 | Superseded by the September 13 decision to maintain `mlnomadpy/nmn` independently; its mirror workflow was removed | Close as not planned; preserve independent history rather than forcing synchronization |

## Local verification

- Broad Torch/Linen/NNX/Keras-on-Torch/conformance run: 1,662 passed, 233 skipped;
  one test-split import error was fixed and its four-test module then passed.
- TF/Keras suite: 677 passed, 13 skipped before the test split. After the split,
  the same import error was isolated and fixed; the affected module plus TF
  all-layer tests passed (42 tests).
- All declared conformance modes available in each environment: 195 passed on
  Torch/JAX/Keras-on-Torch; 146 passed on TF/Keras. Missing backends skip locally;
  CI requires each platform's declared backends.
- Keras 3.0.0 compatibility: 47 passed, 10 skipped, including every convolution
  rank's output-shape checks.
- Unit/policy, constructor and regression-name inventories, documentation links,
  release failure fixtures, package MyPy, formatting and lint passed.
- Pinned release build, strict Twine, wheel/sdist verification and installed-wheel
  consumer typing passed. No package was published.
- `npm ls image-size --all` resolves only 2.0.4; audit reports zero vulnerabilities;
  Docusaurus builds with link errors configured to fail.

The host's native MLX initialization is unavailable inside this sandbox. Metal,
CUDA and TPU execution is not represented as locally verified. See
[mirror setup](mirror-sync.md), [test tiers](../tests/README.md), and
[typing support](typing.md) for the operational boundaries.

Remote CI additionally caught and corrected an accidental cross-backend import.
The latest-version job also exposed Flax 0.12.9's import incompatibility with
JAX 0.11.2; package and accelerator-install requirements now exclude that JAX
release pending upstream-compatible validation.
