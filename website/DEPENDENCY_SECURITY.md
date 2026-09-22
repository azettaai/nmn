# Website dependency security

The documentation site is built from reviewed files in this repository. CI
runs `npm run audit:ci` after every clean install and rejects critical findings
or any high-severity advisory that is not listed in the audit script.

## Applied remediations

- `serialize-javascript` is overridden to 7.1.1, removing the RCE and CPU
  exhaustion advisories inherited through Docusaurus's webpack plugins.
- `uuid` is overridden to 11.1.1, the first patched release for the buffer
  bounds advisory inherited through `sockjs`.
- `qs` is overridden to 6.16.0, the first patched release for
  `GHSA-4mjr-xmp4-gh2g`, inherited through the Docusaurus development server.

These overrides are exercised by the production Docusaurus build. Dependabot
continues to update the npm lockfile weekly.

## Image parser remediation (#170)

The lockfile resolves `image-size` 2.0.4 through Docusaurus's existing compatible
range, without a forced override. This fixes GHSA-w3rx-r6r6-pgpr (ICNS) and
GHSA-5p2g-fcmc-qvqq (JXL/HEIF). The high-advisory allowlist is empty, so either
advisory returning fails the audit gate before artifact upload or deployment.
Production builds process versioned, reviewed repository images and MDX.
