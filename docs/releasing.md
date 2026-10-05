# Releasing

PyPI is published only from a GitHub Release, never from a push or tag.

1. Bump `__version__` in `pymetaheuristics/__init__.py` (and the version
   assertion in `tests/test_pymetaheuristics.py`) and update `CHANGELOG.md`.
2. Merge to `main`.
3. Create a GitHub Release with tag `v<version>` (for example `v0.3.0`). The
   tag must match `__version__` (a leading `v` is ignored) or the workflow
   fails before building.

The `delivery.yml` workflow then verifies the tag, builds the sdist and
wheel, publishes to PyPI with trusted publishing (no secrets) and attaches
both files to the release. Pre-releases are skipped: nothing is published
or attached.

## One-time setup (repository owner)

- On PyPI, register a trusted publisher for `pymetaheuristics`: owner
  `igormcsouza`, repository `pymetaheuristics`, workflow `delivery.yml`,
  environment `pypi`.
- In the GitHub repository settings, create the `pypi` environment.
- For the docs site, enable GitHub Pages with source "GitHub Actions"
  (Settings, Pages). The `docs.yml` workflow deploys on pushes to `main`
  and fails until this is done.
