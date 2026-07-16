# Contributing to Calm-Data-Generator

Thank you for your interest in contributing to **Calm-Data-Generator**! We welcome contributions from the community to help make this library better.

## Reporting Bugs

Calculated a wrong statistic? Found a crash? Please report it!

1.  **Search existing issues** on GitHub to see if the bug has already been reported.
2.  If not, **open a new Issue** using the "Bug Report" template.
3.  Include as much detail as possible: version, code snippet to reproduce, and error logs.

## Feature Requests

Have an idea for a new feature?

1.  Open a new Issue using the "Feature Request" template.
2.  Describe the feature clearly and why it would be useful.

## Pull Requests

We actively welcome your pull requests.

1.  **Fork** the repo and create your branch from `main`.
2.  If you've added code that should be tested, add tests.
3.  Ensure your code passes linting (`ruff`).
4.  **If you edited a documentation file that has an EN/ES pair** (e.g. `README.md` /
    `README_ES.md`, or any file under `calm_data_generator/docs/`), **update both in the
    same PR.** CI will post a non-blocking warning if it detects a one-sided edit — treat
    it as a reminder, not a hard requirement, but keeping both versions in sync is expected.
5.  Issue that pull request!

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/Calm-Data_Generator.git
cd Calm-Data_Generator

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install editable (there is no "dev" extra — install the tooling separately)
pip install -e ".[full]"
pip install pytest ruff pre-commit

# Enable the pre-commit hooks (ruff, trailing-whitespace, end-of-file-fixer)
pre-commit install
```

Before opening a PR, read [ARCHITECTURE.md](./ARCHITECTURE.md) for a map of
the modules — it will tell you which file to touch for your change.

## Running Tests

```bash
pytest tests/
```

River-dependent tests are skipped automatically if `river` isn't installed
(`pip install -e ".[stream]"` to enable them).

## Licensing

By contributing, you agree that your contributions will be licensed under its MIT License.
