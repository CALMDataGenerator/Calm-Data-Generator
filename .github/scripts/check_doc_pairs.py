#!/usr/bin/env python3
"""
Warns (does not fail the build) when a PR touches one side of an EN/ES doc pair
(e.g. README.md / README_ES.md) but not the other. Bilingual docs are a project
requirement here, and a silent one-sided edit is how the two versions drift apart.

Usage: check_doc_pairs.py <file1> <file2> ...
(the list of files changed in the PR, e.g. from `git diff --name-only`)
"""
import sys
from pathlib import Path


def find_pairs(repo_root: Path):
    """Return {en_path: es_path} for every `X.md` that has a sibling `X_ES.md`."""
    pairs = {}
    for md in repo_root.rglob("*.md"):
        if md.name.endswith("_ES.md"):
            continue
        es = md.with_name(md.stem + "_ES.md")
        if es.exists():
            pairs[md.relative_to(repo_root).as_posix()] = es.relative_to(repo_root).as_posix()
    return pairs


def main():
    changed = set(sys.argv[1:])
    if not changed:
        return 0

    repo_root = Path(__file__).resolve().parents[2]
    pairs = find_pairs(repo_root)

    one_sided = []
    for en, es in pairs.items():
        en_changed = en in changed
        es_changed = es in changed
        if en_changed and not es_changed:
            one_sided.append((en, es))
        elif es_changed and not en_changed:
            one_sided.append((es, en))

    if one_sided:
        print("::warning::Bilingual docs may be out of sync — this PR edits one side "
              "of an EN/ES pair without the other. See CONTRIBUTING.md.")
        for changed_file, pair_file in one_sided:
            print(f"  - {changed_file} changed, but its pair {pair_file} was not touched.")
        print("\nIf this is intentional (e.g. a typo-only fix, or the translation is "
              "coming in a follow-up PR), you can ignore this warning — it does not "
              "block the build.")
    else:
        print("Doc pairs check: no one-sided EN/ES edits detected.")

    return 0  # never fail the build — this is advisory only


if __name__ == "__main__":
    sys.exit(main())
