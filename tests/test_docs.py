"""Run every ```python block of the user docs and README, so they can't rot.

Blocks of one page share a namespace and run in order. Non-runnable
snippets use ```text.
"""
import re
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).parent.parent
PAGES = ['README.md', 'docs/index.md', 'docs/guide.md', 'docs/examples.md',
         'docs/extending.md', 'docs/migrating.md']
BLOCK = re.compile(r'^```python\n(.*?)^```', re.S | re.M)


@pytest.mark.parametrize('page', PAGES)
def test_doc_code_runs(page, monkeypatch):
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:  # optional: plotting snippets run against a stub
        stub = mock.MagicMock()
        monkeypatch.setitem(sys.modules, 'matplotlib', stub)
        monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', stub.pyplot)
    blocks = BLOCK.findall((ROOT / page).read_text())
    assert blocks, '%s has no python blocks' % page
    namespace = {'__name__': 'docs'}
    for block in blocks:
        exec(compile(block, page, 'exec'), namespace)
