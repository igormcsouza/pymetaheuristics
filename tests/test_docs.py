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
TUTORIAL = ['problem', 'constraints', 'stopping', 'operators',
            'genetic-algorithm', 'simulated-annealing',
            'artificial-bee-colony', 'results']
# The tutorial pages build on each other, so they share one namespace.
PAGES = ['README.md', 'docs/index.md',
         ['docs/tutorial/%s.md' % name for name in TUTORIAL],
         'docs/examples.md', 'docs/extending.md', 'docs/release-notes.md']
# Blocks may be indented inside a tab or admonition.
BLOCK = re.compile(r'^( *)```python\n(.*?)^\1```', re.S | re.M)


@pytest.mark.parametrize('page', PAGES, ids=str)
def test_doc_code_runs(page, monkeypatch):
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:  # optional: plotting snippets run against a stub
        stub = mock.MagicMock()
        monkeypatch.setitem(sys.modules, 'matplotlib', stub)
        monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', stub.pyplot)
    namespace = {'__name__': 'docs'}
    for name in [page] if isinstance(page, str) else page:
        blocks = BLOCK.findall((ROOT / name).read_text())
        assert blocks, '%s has no python blocks' % name
        for indent, block in blocks:
            block = ''.join(line[len(indent):]
                            for line in block.splitlines(True))
            exec(compile(block, name, 'exec'), namespace)
