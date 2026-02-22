#!/usr/bin/env python3
"""Regression tests for tex2text.py conversion behavior."""

from __future__ import annotations

import types
import unittest

import tex2text as t2t


def _options(
    *,
    abstract: bool = False,
    markdown: bool = False,
    unicode: bool = False,
    math: str | None = None,
    gradescope: bool = False,
    sections: bool = False,
):
    return types.SimpleNamespace(
        abstract=abstract,
        markdown=markdown,
        unicode=unicode,
        math=math,
        gradescope=gradescope,
        sections=sections,
    )


class TestTex2TextUnicode(unittest.TestCase):
    def test_unicode_basic_greeks(self) -> None:
        out = t2t.tex2text(r"$\alpha + \beta$", _options(unicode=True))
        self.assertIn("α", out)
        self.assertIn("β", out)
        self.assertNotIn(r"\alpha", out)
        self.assertNotIn(r"\beta", out)

    def test_bra_ket_and_dagger_unicode(self) -> None:
        src = (
            r"\["
            r"\lvert \Phi_{\mathrm{HF}} \rangle = c^\dagger \lvert \mathrm{vac} \rangle"
            r"\]"
        )
        out = t2t.tex2text(src, _options(unicode=True))
        self.assertIn("|", out)
        self.assertIn("⟩", out)
        self.assertIn("†", out)
        self.assertNotIn("lvert", out)
        self.assertNotIn("mathrm", out)

    def test_prod_unicode(self) -> None:
        out = t2t.tex2text(
            r"\[\prod_{i=0}^{N^\uparrow-1} a_i\]",
            _options(unicode=True),
        )
        self.assertIn("∏", out)
        self.assertNotIn("prod_{", out)


class TestTex2TextModes(unittest.TestCase):
    def test_math_preserve_mode(self) -> None:
        out = t2t.tex2text(r"$\alpha + \beta$", _options(unicode=True, math="$$"))
        self.assertIn("$$", out)
        self.assertIn(r"\alpha", out)
        self.assertIn(r"\beta", out)


class TestExtractAbstract(unittest.TestCase):
    def test_extract_abstract_missing_raises_valueerror(self) -> None:
        with self.assertRaises(ValueError):
            t2t.extract_abstract(r"\section{Intro} no abstract here")


if __name__ == "__main__":
    unittest.main()
