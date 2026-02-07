"""Lightweight shim replacing torchtext.vocab for scgpt compatibility.

torchtext has been deprecated and its C++ extensions are incompatible with
recent PyTorch versions (>= 2.6). scgpt only uses torchtext.vocab.Vocab for
a simple string-to-index mapping. This module provides a pure-Python
replacement and patches sys.modules so that `import torchtext.vocab` resolves
here instead of loading the broken native library.

Usage:
    import src.torchtext_shim  # must be imported BEFORE scgpt
"""

import sys
import types
from collections import OrderedDict
from typing import Dict, Optional


class Vocab:
    """Minimal re-implementation of torchtext.vocab.Vocab."""

    def __init__(self, stoi: Optional[Dict[str, int]] = None):
        self._stoi: Dict[str, int] = dict(stoi) if stoi else {}
        self._itos: Dict[int, str] = {v: k for k, v in self._stoi.items()}
        self._default_index: Optional[int] = None

    # --- core lookup ---
    def __getitem__(self, token: str) -> int:
        if token in self._stoi:
            return self._stoi[token]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(token)

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def __len__(self) -> int:
        return len(self._stoi)

    # --- mutation ---
    def insert_token(self, token: str, index: int) -> None:
        self._stoi[token] = index
        self._itos[index] = token

    def set_default_index(self, index: int) -> None:
        self._default_index = index

    # --- introspection ---
    def get_stoi(self) -> Dict[str, int]:
        return dict(self._stoi)

    def get_itos(self):
        return [self._itos[i] for i in range(len(self._itos))]

    @property
    def vocab(self):
        """Return self so that Vocab(v.vocab) works like torchtext."""
        return self


def vocab(ordered_dict: OrderedDict, min_freq: int = 1) -> Vocab:
    """Build a Vocab from an OrderedDict of {token: freq}, mimicking
    torchtext.vocab.vocab()."""
    stoi = {}
    idx = 0
    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            stoi[token] = idx
            idx += 1
    return Vocab(stoi)


# ---------------------------------------------------------------------------
# Patch sys.modules so `import torchtext.vocab` and `import torchtext`
# resolve without loading the real (broken) torchtext C++ extension.
# ---------------------------------------------------------------------------
def install():
    """Install the shim into sys.modules."""
    # Create a fake torchtext package
    _torchtext = types.ModuleType("torchtext")
    _torchtext.__path__ = []  # mark as package

    # Create a fake torchtext.vocab module
    _vocab_mod = types.ModuleType("torchtext.vocab")
    _vocab_mod.Vocab = Vocab
    _vocab_mod.vocab = vocab

    _torchtext.vocab = _vocab_mod

    # Also need a fake torchtext._extension to prevent the real __init__ from running
    _ext_mod = types.ModuleType("torchtext._extension")
    _torchtext._extension = _ext_mod

    sys.modules["torchtext"] = _torchtext
    sys.modules["torchtext.vocab"] = _vocab_mod
    sys.modules["torchtext._extension"] = _ext_mod


# Auto-install on import
install()
