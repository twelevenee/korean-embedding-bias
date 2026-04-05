"""
Embedding model loading utilities.

Supports:
  - Korean FastText (cc.ko.300.bin) via gensim load_facebook_vectors
  - Korean Word2Vec (Namuwiki / Kyubyong / fallback) via gensim KeyedVectors

The EmbeddingModel wrapper provides a unified interface regardless of the
underlying model type, and handles OOV lookups + L2 normalization.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class ModelNotFoundError(FileNotFoundError):
    pass


# ---------------------------------------------------------------------------
# Unified embedding wrapper
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """
    Thin wrapper around gensim KeyedVectors.
    All returned vectors are L2-normalized (unit norm).
    """

    def __init__(self, keyed_vectors, model_name: str):
        self._kv = keyed_vectors
        self.model_name = model_name
        self.dim: int = keyed_vectors.vector_size

    def get_vector(self, word: str) -> np.ndarray:
        """Return L2-normalized vector for word. Raises KeyError if OOV."""
        v = self._kv[word].astype(np.float32)
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            raise KeyError(f"Zero vector for '{word}' in {self.model_name}")
        return v / norm

    def get_vector_safe(self, word: str) -> Optional[np.ndarray]:
        """Return normalized vector or None if OOV."""
        try:
            return self.get_vector(word)
        except KeyError:
            return None

    def has_word(self, word: str) -> bool:
        return word in self._kv.key_to_index

    def filter_word_list(
        self, words: List[str], aliases: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Return (found_words, missing_words) for a list of words.
        If `aliases` is provided (e.g. {"CEO": ["최고경영자"]}), the first
        available alias is used in place of missing primary forms.
        """
        from src.word_sets import WORD_ALIASES
        alias_map = aliases or WORD_ALIASES

        found, missing = [], []
        for w in words:
            if self.has_word(w):
                found.append(w)
            else:
                # Try aliases
                resolved = None
                for alt in alias_map.get(w, []):
                    if self.has_word(alt):
                        resolved = alt
                        break
                if resolved:
                    found.append(resolved)
                else:
                    missing.append(w)
        return found, missing

    def __repr__(self) -> str:
        return f"EmbeddingModel(name={self.model_name!r}, dim={self.dim})"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_fasttext_korean(
    model_path: Path,
    memory_map: bool = True,
) -> EmbeddingModel:
    """
    Load Korean FastText from cc.ko.300.bin using gensim.

    Uses gensim.models.fasttext.load_facebook_vectors(), which reads Facebook's
    binary format and exposes a KeyedVectors-compatible interface. On the first
    call, gensim writes cc.ko.300.bin.vectors_ngrams.npy and
    cc.ko.300.bin.vectors_vocab.npy alongside the binary (~4 GB total).
    Subsequent loads are near-instant via memory-mapping.

    Args:
        model_path: path to cc.ko.300.bin
        memory_map: if True, memory-map the .npy files (~1.5 GB RAM vs ~6 GB)
    """
    from gensim.models.fasttext import load_facebook_vectors

    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelNotFoundError(
            f"FastText model not found at {model_path}.\n"
            f"Download with: python -c \"from src.load_embeddings import "
            f"download_fasttext_korean; from pathlib import Path; "
            f"download_fasttext_korean(Path('models/'))\""
        )

    print(f"Loading FastText model from {model_path} (this may take a few minutes on first load)...")
    kv = load_facebook_vectors(str(model_path))
    return EmbeddingModel(kv, model_name="fasttext-ko")


def load_word2vec_namuwiki(model_path: Path) -> EmbeddingModel:
    """
    Load a Korean Word2Vec model in gensim KeyedVectors format.

    Supports saved gensim models (new and old), or word2vec format (.txt / .vec).
    Tries KeyedVectors.load first, then old Word2Vec pickle, falls back to load_word2vec_format.

    Fallback model sources (in order):
      1. Kyubyong/wordvectors Korean release (~1.2 GB)
         https://github.com/Kyubyong/wordvectors
      2. snunlp/ko_en_wiki_word2vec on HuggingFace
      3. Raise ModelNotFoundError with download instructions
    """
    from gensim.models import KeyedVectors, Word2Vec
    import pickle

    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelNotFoundError(
            f"Word2Vec model not found at {model_path}.\n\n"
            "Fallback download options:\n"
            "  1. Kyubyong/wordvectors: https://github.com/Kyubyong/wordvectors\n"
            "     Download 'ko.bin' from the releases page.\n"
            "  2. snunlp ko_en_wiki_word2vec on HuggingFace:\n"
            "     https://huggingface.co/snunlp/ko_en_wiki_word2vec\n"
        )

    print(f"Loading Word2Vec model from {model_path}...")
    try:
        # Try loading as old Word2Vec model first
        with open(str(model_path), 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        if isinstance(model, Word2Vec) and hasattr(model, 'syn0'):
            kv = KeyedVectors(vector_size=model.vector_size)
            kv.add_vectors(list(model.vocab.keys()), model.syn0)
        else:
            raise ValueError("Not a compatible Word2Vec model")
    except Exception:
        try:
            # Try loading as saved gensim model
            kv = KeyedVectors.load(str(model_path))
        except Exception:
            # Fall back to word2vec format
            suffix = model_path.suffix.lower()
            binary = suffix in (".bin",)
            try:
                kv = KeyedVectors.load_word2vec_format(str(model_path), binary=binary)
            except Exception:
                # Try the other format
                kv = KeyedVectors.load_word2vec_format(str(model_path), binary=not binary)

    return EmbeddingModel(kv, model_name="word2vec-ko")


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_fasttext_korean(dest_dir: Path, chunk_size_mb: int = 10) -> Path:
    """
    Download cc.ko.300.bin.gz from Facebook Research and decompress it.

    URL: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz
    File size: ~4.2 GB compressed, ~4.5 GB decompressed.
    Estimated time: 10-60 minutes depending on connection speed.

    Args:
        dest_dir: directory to save the decompressed .bin file
        chunk_size_mb: download chunk size in MB

    Returns:
        Path to the decompressed cc.ko.300.bin file
    """
    import requests
    from tqdm import tqdm

    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    gz_path = dest_dir / "cc.ko.300.bin.gz"
    bin_path = dest_dir / "cc.ko.300.bin"

    if bin_path.exists():
        print(f"Model already exists at {bin_path}. Skipping download.")
        return bin_path

    chunk_size = chunk_size_mb * 1024 * 1024

    print(f"Downloading Korean FastText from {url}")
    print("Expected size: ~4.2 GB. This will take a while.")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with open(gz_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Decompressing to {bin_path}...")
    with gzip.open(gz_path, "rb") as f_in, open(bin_path, "wb") as f_out:
        with tqdm(unit="B", unit_scale=True, desc="Decompressing") as pbar:
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                pbar.update(len(chunk))

    gz_path.unlink()
    print(f"Done. Model saved to {bin_path}")
    return bin_path


# ---------------------------------------------------------------------------
# Vocabulary verification
# ---------------------------------------------------------------------------

def verify_model_words(model: EmbeddingModel, word_sets) -> dict:
    """
    Check WEAT word coverage in the model.

    Returns a dict with keys:
      male_attrs, female_attrs, male_occupations, female_occupations,
      neutral_occupations — each with 'found', 'missing', 'coverage_pct'.
    """
    from src.word_sets import WORD_ALIASES

    sets = {
        "male_attrs": word_sets.male_attrs,
        "female_attrs": word_sets.female_attrs,
        "male_occupations": word_sets.male_occupations,
        "female_occupations": word_sets.female_occupations,
        "neutral_occupations": word_sets.neutral_occupations,
    }

    report = {}
    for key, words in sets.items():
        found, missing = model.filter_word_list(words, aliases=WORD_ALIASES)
        report[key] = {
            "found": found,
            "missing": missing,
            "total": len(words),
            "coverage_pct": 100 * len(found) / len(words),
        }

    return report


def print_coverage_report(report: dict, model_name: str = "") -> None:
    """Pretty-print the output of verify_model_words()."""
    header = f"Coverage report — {model_name}" if model_name else "Coverage report"
    print(f"\n{header}")
    print("-" * 50)
    for key, info in report.items():
        pct = info["coverage_pct"]
        found_n = len(info["found"])
        total = info["total"]
        status = "OK" if pct == 100 else ("WARN" if pct >= 60 else "FAIL")
        print(f"  [{status}] {key}: {found_n}/{total} ({pct:.1f}%)")
        if info["missing"]:
            print(f"         missing: {info['missing']}")
    print()
