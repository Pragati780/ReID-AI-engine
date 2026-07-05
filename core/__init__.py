"""
Core face re-identification modules.

Each module has exactly one responsibility, so the pipeline can be
understood, tested, and later swapped out (e.g. matcher.py -> FAISS
lookup) one piece at a time without touching the rest.

    detector.py   -> find faces + landmarks in an image
    aligner.py    -> warp a detected face into a canonical pose (112x112)
    embedder.py   -> turn an aligned face into a 512-d identity vector
    quality.py    -> flag blurry / too-small faces before they poison results
    matcher.py    -> compare embeddings via cosine similarity
    aggregator.py -> turn frame-level matches into clean timestamp ranges
    utils.py      -> video frame sampling, saving crops, I/O helpers
"""
