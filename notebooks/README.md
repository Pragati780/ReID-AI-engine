# Notebooks

This folder is for exploratory work only -- NOT part of the pipeline:

- Threshold calibration: plot similarity score distributions for known
  genuine vs. impostor face pairs to pick `matching.similarity_threshold`
  in `config/config.yaml` empirically instead of guessing.
- Visual inspection: display aligned face crops side by side to sanity
  check `core/aligner.py` output before trusting downstream embeddings.
- Ad-hoc debugging of any specific video/reference pair that misbehaves.

Nothing in the actual pipeline (`pipeline/run_pipeline.py`) depends on
anything in this folder.
