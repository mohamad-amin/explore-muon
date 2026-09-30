# Small optimizer-surrogate study

This directory is the canonical home for this chat's work. Code, data, results,
decision history, and scratch files are isolated here at the user's request.

- [Current state](RESEARCH_STATE.md)
- [Project history and evaluation review](PROJECT_REVIEW.md)
- [Protocol and retained decision history](PROTOCOL.md)
- [Training and analysis code](research/tiny_spectra/README.md)
- [Experiments](logs/tiny_spectra/README.md)
- [Relocation evidence](migration/relocation.json)

Run Python through the local wrapper, which also keeps caches and temporary
files in this directory:

```bash
./run -m unittest research.tiny_spectra.test_model_data research.tiny_spectra.test_optim -v
./run -m research.tiny_spectra.analyze logs/tiny_spectra/stories_c_screen_20260928
```

The `.venv` symlink reads the existing shared environment. The small trainer and
its pinned numerical dependencies are owned here; the parent full training
pipeline is not copied or used. `research/adamw_spectra/train.py` contains only
the tiny `amp_context` compatibility helper used by kernel-reference tests.

Original code/data/log paths remain as compatibility symlinks so saved configs,
checkpoint metadata, old reports and recorded launch commands retain their
meaning. Frozen artifacts were moved without editing their contents. New work
uses this directory. The parent notebooks contain only a pointer to this branch.

The study is not yet qualified. Both independent confirmation populations
remain unscored. Only GPUs strictly below 48 GB are allowed.
