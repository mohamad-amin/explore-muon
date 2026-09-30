# Isolated small-surrogate study

Keep all new code, data, results, notes, plots and temporary work inside this
directory. Use `./run` to invoke Python; it sets the working directory and cache
paths here. Do not modify the parent project's research code or notebooks.

Read RESEARCH_GUIDE.md, RESEARCH_STATE.md and the relevant PROTOCOL.md entries
before research planning or substantial experiments, including after a reset.
Verify live Slurm handles before moving, continuing or restarting work.

The user requires GPUs with strictly less than 48 GB VRAM. Keep development and
sealed confirmation data separate. Never tune recipes or select favorable
windows after opening confirmation scores. Preserve failed comparisons,
predeclared rules, frozen sources and original metadata.

Historical paths outside this directory are compatibility symlinks. Preserve
them and do not rewrite absolute paths in old checkpoints. The `.venv` is a
read-only shared environment; any added preprocessing dependencies belong in a
private target inside this study. Other parent-project studies are out of scope.

Record decisions in PROTOCOL.md and current status in RESEARCH_STATE.md. Seek an
independent high-level peer discussion before a major new scientific direction
or a growing repair cascade. Routine edits and execution checks need no ceremony
or extra user approval. Current user instructions take precedence.
