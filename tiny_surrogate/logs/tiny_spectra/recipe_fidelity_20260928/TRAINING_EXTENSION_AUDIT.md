# Training-data extension feasibility audit

Read-only planning on 2026-09-28 while the fixed-recipe seed replication runs.
No additional text was downloaded, no data was changed, and no confirmation
panel was loaded for scoring. This note authorizes no preparation or experiment.

The current training capacity is 32,939,904 unique target positions. A doubled
version of that horizon requires 65,879,808. A full-goal proposal must instead
calculate capacity from its largest declared batch-times-update budget: a
matched-step control could need more than twice the midpoint horizon.

The pinned source receipt records a 128 MiB training prefix of a
1,924,281,556-byte file at revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. This is historical download evidence,
not a new network availability check. The original source URL, HTTP range,
ETag, byte count, and hash remain in
`data/tiny_stories_20260928/source/source_manifest.json`.

An eventual extension can retain the existing tokenizer, vocabulary, model and
entire development/test populations. It must not rerun `stories.prepare()` as
currently written: that routine refits the tokenizer and gives training
precedence when removing overlaps from validation. For an extension, those
populations are already fixed. New training documents that conflict with them
must be excluded instead. Exact normalized identities and verified 64-word
overlaps need to be checked against the fixed held-out documents, and new
training documents must be deduplicated against the old training inventory.
That is a data-only audit; model losses must not affect membership.

Preserve the old source prefix, tokenizer, encoded train/development/test files,
membership receipts, manifests and scientific artifacts byte-for-byte. An
extension belongs in a new versioned data directory with its own hashes and
exclusion report. Reuse the original tokenizer without fitting it again. Do not
rewrite the sealed panel manifest to make compatibility checks pass.

There is also an order constraint. The current stream is a seeded permutation
of the *entire* training block population. Enlarging that population and drawing
a new permutation changes the first 32.94M targets, even with the same seed.
Therefore old runs cannot silently become matched controls for the new stream.
Two explicit designs are possible:

- Rerun every relevant development/confirmation/control arm on one newly frozen
  enlarged pool. Short and long horizons share the same permutation prefix
  within that new family. Old results remain historical development evidence.
- Introduce a documented two-segment stream whose first segment reproduces the
  old permutation and whose second segment uses the appended targets. This is
  a changed data-order rule, not uniform shuffling of the enlarged pool; its
  possible distribution shift must be considered. It requires new loader
  contracts and cannot be described as an unchanged training recipe.

This audit selects neither design. A later independently reviewed proposal must
choose one before training, with a fixed download/target budget and all necessary
horizon controls. A successful current seed replication would support only its
fixed development contrast, not either extension design automatically.

The confirmation adapter intentionally refuses an extended training manifest
today. It pins the original training hash and checks tokenizer, vocabulary,
membership and source identity against the sealed panel. An extension needs an
explicit compatibility receipt tying the new train inventory to the original
panel and tokenizer, plus synthetic rejection tests. Replacing a hash constant
or dropping membership checks would not provide that evidence. The panel's
one-family binding must remain intact; neither real panel is currently bound.
