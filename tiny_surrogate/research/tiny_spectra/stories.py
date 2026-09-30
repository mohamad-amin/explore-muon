"""Data-only TinyStories preparation and fresh, non-recycling token streams.

Training never loads sealed_test. Tokenizers is a preparation-only dependency;
runtime training uses compact IDs and the frozen vocabulary without importing it.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata

import numpy as np
import torch

from .data import CharacterCorpus

REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
PREFIX_HASHES = {"train": "d37cd3af93083e3504803f4f836755bfafb8b32cd046165f85ad91d6bdd2a1b5",
                 "valid": "a2484bd1135817beb96451a200faae649ef6d341041c9eec4513436fc2de79ff"}
TOKENIZER_VERSION = "0.23.2"
EOT = "<|endoftext|>"
SEQ_LEN = 128
DEFAULT_STREAM_SEED = 20260928 + 1729
MIN_TRAIN_TARGETS = 16_777_216
MASK = (1 << 64) - 1


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))


def artifact(path, relative_to):
    path = Path(path)
    return dict(path=str(path.relative_to(relative_to)), bytes=path.stat().st_size,
                sha256=sha(path.read_bytes()))


def verified(root, spec):
    path = Path(root) / spec["path"]
    if path.resolve().is_relative_to(Path(root).resolve()) is False:
        raise ValueError("Artifact escaped its data directory")
    raw = path.read_bytes()
    if len(raw) != spec["bytes"] or sha(raw) != spec["sha256"]:
        raise ValueError(f"Artifact integrity failure: {path}")
    return raw


def normalized_words(text):
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def document_identity(text):
    return sha(" ".join(normalized_words(text)).encode())


def read_complete_stories(raw, source):
    """Trim at last complete literal EOT delimiter before decoding UTF-8."""
    marker = EOT.encode()
    last = raw.rfind(marker)
    if last < 0:
        raise ValueError("No complete story delimiter")
    end = last + len(marker)
    documents, cursor = [], 0
    for index, body in enumerate(raw[:end].split(marker)[:-1]):
        text = body.decode("utf-8").strip()
        documents.append(dict(source=source, source_index=index, source_byte_start=cursor,
                              source_byte_end=cursor + len(body), text=text,
                              text_sha256=sha(text.encode()), identity=document_identity(text)))
        cursor += len(body) + len(marker)
    return documents, dict(prefix_bytes=len(raw), complete_prefix_bytes=end,
                           dropped_incomplete_suffix_bytes=len(raw)-end,
                           complete_document_count=len(documents))


def assign_exact(train, valid):
    """Train takes precedence; first normalized occurrence survives within source."""
    accepted_train, candidates, audit = [], [], []
    seen_train, seen_valid = {}, {}
    for source, docs in (("train", train), ("valid", valid)):
        for doc in docs:
            row = {k: v for k, v in doc.items() if k != "text"}
            identity = doc["identity"]
            role = "train" if source == "train" else ("dev" if int(identity[-2:], 16) % 2 == 0 else "test")
            row["assigned_role"] = role
            if not normalized_words(doc["text"]):
                row.update(status="excluded", reason="empty_normalized_document")
            elif identity in seen_train:
                row.update(status="excluded", reason="exact_normalized_duplicate_of_train",
                           matched_source_index=seen_train[identity])
            elif identity in seen_valid:
                row.update(status="excluded", reason="exact_normalized_duplicate_of_valid",
                           matched_source_index=seen_valid[identity])
            else:
                row.update(status="candidate", reason="first_normalized_document")
                item = dict(doc, role=role, audit_index=len(audit))
                if source == "train":
                    seen_train[identity] = doc["source_index"]
                    accepted_train.append(item)
                else:
                    seen_valid[identity] = doc["source_index"]
                    candidates.append(item)
            audit.append(row)
    return accepted_train, candidates, audit


def shingles(words, n=64, word_codes=None):
    """64-bit rolling hash is only a lookup accelerator; hits require exact words."""
    if len(words) < n:
        return
    cache = {} if word_codes is None else word_codes
    values = []
    for word in words:
        if word not in cache:
            cache[word] = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")
        values.append(cache[word])
    base, power, value = 1_000_003, pow(1_000_003, n-1, 1 << 64), 0
    for x in values[:n]:
        value = (value * base + x) & MASK
    yield value, 0
    for i in range(n, len(values)):
        value = ((value - values[i-n] * power) * base + values[i]) & MASK
        yield value, i-n+1


def quarantine_overlap(train, candidates, audit, n=64, hash_function=shingles):
    """Drop validation matches to train, then both sides of dev/test matches."""
    words = [normalized_words(doc["text"]) for doc in candidates]
    index = defaultdict(list)
    codes = {}
    for i, ws in enumerate(words):
        for key, offset in hash_function(ws, n, codes):
            index[key].append((i, offset))
    excluded, evidence = set(), []
    for doc in train:
        ws = normalized_words(doc["text"])
        for key, start in hash_function(ws, n, codes):
            for i, offset in index.get(key, ()):
                if i not in excluded and ws[start:start+n] == words[i][offset:offset+n]:
                    excluded.add(i)
                    evidence.append(dict(reason="shared64_words_with_train", candidate=i,
                                         train_source_index=doc["source_index"], train_word_offset=start,
                                         candidate_word_offset=offset,
                                         normalized_shingle_sha256=sha(" ".join(ws[start:start+n]).encode())))
    # Keep the entire remaining graph fixed before applying symmetric exclusions.
    train_excluded = set(excluded)
    for matches in index.values():
        if len(matches) < 2:
            continue
        dev = [(i, j) for i, j in matches if i not in train_excluded and candidates[i]["role"] == "dev"]
        test = [(i, j) for i, j in matches if i not in train_excluded and candidates[i]["role"] == "test"]
        for i, a in dev:
            for j, b in test:
                if words[i][a:a+n] == words[j][b:b+n] and (i not in excluded or j not in excluded):
                    excluded.update((i, j))
                    evidence.append(dict(reason="shared64_words_across_dev_test", candidate=i,
                                         other_candidate=j, candidate_word_offset=a, other_word_offset=b,
                                         normalized_shingle_sha256=sha(" ".join(words[i][a:a+n]).encode())))
    for i, doc in enumerate(candidates):
        audit[doc["audit_index"]].update(status="excluded" if i in excluded else "accepted",
                                        reason=("long_overlap_quarantine" if i in excluded else "independent_document_candidate"))
    for doc in train:
        audit[doc["audit_index"]].update(status="accepted", reason="first_training_document")
    return [doc for i, doc in enumerate(candidates) if i not in excluded], evidence


def train_tokenizer(train, private_target):
    """Iterator has access to accepted training documents only, never valid/test."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "2"
    sys.path.insert(0, str(Path(private_target).resolve()))
    import tokenizers
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    if tokenizers.__version__ != TOKENIZER_VERSION or not Path(tokenizers.__file__).resolve().is_relative_to(Path(private_target).resolve()):
        raise ValueError("Wrong preparation-only tokenizer installation")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=4096, min_frequency=2, show_progress=False,
                                  special_tokens=[EOT], initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()))
    consumed = []
    def iterator():
        for doc in train:
            if doc["role"] != "train":
                raise ValueError("Tokenizer training received a nontraining document")
            consumed.append(doc["identity"])
            yield doc["text"]
    tokenizer.train_from_iterator(iterator(), trainer=trainer, length=len(train))
    if consumed != [doc["identity"] for doc in train] or tokenizer.get_vocab_size() != 4096:
        raise ValueError("Tokenizer fitting count or vocabulary mismatch")
    return tokenizer, dict(version=tokenizers.__version__, training_document_count=len(consumed),
                           training_identity_sequence_sha256=sha("\n".join(consumed).encode()),
                           train_only=True, vocab_size=4096, initial_alphabet="all256bytes",
                           pretokenizer="ByteLevel(add_prefix_space=False,use_regex=True)",
                           normalization="none; original story text", eot_id=tokenizer.token_to_id(EOT))


def encode_documents(root, role, docs, tokenizer):
    destination = root / ("sealed_test" if role == "test" else "")
    path = destination / f"{role}.u16"
    offsets, cursor = [], 0
    eot = tokenizer.token_to_id(EOT)
    with path.open("xb") as stream:
        for begin in range(0, len(docs), 128):
            batch = docs[begin:begin+128]
            for doc, encoded in zip(batch, tokenizer.encode_batch([d["text"] for d in batch], add_special_tokens=False)):
                ids = np.asarray(encoded.ids + [eot], dtype="<u2")
                stream.write(ids.tobytes())
                offsets.append(dict(identity=doc["identity"], source_index=doc["source_index"],
                                    text_sha256=doc["text_sha256"], start=cursor, end=cursor+len(ids),
                                    token_ids_sha256=sha(ids.tobytes())))
                cursor += len(ids)
    member_path = destination / f"{role}.documents.json"
    write_json(member_path, offsets)
    return dict(tokens=artifact(path, destination), documents=artifact(member_path, destination),
                token_count=cursor, document_count=len(docs)), offsets


def evaluation_starts(offsets, seq_len=SEQ_LEN):
    return np.asarray([start for doc in offsets
                       for start in range(doc["start"], doc["end"]-seq_len, seq_len)], dtype="<i8")


def evaluation_plan_all_targets(offsets, seq_len=SEQ_LEN):
    """Each original document target index1..N-1 occurs exactly once."""
    if seq_len < 1:raise ValueError("Positive context required")
    starts,lengths,document_indices=[],[],[]
    for i,doc in enumerate(offsets):
        if doc["start"]<0 or doc["end"]<doc["start"]:raise ValueError("Invalid document boundary")
        for start in range(doc["start"],doc["end"]-1,seq_len):
            starts.append(start)
            lengths.append(min(seq_len,doc["end"]-start-1))
            document_indices.append(i)
    return tuple(np.asarray(values,dtype="<i8") for values in (starts,lengths,document_indices))


def padded_evaluation_batch(token_ids, starts, lengths, seq_len, eot_id, device="cpu"):
    """Causal right padding: input EOT, target -100; only real prefixes count."""
    ids=torch.as_tensor(token_ids,device="cpu")
    starts=torch.as_tensor(starts,device="cpu")
    lengths=torch.as_tensor(lengths,device="cpu")
    if (seq_len<1 or starts.ndim!=1 or lengths.shape!=starts.shape or len(starts)==0
            or starts.dtype not in (torch.int32,torch.int64) or lengths.dtype not in (torch.int32,torch.int64)):
        raise ValueError("Nonempty integer window starts/lengths required")
    if (starts<0).any() or (lengths<1).any() or (lengths>seq_len).any() or (starts+lengths>=len(ids)).any():
        raise ValueError("Invalid complete-prefix window")
    x=torch.full((len(starts),seq_len),int(eot_id),dtype=torch.long)
    y=torch.full_like(x,-100)
    for row,(start,length) in enumerate(zip(starts.tolist(),lengths.tolist())):
        real=ids[start:start+length+1]
        if (real<0).any():raise ValueError("Negative source token is not padding")
        x[row,:length]=real[:-1]
        y[row,:length]=real[1:]
    return x.to(device),y.to(device)


def prepare_evaluation_v2(root):
    """Add complete-target coverage before scoring; preserve every v1 artifact."""
    root=Path(root)
    path=root/"training_manifest_v2.json"
    destination=root/"sealed_test_v2"
    if path.exists() or destination.exists():raise FileExistsError("V2 policy already attempted")
    receipt=json.loads((root/"PREPARATION_COMPLETE.json").read_bytes())
    train_raw=(root/"training_manifest.json").read_bytes()
    test_raw=(root/"sealed_test/manifest.json").read_bytes()
    if sha(train_raw)!=receipt["training_manifest_sha256"] or sha(test_raw)!=receipt["test_manifest_sha256"]:
        raise ValueError("Original manifests changed")
    train,test=json.loads(train_raw),json.loads(test_raw)
    offsets=json.loads(verified(root,train["splits"]["val"]["documents"]))
    starts,lengths,indices=evaluation_plan_all_targets(offsets,train["seq_len"])
    for label,values in (("starts",starts),("lengths",lengths),("document_indices",indices)):
        with (root/f"dev.{label}_v2.i64").open("xb") as stream:stream.write(values.tobytes())
    policy=dict(evaluation_policy_version=2,
                evaluation_boundary_rule="All predictable story tokens including EOT, excluding only eachstory'sfirsttoken; 128stride; tail/shortwindowsrightpadded",
                target_padding_id=-100,input_padding_id=train["eot_id"],target_count_unit="BPE_tokens",
                evaluation_policy_source_sha256=sha(Path(__file__).read_bytes()))
    train_v2=dict(train,**policy,previous_manifest_sha256=sha(train_raw),
                   dev_starts=artifact(root/"dev.starts_v2.i64",root),
                   dev_lengths=artifact(root/"dev.lengths_v2.i64",root),
                   dev_document_indices=artifact(root/"dev.document_indices_v2.i64",root),
                   dev_full_windows=len(starts),dev_target_count=int(lengths.sum()),
                   dev_scored_documents=len(set(indices.tolist())),
                   dev_zero_target_documents=len(offsets)-len(set(indices.tolist())))
    write_json(path,train_v2)
    destination.mkdir()
    # New exact byte copies keep path isolation without hardlink mutation coupling.
    for key in ("tokens","documents"):
        spec=test["splits"]["test"][key]
        raw=verified(root/"sealed_test",spec)
        with (destination/spec["path"]).open("xb") as stream:stream.write(raw)
    rows=[]
    for old in test["documents"]:
        local=[dict(start=0,end=old["end"]-old["start"])]
        ss,ll,_=evaluation_plan_all_targets(local,test["seq_len"])
        row=dict(old,full_starts=ss.tolist(),full_lengths=ll.tolist(),curve_starts=ss.tolist(),curve_lengths=ll.tolist(),
                 counts=dict(full_windows=len(ss),curve_windows=len(ss),full_target_tokens=int(ll.sum()),curve_target_tokens=int(ll.sum())),
                 inclusion_reason="All independently assignedtest stories; every predictable within-storytoken incl shortstories/tails")
        rows.append(row)
    targets=sum(row["counts"]["full_target_tokens"] for row in rows)
    windows=sum(row["counts"]["full_windows"] for row in rows)
    test_v2=dict(test,**policy,previous_manifest_sha256=sha(test_raw),documents=rows,full_windows=windows,
                  scored_document_count=sum(bool(row["full_starts"]) for row in rows),
                  zero_target_documents=sum(not row["full_starts"] for row in rows),
                  curve_rule="all fullwindows incl short/tailwindows; tokenweighted masks, no selected subset",
                  totals=dict(full_windows=windows,curve_windows=windows,full_target_tokens=targets,curve_target_tokens=targets))
    write_json(destination/"manifest.json",test_v2)
    (destination/"manifest.sha256").write_text(sha(canonical(test_v2))+"\n")
    report=dict(training_manifest_v2_sha256=sha(canonical(train_v2)),test_manifest_v2_sha256=sha(canonical(test_v2)),
                original_training_manifest_sha256=sha(train_raw),original_test_manifest_sha256=sha(test_raw),
                tokens_tokenizer_membership_and_training_permutation_unchanged=True,no_model_inference=True,
                dev_windows=len(starts),dev_targets=int(lengths.sum()),test_windows=windows,test_targets=targets,
                source_sha256=policy["evaluation_policy_source_sha256"])
    write_json(root/"EVALUATION_V2_PREPARED.json",report)
    return report


def prepare(root):
    """One bounded preparation from already checked immutable byte-range prefixes."""
    root = Path(root)
    if (root / "PREPARATION_STARTED.json").exists():
        raise FileExistsError("Preparation already attempted; preserve artifacts and inspect its state")
    write_json(root / "PREPARATION_STARTED.json", dict(started_unix=time.time(), source_sha256=sha(Path(__file__).read_bytes())))
    parsed, trimming = {}, {}
    for role in ("train", "valid"):
        raw = (root / "source" / f"{role}.prefix.txt").read_bytes()
        if sha(raw) != PREFIX_HASHES[role]:
            raise ValueError("Downloaded prefix hash changed")
        parsed[role], trimming[role] = read_complete_stories(raw, role)
    train, candidates, membership = assign_exact(parsed["train"], parsed["valid"])
    print(json.dumps(dict(stage="exact_dedup", train=len(train), valid_candidates=len(candidates), trimming=trimming)), flush=True)
    del parsed
    candidates, evidence = quarantine_overlap(train, candidates, membership)
    roles = {"train": train, "dev": [x for x in candidates if x["role"] == "dev"],
             "test": [x for x in candidates if x["role"] == "test"]}
    with (root / "membership.jsonl").open("x") as stream:
        for row in membership:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(root / "overlap_exclusions.json", evidence)
    print(json.dumps(dict(stage="overlap_audit", counts={k: len(v) for k,v in roles.items()},
                          reasons=dict(Counter(row["reason"] for row in membership)), overlap_events=len(evidence))), flush=True)
    tokenizer, fit = train_tokenizer(train, root / "_prep_dependencies/tokenizers_0_23_2")
    tokenizer.save(str(root / "tokenizer.json"))
    write_json(root / "tokenizer_fit.json", fit)
    vocabulary = [tokenizer.id_to_token(i) for i in range(4096)]
    splits, offsets = {}, {}
    for role, docs in roles.items():
        splits[role], offsets[role] = encode_documents(root, role, docs, tokenizer)
        print(json.dumps(dict(stage="encoded", role=role, **{k:v for k,v in splits[role].items() if k.endswith("count")})), flush=True)
    blocks = (splits["train"]["token_count"]-1) // SEQ_LEN
    if blocks * SEQ_LEN < MIN_TRAIN_TARGETS:
        raise ValueError(f"Insufficient fresh targets: {blocks * SEQ_LEN}")
    generator = torch.Generator().manual_seed(DEFAULT_STREAM_SEED)
    permutation = torch.randperm(blocks, generator=generator).numpy().astype("<u4")
    (root / "train.default_block_permutation.u32").write_bytes(permutation.tobytes())
    dev_starts = evaluation_starts(offsets["dev"])
    (root / "dev.starts.i64").write_bytes(dev_starts.tobytes())
    common = dict(schema_version=1, corpus_kind="tiny_stories_byte_bpe_v1", revision=REVISION,
                  vocabulary=vocabulary, vocab_size=4096, seq_len=SEQ_LEN,
                  tokenizer_sha256=sha((root / "tokenizer.json").read_bytes()),
                  tokenizer_fit=fit, eot_id=fit["eot_id"], token_dtype="little-endian uint16",
                  normalization_for_overlap="NFKC,casefold,Unicode\\w+,single-space words",
                  split_rule="officialtrain stays train; normalizedSHA256 lastbyte even→dev,odd→test for officialvalid",
                  dedup_rule="First normalizedwhole-doc identity; training precedence; shared64word quarantine of validation vs train and both dev/test sides",
                  prefix_hashes=PREFIX_HASHES, trimming=trimming,
                  membership_sha256=sha((root / "membership.jsonl").read_bytes()),
                  overlap_exclusions_sha256=sha((root / "overlap_exclusions.json").read_bytes()),
                  source_manifest_sha256=sha((root / "source/source_manifest.json").read_bytes()),
                  preparation_source_sha256=sha(Path(__file__).read_bytes()))
    training = dict(common, role="training_and_development_only", splits={"train":splits["train"], "val":splits["dev"]},
                    tokenizer=artifact(root / "tokenizer.json", root),
                    dev_starts=artifact(root / "dev.starts.i64", root),
                    default_permutation=artifact(root / "train.default_block_permutation.u32", root),
                    default_stream_seed=DEFAULT_STREAM_SEED, torch_version=str(torch.__version__),
                    train_blocks=blocks, fresh_target_capacity=blocks*SEQ_LEN,
                    training_boundary_rule="packedoriginaldocuments+EOT; contiguous128target blocks may crossdocuments; permutationwithoutreplacement",
                    train_stream_rule="torch.randperm(entire_block_population,generator); fixedprefix across horizons/batches; exhaustionraises",
                    evaluation_boundary_rule="Only full128target windows contained within individual story includingEOT",
                    dev_full_windows=len(dev_starts), dev_target_count=len(dev_starts)*SEQ_LEN)
    write_json(root / "training_manifest.json", training)
    test_docs=[]
    for doc in offsets["test"]:
        starts=list(range(0,doc["end"]-doc["start"]-SEQ_LEN,SEQ_LEN))
        test_docs.append(dict(doc, full_starts=starts, curve_starts=starts,
                             counts=dict(full_windows=len(starts),curve_windows=len(starts),
                                         full_target_characters=len(starts)*SEQ_LEN,curve_target_characters=len(starts)*SEQ_LEN),
                             inclusion_reason="All independently assigned and deduplicatedtest stories; allcompletewithin-storywindows"))
    test = dict(common, role="sealed_confirmation", scoring_authorization="requires_separate_frozen_recipe_and_cohort_lock",
                splits={"test":splits["test"]}, documents=test_docs,
                document_count=len(test_docs), full_windows=sum(len(x["full_starts"]) for x in test_docs),
                curve_rule="allfullwindows perstory; no selected curve subset; shortstorieswithzero128targetwindows retained in manifest",
                character_corpus_reused=False)
    write_json(root / "sealed_test/manifest.json", test)
    (root / "sealed_test/manifest.sha256").write_text(sha(canonical(test))+"\n")
    write_json(root / "PREPARATION_COMPLETE.json", dict(training_manifest_sha256=sha(canonical(training)),
               test_manifest_sha256=sha(canonical(test)), fresh_target_capacity=blocks*SEQ_LEN,
               counts={k:len(v) for k,v in roles.items()}, no_model_inference=True, finished_unix=time.time()))
    return training


class TokenCorpus(CharacterCorpus):
    """CharacterCorpus-compatible batches over fixed IDs; no training replacement."""
    def __init__(self, arrays, vocabulary, manifest, dev_starts, default_permutation=None, dev_documents=(),
                 dev_lengths=None, dev_document_indices=None):
        if set(arrays) != {"train", "val"}:
            raise ValueError("Training loader exposes only train and val(development)")
        self._arrays = {key: torch.as_tensor(value.astype(np.int64)) for key,value in arrays.items()}
        self.vocabulary, self.manifest = tuple(vocabulary), manifest
        self._dev_starts = torch.as_tensor(dev_starts.astype(np.int64))
        self._permutation = None if default_permutation is None else torch.as_tensor(default_permutation.astype(np.int64))
        self._streams = {}
        self.development_documents = tuple(dev_documents)
        self._dev_lengths=(torch.full_like(self._dev_starts,manifest["seq_len"]) if dev_lengths is None
                           else torch.as_tensor(dev_lengths.astype(np.int64)))
        self._dev_document_indices=(None if dev_document_indices is None
                                    else torch.as_tensor(dev_document_indices.astype(np.int64)))
        self.boundaries = {key:(0,len(value)) for key,value in self._arrays.items()}

    def tokens(self, split):
        if split not in self._arrays:
            raise ValueError("Sealed test is not available to the training loader")
        return self._arrays[split]

    def evaluation_starts(self, split, seq_len):
        if split != "val" or seq_len != self.manifest["seq_len"]:
            raise ValueError("Only declared document-contained development windows are available")
        return self._dev_starts.clone()

    def evaluation_document_indices(self, split, seq_len):
        self.evaluation_starts(split, seq_len)
        if self._dev_document_indices is not None:return self._dev_document_indices.clone()
        return torch.tensor([i for i,doc in enumerate(self.development_documents)
                             for _ in range(doc["start"],doc["end"]-seq_len,seq_len)], dtype=torch.long)

    def evaluation_lengths(self, split, seq_len):
        self.evaluation_starts(split,seq_len)
        return self._dev_lengths.clone()

    def evaluation_batch(self, split, starts, seq_len, device="cpu"):
        self.evaluation_starts(split,seq_len)
        starts=torch.as_tensor(starts,device="cpu")
        if starts.ndim!=1 or len(starts)==0 or starts.dtype not in (torch.int32,torch.int64):
            raise ValueError("Nonempty integer evaluation starts required")
        indices=torch.searchsorted(self._dev_starts,starts)
        if (indices>=len(self._dev_starts)).any() or not torch.equal(self._dev_starts[indices],starts):
            raise ValueError("Evaluation start outside declared document-contained bank")
        return padded_evaluation_batch(self.tokens(split),starts,self._dev_lengths[indices],seq_len,
                                       self.manifest["eot_id"],device)

    def batch(self, split, batch_size, seq_len, *, generator=None, positions=None,
              device="cpu", return_positions=False):
        if seq_len != self.manifest["seq_len"]:
            raise ValueError("Context length differs from frozen token blocks")
        if positions is not None:
            starts=torch.as_tensor(positions,device="cpu")
            if split=="val" and not torch.isin(starts,self._dev_starts).all():
                raise ValueError("Development positions must be declared document-contained starts")
            if split=="train" and (starts % seq_len != 0).any():
                raise ValueError("Training positions must be canonical nonoverlap block starts")
        if split=="val" and self.manifest.get("evaluation_policy_version",1)==2:
            if positions is not None and generator is not None:raise ValueError("Positions or generator, not both")
            starts=(self.window_positions(split,batch_size,seq_len,generator=generator) if positions is None
                    else torch.as_tensor(positions,device="cpu"))
            if starts.shape!=(batch_size,):raise ValueError("Wrong evaluation batch size")
            x,y=self.evaluation_batch(split,starts,seq_len,device)
            return (x,y,starts.clone()) if return_positions else (x,y)
        return super().batch(split,batch_size,seq_len,generator=generator,positions=positions,
                             device=device,return_positions=return_positions)

    def window_positions(self, split, batch_size, seq_len, *, generator):
        if batch_size < 1 or seq_len != self.manifest["seq_len"] or generator is None or generator.device.type != "cpu":
            raise ValueError("Explicit CPU generator, positive batch and frozen context length required")
        if split == "val":
            indices=torch.randint(len(self._dev_starts),(batch_size,),generator=generator)
            return self._dev_starts[indices]
        if split != "train":
            raise ValueError("Sealed test is not available to the training loader")
        blocks = (len(self.tokens("train"))-1)//seq_len
        if generator not in self._streams:
            permutation = (self._permutation if self._permutation is not None and generator.initial_seed()==self.manifest["default_stream_seed"]
                           else torch.randperm(blocks,generator=generator))
            self._streams[generator] = [permutation,0]
        permutation,cursor=self._streams[generator]
        if cursor+batch_size>blocks:
            raise ValueError("Fresh training block stream exhausted; recycling is forbidden")
        self._streams[generator][1]+=batch_size
        return permutation[cursor:cursor+batch_size].clone()*seq_len


def load_training_corpus(path, *, expected_manifest_sha256=None):
    """No sealed_test path, contents, or arrays are read by this entry point."""
    path=Path(path)
    if path.is_file():
        if path.name not in ("training_manifest.json","training_manifest_v2.json"):raise ValueError("Expected explicit training manifest")
        root=path.parent;manifest_path=path
    else:root=path;manifest_path=root/"training_manifest.json"
    raw=manifest_path.read_bytes()
    if expected_manifest_sha256 is not None and sha(raw)!=expected_manifest_sha256:
        raise ValueError("Training manifest hash mismatch")
    m=json.loads(raw)
    if m["role"]!="training_and_development_only" or set(m["splits"])!={"train","val"}:
        raise ValueError("Wrong training manifest role")
    verified(root,m["tokenizer"])
    arrays={key:np.frombuffer(verified(root,spec["tokens"]),dtype="<u2") for key,spec in m["splits"].items()}
    offsets={key:json.loads(verified(root,spec["documents"])) for key,spec in m["splits"].items()}
    for key,value in arrays.items():
        if len(value)!=m["splits"][key]["token_count"] or len(value)==0 or value.max()>=m["vocab_size"]:
            raise ValueError("Invalid token array")
        cursor=0
        for doc in offsets[key]:
            if doc["start"]!=cursor or sha(value[doc["start"]:doc["end"]].tobytes())!=doc["token_ids_sha256"]:
                raise ValueError("Document token integrity failure")
            cursor=doc["end"]
        if cursor!=len(value):raise ValueError("Document boundaries incomplete")
    starts=np.frombuffer(verified(root,m["dev_starts"]),dtype="<i8")
    lengths,document_indices=None,None
    if m.get("evaluation_policy_version",1)==2:
        lengths=np.frombuffer(verified(root,m["dev_lengths"]),dtype="<i8")
        document_indices=np.frombuffer(verified(root,m["dev_document_indices"]),dtype="<i8")
        expected_starts,expected_lengths,expected_indices=evaluation_plan_all_targets(offsets["val"],m["seq_len"])
        if not np.array_equal(lengths,expected_lengths) or not np.array_equal(document_indices,expected_indices):
            raise ValueError("Development target coverage mismatch")
        if int(lengths.sum())!=m["dev_target_count"] or len(starts)!=m["dev_full_windows"]:
            raise ValueError("Development evaluation denominator mismatch")
    else:expected_starts=evaluation_starts(offsets["val"],m["seq_len"])
    if not np.array_equal(starts,expected_starts):
        raise ValueError("Development start bank mismatch")
    permutation=np.frombuffer(verified(root,m["default_permutation"]),dtype="<u4")
    actual_blocks=(len(arrays["train"])-1)//m["seq_len"]
    if m["train_blocks"]!=actual_blocks or m["fresh_target_capacity"]!=actual_blocks*m["seq_len"]:
        raise ValueError("Fresh token capacity disagrees with actual packed stream")
    if not np.array_equal(np.sort(permutation),np.arange(m["train_blocks"])):
        raise ValueError("Training permutation is not bijective")
    return TokenCorpus(arrays,m["vocabulary"],m,starts,permutation,offsets["val"],lengths,document_indices)


@dataclass(frozen=True)
class StoryTestDocument:
    slug:str
    token_ids:np.ndarray
    full_starts:np.ndarray
    curve_starts:np.ndarray
    manifest:dict
    full_lengths:np.ndarray=None
    curve_lengths:np.ndarray=None
    def starts(self,bank):
        if bank not in ("full","curve"):raise ValueError("Unknown bank")
        return self.full_starts if bank=="full" else self.curve_starts
    def lengths(self,bank):
        if bank not in ("full","curve"):raise ValueError("Unknown bank")
        return self.full_lengths if bank=="full" else self.curve_lengths


@dataclass(frozen=True)
class StoryTestPanel:
    path:Path
    manifest:dict
    manifest_sha256:str
    documents:tuple
    @property
    def vocabulary(self):return tuple(self.manifest["vocabulary"])
    @property
    def seq_len(self):return self.manifest["seq_len"]


def load_test_panel(path, *, expected_manifest_sha256):
    """Explicit sealed access: evaluator must separately enforce the recipe lock."""
    path=Path(path);root=path.parent if path.is_file() else path
    if path.is_file() and path.name!="manifest.json":raise ValueError("Expected sealed manifest.json")
    raw=(root/"manifest.json").read_bytes()
    if sha(raw)!=expected_manifest_sha256 or sha(raw)!=(root/"manifest.sha256").read_text().strip():
        raise ValueError("Sealed test manifest mismatch")
    m=json.loads(raw)
    if m["role"]!="sealed_confirmation":raise ValueError("Not a sealed panel")
    ids=np.frombuffer(verified(root,m["splits"]["test"]["tokens"]),dtype="<u2")
    offsets=json.loads(verified(root,m["splits"]["test"]["documents"]))
    if len(ids)!=m["splits"]["test"]["token_count"] or len(offsets)!=len(m["documents"]):
        raise ValueError("Test size mismatch")
    documents=[];cursor=0
    for base,row in zip(offsets,m["documents"]):
        if any(row[key]!=value for key,value in base.items()) or row["start"]!=cursor:
            raise ValueError("Test document membership mismatch")
        docids=ids[row["start"]:row["end"]];cursor=row["end"]
        if sha(docids.tobytes())!=row["token_ids_sha256"] or (len(docids) and docids.max()>=m["vocab_size"]):
            raise ValueError("Test token integrity failure")
        if m.get("evaluation_policy_version",1)==2:
            starts,lengths,_=evaluation_plan_all_targets([dict(start=0,end=len(docids))],m["seq_len"])
            if lengths.tolist()!=row["full_lengths"] or row["curve_lengths"]!=row["full_lengths"]:
                raise ValueError("Test target lengths mismatch")
        else:
            starts=np.arange(0,max(0,len(docids)-m["seq_len"]),m["seq_len"],dtype=np.int64)
            lengths=np.full(len(starts),m["seq_len"],dtype=np.int64)
        if starts.tolist()!=row["full_starts"] or row["curve_starts"]!=row["full_starts"]:
            raise ValueError("Test starts mismatch")
        starts.flags.writeable=False
        lengths.flags.writeable=False
        if len(starts):
            targets=int(lengths.sum())
            if m.get("evaluation_policy_version",1)==2 and (row["counts"]["full_target_tokens"]!=targets or row["counts"]["curve_target_tokens"]!=targets):
                raise ValueError("Test document denominator mismatch")
            view=dict(row,counts=dict(row["counts"],full_target_tokens=targets,curve_target_tokens=targets,
                                     full_target_characters=targets,curve_target_characters=targets))
            documents.append(StoryTestDocument(row["identity"],docids,starts,starts,view,lengths,lengths))
    if cursor!=len(ids):raise ValueError("Test boundaries incomplete")
    # Derived compatibility fields do not alter the byte-pinned source manifest.
    # All starts are validated above; zero-window stories remain in its inventory.
    total_windows=sum(len(d.full_starts) for d in documents)
    total_targets=sum(int(d.full_lengths.sum()) for d in documents)
    if total_windows!=m["full_windows"]:raise ValueError("Test full-window denominator mismatch")
    if m.get("evaluation_policy_version",1)==2 and (m["totals"]["full_target_tokens"]!=total_targets or m["totals"]["curve_target_tokens"]!=total_targets):
        raise ValueError("Test total target denominator mismatch")
    m=dict(m,totals=dict(full_windows=total_windows,curve_windows=total_windows,
                         full_target_tokens=total_targets,curve_target_tokens=total_targets,
                         full_target_characters=total_targets,curve_target_characters=total_targets),
           scored_document_count=len(documents),target_count_unit="BPE_tokens",
           compatibility_aliases={"full_target_characters":"full_target_tokens", "curve_target_characters":"curve_target_tokens"})
    return StoryTestPanel(root.resolve(),m,sha(raw),tuple(documents))


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args=parser.parse_args()
    prepare(args.root)
