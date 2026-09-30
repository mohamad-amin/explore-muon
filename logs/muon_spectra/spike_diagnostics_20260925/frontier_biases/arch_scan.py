import json, re, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
H = {"User-Agent": "Mozilla/5.0"}
REPOS = ["deepseek-ai/DeepSeek-V4-Pro-0813", "Qwen/Qwen3.8-2.4T-A95B", "Qwen/Qwen3.8-27B", "zai-org/GLM-5.3",
         "moonshotai/Kimi-K3", "google/gemma-4-31B-it", "unsloth/Llama-4-Maverick-17B-128E-Instruct",
         "mistralai/Mistral-Medium-3.5-128B", "openai/gpt-oss-120b", "MiniMaxAI/MiniMax-M2.7",
         "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "tencent/Hy4-preview", "XiaomiMiMo/MiMo-V2.6-Pro-RL",
         "allenai/Olmo-3.1-32B-Instruct", "ibm-granite/granite-4.2-30b", "CohereLabs/command-a-plus-05-2026-bf16"]
KEYS = re.compile(r"^(hidden_size|num_hidden_layers|n_layers|num_attention_heads|num_key_value_heads|head_dim|hidden_act|"
                  r"hidden_activation|activation|intermediate_size|moe_intermediate_size|rope_theta|partial_rotary_factor|"
                  r"rope_scaling|rotary_dim|no_rope_layers|nope_layer_interval|use_qk_norm|qk_norm|qk_layernorm|"
                  r"tie_word_embeddings|sliding_window|attn_logit_softcapping|final_logit_softcapping|attn_output_gate|"
                  r"use_gated_attention|layer_types|full_attention_interval|kv_lora_rank|q_lora_rank|qk_rope_head_dim|"
                  r"n_routed_experts|num_experts|num_local_experts|vocab_size|max_position_embeddings|mlp_bias|"
                  r"embedding_multiplier|residual_multiplier|logits_scaling|attention_multiplier|norm_type|use_sink|learnable_sink)$")
def get(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=180) as r:
        return json.load(r)
def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k in ("text_config", "language_config", "llm_config", "text_model_config"):
            out.update(flat(v, pre))
        else:
            out.setdefault(k, v)
    return out
def scan(repo):
    try:
        cfg = flat(get(f"https://huggingface.co/{repo}/resolve/main/config.json"))
        sib = [s["rfilename"] for s in get(f"https://huggingface.co/api/models/{repo}")["siblings"]]
        idx = sorted([f for f in sib if f.endswith(".safetensors.index.json")], key=lambda f: (not f.startswith("model"), f))[0]
        names = get(f"https://huggingface.co/{repo}/resolve/main/{idx}")["weight_map"]
        lm = [n for n in names if not re.search(r"vision|visual|audio|mm_projector|multi_modal|aligner|patch_embed|mtp", n)]
        layer = Counter()
        for n in lm:
            m = re.search(r"layers\.(\d+)\.(.*)$", n)
            if m:
                s = re.sub(r"\.\d+\.", ".#.", m.group(2))
                s = re.sub(r"experts\.#\.", "experts.#.", s)
                layer[s] += 1
        top = sorted(k for k in layer if not re.search(r"weight_scale|scale_inv|\.scales$|input_scale", k))
        other = sorted({re.sub(r"\.\d+\.", ".#.", n) for n in lm if "layers." not in n})
        conf = {k: v for k, v in cfg.items() if KEYS.match(k)}
        if isinstance(conf.get("layer_types"), list):
            conf["layer_types"] = dict(Counter(conf["layer_types"]))
        return repo, conf, top, other
    except Exception as e:
        return repo, {"ERROR": str(e)}, [], []
with ThreadPoolExecutor(8) as pool:
    res = list(pool.map(scan, REPOS))
json.dump([{"repo": r, "config": c, "layer_tensors": t, "other_tensors": o} for r, c, t, o in res], open("arch_scan.json", "w"), indent=1)
for r, c, t, o in res:
    print(f"## {r}\n   config: {json.dumps(c)}")
    print("   layer tensors:", ", ".join(x for x in t if "experts.#." not in x or x.endswith(("gate_proj.weight", "w1.weight", "gate_up_proj", "down_proj.weight")))[:1400])
    print("   non-layer tensors:", ", ".join(o)[:400])
