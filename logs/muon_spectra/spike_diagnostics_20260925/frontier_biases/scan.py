import json, re, struct, sys, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

REPOS = ["deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-ai/DeepSeek-V4-Pro-0813", "moonshotai/Kimi-K3", "zai-org/GLM-5.3",
         "Qwen/Qwen3.8-2.4T-A95B", "Qwen/Qwen3.8-27B", "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
         "google/gemma-4-31B-it", "google/gemma-4-26B-A4B-it", "mistralai/Mistral-Medium-3.5-128B",
         "mistralai/Mistral-Large-3-675B-Instruct-2512", "openai/gpt-oss-120b", "MiniMaxAI/MiniMax-M2.7",
         "xai-org/grok-2", "CohereLabs/command-a-plus-05-2026-bf16", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
         "ibm-granite/granite-4.2-30b", "tencent/Hy4-preview", "meituan-longcat/LongCat-2.0", "stepfun-ai/Step-3.5-Flash",
         "inclusionAI/Ling-3.0-flash", "XiaomiMiMo/MiMo-V2.6-Pro-RL", "upstage/Solar-Open2-250B",
         "swiss-ai/Apertus-v1.5-70B", "allenai/Olmo-3.1-32B-Instruct", "microsoft/Phi-4-reasoning-vision-15B",
         "baidu/ERNIE-4.5-300B-A47B-PT"]
H = {"User-Agent": "Mozilla/5.0"}

def get(url, rng=None):
    h = dict(H)
    if rng: h["Range"] = rng
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
        return r.read()

def files(repo):
    return [s["rfilename"] for s in json.loads(get(f"https://huggingface.co/api/models/{repo}"))["siblings"]]

def tensor_names(repo, fl):
    idx = [f for f in fl if f.endswith(".safetensors.index.json")]
    idx.sort(key=lambda f: (not f.startswith("model"), f))
    if idx:
        return list(json.loads(get(f"https://huggingface.co/{repo}/resolve/main/{idx[0]}"))["weight_map"]), idx[0]
    st = [f for f in fl if f.endswith(".safetensors")]
    names = []
    for f in st[:4]:
        url = f"https://huggingface.co/{repo}/resolve/main/{f}"
        n = struct.unpack("<Q", get(url, "bytes=0-7"))[0]
        names += [k for k in json.loads(get(url, f"bytes=8-{7 + n}")) if k != "__metadata__"]
    return names, ",".join(st[:4])

def scan(repo):
    tried = [repo]
    for candidate in (repo, "unsloth/" + repo.split("/")[1]):
        try:
            fl = files(candidate)
            names, src = tensor_names(candidate, fl)
            try:
                cfg = json.loads(get(f"https://huggingface.co/{candidate}/resolve/main/config.json"))
            except Exception:
                cfg = {}
            pats = {}
            for n in names:
                if re.search(r"bias|sink", n, re.I):
                    p = re.sub(r"\.\d+\.", ".#.", n)
                    p = re.sub(r"\.\d+\.", ".#.", p)
                    pats[p] = pats.get(p, 0) + 1
            flat = json.dumps(cfg)
            keys = {k: v for k, v in re.findall(r'"([a-z_]*(?:bias|norm_type|layer_norm_eps|rms_norm_eps|sink)[a-z_]*)":\s*("[^"]*"|[\w.\-]+)', flat)}
            return {"repo": repo, "source": candidate, "index": src, "tensors": len(names), "bias_patterns": pats,
                    "config_keys": keys, "arch": cfg.get("architectures") or cfg.get("text_config", {}).get("architectures")}
        except urllib.error.HTTPError as e:
            tried.append(f"{candidate}: HTTP {e.code}")
        except Exception as e:
            tried.append(f"{candidate}: {type(e).__name__} {e}")
    return {"repo": repo, "error": tried}

with ThreadPoolExecutor(8) as pool:
    results = list(pool.map(scan, REPOS))
json.dump(results, open("scan.json", "w"), indent=1)
for r in results:
    if "error" in r:
        print(f"## {r['repo']}: FAILED {r['error']}"); continue
    print(f"## {r['repo']} (from {r['source']}; {r['tensors']} tensors; arch {r['arch']})")
    print("   config:", r["config_keys"])
    pats = sorted(r["bias_patterns"].items(), key=lambda kv: -kv[1])
    print("   bias/sink tensors:", pats[:12] if pats else "NONE")
