# Bias parameters in the latest open-weight frontier checkpoints (scan of 2026-09-25)

`scan.py` lists each model's tensor names from its Hugging Face safetensors index (or header) and its config. It records every tensor whose name contains "bias" or "sink" (`scan.json`). Gated repos fall back to an ungated `unsloth/` mirror; Llama 4 was read this way. Apertus v1.5 stayed gated and was not scanned. Tensor shapes were spot-checked by hand:
- DeepSeek-V4 `ffn.gate.bias` is [384] = one per routed expert;
- the `attn_sink`, `learnable_sink_param` and `attention_sink_bias` tensors are one per attention head;
- gpt-oss `q_proj.bias` is [4096] and `experts.down_proj_bias` is [128, 2880].

Result: only gpt-oss-120b has linear biases in its language-model attention/MLP. Every other latest flagship scanned has none, and uses RMSNorm without bias; Cohere uses LayerNorm without bias. The remaining bias/sink tensors are:
- per-expert routing biases;
- linear-attention/SSM `dt_bias`/`conv1d.bias`;
- per-head learned attention sinks (DeepSeek-V4/V4.1, Hy4, MiMo, gpt-oss);
- vision-encoder biases.

Earlier checkpoints GLM-4.5 and Qwen2.5-72B had Q/K/V biases; GLM-5.x and Qwen3/3.8 do not.

## Architecture scan (`arch_scan.py`, `arch_scan.json`)

The scan covers the language-model config fields and per-layer tensor names of the latest flagships from 15 labs, including a dense Qwen3.8-27B. Counts are out of those 15:

**Near-universal**
- **Position encoding:** 0/15 have a learned absolute position table; all use RoPE. Some use partial RoPE (Qwen3.8, MiMo, MiniMax) or decoupled RoPE dims in MLA (DeepSeek, GLM, Kimi, Hy4); Llama 4 skips RoPE on every 4th layer.
- **MLP:** 15/15 use a gated MLP (SwiGLU; GeGLU in Gemma 4). Kimi K3's routed experts are ungated.
- **Normalization:** 14/15 use RMSNorm with no bias (Cohere: LayerNorm without bias).
- **Biases:** 14/15 have no linear biases.
- **Embeddings:** 13/15 are untied.
- **Attention:** 15/15 use fewer KV heads than query heads (GQA/MQA) or MLA.

**Split**
- **QK normalization:** explicit q/k norm in 5 (Qwen3.8, Gemma 4, MiniMax, OLMo 3.1, DeepSeek-V4's q/kv norms); normalized MLA latents in 3 (GLM, Kimi, Hy4); absent in the rest.
- **Sinks and gates:** learned attention sinks in 4; attention output gates in 3 (Qwen3.8, Kimi K3, Hy4).
- **Local attention:** interleaved sliding-window/local attention in 6.
- **Extra norms:** sandwich or post-branch norms in 2 (Gemma 4, OLMo 3.1).
- **Residual streams:** hyper-connection multi-stream residuals in 2 (DeepSeek-V4, Hy4).

**Scale features**
- MoE in 11.
- Linear-attention or SSM hybrids in 3.
- Sparse-attention indexers and MTP heads in several.
