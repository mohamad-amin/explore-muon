# Explain Paper Gist

- **Source:** https://chatgpt.com/g/g-p-6ab59590b82081919449faee9b30cd39-muon-v2/c/6ab55118-7e80-83ea-9138-f8b0e981df35
- **Created:** 2026-09-24 11:34 (UTC-05:00)
- **Updated:** 2026-09-24 17:16 (UTC-05:00)
- **Exported:** 2026-09-24 19:51 (UTC-05:00)

---

## User

Can you explain the gist of this paper? https://arxiv.org/pdf/2606.04058

## ChatGPT · gpt-5-6-thinking · Worked for 27s

<details>
<summary>Reasoning</summary>

*Progress note:* I’ll read the paper itself and focus on the core mechanism, what is actually new, and the main empirical/theoretical takeaway rather than walking through every section.

**Framing spectral findings**

I’m organizing the paper’s main points: singular values remain below one, update directions are suppressed, and layer-dependent scaling follows a power law. I’ll connect this to rank-controlled experiments and practical iteration choices.

**Framed spectral findings**

**Searched 3 websites**

</details>

Yes. The paper is **“Spectral Scaling Laws of Muon”** by Magakyan, Parrilo, and Ozdaglar. Its central question is surprisingly narrow but practically important:

> **As models get larger, does Muon’s cheap approximate orthogonalization still work accurately enough?**

Their answer is: **for most layers, yes; for some late layers, increasingly no. And the failure is predictable from scaling laws.** ([arXiv](https://arxiv.org/pdf/2606.04058))

### The core idea

Recall what Muon does. Suppose the momentum matrix has an SVD

$$
M = U\,\mathrm{diag}(\sigma_i)\,V^\top.
$$

Ideal Muon would essentially replace it with its polar factor

$$
M \quad\longrightarrow\quad UV^\top,
$$

so **every singular value becomes 1**. The large and small singular directions therefore contribute on roughly equal footing.

But computing the exact polar decomposition/SVD would be expensive. Muon instead runs a handful of **Newton–Schulz (NS) polynomial iterations**. Importantly, NS doesn't change $U,V$; it applies some scalar function $f$ to every singular value:

$$
U\,\mathrm{diag}(\sigma_i)\,V^\top
\quad\rightarrow\quad
U\,\mathrm{diag}(f(\sigma_i))\,V^\top.
$$

Ideally,

$$
f(\sigma)\approx1.
$$

For reasonably large $\sigma$, that's exactly what happens. But near zero, a finite sequence of odd polynomials **cannot lift arbitrarily tiny singular values to 1**. With a standard five-step NS configuration, sufficiently small directions remain tiny and are effectively suppressed. ([arXiv](https://arxiv.org/pdf/2606.04058))

So the paper asks:

> **How small are Muon's momentum singular values in real LLM training, and how does this change as the model scales?**

---

### Their main empirical discovery

They train GPT-2-style models from **77M to 2.8B parameters** and continuously measure the normalized singular-value spectrum of Muon's momentum matrices at different depths and for Q/K/V/O/MLP matrices. ([arXiv](https://arxiv.org/pdf/2606.04058))

They discover something very clean.

After a short transient at the beginning of training, quantities such as the median singular value essentially settle down:

$$
\sigma_{0.5}(M_t)\longrightarrow \text{approximately constant}.
$$

And that constant depends very predictably on model size.

In fact,

$$
\sigma_q(M) \approx C_q M^{-\alpha}.
$$

So they find **power-law scaling of the momentum spectrum with model size**. Even more interestingly, within a given layer type, different spectral quantiles have approximately the **same exponent** $\alpha$. ([arXiv](https://arxiv.org/pdf/2606.04058))

The striking part is that $\alpha$ varies enormously with network depth.

For many middle layers:

$$
\alpha\approx0.25.
$$

But for some final layers:

$$
\alpha\approx0.66-0.96.
$$

For example, their final MLP layer has approximately

$$
\sigma_q(M)\propto M^{-0.96}.
$$

So as the model grows, the singular values in those late-layer momentum matrices are collapsing toward zero **much faster** than those in ordinary middle layers. ([arXiv](https://arxiv.org/pdf/2606.04058))

---

### Why does that matter?

Because Newton–Schulz has an effective accuracy floor.

Imagine five-step NS works well whenever

$$
\sigma \gtrsim 10^{-3},
$$

but poorly below that.

A middle-layer singular value might scale like

$$
5\times10^{-3} M^{-0.25}.
$$

Even after scaling the model dramatically, it stays above the danger zone.

But a late layer scaling like

$$
10^{-3}M^{-0.66}
$$

can quickly fall below it.

Their 300B extrapolation illustrates this. For a mid-late Q layer, the median singular value is predicted to stay around

$$
1.4\times10^{-3},
$$

which is still handled by the five-step NanoGPT-style NS scheme.

For the **final O projection**, however, they predict roughly

$$
5\times10^{-5},
$$

which is well inside the failure regime. There you'd want something more accurate, such as the **10-step NS composition used by DeepSeek-V4**. ([arXiv](https://arxiv.org/pdf/2606.04058))

So their conclusion is basically:

$$
\boxed{\text{NS accuracy should be layer-dependent.}}
$$

Don't spend 10 NS iterations everywhere. Use cheap NS for most of the network and accurate NS only where the spectrum requires it.

---

### There's one clever experiment that makes the argument much stronger

You might ask:

> Why should we care if a few small singular directions aren't orthogonalized? Maybe those directions aren't important anyway.

They test exactly this.

Instead of full Muon, they explicitly construct an update using only the largest fraction $p$ of singular directions:

$$
UV^\top
\quad\rightarrow\quad
U_{1:k}V_{1:k}^\top.
$$

They find roughly:

- top **90%** of directions → essentially full Muon performance;
- top **50%** → only a small degradation;
- top **25%** → about **10–20% worse token efficiency**;
- top **10%** → roughly **50% worse efficiency**.

So you don't need NS to perfectly orthogonalize absolutely every tiny mode. But you probably want it to reach something like the **median of the spectrum**. ([arXiv](https://arxiv.org/pdf/2606.04058))

That gives meaning to their scaling law: you can track the $q=0.5$ singular value and ask,

$$
f(\sigma_{0.5})\approx1?
$$

If yes, your NS scheme is probably sufficiently accurate.

---

### The deeper picture I take from the paper

There's a useful way of thinking about their result.

Muon is often described abstractly as

$$
M\rightarrow UV^\top.
$$

But **actual Muon is not really $UV^\top$**. It is

$$
M
\rightarrow
U f(\Sigma)V^\top,
$$

where $f$ is determined by your NS implementation.

Consequently, Muon has a hidden hyperparameter:

$$
\boxed{\text{the spectral resolution of its matrix-sign approximation}.}
$$

At small scale this distinction barely matters because most singular values live where $f(\sigma)\simeq1$.

As model width/depth grows, the normalized momentum spectrum shifts toward zero—especially in late layers—so the exact choice of $f$ starts mattering. The implementation of the polar approximation therefore becomes part of the **optimizer's scaling behavior**, not merely a numerical detail.

The paper's Figure 6 makes this particularly clear: the momentum spectrum has one very large outlier and then a huge bulk packed near zero. ([arXiv](https://arxiv.org/pdf/2606.04058))

---

### In one sentence

**Muon's momentum spectra obey surprisingly clean, layer-dependent scaling laws; because approximate Newton–Schulz fails on very small singular values, scaling the model eventually makes the usual cheap Muon approximation inadequate in a few late layers, suggesting adaptive/layer-wise NS accuracy rather than one uniform configuration.**

One important caveat is that the strongest frontier-scale conclusions are **extrapolations**: their actual models stop at 2.8B, and they're using GPT-2-style architectures. The proposed 300B behavior isn't directly measured. The authors explicitly flag architecture generalization as an open question. ([arXiv](https://arxiv.org/pdf/2606.04058))

Also, given our earlier Muon discussion: **this paper is not evidence that Muon's optimization advantage grows superlinearly with batch size or model size.** It is studying how the *numerical approximation inside Muon* behaves as model size increases. That's an important distinction.

---

## User

> *Replying to:*
> So as the model grows, the singular values in those late-layer momentum matrices are collapsing toward zero **much faster** than those in ordinary middle layers

Is this during training, after the short stabilization, or at init?

## ChatGPT · gpt-5-6-thinking · Worked for a few seconds

<details>
<summary>Reasoning</summary>

**Searched 1 website**

</details>

It is **during training, after a short burn-in/stabilization period**, not at initialization.

What they do is track the singular-value quantiles of Muon’s **momentum buffer $M_t$** throughout training. They observe that, for each layer/model size, the spectrum moves around initially but then fairly quickly settles to an approximately stationary level. Their scaling laws are fitted to those **post-burn-in stabilization values**. ([arXiv](https://arxiv.org/pdf/2606.04058))

So when I said

$$
\sigma_q(M)\propto \text{model size}^{-\alpha},
$$

I specifically meant roughly

$$
\boxed{
\text{stabilized training-time singular-value quantile}
\;\propto\;
\text{model size}^{-\alpha}.
}
$$

For example, the claim that the final MLP has $\alpha\approx0.96$ means: train models of 77M, 160M, …, 2.8B; wait until the momentum spectrum has passed its initial transient; measure its stabilized quantiles; then those stabilized values fall almost like $M^{-0.96}$ as model size increases. ([arXiv](https://arxiv.org/pdf/2606.04058))

This is actually an important distinction because **there isn't really a meaningful Muon momentum spectrum “at initialization”** in the same sense—the momentum buffer starts from zero and is built from gradients. The phenomenon is about the characteristic spectrum that emerges under ongoing optimization.

And perhaps the most interesting part is that they claim it becomes **quite stationary after burn-in**: it is not primarily a “late training” phenomenon where singular values progressively collapse. Rather,

$$
\text{early transient}
\rightarrow
\boxed{\text{stable spectral regime}}
$$

and **the location of that stable regime shifts downward with model scale**, especially for the final layers. ([arXiv](https://arxiv.org/pdf/2606.04058))

So I would phrase my previous statement more carefully:

> **As model size grows, the training-time equilibrium/stabilized momentum spectrum of late layers lies increasingly closer to zero.**

Not:

> singular values continuously collapse toward zero over the course of training.

That distinction matters a lot for interpreting what mechanism might be causing it.

---

## User

Is this stabilization initialization dependent?

## ChatGPT · gpt-5-6-thinking · Worked for 21s

<details>
<summary>Reasoning</summary>

**Assessing initialization effects**

I’m treating the stabilization claims as conditional, because the study uses specific scale-aware initialization and doesn’t report seed sweeps or alternate schemes. Momentum normalization removes global scale, but not spectral shape or initialization-dependent dynamics.

**Searched 6 websites**

</details>

That is **not established by the paper**, and it is an important caveat.

There are really two different questions here.

**1. Dependence on the random initialization seed.**  
They do not appear to run a multi-seed experiment where the same architecture/hyperparameters are trained from several independent initializations and the eventual momentum spectra are compared. The figures show the trajectory for each model configuration, rather than mean ± variation over seeds. So the paper does **not demonstrate**

$$
\sigma_q(M_t)
\xrightarrow[t\gg t_{\rm burnin}]{}
s_q
$$

with the *same* $s_q$ independent of the initial draw.

What they establish is only that, **within their runs**, after an initial transient the normalized momentum-spectrum quantiles become very stable. ([arXiv](https://arxiv.org/pdf/2606.04058))

**2. Dependence on the initialization scheme / scaling.**  
This is even more clearly unresolved. They actually use a particular scale-aware initialization: matrix parameters are scaled by

$$
\sqrt{\frac{d_{\rm out}}{d_{\rm in}}},
$$

and the LM head by $1/\sqrt{d_{\rm in}}$, specifically to obtain good transfer across model scales. ([arXiv](https://arxiv.org/pdf/2606.04058))

So their claimed scaling law should really be read as

$$
s_q(M;\ \text{architecture, Muon recipe, init parameterization, data, etc.})
\propto M^{-\alpha}.
$$

It is **not yet shown** that $\alpha$, or even the stabilization value, is invariant to initialization.

There is one subtlety: they measure the spectrum of the **Frobenius-normalized momentum**

$$
\tilde M_t=\frac{M_t}{\|M_t\|_F}.
$$

([arXiv](https://arxiv.org/pdf/2606.04058))

Therefore, if changing initialization merely multiplied all gradients/momentum singular values by a common scalar,

$$
M_t' = cM_t,
$$

the quantity they measure would be exactly unchanged:

$$
\frac{cM_t}{\|cM_t\|_F}
=
\frac{M_t}{\|M_t\|_F}.
$$

But initialization generally does **more** than globally rescale gradients. It changes activation statistics, Jacobians, correlations between directions, layerwise gradient anisotropy, etc., so it can change the **shape** of the normalized spectrum:

$$
\frac{\sigma_i(M)}
{\sqrt{\sum_j\sigma_j(M)^2}}.
$$

And that is precisely what they are measuring.

### So I would interpret their observation as

They have found something that **looks like a training-induced spectral attractor**:

$$
\text{initial condition}
\longrightarrow
\text{transient}
\longrightarrow
\boxed{\text{apparently stationary normalized spectrum}}.
$$

But they have **not shown that it is actually an attractor with respect to initialization**.

In fact, this seems like one of the most natural missing experiments in the paper. For one fixed model size, I would run:

$$
\begin{array}{c}
\text{different seeds}\\
\text{different init variance}\\
\mu\text{P/modded-NanoGPT init}\\
\text{standard GPT init}\\
\text{orthogonal init}
\end{array}
\quad\longrightarrow\quad
\sigma_q\!\left(
M_t/\|M_t\|_F
\right)
$$

and ask whether all trajectories converge to the same $s_q$.

If **different initializations give different early trajectories but converge to essentially the same spectrum**, then the result becomes much more interesting: the spectrum is probably an emergent property of the optimization/training dynamics rather than inherited from initialization.

If instead

$$
s_q = s_q(\text{initialization}),
$$

then their reported power laws may partly be a consequence of the particular scale-aware parameterization they used.

So I would currently call it **“stabilization under a fixed initialization/training recipe,” not initialization-independent stabilization.**

---

## User

Question: needing more iterations means being ill-conditioned, right?

## ChatGPT · gpt-5-6-thinking · Worked for 8s

<details>
<summary>Reasoning</summary>

**Clarifying conditioning**

I’m distinguishing iteration count from condition number. More Newton–Schulz steps usually indicate singular values near zero and poorer conditioning, but the relationship isn’t exact because convergence also depends on normalization and which spectral range the method targets.

**Clarified conditioning concepts**

</details>

Yes, **roughly**, but there is an important distinction.

For Newton–Schulz polar orthogonalization, needing more iterations means that **some singular values are very close to zero relative to the top of the spectrum**. That is closely related to ill-conditioning.

For a matrix

$$
M=U\operatorname{diag}(\sigma_1,\ldots,\sigma_r)V^\top,
$$

the usual condition number is

$$
\kappa(M)=\frac{\sigma_{\max}}{\sigma_{\min}}.
$$

So if $\sigma_{\min}$ gets very small while $\sigma_{\max}$ stays comparable, then

$$
\kappa\gg 1,
$$

and yes, the matrix is ill-conditioned.

The reason this hurts Newton–Schulz is simple. For the standard polar iteration, the scalar dynamics look approximately like

$$
x_{k+1}=\frac12 x_k(3-x_k^2).
$$

Near $x=0$,

$$
x_{k+1}\approx 1.5x_k.
$$

So if you start at

$$
x_0=10^{-4},
$$

it takes several iterations just to get out of the tiny-$x$ regime. By contrast, a singular value starting at $0.3$ gets close to $1$ very quickly.

So:

$$
\boxed{\text{smaller singular values} \Rightarrow
\text{more NS iterations}}
$$

and usually

$$
\boxed{\text{smaller }\sigma_{\min}
\Rightarrow \text{larger condition number}.}
$$

But for **this Muon paper**, I'd use a slightly different phrase than classical ill-conditioning.

### They care about an “effective” condition number

Muon doesn't necessarily care about the **single smallest singular value**.

Suppose the spectrum is

$$
[1,\;0.8,\;0.7,\;0.6,\;\ldots,\;0.4,\;10^{-12}].
$$

Classically,

$$
\kappa\sim10^{12},
$$

so this matrix is horrifically ill-conditioned.

But that one tiny direction may be irrelevant for Muon. Five NS steps can still properly orthogonalize essentially all the other directions.

Conversely, suppose

$$
[1,\;0.02,\;0.015,\;0.01,\;0.008,\ldots].
$$

Now a **large fraction of the spectrum** is small. That's much more consequential for Muon.

So the quantity relevant to their argument is closer to something like

$$
\kappa_q
=
\frac{\sigma_{\max}}{\sigma_q},
$$

where $\sigma_q$ might be the median singular value, rather than

$$
\frac{\sigma_{\max}}{\sigma_{\min}}.
$$

If the median falls from $10^{-2}$ to $10^{-4}$, you need substantially more NS iterations to orthogonalize half of the singular directions.

### So the clean interpretation is

In the paper, when they say larger models/later layers need more NS iterations, they are effectively saying:

$$
\boxed{
\text{their momentum matrices become increasingly spectrally anisotropic / effectively ill-conditioned}.
}
$$

But it's **not necessarily classical condition-number blowup caused by one pathological smallest singular value**. Their result is stronger/more interesting in a sense: **a nontrivial portion of the spectrum is moving toward zero.**

And this connection raises an interesting mechanistic question: *why are late-layer Muon momentum matrices becoming increasingly ill-conditioned with scale?* That seems more fundamental than the NS-iteration observation itself.

---

## User

What do we know about ill-conditioned gradients during training from the literature? This question needs a lot of exploration.

## ChatGPT · gpt-5-6-thinking · Worked for 2m 33s

<details>
<summary>Reasoning</summary>

*Progress note:* I’m going to separate three notions that the literature often conflates: **conditioning of the loss/Hessian**, **conditioning of gradient covariance/Fisher**, and **conditioning of an individual layer’s gradient or momentum matrix**. The Muon question is mostly the third, but the first two may explain why it happens.

**Planning scholarly research**

I’m broadening the search across gradient rank, covariance and Fisher spectra, Hessian conditioning, Jacobian dynamics, gradient confusion, and optimizer geometry. I’ll connect foundational theory with newer methods such as Shampoo, K-FAC, SOAP, and Muon.

**Planned scholarly research**

**Searched 15 websites**

*Progress note:* A first important finding: the phrase **“ill-conditioned gradient” is not standard across deep-learning papers**. Older work mostly studies *curvature* or *Jacobian/Fisher conditioning*; direct conditioning of the **matrix-valued gradient itself** is much less studied, and the Muon-era literature is making that object explicit.

**Connecting gradient geometry**

I’m adding work on stochastic-gradient noise, anisotropy, Hessian alignment, and batch-size effects. This should clarify whether ill-conditioning reflects curvature, gradient covariance, optimization noise, or interactions among them.

**Broadening literature coverage**

I’m including direct studies of low-rank gradients, Fisher spectra, natural-gradient preconditioning, K-FAC, GaLore, and related methods. I’m separating observed gradient structure from curvature and optimizer-induced conditioning.

**Connected gradient geometry and broadened literature coverage**

**Searched 21 websites**

*Progress note:* There is a particularly relevant modern branch I’m checking now: **GaLore/low-rank gradient training, Shampoo/SOAP, and Muon analyses**. These study essentially the same matrix structure from three different angles—compression, second-moment preconditioning, and first-moment spectral flattening.

**Mapping key concepts**

I’m building a conceptual map around gradient matrices, low-rank structure, Fisher and Hessian spectra, stochastic anisotropy, and preconditioning. I’ll distinguish empirical findings from claims that remain architecture- or optimizer-dependent.

**Defining conditioning carefully**

I’m distinguishing gradient-matrix spectral imbalance from Hessian, covariance, Fisher, and Jacobian conditioning. For a linear layer, batch gradients are activation–error cross-covariances, so low rank or ill-conditioning can arise from either side or their alignment.

**Mapped key concepts**

**Linking Fisher structure**

I’m connecting layerwise gradient statistics to Kronecker-factored Fisher approximations: forward activation covariance and backward-error covariance can each create anisotropy, while Jacobian and Hessian conditioning describe related but distinct phenomena.

**Linked Fisher structure**

**Tracing training dynamics**

I’m focusing next on evidence about how gradient spectra, Jacobian singular values, and neural-tangent-kernel conditioning evolve during training, then connecting those trends to optimizer behavior and emerging low-rank methods.

**Traced training dynamics**

**Assessing spectral dynamics**

I’m treating anisotropy as dynamic rather than purely initialization-driven. Low-rank gradients may encode meaningful features, while aggressive spectral flattening could amplify weak or noisy directions, so condition number alone doesn’t establish pathology.

**Assessed spectral dynamics**

**Separating signal from noise**

I’m emphasizing per-mode signal-to-noise ratio: a weak singular direction may represent coherent learning or cancellation-driven noise. Recent work cautions that flattening spectra can help the former but amplify the latter.

**Comparing conditioning notions**

I’m separating Jacobian, Hessian, Fisher, covariance, and gradient-matrix conditioning. They overlap through training dynamics, but a high gradient-matrix condition number alone doesn’t prove that the loss landscape is difficult or optimization is unstable.

**Compared conditioning notions and separated signal from noise**

**Searched 49 websites**

</details>

Yes. I dug through the literature with the Muon question specifically in mind, and there is a useful distinction that changes how I would frame our previous discussion.

**If by “ill-conditioned gradients” we mean a matrix gradient $G=\partial L/\partial W$ having a very unequal singular-value spectrum, this is a real and increasingly well-documented phenomenon. But most of the historical literature on “ill-conditioning in deep learning” is actually about the Hessian, Fisher/gradient covariance, or network Jacobian—not the condition number of $G$ itself.** Direct study of the latter has really accelerated only in the last few years with GaLore, Shampoo/SOAP, and especially Muon.

## 1. Four different kinds of “conditioning”

It helps to keep these separate:

| object | spectrum tells you | classic literature |
|---|---|---|
| $J=\partial f/\partial x$ | whether forward/backward signals explode or vanish | dynamical isometry |
| $H=\nabla^2 L$ | curvature anisotropy of the optimization problem | Hessian conditioning, edge of stability |
| $F$, $C_g$ | anisotropy of per-example gradients / stochastic gradient noise | natural gradient, K-FAC, Shampoo |
| $G=\partial L/\partial W$, or momentum $M$ | anisotropy **inside one matrix-valued update** | GaLore, Muon, PolarGrad |

This distinction is now made explicitly in recent matrix-optimizer work. PolarGrad, for example, calls the first kind **curvature anisotropy** and the last kind **gradient anisotropy**, emphasizing that

$$
\kappa_H=\kappa(\nabla^2L)
$$

and

$$
\kappa_G=\kappa(\nabla_W L)
$$

are genuinely different quantities. ([arXiv](https://arxiv.org/abs/2505.21799))

So, revising my answer from before:

> Needing more Newton–Schulz iterations tells you that the **Muon momentum matrix itself is spectrally ill-conditioned**.

It does **not automatically imply that the loss landscape/Hessian is ill-conditioned**.

They can certainly be related, but equivalence has not been established.

---

# 2. The old literature: loss landscapes are extremely anisotropic

This is the oldest version of the story.

For an ordinary quadratic

$$
L(\theta)
=
\frac12(\theta-\theta^\star)^\top H(\theta-\theta^\star),
$$

gradient descent evolves independently along the Hessian eigenvectors:

$$
e_{t+1,i}
=
(1-\eta\lambda_i)e_{t,i}.
$$

If

$$
\kappa_H=\frac{\lambda_{\max}}{\lambda_{\min}}\gg 1,
$$

then the learning rate is limited by $\lambda_{\max}$, while progress along $\lambda_{\min}$ is painfully slow.

This was already emphasized as a central deep-learning optimization problem by Sutskever et al. in 2013: initialization and momentum could make an enormous difference precisely because neural-network objectives had severe curvature problems. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v28/sutskever13.html))

Empirical Hessian studies subsequently found an extremely characteristic spectrum:

$$
\text{huge near-zero bulk}
+
\text{small number of large outliers}.
$$

Sagun et al. found this bulk-plus-outliers organization, and Ghorbani et al. later showed something even more relevant: **the gradient becomes strongly concentrated in the eigenspaces corresponding to those large Hessian outliers**. ([Hugging Face](https://huggingface.co/papers/1706.04454))

That is already a form of low-dimensional gradient dynamics.

Gur-Ari, Roberts & Dyer went further: across several networks, after an initial transient,

> the gradient dynamically moves into a very small subspace spanned by leading Hessian eigenvectors

and remains there for substantial portions of training. ([arXiv](https://arxiv.org/abs/1812.04754))

So there was evidence as early as 2018–2019 for

$$
\boxed{
\text{training gradient signal becomes strongly directionally concentrated}
}
$$

even though researchers weren't expressing this as the singular-value condition number of an individual layer gradient.

---

# 3. Importantly, this is a **during-training** phenomenon

This connects directly to your previous question about initialization.

There is a separate, very mature literature showing that initialization can create terrible conditioning of the network Jacobian.

Pennington, Schoenholz & Ganguli's dynamical-isometry work showed that controlling the entire Jacobian singular-value distribution—not merely its mean—is crucial for gradient propagation:

$$
\sigma_i(J)\approx 1
$$

is far better than having some singular values $\ll1$ and others $\gg1$, even if

$$
E[\sigma_i^2]\approx1.
$$

Properly conditioned Jacobians can make very deep networks train dramatically faster. ([arXiv](https://arxiv.org/abs/1711.04735))

Transformer work made a similar point: Pre-LN versus Post-LN drastically changes gradient behavior near initialization, and residual scaling/normalization can prevent vanishing, exploding, and rank-collapse effects. ([Microsoft](https://www.microsoft.com/en-us/research/publication/on-layer-normalization-in-the-transformer-architecture/))

But that is **not the whole story**.

The Gur-Ari/Ghorbani results show anisotropy *emerging during training*. And the paper we're discussing, *Spectral Scaling Laws of Muon*, finds that the momentum spectrum goes through

$$
\text{burn-in}
\rightarrow
\boxed{\text{stable anisotropic spectrum}},
$$

with the stationary spectrum depending systematically on model size and layer depth. ([alphaXiv](https://www.alphaxiv.org/audio/2606.04058v1))

So there are at least two distinct mechanisms:

$$
\text{initialization-induced conditioning}
$$

and

$$
\text{training-induced spectral organization}.
$$

The latter appears very real.

---

# 4. Gradient covariance/Fisher gives an even clearer picture

Now move from the mean gradient $g$ to individual sample gradients $g_i$.

Define

$$
F \approx E[g_i g_i^\top]
$$

or noise covariance

$$
C
=
E[(g_i-g)(g_i-g)^\top].
$$

These matrices are also famously very anisotropic.

Karakida, Akaho & Amari found that Fisher spectra in wide networks have pathological scale separation: a small number of very large eigenvalues sit above a huge low-eigenvalue bulk. ([PubMed](https://pubmed.ncbi.nlm.nih.gov/34310678/))

More recently, Xie et al. found that the **covariance eigenvalue spectrum itself often exhibits power-law structure** during neural-network training. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d0b2eda0386f477ab14d7e181e16c899-Abstract.html))

And Feinberg et al.'s *Sketchy* work observed that Kronecker-factored gradient covariance is concentrated in a relatively small leading eigenspace, although that leading eigenspace **moves throughout training**. This motivated explicitly maintaining a low-rank second-moment approximation. ([NeurIPS Papers](https://papers.nips.cc/paper_files/paper/2023/hash/ef72fa6579401ffff9da246a5014f055-Abstract-Conference.html))

There is also substantial work showing that gradient-noise covariance and Hessian curvature are related and often strongly aligned, although they are not identical. SGD noise is highly anisotropic rather than isotropic. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v97/zhu19e))

So a fairly robust empirical picture is:

$$
\boxed{
\text{gradient statistics have a few strong directions and many weak directions.}
}
$$

---

# 5. Now we get to the object Muon actually acts on: the matrix gradient

For a linear layer

$$
y=Wx,
$$

the gradient from one example/token is

$$
G
=
\delta x^\top.
$$

It is **rank one**.

For $N$ examples/tokens, if we stack activations and backward errors,

$$
X=[x_1,\ldots,x_N],
\qquad
\Delta=[\delta_1,\ldots,\delta_N],
$$

then

$$
\boxed{
G=\frac{1}{N}\Delta X^\top.
}
$$

This equation is extremely useful.

It means the gradient spectrum is determined by the interaction between:

$$
\underbrace{X}_{\text{forward representations}}
\qquad\text{and}\qquad
\underbrace{\Delta}_{\text{backpropagated errors}}.
$$

Immediately,

$$
\operatorname{rank}(G)
\le
\min\{
\operatorname{rank}(X),
\operatorname{rank}(\Delta),
N
\}.
$$

So low-rank or badly conditioned matrix gradients are actually quite natural.

They can arise because:

$$
X
\text{ is anisotropic},
$$

or

$$
\Delta
\text{ is anisotropic},
$$

or because only certain combinations of the two are coherently correlated.

This is also closely related to why K-FAC works. K-FAC approximates a layer Fisher block as

$$
F_W
\approx
E[xx^\top]
\otimes
E[\delta\delta^\top].
$$

So it separately models the forward and backward sides of exactly this outer-product structure. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v37/martens15.html))

In the ideal Kronecker setting,

$$
\kappa(F_W)
\approx
\kappa(E[xx^\top])
\;
\kappa(E[\delta\delta^\top]).
$$

That is one plausible bridge between the old curvature-conditioning literature and the modern Muon gradient-spectrum observation.

---

# 6. GaLore gave some of the first direct LLM evidence

This is very relevant.

GaLore was built around the empirical observation that LLM weight gradients can be represented well in a much lower-dimensional subspace than the full matrix dimensions.

Instead of storing optimizer state for

$$
G_t\in\mathbb R^{m\times n},
$$

GaLore finds a truncated SVD-based projection and performs optimization in that low-dimensional gradient subspace. It works for full-parameter LLM pretraining up to billions of parameters. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v235/zhao24s.html))

And GaLore 2 continued scaling this approach to 7B pretraining over hundreds of billions of tokens. ([Hugging Face](https://huggingface.co/papers/2504.20437))

This does **not** mean gradients literally have fixed low rank throughout training.

The important observation is closer to:

$$
\boxed{
\text{most gradient energy often occupies relatively few singular directions.}
}
$$

And the relevant subspace changes with training.

That is already very close to saying the matrix is **effectively ill-conditioned**.

There is now also theoretical work. *Low Rank Gradients and Where to Find Them* proves approximate low-rank gradient structure in two-layer networks under fairly general anisotropic data assumptions and identifies dominant rank-one components linked to data/residual structure. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/df7ac32d352db77aad1f355ad43206a9-Abstract-Conference.html))

---

# 7. Some newer LLM evidence is even more direct

Several 2026 papers are now explicitly talking about **spectral anisotropy of LLM gradients**.

For example, *Spectra: Rethinking Optimizers for LLMs Under Spectral Anisotropy* reports a spike–tail structure in LLM gradients that persists through training: roughly a very small fraction of spectral directions carries a disproportionate amount of gradient energy. It proposes treating the strong spike and weak tail differently. This is recent preprint evidence, so I would not treat its precise quantitative claims as settled yet. ([arXiv](https://arxiv.org/abs/2602.11185))

Likewise, *Muon in Vision Transformers* directly measures matrix-gradient singular spectra and finds that training recipe and optimizer can substantially alter spectral concentration. In Muon runs without strong augmentation, deep MLP-down gradients undergo late-training **spectral concentration / mode collapse**; with AdamW versus Muon, QKV gradients occupy noticeably different spectral bases. ([arXiv](https://arxiv.org/abs/2605.24770))

And the paper we're discussing gives perhaps the cleanest systematic LLM result for **momentum rather than raw gradient**:

$$
\frac{M_t}{\|M_t\|_F}
$$

develops a stable spectrum after burn-in, and characteristic quantiles shrink with model scale according to approximately power-law relations that depend strongly on layer depth/type. ([alphaXiv](https://www.alphaxiv.org/audio/2606.04058v1))

So the direct matrix-gradient/momentum evidence now lines up quite well with the older Fisher/Hessian evidence.

---

# 8. But there is an important trap: **high gradient condition number is not necessarily bad**

This is perhaps the most important conceptual point.

Suppose

$$
G
=
U
\begin{bmatrix}
100 &&\\
&10&\\
&&10^{-3}
\end{bmatrix}
V^\top.
$$

Why is the third mode tiny?

There are at least two radically different possibilities.

### Case A: weak but real learning signal

The third mode represents an important feature whose gradient happens to be systematically weak.

Then ordinary gradient descent barely learns it:

$$
\Delta W_3\propto10^{-3}.
$$

Muon changes this to approximately

$$
\Delta W_3\propto1.
$$

This is potentially extremely useful.

### Case B: cancellation/noise

The third mode has

$$
E[g_3]\approx0
$$

because individual examples produce large gradients with inconsistent signs.

Then its signal-to-noise ratio may be terrible:

$$
\mathrm{SNR}_3
=
\frac{|E[g_3]|}
{\sqrt{\operatorname{Var}(g_3)}}\ll1.
$$

Muon forcing this mode to unit magnitude can amount to **amplifying noise**.

These cases have the same singular value of the *mean gradient*.

So

$$
\kappa(G)
$$

alone cannot tell us whether flattening is beneficial.

This is one reason I find the recent SpecMuon/Spectra line interesting: they explicitly question whether blindly making every singular mode equally large is always desirable. SpecMuon introduces mode-dependent control rather than simply forcing all singular values to one. ([arXiv](https://arxiv.org/abs/2602.16167))

---

# 9. This changes how I think about Muon

Muon takes

$$
M=U\Sigma V^\top
$$

and approximately maps

$$
M
\longrightarrow
UV^\top.
$$

In other words,

$$
\kappa(M)
\gg 1
\quad\longrightarrow\quad
\kappa(\operatorname{Muon}(M))
\approx1.
$$

It is the ultimate **gradient-anisotropy preconditioner**.

There is growing theory suggesting that this can remove condition-number dependence in certain structured problems. Ma et al. prove, for matrix factorization and simplified linear-transformer problems, that spectral orthogonalization yields convergence rates independent of the relevant condition number under their assumptions. ([arXiv](https://arxiv.org/abs/2601.13474))

Another recent analysis argues that spectral flattening allows Muon to tolerate larger learning rates because ordinary gradient descent is constrained by the largest spectral direction, whereas the orthogonalized update effectively depends on a much less extreme scale; their Kronecker model connects the gain to the gradient-covariance spectrum. ([arXiv](https://arxiv.org/abs/2605.13079))

So there is a plausible high-level story:

$$
\text{gradient spectrum becomes anisotropic}
$$

$$
\Downarrow
$$

$$
\text{ordinary optimizers make much more progress in strong modes}
$$

$$
\Downarrow
$$

$$
\text{Muon equalizes progress across matrix singular modes}.
$$

But we do **not** yet have a general theorem saying that this is the reason Muon beats Adam in real LLMs.

---

# 10. Adam, Shampoo, SOAP and Muon actually attack different versions of the same problem

This comparison is illuminating.

### Adam

Adam performs coordinatewise normalization:

$$
g_i
\mapsto
\frac{m_i}{\sqrt{v_i}}.
$$

It's invariant to diagonal rescaling of parameters/gradients, but it does not know the singular vectors of a matrix. ([arXiv](https://arxiv.org/abs/1412.6980))

If the problematic directions are rotated relative to coordinate axes, Adam can't directly whiten them.

### Shampoo

Shampoo accumulates things like

$$
L_t=\sum_\tau G_\tau G_\tau^\top,
\qquad
R_t=\sum_\tau G_\tau^\top G_\tau,
$$

and preconditions using matrix inverse roots.

It therefore learns a **second-moment spectral geometry**. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v80/gupta18a))

### SOAP

SOAP essentially combines this matrix eigenbasis with Adam-like adaptive scaling inside that rotated basis. ([arXiv](https://arxiv.org/abs/2409.11321))

### Muon

Muon instead operates primarily on the **first-moment/momentum matrix**

$$
M_t
$$

and discards its singular magnitudes.

So you could crudely summarize them as

$$
\begin{array}{c|c}
\text{optimizer}&\text{anisotropy it handles}\\
\hline
\text{Adam}&\text{coordinate second moment}\\
\text{K-FAC}&\text{activation/error curvature}\\
\text{Shampoo/SOAP}&\text{matrix second moment}\\
\text{Muon}&\text{matrix first-moment spectrum}
\end{array}
$$

That is why the Muon spectrum paper is interesting: it is measuring **the exact thing its optimizer destroys**.

---

# 11. Another subtlety: classical condition number may actually be the wrong metric

For neural-network gradients, exact rank deficiency is common.

If

$$
\sigma_{\min}=0,
$$

then

$$
\kappa(G)=\infty.
$$

That doesn't mean training is infinitely pathological.

So for this context I would prefer quantities such as

$$
\boxed{
\kappa_q(G)
=
\frac{\sigma_1}{\sigma_q}
}
$$

where $q$ might be the median or 25th percentile.

Or stable rank:

$$
r_{\mathrm{stable}}
=
\frac{\|G\|_F^2}{\|G\|_2^2}
=
\frac{\sum_i\sigma_i^2}
{\sigma_1^2}.
$$

Or entropy/effective rank.

This is exactly why the Muon scaling paper's choice to look at **singular-value quantiles** is much more meaningful than just reporting $\sigma_{\min}$.

For NS specifically, what matters is something like

$$
\frac{\sigma_q}{\sigma_{\max}}
$$

for the fraction $q$ of modes you care about orthogonalizing.

---

# 12. Momentum makes the problem even more interesting

Muon doesn't orthogonalize a single-batch gradient:

$$
G_t.
$$

It orthogonalizes approximately

$$
M_t
=
\beta M_{t-1}
+
(1-\beta)G_t.
$$

Suppose each $G_t$ is low rank, but its singular vectors slowly rotate.

Then

$$
M_t
=
\sum_{k\ge0}
(1-\beta)\beta^kG_{t-k}
$$

can actually have substantially **higher rank** than each constituent gradient.

So the paper finding that **momentum itself** remains highly spectrally concentrated is stronger than simply saying individual minibatch gradients are low rank.

It suggests persistent coherence:

$$
\text{the same spectral directions remain important across many consecutive steps}.
$$

That's consistent with GaLore/Sketchy observations that dominant gradient subspaces evolve relatively slowly rather than being completely random from batch to batch. ([NeurIPS Papers](https://papers.nips.cc/paper_files/paper/2023/hash/ef72fa6579401ffff9da246a5014f055-Abstract-Conference.html))

---

# 13. What seems reasonably established now

I would separate the evidence into confidence levels.

### Fairly established

Deep-learning optimization geometry is extremely anisotropic. Hessians, Fishers, gradient covariance, and Jacobians typically possess broad spectra with strong leading directions. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v97/ghorbani19b))

Gradient dynamics concentrate into low-dimensional directions during training, and this concentration is not merely an initialization effect. ([arXiv](https://arxiv.org/abs/1812.04754))

Layerwise matrix gradients in modern neural networks often have strong low-rank/effective-low-rank structure; this is sufficiently strong that methods such as GaLore can exploit it in LLM pretraining. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v235/zhao24s.html))

### Strong recent evidence, but younger literature

Muon-style gradient/momentum matrices have strongly unequal singular spectra, and spectral flattening seems to explain a meaningful part of Muon's behavior. ([arXiv](https://arxiv.org/abs/2601.13474))

Those spectra depend on layer, architecture, data recipe, optimizer, and model scale. ([arXiv](https://arxiv.org/abs/2605.24770))

### Not established

We do **not** know a universal causal chain

$$
\text{Hessian ill-conditioning}
\Rightarrow
\text{gradient matrix ill-conditioning}
\Rightarrow
\text{Muon advantage}.
$$

Nor do we know whether the small singular modes of real LLM gradients are predominantly

$$
\text{weak useful signal}
$$

or

$$
\text{noise/cancellation}.
$$

That distinction is crucial.

---

# 14. And this is where I think there is a real missing research question

The 2606.04058 paper tells us:

$$
\boxed{
\text{as model size increases, some layer momentum spectra become increasingly skewed.}
}
$$

But it mostly stops at describing the scaling law.

The deeper question is:

> **What generates this spectral skew?**

Because for a layer,

$$
G=\Delta X^\top,
$$

you could decompose the phenomenon into three possibilities:

$$
\boxed{
\begin{aligned}
&\text{forward anisotropy: } &&X X^\top\\
&\text{backward anisotropy: }&&\Delta\Delta^\top\\
&\text{forward/backward alignment: }&&\Delta X^\top
\end{aligned}
}
$$

and ask which one scales with model size.

That experiment seems extremely informative.

For each layer and checkpoint, measure simultaneously:

$$
\operatorname{spec}(XX^\top),
$$

$$
\operatorname{spec}(\Delta\Delta^\top),
$$

$$
\operatorname{spec}(G),
$$

$$
\operatorname{spec}(M),
$$

and ideally the Shampoo/K-FAC factors.

Then ask whether the paper's observed relation

$$
\sigma_q(M)\sim N^{-\alpha_\ell}
$$

can be explained by

$$
\sigma_q(XX^\top),
$$

by

$$
\sigma_q(\Delta\Delta^\top),
$$

or by increasingly strong **alignment/correlation between them**.

That would connect three largely separate literatures:

$$
\text{representation geometry}
\leftrightarrow
\text{backpropagation geometry}
\leftrightarrow
\text{Muon gradient geometry}.
$$

---

# 15. The experiment I find even more compelling

For each singular mode

$$
G = \sum_i \sigma_i u_i v_i^\top,
$$

measure its **per-example or per-microbatch signal-to-noise ratio**:

$$
s_{i,b}
=
u_i^\top G_b v_i,
$$

then

$$
\mathrm{SNR}_i
=
\frac{|E_b[s_{i,b}]|}
{\sqrt{\operatorname{Var}_b(s_{i,b})}}.
$$

Now compare

$$
\sigma_i
\quad\text{versus}\quad
\mathrm{SNR}_i.
$$

That would tell us what the tiny singular values actually mean.

If

$$
\sigma_i\downarrow
\quad\text{but}\quad
\mathrm{SNR}_i \text{ stays high},
$$

then Muon's amplification of weak modes is probably performing useful preconditioning.

If instead

$$
\sigma_i\downarrow
\quad\text{and}\quad
\mathrm{SNR}_i\downarrow,
$$

then increasingly accurate NS may actually be **amplifying increasingly noisy modes**.

And then the conclusion of the scaling-law paper—

> larger models need more accurate NS

—would need an important qualification:

$$
\boxed{
\text{only if those newly recovered singular directions contain useful signal.}
}
$$

I don't think the current literature answers that.

---

## So my current overall view

I think there is a real phenomenon underneath the Muon paper, rather than merely a numerical Newton–Schulz curiosity:

$$
\boxed{
\text{neural-network training naturally produces strongly anisotropic matrix gradients.}
}
$$

We have seen pieces of this for a decade in Hessian/Fisher/Jacobian work, and direct evidence for **matrix gradient/momentum anisotropy** is now becoming much stronger through GaLore, Shampoo/SOAP, PolarGrad and Muon.

What is surprisingly **not** well understood is why the matrix-gradient spectrum takes the shape it does, why its conditioning changes so strongly with depth and model scale, and whether the weak modes are useful signal or stochastic junk.

That, to me, is the scientifically interesting question behind the paper—not “how many Newton–Schulz steps should we use?”

---

## User

And what causes this?

## ChatGPT · gpt-5-6-thinking · Worked for 2m 14s

<details>
<summary>Reasoning</summary>

*Progress note:* There are several plausible causes, and they are not all equivalent. I’m checking which ones have actual evidence behind them—especially whether the skew comes from **activations $X$**, **backpropagated errors $\Delta$**, their alignment, or the curvature/noise geometry around them.

**Explaining gradient spectra**

I’m connecting the spectrum to activation and error covariances, while separating algebraic rank-one token contributions from anisotropic activations, correlated backpropagated errors, downstream contraction, and task-driven low-dimensional structure.

**Explained gradient spectra**

**Searched 29 websites**

*Progress note:* I found a crucial correction to our framing: **the $M^{-0.25}$-type shrinkage in this paper does not by itself imply worsening condition number.** Because they Frobenius-normalize the momentum, even a *perfectly conditioned* $d\times d$ orthogonal matrix has every singular value equal to $1/\sqrt d$, which shrinks automatically as width grows. I’m checking how much of their scaling is just this dimensional baseline versus genuine spectral concentration.

**Refining the scaling analysis**

I’m checking the paper’s detailed evidence and distinguishing normalized singular-value behavior from raw gradient scale. I’m also evaluating whether width scaling, outliers, bulk spectra, and conditioning explain the observed exponent near −0.25.

**Clarifying normalization effects**

I’m separating scale from conditioning: normalized singular values shrink with matrix dimension even when condition numbers stay fixed. The observed outlier-plus-bulk spectrum suggests genuine anisotropy, but the exponent may partly reflect dimensional normalization.

**Refined scaling analysis and clarified normalization effects**

**Searched 11 websites**

**Computed the diameter–mass scaling exponent**

**Interpreting the exponent**

I’m comparing the observed scaling with a dimension-only baseline: normalized singular values naturally shrink like $d^{-1/2}$. A mid-layer exponent near −0.25 may largely reflect width growth, while late layers show stronger spectral concentration.

**Explaining spectral structure**

I’m relating each token gradient to an outer product of activation and error vectors. Low effective rank requires aligned, anisotropic covariances—not merely random independent factors—because normalization doesn’t whiten features, and Jacobian products can amplify selected directions.

**Explained the exponent and spectral structure**

**Checking scaling evidence**

I’m looking for the reported model configurations and quantile behavior to test whether the exponent reflects dimension alone. If all quantiles share one slope, that supports broad scale effects; deviations in late layers would indicate stronger anisotropy.

**Checked scaling evidence**

**Reassessing conditioning claims**

I’m noting that equal slopes across bulk quantiles imply roughly self-similar shape, not necessarily worsening conditioning. Faster late-layer decay may instead reflect dimensional growth or increasing dominance by leading singular directions.

**Reassessed conditioning claims**

**Searched 17 websites**

</details>

Yes—and after looking more carefully, I think there is a much cleaner answer than “the gradients become ill-conditioned.”

I need to correct one thing from my previous answer:

> **Needing more Newton–Schulz iterations in this paper is not equivalent to the momentum matrix becoming more ill-conditioned.**

That distinction turns out to be important for understanding what causes the phenomenon.

### 1. There is a completely trivial dimensional cause

The paper feeds Newton–Schulz

$$
\widetilde M=\frac{M}{\|M\|_F}.
$$

([arXiv](https://arxiv.org/html/2606.04058))

Now imagine the *best-conditioned matrix possible*: an $r\times r$ matrix with

$$
\sigma_1=\sigma_2=\cdots=\sigma_r=c.
$$

Its condition number is

$$
\kappa(M)=1.
$$

But after Frobenius normalization,

$$
\|M\|_F=c\sqrt r
$$

and therefore every singular value becomes

$$
\boxed{
\sigma_i(\widetilde M)=\frac1{\sqrt r}.
}
$$

So as the matrix gets wider,

$$
r\uparrow
\quad\Rightarrow\quad
\sigma_i(\widetilde M)\downarrow,
$$

**even though its condition number stays exactly 1.**

And Newton–Schulz cares about that absolute normalized value: its polynomial is bad near zero. So a larger *perfectly conditioned* matrix can require more NS iterations.

That means there are two separate quantities:

$$
\boxed{
\text{NS difficulty}
\neq
\text{condition number}.
}
$$

NS difficulty here depends on something like

$$
\frac{\sigma_q(M)}{\|M\|_F},
$$

whereas classical conditioning depends on

$$
\frac{\sigma_1(M)}{\sigma_r(M)}.
$$

---

## 2. And remarkably, this may explain most of their $M^{-0.25}$ result

Look at their architecture scaling:

$$
d:
512,\ 768,\ 1024,\ 1280,\ 1792,\ 2048,\ 2560
$$

while model size goes from 77M to 2.8B. ([arXiv](https://arxiv.org/html/2606.04058))

If I fit those numbers, approximately

$$
d\propto M^{0.443}.
$$

For a roughly full-rank $d\times d$ matrix whose spectral shape remains fixed,

$$
\sigma_q\left(\frac{M}{\|M\|_F}\right)
\sim
\frac{1}{\sqrt d},
$$

so we'd expect

$$
\frac1{\sqrt d}
\sim
M^{-0.443/2}
=
\boxed{M^{-0.221}}.
$$

Their ordinary mid-network layers empirically give approximately

$$
M^{-0.25}.
$$

([arXiv](https://arxiv.org/html/2606.04058))

Those are strikingly close.

So I would now interpret their mid-layer scaling very differently:

$$
\boxed{
M^{-0.25}
\text{ may be largely ordinary width/rank scaling under Frobenius normalization.}
}
$$

It does **not** necessarily say that these matrices become increasingly ill-conditioned.

This is something I wish the paper had normalized out explicitly.

---

# 3. There *is* genuine spectral anisotropy, though

This doesn't make the whole observation trivial.

They also observe that the momentum spectra have:

- one extremely large singular-value outlier, often $10\times+$ the bulk;
- then a bulk heavily concentrated near zero. ([arXiv](https://arxiv.org/html/2606.04058))

That's genuine anisotropy.

And, more importantly, some late layers scale like

$$
M^{-0.66}
$$

or even

$$
M^{-0.96}.
$$

([arXiv](https://arxiv.org/html/2606.04058))

Width normalization can't plausibly account for that if its baseline is roughly $M^{-0.22}$.

So I'd decompose the observed exponent as conceptually

$$
\alpha_{\rm observed}
\approx
\underbrace{\alpha_{\rm dimension}}_{\sim -0.22}
+
\underbrace{\alpha_{\rm concentration}}_{\text{actual change in spectral shape}}.
$$

Thus:

$$
\begin{array}{c|c}
\text{layer} & \text{interpretation}\\ \hline
\text{middle, }-0.25 & \text{mostly dimensional baseline?}\\
\text{final O, }-0.66 & \text{substantial extra concentration}\\
\text{final MLP, }-0.96 & \text{very strong extra concentration}
\end{array}
$$

The late-layer effect is therefore the genuinely mysterious part.

---

# 4. There is another clue in the paper supporting this interpretation

They report that **all five quantiles $q=.1,.25,.5,.75,.9$ have approximately the same scaling exponent within each layer type.** ([arXiv](https://arxiv.org/html/2606.04058))

That's interesting.

If

$$
\sigma_{.1}\sim M^{-\alpha},
\qquad
\sigma_{.5}\sim M^{-\alpha},
\qquad
\sigma_{.9}\sim M^{-\alpha},
$$

then ratios such as

$$
\frac{\sigma_{.1}}{\sigma_{.9}}
$$

are approximately constant with scale.

So the **shape of the bulk is roughly self-similar**.

Again, that is not what I would expect if the entire bulk were progressively becoming more and more ill-conditioned internally.

Rather, it looks something like

$$
\boxed{
\text{same-shaped bulk}
\times
\text{an overall shrinking scale}.
}
$$

Because everything is Frobenius normalized, one obvious way this can happen is that increasing Frobenius energy is sitting elsewhere—especially in the giant leading singular direction.

The paper doesn't directly establish that explanation because it doesn't give a corresponding scaling law for $\sigma_1$, but I think it is an important possibility.

---

# So where does the *actual anisotropy* come from?

Now we get to the deeper question.

For essentially every transformer linear layer,

$$
y=Wx,
$$

one token/example gives

$$
\nabla_W L
=
\delta x^\top,
$$

where

$$
x=\text{input activation},
\qquad
\delta=\frac{\partial L}{\partial y}.
$$

That's rank one.

Over many tokens,

$$
\boxed{
G
=
\sum_b\delta_bx_b^\top
=
\Delta X^\top.
}
$$

This is, in my view, the right equation for understanding Muon's spectrum.

There are therefore only three places spectral concentration can fundamentally come from:

$$
\boxed{
X,\qquad \Delta,\qquad \text{their correlation}.
}
$$

---

## 5. Cause #1: the activations $X$ are highly anisotropic

Suppose activations occupy only a few strong feature directions.

Write

$$
C_x
=
E[xx^\top].
$$

If

$$
\lambda_1(C_x)\gg\lambda_{100}(C_x),
$$

then a large amount of input variation lives in a low-dimensional subspace.

Transformers are known to develop highly anisotropic hidden representations; importantly, LayerNorm/RMSNorm controls token norms but **does not whiten the feature covariance matrix**. Transformer self-attention itself has also been implicated in producing representation anisotropy. ([ACL Anthology](https://aclanthology.org/2024.eacl-long.3/))

So even if

$$
\|x\|\approx\text{constant},
$$

you can have

$$
C_x=
\begin{pmatrix}
100&&\\
&1&\\
&&0.001\\
&&&\ddots
\end{pmatrix}.
$$

That immediately makes

$$
G=\Delta X^\top
$$

anisotropic.

This is essentially one half of the classical K-FAC picture: layerwise curvature factors into activation statistics and backpropagated-error statistics. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v37/martens15.html))

---

# 6. Cause #2: the backward signals $\Delta$ are even more structured

Now consider

$$
\delta_\ell
=
J_{\ell+1:L}^{\top}\delta_L.
$$

The downstream network acts as a matrix-valued filter.

Some directions of $\delta_L$ are preserved or amplified, while others contract:

$$
\delta_\ell
=
J^\top\delta_L.
$$

If

$$
J
=
U
\begin{bmatrix}
1&&\\
&0.3&\\
&&10^{-3}
\end{bmatrix}
V^\top,
$$

then after backpropagation,

$$
\Delta
$$

naturally becomes spectrally uneven.

This was one of the original motivations behind dynamical-isometry/Jacobian-conditioning work, and transformer rank-collapse analyses show concrete cases where representation geometry causes certain gradient directions—e.g. Q/K gradients—to become very small. ([arXiv](https://arxiv.org/abs/2206.03126))

So another plausible chain is

$$
\text{structured output error}
\rightarrow
\text{anisotropic Jacobians}
\rightarrow
\text{anisotropic }\Delta
\rightarrow
\text{anisotropic }G.
$$

---

# 7. Cause #3: forward and backward directions align

Even if both $X$ and $\Delta$ have moderate anisotropy, their cross-correlation can be extremely concentrated.

Consider

$$
G
=
E[\delta x^\top].
$$

If a particular activation feature $v$ reliably predicts a particular error feature $u$, then

$$
E[(u^\top\delta)(v^\top x)]
$$

is large.

That produces a singular component

$$
\sigma\,uv^\top.
$$

A few highly reproducible input/error associations can therefore create

$$
G
\simeq
\sigma_1u_1v_1^\top
+
\sigma_2u_2v_2^\top
+\text{small residue}.
$$

This is closely related to recent theory showing approximately low-rank gradients arising from structured **data–residual correlations**, not merely from low-rank inputs themselves. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/df7ac32d352db77aad1f355ad43206a9-Abstract-Conference.html))

I suspect this is more important than simple activation anisotropy.

---

# 8. Training itself can *create* these strong directions

There's a second level of explanation.

Why would $X$ and $\Delta$ become so structured during training?

Because learning creates features that are repeatedly useful.

Suppose a feature is useful across millions of examples. Its gradient contribution has coherent sign:

$$
g_t^{\rm signal}\approx s.
$$

Another direction has weak, example-specific contributions:

$$
E[g_t^{\rm noise}]\approx0.
$$

After averaging many examples,

$$
\sum_t g_t^{\rm signal}
\sim T,
$$

while random components grow only like

$$
\left\|\sum_t g_t^{\rm noise}\right\|
\sim \sqrt T.
$$

Hence

$$
\frac{\text{coherent direction}}
{\text{incoherent direction}}
\sim\sqrt T.
$$

Learning therefore naturally produces a few **coherent** gradient directions sitting above a much broader noise floor.

GaLore's empirical success is consistent with gradients having substantial transient low-dimensional structure during LLM training. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v235/zhao24s.html))

---

# 9. SGD noise can itself manufacture very large gradient components

This is a subtler and important mechanism.

Older work found that gradients align strongly with the large-eigenvalue Hessian subspace during training. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v97/ghorbani19b))

It was tempting to interpret this as:

> those directions must be where useful learning happens.

But more recent work challenged that.

Song, Ahn & Yun found that the alignment largely disappears when switching from SGD to full-batch GD, and reappears when switching back to SGD. They conclude that **stochastic minibatch noise is a main cause of the dominant-subspace alignment**. ([OpenReview](https://openreview.net/pdf/8ffb5bb6c18e6af7a0141e60e32085de7baa63e3.pdf))

A simple quadratic explains why.

Take a curvature direction $i$:

$$
L_i=\frac12\lambda_i\theta_i^2.
$$

SGD gives roughly

$$
\theta_{t+1,i}
=
(1-\eta\lambda_i)\theta_{t,i}
-
\eta\xi_{t,i}.
$$

In stationarity, minibatch noise continually kicks the parameters away from the valley floor, while curvature pulls them back.

For small $\eta$,

$$
\operatorname{Var}(\theta_i)
\approx
\frac{\eta\,\operatorname{Var}(\xi_i)}
{2\lambda_i}.
$$

But the gradient is

$$
g_i=\lambda_i\theta_i,
$$

so

$$
\operatorname{Var}(g_i)
\approx
\frac{\eta\lambda_i}{2}
\operatorname{Var}(\xi_i).
$$

Thus, all else equal,

$$
\boxed{
\lambda_i\uparrow
\quad\Rightarrow\quad
\text{gradient fluctuations}\uparrow.
}
$$

So sharp directions can dominate gradient magnitude because SGD **oscillates in them**, not because they are the directions responsible for long-run optimization progress.

This can create huge spectral outliers.

---

# 10. And then Muon's momentum makes the spectrum *more* concentrated

This is especially relevant because the scaling paper is **not measuring raw gradients**.

It measures

$$
M_t
=
\mu M_{t-1}+G_t.
$$

That changes the story materially.

Suppose

$$
G_t
=
S + N_t,
$$

where $S$ is a temporally persistent gradient component and

$$
E[N_t]=0.
$$

Then momentum gives

$$
M_t
\approx
\frac{1}{1-\mu}S
+
\sum_k\mu^kN_{t-k}.
$$

The persistent component adds coherently.

The random component partially cancels.

A very recent 2026 analysis of Muon formalizes essentially this mechanism: **momentum acts as a spectral filter**, suppressing stochastic perturbations while preserving persistent signal, thereby increasing the spectral gap before orthogonalization. ([arXiv](https://arxiv.org/abs/2606.03899))

So:

$$
\boxed{
\text{raw gradient}
\rightarrow
\text{momentum}
\rightarrow
\text{more pronounced spectral signal/noise separation}.
}
$$

This may explain a substantial amount of the giant outlier + near-zero bulk seen in the Muon paper.

---

# 11. It also gives a nice explanation for “stabilization”

Why does the spectrum do

$$
\text{transient}
\rightarrow
\text{stable plateau}?
$$

A plausible interpretation is that after early feature formation:

1. activation covariance $C_x$ becomes relatively stable;
2. backward-error covariance $C_\delta$ becomes relatively stable;
3. gradient noise scale becomes relatively stable;
4. persistent gradient subspaces rotate only slowly;
5. the momentum EMA reaches its stationary filtering regime.

Then

$$
M_t
=
\sum_k\mu^kG_{t-k}
$$

has a stationary-ish spectrum.

So I wouldn't necessarily think of the plateau as some mysterious optimizer attractor.

It could be the straightforward steady state of

$$
\boxed{
\text{representation statistics}
+
\text{error statistics}
+
\text{SGD noise}
+
\text{EMA filtering}.
}
$$

---

# 12. The really unexplained phenomenon is the **late-layer exponent**

This, I think, is where the paper becomes scientifically interesting.

The generic dimensional argument can explain something around

$$
-0.22.
$$

It cannot naturally explain

$$
-0.66
\quad\text{or}\quad
-0.96.
$$

So something qualitatively different is occurring in the final transformer blocks.

My leading hypotheses would be:

$$
\boxed{
\begin{array}{ll}
1.& \Delta_\ell\text{ becomes increasingly low-effective-rank with scale}\\
2.& X_\ell\text{ becomes increasingly anisotropic with scale}\\
3.& E[\delta x^\top]\text{ becomes more concentrated}\\
4.& \text{one persistent momentum mode captures more Frobenius energy}\\
5.& \text{some combination of these}.
\end{array}
}
$$

I would lean toward **the backward/error side plus momentum**, especially for final layers.

Why? The last block is most directly coupled to

$$
\text{final norm}
\rightarrow
\text{LM head}
\rightarrow
\text{cross entropy}.
$$

So its error signal is much more directly determined by the highly structured next-token prediction residual.

Earlier layers pass through many residual branches, attention/MLP transformations, etc., which can remix that signal.

But this is currently a **hypothesis**, not something demonstrated by the scaling-law paper.

---

# 13. There is a very simple experiment that would tell us

I now think their plots should have included

$$
\boxed{
\sqrt r\,
\sigma_q\left(
\frac{M}{\|M\|_F}
\right).
}
$$

Why?

For a fixed-shape, full-rank spectrum,

$$
\sigma_q(M/\|M\|_F)\propto r^{-1/2}.
$$

Multiplying by $\sqrt r$ removes that trivial effect.

Then:

### If the mid-layer curves become flat with scale

that would confirm

$$
M^{-0.25}
\approx
\text{mostly dimensional normalization}.
$$

### If final-layer curves still fall strongly

then that is actual scale-dependent spectral concentration.

I would additionally measure

$$
\frac{\sigma_q}{\sigma_1},
\qquad
\frac{\sigma_1^2}{\|M\|_F^2},
$$

stable rank

$$
r_{\rm stable}
=
\frac{\|M\|_F^2}{\sigma_1^2},
$$

and effective rank.

Those would distinguish

$$
\text{“matrix got bigger”}
$$

from

$$
\text{“matrix genuinely became more spectrally concentrated.”}
$$

---

## So my updated answer is

The apparent small singular values in this paper have **at least three layers of cause**:

$$
\boxed{
\begin{aligned}
\textbf{1. Geometry:}&\quad
\text{Frobenius normalization automatically gives }\sigma\sim1/\sqrt d.\\[3pt]
\textbf{2. Network structure:}&\quad
G=\Delta X^\top,\text{ and }X,\Delta,\text{ and their correlation are anisotropic.}\\[3pt]
\textbf{3. Training dynamics:}&\quad
\text{SGD noise + temporal coherence + momentum enlarge spectral gaps.}
\end{aligned}
}
$$

The first one may explain **most of the ordinary $-0.25$ scaling**.

The third and second are probably responsible for the actual giant-outlier/near-zero-bulk structure.

And the genuinely unresolved phenomenon is:

$$
\boxed{
\text{Why does this extra concentration become dramatically stronger in the final layers as model scale grows?}
}
$$

That is much more interesting to me than the Newton–Schulz iteration-count question itself.

---

## User

And why does the concentration become dramatically stronger in the final layers as model scale grows?

## ChatGPT · gpt-5-6-thinking · Worked for 1m 43s

<details>
<summary>Reasoning</summary>

*Progress note:* The paper itself does **not** really explain this; it mainly documents it. I’m checking what mechanisms are actually supported by adjacent work, then I’ll separate what is established from what is still a plausible hypothesis.

**Analyzing gradient concentration**

I’m examining whether late-layer gradients become lower-rank because final readout errors concentrate in fewer directions. I’m considering activation anisotropy, residual-stream geometry, prediction confidence, and how parameter scale or width may affect this behavior.

**Analyzed gradient concentration**

**Comparing depth effects**

I’m considering whether deeper networks mix gradient directions more broadly, while final layers preserve concentrated prediction errors. Larger models may strengthen this contrast through improved confidence and sharper output representations.

**Compared depth effects**

**Investigating scaling causes**

I’m checking whether the effect reflects true size scaling or confounds such as greater training budgets, layer position, width, depth, and token exposure. I’m aiming to separate architectural from optimization-driven explanations.

**Investigated scaling causes**

**Explaining the pattern**

My leading explanation is that larger models produce sharper, more structured prediction errors. Final layers receive this concentrated signal directly, while earlier layers mix it across many transformations, increasing effective rank and weakening concentration.

**Explained the pattern**

**Refining the mechanism**

I’m noting a subtle constraint: linear transformations cannot increase rank, but token-dependent Jacobians can spread directions across examples. Neural-collapse-like output structure may contribute, though language modeling makes the analogy imperfect because targets and contexts vary widely.

**Refined the mechanism**

**Testing the hypothesis**

I’m connecting sharper prediction confidence, hard-token gradient dominance, and late-layer specialization. I’m treating these as plausible mechanisms rather than established facts, especially because language modeling differs from ordinary classification.

**Explaining late-layer concentration**

I’m framing this as a boundary-layer effect: cross-entropy gradients shrink on easy tokens, leaving larger models dominated by correlated hard-token errors. Final layers inherit that signal directly, while earlier layers mix it across contexts.

**Tested the hypothesis and explained late-layer concentration**

**Connecting the mechanisms**

I’m combining four effects: sharper predictions remove easy-token gradients, hard-token errors become more correlated, final layers receive them directly, and momentum preserves persistent directions while averaging diffuse noise.

**Synthesizing the explanation**

I’m relating sharper predictions, correlated hard-token errors, direct final-layer access, context-dependent gradient mixing, and momentum filtering. Together, these effects could make late-layer gradients increasingly spectrally concentrated as models scale.

**Synthesized the mechanisms into a cohesive explanation**

**Searched 3 websites**

</details>

I think there is a fairly compelling mechanism, but it is **not yet causally established by the paper**. The best synthesis I can make from the current literature is:

$$
\boxed{
\text{larger model}
\rightarrow
\text{more concentrated output error}
\rightarrow
\text{late-layer gradients inherit it directly}
\rightarrow
\text{momentum further filters toward persistent modes}
}
$$

while earlier layers partially “mix” that terminal error through input-dependent Jacobians.

That would naturally create a **spectral boundary layer near the output** whose concentration strengthens with scale.

### Start from the last layer

For any matrix $W_\ell$,

$$
G_\ell=\frac{\partial L}{\partial W_\ell}
=\sum_t \delta_{\ell,t}x_{\ell,t}^{\top}.
$$

For the final transformer block, the backward signal is very close to

$$
\delta_{L,t}
\approx
J_{\mathrm{norm},t}^{\top}
W_U^{\top}
\underbrace{(p_t-y_t)}_{e_t}.
$$

So the entire gradient geometry of the final block is being driven by the next-token error vectors

$$
e_t=p_t-y_t.
$$

And these become highly nonuniform as the model improves.

For an easy token,

$$
p_t(y_t)\rightarrow1
\quad\Rightarrow\quad
e_t\rightarrow0.
$$

For a token on which the model is uncertain,

$$
p_t(y_t)\ll1,
$$

its gradient remains large. This gradient-probability relationship is exact for cross entropy and is observed directly in LLM training: low-probability tokens generate much larger gradients. ([arXiv](https://arxiv.org/abs/2505.12929))

Now combine that with scaling. Larger language models achieve systematically lower cross-entropy, and recent token-level scaling work shows that increasing scale progressively resolves subsets of contextualized tokens rather than uniformly reducing every token's loss. ([arXiv](https://arxiv.org/abs/2001.08361))

So imagine a batch containing 10,000 tokens.

Small model:

$$
\begin{array}{c}
\text{many tokens still uncertain}\\
\Downarrow\\
\{e_1,e_2,\ldots,e_{10000}\}
\text{ all contribute substantially}
\end{array}
$$

Large model:

$$
\begin{array}{c}
\text{most easy tokens nearly solved}\\
\Downarrow\\
\text{gradient dominated by a much smaller subset of hard/ambiguous tokens}
\end{array}
$$

That alone can reduce the **effective sample rank** of

$$
\Delta_L=[\delta_{L,1},\ldots,\delta_{L,B}].
$$

---

## And those remaining errors are probably not isotropic

This is the second crucial ingredient.

Suppose the 1,000 remaining difficult tokens produced totally random error directions in $d=4096$ dimensions. You might still get a fairly broad gradient spectrum.

But language errors aren't random.

Many hard examples involve recurring structures:

- frequency-related corrections;
- semantic competitors;
- syntactic ambiguities;
- uncertainty between similar tokens;
- recurring contextual patterns.

The output embedding itself is known to become strongly anisotropic and frequency structured; frequent tokens develop systematically different norms and alignment with final hidden states. ([OpenReview](https://openreview.net/pdf?id=TkHcdBLsJJ))

Therefore it is plausible that

$$
C_{\delta,L}
=
E[\delta_L\delta_L^\top]
$$

looks increasingly like

$$
C_{\delta,L}
\approx
\lambda_1u_1u_1^\top
+
\lambda_2u_2u_2^\top
+\cdots
$$

with

$$
\lambda_1,\lambda_2,\ldots
$$

dominating a long weak tail.

Then

$$
G_L=\Delta_L X_L^\top
$$

inherits that concentration.

---

# Why doesn't the same thing happen equally in earlier layers?

This is where I think the **depth effect** comes from.

For an earlier layer,

$$
\delta_{\ell,t}
=
J_{\ell+1:L,t}^{\top}\delta_{L,t}.
$$

Crucially, that Jacobian is **token/context dependent**:

$$
J_{\ell+1:L,t}\neq J_{\ell+1:L,t'}.
$$

So even if two tokens begin with roughly the same terminal error direction $u$,

$$
\delta_{L,t}\approx u,
\qquad
\delta_{L,t'}\approx u,
$$

by the time that signal reaches an earlier layer,

$$
\delta_{\ell,t}
=
J_t^\top u,
$$

$$
\delta_{\ell,t'}
=
J_{t'}^\top u.
$$

Those can point in rather different directions.

Attention, nonlinear MLPs, token interactions and residual paths act as a **context-dependent remixing operation**.

Schematically,

$$
\text{concentrated terminal error}
\overset{\text{backprop through many varying Jacobians}}{\longrightarrow}
\text{broader set of directions}.
$$

But the final block sees essentially the raw terminal error before that repeated remixing happens.

So there is a natural output-boundary effect:

$$
\boxed{
\begin{array}{ccc}
\text{early} & \cdots & \text{late}\\
\text{mixed error geometry} &&
\text{direct output-error geometry}\\
\text{broad} &&
\text{concentrated}
\end{array}}
$$

This fits independent evidence that transformers develop strong depth-dependent spectral organization during training. One 2026 study finds late transformer weight matrices eventually **over-compress relative to early layers**, and that this depth structure strengthens with training/model depth. ([arXiv](https://arxiv.org/abs/2604.22778))

There is also growing evidence that later LLM layers tend increasingly toward **refinement of already-formed representations**, rather than creating entirely new representations. ([arXiv](https://arxiv.org/abs/2512.14064))

Those observations aren't proof of our mechanism, but they're consistent with it.

---

# Why is **MLP-down** particularly extreme?

This part is especially interesting.

For the MLP down projection,

$$
h_{\rm MLP}
=
W_{\rm down}\,
\phi(W_{\rm up}x),
$$

so

$$
\boxed{
G_{\rm down}
=
\Delta_{\rm resid}
H_{\rm MLP}^{\top}.
}
$$

Notice what happens in the final block.

The **left side**

$$
\Delta_{\rm resid}
$$

is already highly constrained by the output-loss geometry we just discussed.

And the **right side**

$$
H_{\rm MLP}
$$

consists of nonlinear/gated features.

If late-layer MLPs become specialized so that only a relatively small family of features fires coherently on the remaining training errors, then **both sides of the outer product become low-effective-dimensional**:

$$
\underbrace{\Delta_{\rm resid}}_{\text{concentrated}}
\quad
\underbrace{H_{\rm MLP}^\top}_{\text{specialized}}
$$

giving

$$
G_{\rm down}
$$

an exceptionally concentrated spectrum.

That could explain why the Muon paper gets its most extreme exponent,

$$
\boxed{-0.96},
$$

specifically for the **final MLP projection**, rather than uniformly for every late-layer matrix. The paper documents this exponent but does not explain its mechanism. ([arXiv](https://arxiv.org/html/2606.04058))

There is striking independent evidence for this exact architectural localization: a recent ViT/Muon study found that when gradient spectral collapse occurs, **deep MLP-down matrices are by far the most susceptible components**; strong augmentation expands their active gradient subspace by as much as $10\!-\!16\times$. ([arXiv](https://arxiv.org/abs/2605.24770))

That is difficult to dismiss as purely an artifact of this one LLM experiment.

---

# Momentum then amplifies the phenomenon

Remember that the scaling-law paper isn't measuring $G_t$.

It's measuring

$$
M_t
=
\beta M_{t-1}
+
(1-\beta)G_t.
$$

Suppose a dominant gradient mode is temporally coherent:

$$
G_t^{\rm signal}\sim uv^\top
$$

at many successive steps.

It accumulates coherently:

$$
M_t^{\rm signal}
\sim
\sum_k\beta^k uv^\top
\sim
\frac{1}{1-\beta}uv^\top.
$$

Now suppose weak directions fluctuate from batch to batch:

$$
G_t^{\rm noise}=N_t,
\qquad E[N_t]\approx0.
$$

Those partially average away.

So momentum performs roughly

$$
\boxed{
\text{persistent directions}\uparrow,
\qquad
\text{incoherent directions}\downarrow.
}
$$

A 2026 Muon analysis formalizes exactly this signal-plus-perturbation argument: momentum before orthogonalization acts as a **spectral filter**, suppressing perturbations and increasing the spectral gap around persistent gradient directions. ([arXiv](https://arxiv.org/abs/2606.03899))

Thus the raw-gradient concentration created near the output can become even stronger in the momentum matrix that the scaling-law paper measures.

---

# Now put model scale into the picture

This gives a possible feedback chain:

$$
\boxed{
\begin{aligned}
\text{model gets larger}
&\Rightarrow \text{more tokens become easy}\\
&\Rightarrow \text{easy-token CE gradients vanish}\\
&\Rightarrow \text{fewer hard structures dominate}\\
&\Rightarrow C_{\delta,L}\text{ becomes more concentrated}\\
&\Rightarrow G_{\rm final}\text{ becomes more concentrated}\\
&\Rightarrow \text{EMA preserves recurring modes}\\
&\Rightarrow M_{\rm final}\text{ becomes very concentrated}.
\end{aligned}}
$$

Earlier layers are partially protected from this because context-dependent Jacobians diversify the terminal error signal.

So I think the key idea is **not simply “later layers are more ill-conditioned.”**

It's:

> **The end of the network is a boundary where the increasingly sparse/structured residual prediction error enters backpropagation. Scaling makes that residual error more selective, and the final few layers see it before the network has had an opportunity to remix it.**

That is a much more mechanistic explanation.

---

## One observation makes me particularly interested in this hypothesis

Recall what the Muon paper says: **all five bulk quantiles in a given layer have almost the same scaling exponent.** ([arXiv](https://arxiv.org/html/2606.04058))

For the final MLP, roughly

$$
\sigma_{.1},\sigma_{.25},\sigma_{.5},
\sigma_{.75},\sigma_{.9}
\propto M^{-0.96}.
$$

This means it isn't merely the smallest singular values collapsing.

The **entire bulk is being pushed downward together** relative to the Frobenius norm.

That looks exactly like what would happen if

$$
\frac{\sigma_1^2}{\|M\|_F^2}
$$

were increasing sharply with scale: one or a few coherent modes absorb more and more of the available spectral energy, leaving the whole bulk uniformly smaller.

I would very much like to see the missing plot:

$$
\boxed{
\rho_1(M)
=
\frac{\sigma_1(M)^2}{\|M\|_F^2}
}
$$

versus model size and depth.

If

$$
\rho_1^{\rm final}
\uparrow
$$

with a strong power law while

$$
\rho_1^{\rm mid}
\approx\text{constant},
$$

then we'd know the dramatic $-0.96$ isn't “all singular values independently disappearing.” It's **spectral energy migrating into a tiny coherent subspace**.

And then the next experiment almost writes itself: decompose

$$
G_\ell=\Delta_\ell X_\ell^\top
$$

and measure, across scale,

$$
\operatorname{erank}(\Delta_\ell),
\qquad
\operatorname{erank}(X_\ell),
\qquad
\operatorname{erank}(G_\ell),
$$

plus per-token gradient participation.

If the hypothesis above is correct, I would predict a very specific result:

$$
\boxed{
\operatorname{erank}(\Delta_L)
\text{ falls unusually quickly with model scale},
}
$$

with a strong recovery in effective rank as you move backward through the network. **That experiment would get much closer to explaining the scaling law than simply measuring more momentum spectra.**

One caveat is important: the original scaling experiment changes width, depth, sequence length, and total training-token budget together—models span 512→2560 width, 8→32 layers, sequence length 512→1024, and each is trained on $20\times$ parameters in tokens. ([arXiv](https://arxiv.org/html/2606.04058)) So before calling any mechanism a *model-size law*, I'd want width-only and depth-only sweeps. The $-0.96$ phenomenon is real in their setup; **why it scales that way remains open.**

---

## User

Does the paper keep other confounds such as batch size, etc controlled? Could these. be the reason for the concentration along depth?

## ChatGPT · gpt-5-6-thinking · Worked for 3m

<details>
<summary>Reasoning</summary>

*Progress note:* Yes—this is exactly the right thing to check. I’m looking at the training setup across scales and separating **confounds that are held fixed** from ones that co-vary with model size, because the late-layer concentration could absolutely be partly induced by the scaling recipe rather than width alone.

**Assessing scaling confounds**

I’m checking whether fixed optimization settings, initialization, momentum, architecture, and batch size are fully specified. If batch size varies or is omitted, observed scaling may reflect compound training changes rather than model size alone.

**Estimating batch effects**

I’m considering whether growing batch size could amplify spectral concentration by reducing gradient noise. That might steepen lower-quantile scaling, though the required batch-growth rate may be large, and global batch changes don’t neatly explain final-layer specificity.

**Assessed scaling confounds and estimated batch effects**

**Evaluating gradient noise**

I’m adding sequence length as another effective batch-size factor: doubling tokens can reduce noise and increase spectral concentration. This may create a step change, but it still doesn’t fully explain smooth, layer-specific scaling.

**Checking supporting evidence**

I’m looking for direct evidence that batch size changes gradient singular spectra, rather than merely affecting related representations. That distinction matters before treating batch-related noise reduction as a sufficient explanation.

**Separating scale effects**

I’m distinguishing two claims: batch or token-count changes could influence concentration, especially in later layers, but the paper’s omissions prevent attribution. Fixed optimizer steps also mean models may receive unequal training exposure.

**Checked supporting evidence and separated scale effects**

**Quantifying noise reduction**

I’m modeling batch averaging and momentum as increasing signal-to-noise roughly with the square root of batch size. If coherent gradients are low-rank, this can suppress spectral bulk and concentrate normalized spectra, especially in later layers.

**Quantified noise reduction**

**Searched 16 websites**

**Estimated rank concentration and calculated scaled ratio power**

**Designing decisive ablations**

I’m treating batch size as a plausible contributor, not a complete explanation. I’d isolate it by varying batch and sequence length independently, matching loss and architecture, then comparing depth-wise spectral measures.

**Identifying missing controls**

I’m noting that omitted batch size, token exposure, learning-rate phase, initialization details, and random seeds limit causal interpretation. A depth-dependent concentration pattern may reflect optimization history rather than scale alone.

**Designed decisive ablations**

**Searched 1 website**

</details>

Yes. After checking the experimental setup carefully, I would **not interpret the paper as having isolated “model scale” as the cause of the late-layer spectral effect**. It is a compound scaling sweep, and batch size is actually one of the things the paper does **not report**.

The paper does keep some important things fixed: the dataset is FineWeb, the Muon learning rate for the spectral experiments is fixed at $0.01$, weight decay is $0.01$, the schedule form is fixed, non-matrix parameters use the same AdamW settings, and every model gets a $20\times$-parameters token budget. ([arXiv](https://arxiv.org/pdf/2606.04058))

But many things co-vary with model size:

| quantity | 77M | 2.8B | controlled? |
|---|---:|---:|---|
| width $d$ | 512 | 2560 | **No** |
| depth | 8 | 32 | **No** |
| heads | 8 | 40 | **No** (head dim stays 64) |
| sequence length | 512 | 1024 | **No** |
| total training tokens | $20\times77M$ | $20\times2.8B$ | scales with $M$ |
| batch size | — | — | **Not reported** |
| loss at measurement | — | — | **Not matched** |
| random seeds | — | — | apparently no replicated-seed analysis |

The architecture table explicitly shows width, depth, head count, and sequence length changing together. ([arXiv](https://arxiv.org/pdf/2606.04058)) The scaling law is then fit by taking the average spectrum over **the same absolute steps 1300–1500** for every model. ([arXiv](https://arxiv.org/pdf/2606.04058))

So there are several quite serious possible confounds.

### Batch size is particularly relevant

The appendix specifies learning rates, weight decay, initialization scaling, data, token budget, and hardware—but **never states the training batch size**. ([arXiv](https://arxiv.org/pdf/2606.04058))

That is not a minor omission for this particular measurement.

Write a layer's minibatch gradient as

$$
G_B=\bar G+\epsilon_B,
$$

where

$$
E[\epsilon_B]=0,\qquad
\operatorname{Var}(\epsilon_B)\propto\frac1B.
$$

If the population/mean gradient $\bar G$ is spectrally concentrated while stochastic gradient noise fills many weak directions, then increasing $B$ does exactly this:

$$
B\uparrow
\quad\Rightarrow\quad
\text{noise floor}\downarrow
\quad\Rightarrow\quad
\text{dominant modes become more prominent}.
$$

After Frobenius normalization,

$$
\frac{G_B}{\|G_B\|_F},
$$

the bulk singular values can therefore become smaller relative to the leading modes.

And then Muon's momentum further performs temporal averaging:

$$
M_t=\mu M_{t-1}+G_t,
$$

which suppresses temporally incoherent noise even more.

So if their effective batch size increases with model scale, **yes, that could produce exactly the kind of increasing spectral concentration they're measuring.**

This isn't just a hypothetical concern. A separate 2026 LLM study, *Spectral Lens*, performs controlled batch-size experiments and finds that batch size produces systematically different spectral representation geometries even at matched loss; larger batches produce more concentrated activation spectra. They also explicitly examine gradient SVD spectra. ([arXiv](https://arxiv.org/abs/2605.05683))

---

### But batch size alone cannot straightforwardly explain the *depth* dependence

There's an important distinction.

Within a single model,

$$
B_{\rm layer\,1}=B_{\rm layer\,32}.
$$

So the fact that, say, the final MLP has a much more concentrated spectrum than the middle MLP **cannot simply be because the final layer has a bigger minibatch**.

However, there can be an interaction:

$$
G_{\ell,B}
=
\underbrace{\bar G_\ell}_{\text{signal geometry depends on layer}}
+
\underbrace{\epsilon_{\ell,B}}_{\text{noise geometry depends on layer}}.
$$

Suppose the final-layer population gradient is very low-rank:

$$
\bar G_L\approx
\sigma_1u_1v_1^\top+\sigma_2u_2v_2^\top,
$$

while its spectral tail is mostly minibatch noise.

But an earlier layer has a genuinely high-rank mean gradient because downstream Jacobians remix the errors:

$$
\operatorname{erank}(\bar G_{\rm mid})
\gg
\operatorname{erank}(\bar G_L).
$$

Increasing batch size then does:

$$
\text{final layer: }
\boxed{\text{dramatic concentration}}
$$

while

$$
\text{middle layer: }
\boxed{\text{much smaller change}}.
$$

So a batch-size confound **can interact with depth and generate precisely a stronger scale exponent at the end of the network.**

That makes this a genuine concern.

---

## I think there are actually three bigger confounds

### 1. Depth itself changes from 8 to 32

This may be the most obvious one.

The paper calls

$$
N/4,\quad N/2,\quad3N/4,\quad N
$$

“mid-early,” “mid,” “mid-late,” and “final.” ([arXiv](https://arxiv.org/pdf/2606.04058))

But the “final layer” comparison is therefore

$$
\text{layer 8}
\rightarrow
\text{layer 12}
\rightarrow
\text{layer 20}
\rightarrow
\cdots
\rightarrow
\text{layer 32}.
$$

Those aren't identical layers in wider models.

If gradient geometry changes as a function of **absolute network depth**, then

$$
\sigma_q(M_{\rm final})
\sim M^{-0.96}
$$

could partly be

$$
\boxed{\text{depth scaling}}
$$

rather than parameter-count scaling.

A clean experiment would independently vary

$$
d\quad\text{at fixed }L
$$

and

$$
L\quad\text{at fixed }d.
$$

The paper doesn't do this.

---

### 2. Sequence length doubles at exactly the larger scales

The first four models use

$$
T=512
$$

and the last three use

$$
T=1024.
$$

([arXiv](https://arxiv.org/pdf/2606.04058))

This matters twice.

If the number of sequences per batch stays fixed, the effective token batch doubles:

$$
B_{\rm tokens}=B_{\rm seq}T.
$$

So it has exactly the noise-reduction effect discussed above.

Even if they compensate by halving the number of sequences so that $B_{\rm tokens}$ stays constant, sequence length itself changes the learning problem. A 1024-token context can produce different attention, representation, and error geometry than a 512-token context.

And the final layers could plausibly be more sensitive to this because they're closest to the contextual next-token prediction error.

So the discontinuity

$$
600M,\ T=512
\quad\rightarrow\quad
1.2B,\ T=1024
$$

is a real confound.

---

### 3. They do not match models by **loss**

I think this one may be even more important mechanistically.

They estimate the “stable” spectrum from

$$
t=1300\ldots1500
$$

for every model. ([arXiv](https://arxiv.org/pdf/2606.04058))

But a 2.8B model and a 77M model are not necessarily in the same functional state at step 1400.

If our earlier hypothesis is right—that final-layer spectral concentration arises because a stronger model has solved most easy tokens and its gradient becomes dominated by a smaller set of hard/coherent errors—then you could get

$$
\text{larger model}
\rightarrow
\text{lower loss at step 1400}
\rightarrow
\text{more selective output errors}
\rightarrow
\text{more concentrated final gradient}.
$$

In that case the apparent “model-size scaling law” is really partly

$$
\boxed{
\text{prediction-quality / training-state scaling}.
}
$$

A very revealing control would be:

$$
\text{compare all model sizes at the same validation loss}.
$$

For example,

$$
L_{\rm val}=3.5
$$

for every model.

If the final-layer exponent mostly disappears, then the phenomenon is about **how solved the language-modeling problem is**, not architectural size per se.

The paper doesn't perform that control.

---

## And remember the deterministic width confound we found earlier

Even before any of this,

$$
\tilde M=\frac{M}{\|M\|_F}
$$

automatically gives singular values of order

$$
1/\sqrt d
$$

for a fixed-shape spectrum.

Their width itself scales approximately as

$$
d\sim M^{0.44},
$$

so simply from Frobenius normalization you'd predict roughly

$$
\sigma_q(\tilde M)
\sim
M^{-0.22}.
$$

And indeed most middle layers have exponents around

$$
-0.2\text{ to }-0.3.
$$

The full appendix makes this striking: mid-layer exponents cluster around that range, while final layers jump to $0.38$–$0.96$ depending on matrix type. For example, final O is $0.66$, final MLP Down $0.52$, and final MLP Up $0.96$, while their mid/mid-late counterparts are roughly $0.2$–$0.26$. ([arXiv](https://arxiv.org/pdf/2606.04058))

So I'd decompose their observation conceptually as

$$
\alpha_{\rm measured}
=
\underbrace{\alpha_{\rm width/Frob}}_{\sim0.22}
+
\underbrace{\alpha_{\rm training/geometry}}_{\text{what we actually care about}}
+
\underbrace{\alpha_{\rm confounds}}_{\text{batch, depth, seq len, loss,\ldots}}.
$$

They haven't separated those terms.

---

## Could the confounds explain *all* of the final-layer effect?

I wouldn't go that far.

The fact that **every final matrix type** tends to have a larger exponent than its mid-layer counterpart is suggestive of a genuine output-boundary phenomenon:

$$
\begin{aligned}
Q:&\quad .27_{\rm midlate}\rightarrow .46_{\rm final}\\
K:&\quad .24\rightarrow .38\\
V:&\quad .26\rightarrow .58\\
O:&\quad .26\rightarrow .66\\
\mathrm{MLPdown}:&\quad .23\rightarrow .52\\
\mathrm{MLPup}:&\quad .20\rightarrow .96.
\end{aligned}
$$

([arXiv](https://arxiv.org/pdf/2606.04058))

A purely global batch-size effect would need a **strong layer-dependent interaction** to generate that pattern.

And the extraordinary MLP-Up effect looks particularly layer-specific.

But the current experiment does **not tell us whether the causal statement is**

$$
\boxed{\text{increasing model size causes late-layer spectral concentration}}
$$

or instead something like

$$
\boxed{
\begin{aligned}
&\text{increasing depth}\\
+{}&\text{changing effective batch}\\
+{}&\text{longer context}\\
+{}&\text{lower loss / more mature predictions}\\
+{}&\text{increasing width}
\end{aligned}
\Rightarrow
\text{late-layer concentration}.
}
$$

I would regard the latter as entirely open.

### The experiment I'd want

The clean follow-up is surprisingly small. Take, say, a 350M model and separately vary **batch size by $8\times$**, then vary **sequence length 512 vs 1024 while holding tokens/update fixed**, and measure

$$
\sqrt d\,\sigma_q(M/\|M\|_F)
$$

at every depth. Then do a width-only sweep at fixed 20 layers and a depth-only sweep at fixed $d=1024$, and compare checkpoints at both **fixed step and matched validation loss**.

If the final-layer collapse survives all four controls, then I would become much more convinced that we've uncovered an intrinsic scale-dependent property of backpropagation near the LM head.

At the moment, I think **the empirical phenomenon is real, but the “model-size scaling law” interpretation is substantially less causally identified than the paper makes it sound.**

---

## User

What could have a layer-dependent interaction that in deeper nets causes this?

## ChatGPT · gpt-5-6-thinking · Worked for 2m 9s

<details>
<summary>Reasoning</summary>

*Progress note:* A plausible answer has to explain **two facts at once**: why the last layers are special, and why the contrast gets stronger as total depth increases. The most promising mechanisms are residual/Jacobian mixing, token-correlation structure, and normalization—not batch size alone.

**Exploring layer interactions**

I’m examining how depth could make late-layer gains appear stronger, especially through normalization, residual-stream scaling, gradient propagation, optimizer noise, and layer-dependent signal-to-noise. I’m also checking whether this reflects genuine optimization dynamics or measurement effects.

**Refining the mechanism**

I’m questioning whether normalization alone explains spectral concentration: it mostly rescales gradients. A stronger explanation may involve residual dominance, selective activations, or low-rank downstream Jacobians that preserve only a few gradient directions in late layers.

**Explored layer interactions and refined the mechanism**

**Formulating hypotheses**

I’m considering two linked effects: deeper residual streams can make late blocks near-identity, while selective MLP gating can reduce effective rank. Together, correlated error signals and sparse feature participation may amplify late-layer spectral concentration as depth increases.

**Formulated hypotheses**

**Analyzing momentum effects**

I’m evaluating whether momentum selectively amplifies persistent late-layer directions. If late gradients remain temporally aligned while earlier gradients drift through changing Jacobians, exponential averaging could strengthen late-layer spectral concentration.

**Prioritizing explanations**

I’m emphasizing two depth-amplified mechanisms: downstream Jacobian mixing makes earlier gradients less stable, while late residual blocks remain more directly aligned with output errors. Momentum can then preserve late directions and cancel drifting early ones.

**Analyzed momentum effects and prioritized explanations**

**Searched 23 websites**

*Progress note:* One mechanism is especially easy to miss because it is **intrinsically depth-indexed**: in a residual Transformer, a layer’s gradient depends on how many downstream Jacobians sit between it and the loss. As total depth grows, that number changes differently for early/middle versus final layers.

**Connecting signal concentration**

I’m also considering loss heterogeneity: deeper or better-trained models may concentrate gradient magnitude on fewer difficult examples. That lowers the effective sample rank, especially when late-layer errors remain more directly aligned with the output signal.

**Testing causal mechanisms**

I’m identifying a likely feedback loop: low-rank late updates may receive weaker spectral correction, remain concentrated, and increasingly shape future gradients. Greater depth could make this regime layer-specific through downstream Jacobians and residual interactions.

**Connected signal concentration and tested causal mechanisms**

**Searched 1 website**

</details>

Yes. There are several mechanisms that are **intrinsically layer-dependent and become stronger as total depth increases**. The cleanest one is downstream-Jacobian mixing; after that, Pre-LN residual growth, gradient-noise filtering, training-state effects, and even Muon itself can amplify the pattern.

## 1. The cleanest mechanism: downstream Jacobians “deconcentrate” earlier-layer gradients

For token/example $b$, a layer gradient has the form

$$
G_{\ell,b}
=
\delta_{\ell,b}x_{\ell,b}^{\top},
$$

with

$$
\delta_{\ell,b}
=
J_{\ell+1:L,b}^{\top} e_b,
$$

where $e_b$ is the loss gradient at the output and $J_{\ell+1:L,b}$ is the Jacobian of everything downstream of layer $\ell$.

Suppose the terminal errors $e_b$ occupy a relatively small subspace $E$. Then at the **final layer**,

$$
\delta_{L,b}\in E.
$$

So all examples are producing gradients whose left factors live in roughly the same low-dimensional space.

But an earlier layer sees

$$
\delta_{\ell,b}
=
J_b^\top e_b.
$$

And crucially,

$$
J_b \neq J_{b'}
$$

because attention patterns, MLP derivatives, LayerNorm statistics, etc. are input dependent.

Thus different examples rotate/remix that same terminal subspace differently:

$$
E
\rightarrow J_1^\top E,\quad
J_2^\top E,\quad
J_3^\top E,\ldots
$$

The union

$$
\operatorname{span}_b J_b^\top E
$$

can be much higher dimensional than $E$ itself.

Now look at what happens as the network becomes deeper. At relative depth $1/2$,

$$
J_{\ell+1:L}
$$

contains 4 downstream blocks in an 8-layer network but 16 blocks in a 32-layer network.

So there are many more opportunities for input-dependent remixing.

Schematically:

$$
\boxed{
\begin{array}{rcl}
\text{final layer} &:& E\\
\text{a few layers back} &:& J_1^\top E\\
\text{many layers back} &:& J_2^\top J_1^\top E\\
\vdots&&
\end{array}}
$$

The **final layer always has zero downstream mixing**, regardless of total depth.

The mid-layer gets increasingly much more downstream mixing as total depth increases.

Therefore:

$$
\boxed{
L\uparrow
\quad\Longrightarrow\quad
\text{larger spectral-rank gap between final and middle layers}.
}
$$

This is conceptually related to the older “shattered gradients” literature: gradient correlations change systematically with propagation depth; residual connections greatly mitigate the effect, but do not eliminate depth-dependent changes in gradient correlation structure. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v70/balduzzi17b.html))

I think this is one of the strongest candidate explanations.

---

## 2. Pre-LN gives an additional depth-dependent effect in exactly the right location

For a Pre-LN Transformer,

$$
h_{\ell+1}
=
h_\ell
+
F_\ell(\operatorname{LN}(h_\ell)).
$$

The residual stream accumulates contributions as depth increases.

So typically

$$
\|h_\ell\|
\uparrow
$$

with depth.

Meanwhile approximately,

$$
J_{\rm LN}(h_\ell)
\sim
\frac{1}{\operatorname{std}(h_\ell)}P_\ell,
$$

where $P_\ell$ is mostly a projection-like operator.

Hence as the residual stream grows,

$$
\left\|
J_{F_\ell}J_{\rm LN}
\right\|
$$

becomes smaller relative to the identity skip path.

The full block Jacobian is

$$
J_{\rm block}
=
I+
J_FJ_{\rm LN},
$$

and therefore increasingly behaves like

$$
J_{\rm block}\approx I.
$$

This isn't just theoretical speculation: *The Curse of Depth* reports that in Pre-LN LLMs, deep blocks become increasingly identity-like and make smaller effective contributions, with the effect worsening with depth. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eeb57fdf745eb31a3c7ef22c59a4661d-Abstract-Conference.html))

Why could this produce gradient concentration?

Because the upper part of a very deep model can start looking like

$$
h_{L-k}
\approx h_{L-k+1}
\approx\cdots\approx h_L
$$

in relative geometric terms.

Likewise,

$$
\delta_{L-k}
\approx\delta_{L-k+1}
\approx\cdots\approx\delta_L.
$$

So the last several layers repeatedly see **similar activations and similar error directions**.

Then

$$
G_\ell=\Delta_\ell X_\ell^\top
$$

can repeatedly contain the same strong modes.

As total depth increases, residual-stream growth near the top can become stronger, which could make these upper-layer gradients increasingly redundant/coherent.

Important caveat: **identity-like blocks alone don't mathematically imply a low-rank gradient**. But they create exactly the conditions—high inter-layer redundancy and persistent directions—that can make the gradient/momentum spectrum concentrate.

---

## 3. This creates a very natural interaction with batch size

This is how an unreported batch-size change could produce a **layer-specific** effect rather than simply changing every layer equally.

Write

$$
G_{\ell,B}
=
\bar G_\ell
+
\frac{1}{\sqrt B}\Xi_\ell.
$$

Suppose the population gradient at the final layer is strongly low-rank:

$$
\bar G_L
\approx
\sum_{i=1}^{r}
\sigma_i u_i v_i^\top,
\qquad r\ll d.
$$

The rest of its observed spectrum is substantially minibatch noise.

But suppose downstream Jacobian mixing makes the middle-layer population gradient itself relatively broad:

$$
\operatorname{erank}(\bar G_{\rm mid})
\gg
\operatorname{erank}(\bar G_L).
$$

Now increase $B$.

At the final layer:

$$
\frac1{\sqrt B}\Xi_L\downarrow
$$

removes the broad noise floor, exposing a few strong population modes:

$$
\boxed{\text{dramatic spectral concentration}.}
$$

At the middle layer, much of the broad spectrum is actual signal, so removing noise has much less effect:

$$
\boxed{\text{modest spectral concentration}.}
$$

Thus

$$
\boxed{
\text{batch size}\times\text{layer SNR}
}
$$

is a perfectly plausible layer-dependent interaction.

This would be especially concerning if token batch effectively changes when sequence length goes from 512 to 1024 in the scaling sweep.

---

# 4. Momentum can magnify that layer dependence enormously

This may be even more relevant because the paper measures $M_t$, not $G_t$:

$$
M_{\ell,t}
=
\sum_{k=0}^{\infty}\beta^k
G_{\ell,t-k}.
$$

Imagine a late-layer dominant direction is highly stable:

$$
G_{L,t}^{\rm signal}
\approx
\sigma_t uv^\top
$$

for hundreds of steps.

Momentum adds it coherently.

But suppose a weak direction changes orientation every batch:

$$
N_t,\quad N_{t-1},\quad N_{t-2},\ldots
$$

It cancels under averaging.

Recent theory specifically analyzes this effect in Muon and shows that momentum can enlarge the spectral gap between persistent signal and perturbations. ([Hugging Face](https://huggingface.co/papers/2606.03899))

Now add depth.

A final-layer gradient is directly tied to the relatively stable prediction-error geometry.

An earlier gradient contains

$$
J_{\ell+1:L,t}^{\top} e_t.
$$

And every one of those downstream Jacobians changes as training progresses.

So its singular subspace may wander more over time.

You could therefore get

$$
\text{temporal coherence}_{\rm final}
>
\text{temporal coherence}_{\rm middle}.
$$

As total depth grows, there are more moving downstream transformations for the middle layers:

$$
\boxed{
L\uparrow
\Rightarrow
\text{more temporal rotation of middle-layer gradients}
}
$$

while the final layer remains directly coupled to the output.

Then momentum would preferentially concentrate the late-layer spectrum.

This is a very natural explanation for why the effect could be much stronger in **momentum** than in raw gradients.

---

# 5. Matching models by step rather than loss gives another layer-dependent interaction

Suppose larger/deeper models are simply better at step 1400.

For cross entropy,

$$
e_b=p_b-y_b.
$$

For already-solved tokens,

$$
p_b(y_b)\approx1
\quad\Rightarrow\quad
\|e_b\|\approx0.
$$

So as prediction quality improves, more tokens effectively disappear from the gradient.

Suppose initially 10,000 tokens contribute materially, while later only 1,000 difficult tokens dominate.

A useful effective count would be something like

$$
N_{\rm eff}
=
\frac{(\sum_b a_b)^2}{\sum_ba_b^2},
\qquad
a_b=\|e_b\|.
$$

As errors become heterogeneous,

$$
N_{\rm eff}\downarrow.
$$

That naturally lowers the effective rank of

$$
\Delta_LX_L^\top.
$$

Again, the final layers see this directly.

Earlier layers take those few remaining errors and map them through many different context-dependent Jacobians, partially redistributing them.

So comparing a weak 77M model and a much stronger 2.8B model at the **same optimizer step** can create:

$$
\boxed{
\text{training progress}
\times
\text{depth}
}
$$

which masquerades as

$$
\text{parameter count}
\times
\text{depth}.
$$

That's why the matched-loss control we discussed would be extremely informative.

---

# 6. There is also a potentially nasty **Muon feedback loop**

This one is directly suggested by the paper itself.

Suppose a late layer starts with a more concentrated momentum spectrum:

$$
\sigma_{\rm bulk}\downarrow.
$$

The finite Newton–Schulz approximation fails to lift those tiny modes fully:

$$
f(\sigma_{\rm bulk})\ll1.
$$

So although ideal Muon would use

$$
UV^\top,
$$

actual Muon gives something more like

$$
U
\operatorname{diag}
(1,1,1,\ldots,0.4,0.1,0.02)
V^\top.
$$

In other words, the optimizer itself starts giving the late layer a **lower-effective-rank update**.

That can change the model so that subsequent gradients are even more concentrated:

$$
\boxed{
\text{small tail}
\rightarrow
\text{NS suppresses tail}
\rightarrow
\text{lower-rank learning}
\rightarrow
\text{next gradient has smaller tail}.
}
$$

And the paper contains evidence that such optimizer feedback exists.

When they deliberately train with rank-$p$ updates, the momentum spectra stay close to full Muon for $p=.9,.5$, but at $p=.25$ the median singular-value trajectory begins moving downward; at $p=.1$ it moves down even more. ([arXiv](https://arxiv.org/html/2606.04058))

So the gradient/momentum spectrum is **not exogenous to the optimizer**.

This matters enormously for interpreting their scale law.

As model scale grows:

$$
\text{late layers reach NS's bad region first}
$$

so only those layers start experiencing increasingly low-rank effective updates.

Thus you could get a self-reinforcing layer-specific scale effect:

$$
\boxed{
\text{scale}
\rightarrow
\text{smaller normalized late spectrum}
\rightarrow
\text{poorer NS}
\rightarrow
\text{more concentrated late spectrum}.
}
$$

That is a surprisingly plausible contributor to the very steep late-layer exponents.

---

# 7. The extreme MLP result suggests one additional mechanism

Their most extreme scaling is not generic—it is especially strong for the final MLP projection.

For an MLP,

$$
z=W_{\rm up}x,
\qquad
a=\phi(z),
\qquad
y=W_{\rm down}a.
$$

Then

$$
G_{\rm up}
=
\underbrace{
\left[
W_{\rm down}^{\top}\delta
\odot
\phi'(z)
\right]
}_{\Delta_{\rm hidden}}
x^\top.
$$

So there's an extra **feature gate**

$$
\phi'(z).
$$

If late-layer neurons/features become increasingly specialized, then only a subset of hidden features may receive coherent gradients.

That would make

$$
\Delta_{\rm hidden}
$$

more concentrated than the residual-stream error itself.

That could explain why the final MLP has an exponent far steeper than ordinary attention layers.

I would put less confidence on this one because the causal evidence is weaker, but the localization to MLP matrices makes some feature-selectivity mechanism quite plausible.

---

## So if I had to prioritize mechanisms

I would investigate them in this order:

1. **Downstream-Jacobian mixing** — because it naturally predicts an effect that depends jointly on relative layer and total depth.
2. **Pre-LN residual growth / effective-depth collapse** — because it specifically worsens in upper layers as networks deepen and is independently observed in LLMs. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eeb57fdf745eb31a3c7ef22c59a4661d-Abstract-Conference.html))
3. **Layer-dependent temporal coherence + momentum filtering** — because the measured object is a momentum buffer, and momentum is known to amplify persistent spectral modes. ([Hugging Face](https://huggingface.co/papers/2606.03899))
4. **Batch size or token-batch × layer SNR** — a serious experimental confound.
5. **Matched-step rather than matched-loss effects.**
6. **Finite-NS optimizer feedback** — especially interesting because the paper itself demonstrates that restricting spectral rank changes future momentum spectra. ([arXiv](https://arxiv.org/html/2606.04058))

And there is a very clean experiment that would distinguish almost all of these.

For fixed width, train **8-, 16-, 24-, and 32-layer models** with exactly the same token batch. At each relative depth record both raw $G_\ell$ and momentum $M_\ell$, and measure

$$
\operatorname{erank}(X_\ell),\quad
\operatorname{erank}(\Delta_\ell),\quad
\operatorname{erank}(G_\ell),\quad
\operatorname{erank}(M_\ell),
$$

plus

$$
\cos\angle\big(
\mathcal U(G_{\ell,t}),
\mathcal U(G_{\ell,t+1})
\big)
$$

as a measure of temporal singular-subspace stability. Repeat once with exact SVD/polar updates instead of 5-step NS.

The outcomes would be highly diagnostic:

- $G$ already concentrates with depth → **network/backprop mechanism**.
- $G$ does not, but $M$ does → **temporal-coherence/momentum mechanism**.
- concentration vanishes under exact polar → **NS feedback mechanism**.
- effect collapses at matched loss → **training-state mechanism**.
- effect changes dramatically with batch → **SNR/noise-floor mechanism**.

That would tell us much more about the cause than another model-size scaling sweep.

---

## User

Do these observations, overall, lead to any clear improvement that we could have for training?

## ChatGPT · gpt-5-6-thinking · Worked for 32s

<details>
<summary>Reasoning</summary>

*Progress note:* Yes. The observations point to a few concrete optimizer changes, but only one or two look “low-regret” enough to try immediately. I’m separating changes that follow directly from the measurements from more speculative interventions that depend on the causal story being right.

**Improving training design**

I’m turning these observations into concrete experiments: use layer-adaptive Newton–Schulz iterations, monitor spectral diagnostics, apply partial orthogonalization when tails are noisy, and avoid confounding optimizer changes with batch-size or augmentation changes.

**Improved training design**

**Searched 1 website**

*Progress note:* The clearest immediate improvement is **not “make gradients less concentrated.”** It is: *measure the spectrum you actually have, then spend orthogonalization compute only where it buys useful modes.* That follows directly from the paper without assuming we already know the causal mechanism.

**Refining training improvements**

I’m narrowing this to three actionable ideas: use layer-adaptive orthogonalization depth, stop when a spectral target is reached, and avoid forcing noisy tail singular values fully flat. These should reduce cost while preserving useful conditioning.

**Searched 1 website**

</details>

Yes. The observations suggest a fairly clear optimizer direction:

$$
\boxed{\text{Make Muon spectral- and layer-adaptive rather than applying the same polar approximation everywhere.}}
$$

There are several versions of that idea, with very different confidence levels.

### 1. Most immediate: adaptive Newton–Schulz depth

This is the improvement most directly supported by the paper.

Right now Muon effectively does

$$
M_\ell \xrightarrow[\text{same }k]{\text{NS}} \widehat M_\ell
$$

with the same number of Newton–Schulz steps for every matrix.

But the spectra differ enormously by layer. So instead use

$$
k_\ell =
\min\left\{
k:
f_k(\sigma_{q,\ell})\ge c
\right\},
$$

where $q$ might be $0.5$ and $c$ something like $0.9$–$0.99$.

Then you might get conceptually

$$
\begin{array}{c|c}
\text{layer region}&\text{NS steps}\\
\hline
\text{early/middle}&4\text{--}5\\
\text{late}&6\text{--}8\\
\text{very final problematic matrices}&10
\end{array}
$$

rather than paying 10 steps everywhere.

The paper's rank experiments give a useful empirical target: preserving roughly the top half of singular directions gives performance close to full Muon, whereas dropping to only the top quarter begins to hurt substantially. ([arXiv](https://arxiv.org/abs/2606.04058)) Their frontier-scale example therefore explicitly recommends more accurate NS only for late layers whose median falls below the five-step approximation's useful range. ([arXiv](https://arxiv.org/abs/2606.04058))

This seems like a genuine **compute-efficiency win with relatively little conceptual risk**.

---

## 2. Better still: don't use layer number; adapt to the measured spectrum

Instead of

> last four blocks get 10 steps,

periodically measure a cheap spectral statistic for each momentum matrix.

For example,

$$
s_\ell =
\sigma_{0.5}\left(
\frac{M_\ell}{\|M_\ell\|_F}
\right).
$$

Then choose the cheapest NS polynomial that works at $s_\ell$.

You wouldn't need an SVD every step. The paper finds that the spectrum stabilizes fairly rapidly, so you could calibrate occasionally—say after burn-in and perhaps at sparse checkpoints—and keep the configuration fixed between measurements. ([arXiv](https://arxiv.org/abs/2606.04058))

That turns Muon into something like

$$
\boxed{\text{spectrally budgeted Muon}.}
$$

It is more robust than fitting a rule to “final layers,” because architecture changes, MoE, normalization, batch size, etc. might move the problematic location.

---

# 3. But I think there is a potentially *better* optimizer change hiding here

Muon currently approximately performs

$$
M=U\Sigma V^\top
\quad\longrightarrow\quad
UV^\top.
$$

Thus it treats

$$
\sigma=10
$$

and

$$
\sigma=10^{-8}
$$

identically.

That is a very aggressive assumption:

> **every nonzero singular direction deserves equal update magnitude.**

The observations we've been discussing make me less convinced that's always optimal.

A tiny singular mode could be:

$$
\text{weak but useful signal}
$$

or

$$
\text{minibatch noise}.
$$

Muon cannot distinguish them.

So rather than the hard polar map

$$
f(\sigma)=1,
$$

I'd seriously consider a **regularized polar transform** such as

$$
\boxed{
f_\tau(\sigma)
=
\frac{\sigma}
{\sqrt{\sigma^2+\tau^2}}.
}
$$

Equivalently,

$$
\Delta W
=
M(M^\top M+\tau^2I)^{-1/2}.
$$

Now:

for

$$
\sigma\gg\tau,
$$

you recover Muon:

$$
f_\tau(\sigma)\approx1.
$$

But for

$$
\sigma\ll\tau,
$$

you get

$$
f_\tau(\sigma)\approx\frac{\sigma}{\tau},
$$

so you **don't blow a vanishing/noisy mode all the way up to one**.

This gives you:

$$
\text{gradient}
\quad\longleftrightarrow\quad
\text{regularized Muon}
\quad\longleftrightarrow\quad
\text{full Muon}
$$

through one parameter $\tau$.

---

## Why I think this is especially promising

The paper actually tells us full orthogonalization isn't necessary.

Its experiments find:

$$
p=0.9 \approx \text{full Muon},
$$

and

$$
p=0.5
$$

has only a small gap, while serious degradation appears once only $\sim25\%$ or $10\%$ of directions remain. ([arXiv](https://arxiv.org/abs/2606.04058))

So there appears to be a fairly wide region where the tiniest modes simply don't need exact unit magnitude.

Meanwhile, very recent optimizers are independently moving toward this same broad conclusion. Spectra argues that LLM gradients have a spike–tail structure and explicitly avoids indiscriminately amplifying the noise-sensitive spectral tail; in its reported 8B/50B-token experiment it obtains better efficiency than its AdamW and Muon baselines, although that is currently evidence from a recent preprint rather than a settled result. ([arXiv](https://arxiv.org/abs/2602.11185))

So I think the emerging lesson may be:

$$
\boxed{
\text{flatten the useful spectrum, not necessarily the entire spectrum.}
}
$$

---

# 4. Make that regularization layer-dependent

Now connect this with everything we discussed about late-layer concentration.

Suppose middle layers have a healthy broad spectrum. Use almost pure Muon:

$$
\tau_{\rm mid}\approx0.
$$

But if the final MLP has a huge dominant spike and an extremely weak tail,

$$
\tau_{\rm final}>0.
$$

Then you avoid spending enormous NS compute trying to recover directions that might simply be noise.

An especially natural choice would be

$$
\tau_\ell
=
c\,\sigma_{q,\ell}
$$

or something based on a noise estimate.

Even better:

$$
\tau_\ell
\propto
\sigma_{\rm noise,\ell}.
$$

That would make Muon a kind of **spectral signal-to-noise optimizer**.

---

# 5. The most ambitious improvement: use per-mode SNR

If we really want to exploit the mechanism rather than just the empirical spectrum, I'd estimate

$$
G_b
$$

over several microbatches.

Take the dominant singular vectors $u_i,v_i$ of the momentum and measure

$$
z_{i,b}
=
u_i^\top G_bv_i.
$$

Then estimate

$$
\mathrm{SNR}_i
=
\frac{|\mathbb E[z_{i,b}]|}
{\sqrt{\operatorname{Var}(z_{i,b})}}.
$$

Now the optimizer knows whether a tiny spectral mode is:

**small but consistent**

$$
\sigma_i\ll1,\qquad \mathrm{SNR}_i\gg1
$$

or

**small and noisy**

$$
\sigma_i\ll1,\qquad \mathrm{SNR}_i\ll1.
$$

You would strongly orthogonalize the former and suppress the latter.

Something like

$$
\Delta W
=
\sum_i
w(\mathrm{SNR}_i)
u_iv_i^\top.
$$

That is much more principled than

$$
\Delta W
=
\sum_i u_iv_i^\top.
$$

The practical challenge is doing it cheaply enough.

---

# 6. Another cheap knob worth testing: layer-dependent momentum

Our earlier causal story suggests momentum may itself be making late-layer spectra more concentrated.

If

$$
M_t=\beta M_{t-1}+(1-\beta)G_t,
$$

a highly temporally coherent mode survives, while incoherent directions are suppressed.

So if final-layer gradients are unusually temporally coherent,

$$
\beta=0.95
$$

may make their spike–tail separation enormous.

A simple experiment would be

$$
\beta_{\rm late}<\beta_{\rm mid}.
$$

For example—not as a recommendation, just an ablation—

$$
\beta_{\rm mid}=0.95,
\qquad
\beta_{\rm final}=0.8.
$$

Then ask whether

$$
\frac{\sigma_1^2}{\|M\|_F^2}
$$

falls and whether validation efficiency improves.

I would consider this **much more speculative** than adaptive NS. Momentum filtering can also be precisely what removes stochastic noise, so reducing it might make things worse.

---

# 7. I would *not* immediately try to “fix” gradient concentration itself

This is an important distinction.

It would be tempting to conclude:

$$
\text{concentrated gradient}
=
\text{bad}
$$

and therefore try to force

$$
\operatorname{erank}(G)\uparrow.
$$

I don't think the evidence supports that.

The dominant low-rank modes may be the useful learning signal.

In fact, Spectra takes almost the opposite approach: it explicitly recognizes a dominant spike and treats spike and tail differently rather than trying to make the underlying gradient isotropic. ([arXiv](https://arxiv.org/abs/2602.11185))

And in vision, training recipe changes such as strong augmentation can prevent deep MLP gradient mode collapse, showing that the gradient spectrum is responsive to data/training diversity—but that does not yet establish that artificially maximizing rank is desirable in LLMs. ([arXiv](https://arxiv.org/abs/2605.24770))

So I'd optimize **how we consume the spectrum**, not force the spectrum itself to look pretty.

---

# 8. The practical Muon recipe I would test now

If I were running an LLM pretraining experiment based on everything we've learned, I'd start with:

$$
\boxed{\textbf{Adaptive spectral Muon}}
$$

with three components:

1. **Periodically estimate** per-layer
   $$
   \sigma_{.5}/\|M\|_F,\quad
   \sigma_{.1}/\|M\|_F,\quad
   \rho_1=\sigma_1^2/\|M\|_F^2.
   $$

2. **Choose the minimum NS accuracy per matrix** needed to orthogonalize roughly the top $50\%$–$75\%$ of directions.

3. **Do not chase the extreme spectral tail.** Either simply stop NS once the target quantile is handled, or test a soft/regularized polar map for tiny modes.

So rather than:

$$
\boxed{
\text{same 5-step Muon on every matrix}
}
$$

or

$$
\boxed{
\text{10-step Muon on every matrix},
}
$$

you'd get something like

$$
\boxed{
\begin{array}{c}
\text{cheap Muon where the spectrum is easy}\\
\text{accurate Muon where useful modes are small}\\
\text{regularized treatment of the extreme tail}
\end{array}}
$$

That is the clearest engineering consequence I see.

And interestingly, the first two pieces are already essentially implied by the scaling paper itself: the authors explicitly conclude that uniform NS is suboptimal and propose layer-aware configurations, while also noting that coefficients could be designed specifically for each layer's empirical singular-value distribution. ([arXiv](https://arxiv.org/abs/2606.04058))

The **next leap**, in my view, is not simply “more NS in late layers.” It is recognizing that **spectral magnitude contains information**, and full Muon intentionally throws all of that information away. A regularized/SNR-aware polar update could preserve Muon's main benefit—removing harmful spectral imbalance—without automatically promoting every tiny direction to unit importance.

---

## User

Can you check this? https://arxiv.org/abs/2505.16932
This paper also has some ablations on truncated Muon, or full SVD. Can you read them?

## ChatGPT · gpt-5-6-thinking · Worked for 40s

<details>
<summary>Reasoning</summary>

*Progress note:* Yes. I’ll read the ablation sections directly and compare their **truncated-SVD / exact-SVD Muon** results to the interpretation we’ve been building from the spectral-scaling paper.

**Reviewing ablation pages**

I’m locating the ablation sections on truncated Muon and full SVD, then checking the relevant figures and surrounding text. I’ll summarize what they test and how those results affect the paper’s conclusions.

**Reviewed ablation pages**

**Interpreting the ablations**

I’m finding that exact SVD offers no meaningful validation improvement while roughly doubling step time. Truncation is nearly harmless below a small threshold, but aggressive truncation degrades modestly; reversing singular-value ordering is far worse.

**Refining the conclusion**

I’m distinguishing numerical accuracy from training impact: Polar Express may differ on tiny singular modes, yet those modes barely matter. Truncation at $10^{-4}$–$10^{-3}$ is effectively harmless, while $10^{-2}$ becomes meaningfully worse.

**Interpreted ablations and refined the conclusion**

**Searched 1 website**

</details>

Yes. These ablations are **directly relevant to what we were just discussing**, and in fact they provide stronger evidence than the scaling-law paper that the extreme spectral tail often does **not** deserve exact orthogonalization.

There are two distinct experiments.

### 1. Exact SVD / exact polar does not beat an approximate polar

In §4.3 / Figure 5, they train GPT-2 Small on 1B FineWeb tokens and vary Polar Express from 2 to 30 iterations. They also run **exact Muon**, computing

$$
M=U\Sigma V^\top
\quad\mapsto\quad
UV^\top
$$

using `torch.linalg.svd`. So “SVD” here means **exact polar Muon**, not an optimizer that retains the singular magnitudes. ([arXiv](https://arxiv.org/pdf/2505.16932))

The result is very clean:

- 2 iterations is clearly worse.
- 3 is still worse.
- around 5–6 iterations, performance saturates.
- 7, 10, 20, 30 iterations do essentially nothing further.
- **exact SVD does essentially nothing further either.**

The paper explicitly says that although 5 iterations have **not numerically converged to the exact polar factor**, the resulting optimizer trains just as well. ([arXiv](https://arxiv.org/pdf/2505.16932))

And exact SVD is expensive: in their setup it roughly **doubles step time**, while changing Polar Express from a few iterations to many barely changes overall training time because forward/backward dominate. ([arXiv](https://arxiv.org/pdf/2505.16932))

So:

$$
\boxed{
\text{better numerical approximation of }UV^\top
\not\Rightarrow
\text{better training}
}
$$

past a fairly modest accuracy threshold.

That is already important for our previous discussion about “late layers need more NS iterations.” More iterations are only useful if the additional singular directions they recover actually matter.

---

## 2. Their truncated-Muon ablation is even more interesting

They explicitly decompose

$$
M
=
U_1\Sigma_1V_1^\top+
U_2\Sigma_2V_2^\top
$$

where $U_1,\Sigma_1,V_1$ correspond to singular values

$$
\sigma_i>\gamma\sigma_{\max},
$$

and $U_2,\Sigma_2,V_2$ correspond to

$$
\sigma_i<\gamma\sigma_{\max}.
$$

Then they compare three optimizers.

**Exact Muon:**

$$
\Delta W\propto
U_1V_1^\top+U_2V_2^\top.
$$

Every singular direction maps to $+1$.

**Truncated Muon:**

$$
\Delta W\propto U_1V_1^\top.
$$

The small singular directions map to **zero**.

And, very cleverly, **Reverse Muon:**

$$
\Delta W
\propto
U_1V_1^\top-U_2V_2^\top.
$$

The small singular directions literally point in the **opposite direction from Muon**. ([arXiv](https://arxiv.org/pdf/2505.16932))

This third condition is much more informative than simple truncation.

If the tiny singular directions carry useful gradient information, turning them around should hurt badly.

---

## What happens?

They sweep

$$
\gamma=
10^{-4},10^{-3},10^{-2}.
$$

The plotted final losses are approximately:

| cutoff | Truncated | Reverse | Exact polar |
|---|---:|---:|---:|
| $10^{-4}\sigma_{\max}$ | 3.587 | 3.589 | 3.588 |
| $10^{-3}\sigma_{\max}$ | 3.594 | 3.612 | 3.588 |
| $10^{-2}\sigma_{\max}$ | 3.669 | **4.884** | 3.588 |

These are GPT-2 Small, 1B FineWeb tokens, LR $0.05$, weight decay $0.1$. ([arXiv](https://arxiv.org/pdf/2505.16932))

That's a remarkably informative result.

At

$$
\sigma_i<10^{-4}\sigma_{\max},
$$

you can:

$$
+u_iv_i^\top,
\qquad
0,
\qquad\text{or even}\qquad
-u_iv_i^\top
$$

and **training basically cannot tell the difference**.

At

$$
\sigma_i<10^{-3}\sigma_{\max},
$$

there's a small effect, but even **reversing those directions** only moves final loss from roughly 3.588 to 3.612.

But once you start messing with everything below

$$
10^{-2}\sigma_{\max},
$$

there is clearly useful information there. Truncation hurts moderately; reversing it destroys training. ([arXiv](https://arxiv.org/pdf/2505.16932))

So empirically, in this particular GPT-2 experiment, there's something like:

$$
\boxed{
\begin{array}{ll}
\sigma/\sigma_{\max}\lesssim10^{-4}
& \text{apparently irrelevant}\\[2mm]
\sim10^{-3}
& \text{weakly useful at most}\\[2mm]
\sim10^{-2}
& \text{clearly useful}
\end{array}}
$$

I would not treat those numerical thresholds as universal, but the qualitative result is quite strong.

---

# This changes how I'd interpret finite-step Muon

They make exactly this point.

Five Polar Express iterations **do not converge on the entire spectrum**. If you measure Frobenius error against exact $UV^\top$, there is still substantial error because the tiny singular values aren't fully pushed to one. ([arXiv](https://arxiv.org/pdf/2505.16932))

But look only at singular values above

$$
10^{-3}\sigma_{\max},
$$

and Polar Express is essentially converged after about 5–6 iterations. ([arXiv](https://arxiv.org/pdf/2505.16932))

So finite-step Muon is effectively behaving like a **soft truncated polar transform**:

$$
f(\sigma)\approx
\begin{cases}
1,&\sigma\text{ sufficiently large}\\
\text{intermediate},&\text{transition}\\
0,&\sigma\text{ extremely small}.
\end{cases}
$$

And their training experiments suggest:

$$
\boxed{
\text{this may be a feature rather than merely an approximation error}.
}
$$

That's quite important.

---

# It actually strengthens the regularized-Muon idea we were discussing

Previously I suggested something like

$$
f_\tau(\sigma)
=
\frac{\sigma}{\sqrt{\sigma^2+\tau^2}}
$$

instead of exact

$$
f(\sigma)=1.
$$

This paper provides surprisingly direct empirical support for the motivation.

Exact Muon says:

$$
\sigma=1
\rightarrow1
$$

and

$$
\sigma=10^{-5}
\rightarrow1.
$$

Their ablation asks whether that second transformation is useful.

For sufficiently tiny $\sigma$, the answer appears to be:

$$
\boxed{\text{no.}}
$$

Indeed, the Reverse Muon experiment says something even stronger. Those directions don't merely need less amplification: at sufficiently small magnitude, **the sign of the update along them barely matters**.

That's exactly what you would expect if those modes were dominated by noise or nearly uninformative stochastic variation.

---

# But there's a crucial subtlety when connecting this to the 2606 scaling paper

The two papers normalize their spectral claims differently.

**Polar Express defines “small” relative to the top singular value:**

$$
r_i
=
\frac{\sigma_i}{\sigma_{\max}}.
$$

The scaling-law paper studies singular values after **Frobenius normalization**:

$$
s_i
=
\frac{\sigma_i}{\|M\|_F}.
$$

These are related by

$$
\frac{\sigma_i}{\sigma_{\max}}
=
\frac{\sigma_i/\|M\|_F}
{\sigma_{\max}/\|M\|_F}.
$$

So we **cannot directly say**

> the 2606 median hits $10^{-3}$, therefore Polar Express says it is irrelevant.

We need

$$
\frac{\sigma_{0.5}}{\sigma_{\max}},
$$

not merely

$$
\frac{\sigma_{0.5}}{\|M\|_F}.
$$

And this brings us back to the quantity I thought was missing from the scaling paper:

$$
\rho_1
=
\frac{\sigma_1^2}{\|M\|_F^2}.
$$

Without $\sigma_{\max}/\|M\|_F$, we can't translate between these two papers cleanly.

---

## There is also a really interesting tension between the papers

The newer scaling paper argues roughly:

> as models scale, increasingly small bulk singular values may require more accurate NS so that Muon reaches those modes.

Polar Express gives evidence for almost the opposite instinct:

> **don't bother converging on sufficiently small modes, because they may contain no useful optimization information.**

These are not actually contradictory.

Imagine the spectrum at small scale is

$$
\frac{\sigma_i}{\sigma_1}
=
[1,.2,.1,.05,.02,.01,.005,\ldots,10^{-5}].
$$

If only 5% of modes lie below $10^{-3}$, throwing them away is harmless.

Now suppose scaling causes

$$
60\%
$$

of the spectrum to fall below $10^{-3}$.

At that point, one of two things could be true:

1. those newly small modes are also useless → **do not increase NS accuracy**;
2. they're useful modes whose absolute/relative magnitude shrank due to scaling → **you do need greater NS accuracy**.

The spectral-scaling paper's top-$p$ ablation suggests throwing away **too large a fraction** eventually hurts.

The Polar Express paper says throwing away **sufficiently tiny modes** doesn't.

The missing quantity is therefore not just rank or singular value. It's something like

$$
\boxed{
\text{utility of a mode as a function of its singular magnitude}.
}
$$

---

# And that points to a better experiment than either paper performs

For each mode $i$, measure

$$
r_i=\frac{\sigma_i}{\sigma_1}
$$

and its minibatch consistency

$$
\mathrm{SNR}_i
=
\frac{
\left|\mathbb E_b[u_i^\top G_bv_i]\right|
}{
\sqrt{
\operatorname{Var}_b(u_i^\top G_bv_i)
}
}.
$$

Then plot

$$
\mathrm{SNR}_i
\quad\text{vs.}\quad
\frac{\sigma_i}{\sigma_1}
$$

for different depths and model scales.

Polar Express is implicitly hypothesizing something like

$$
\sigma_i/\sigma_1\ll10^{-3}
\quad\Rightarrow\quad
\mathrm{SNR}_i\approx0.
$$

Their Reverse Muon result is surprisingly good indirect evidence for that.

But **whether that relationship continues to hold in large, deep models is precisely what the scaling-law paper leaves unresolved.**

### My updated optimizer conclusion

After reading this paper, I would **downgrade “use more NS iterations on the final layers” from an obvious improvement to a hypothesis that needs testing**.

A more promising default is:

$$
\boxed{
\text{accurately flatten the informative spectral bulk,
but deliberately leave the extreme tail suppressed.}
}
$$

In other words, instead of asking

> “How many iterations until we reproduce exact $UV^\top$?”

I would ask

> **“How many iterations until we've recovered every singular direction that carries reproducible learning signal?”**

The Polar Express ablations strongly suggest those are **not the same criterion**. ([arXiv](https://arxiv.org/pdf/2505.16932))

---

## User

Can you research the literature more on these?

## ChatGPT · gpt-5-6-thinking · Worked for 3m 10s

<details>
<summary>Reasoning</summary>

*Progress note:* Yes. I’m going to broaden this beyond Muon-specific papers and look for three things: **(1)** evidence that tiny singular modes are useful vs noise, **(2)** optimizers that deliberately truncate/regularize spectral tails, and **(3)** theory on when exact polar/SVD should or should not help.

**Reviewing targeted papers**

I’m narrowing the literature to work on Muon variants, polar-express approximations, singular-value clipping, and spectral regularization. I’m checking theory and experiments separately so the comparison stays precise.

**Reviewed targeted papers**

**Connecting spectral results**

I’m comparing evidence on exact versus approximate orthogonalization, clipped updates, and dynamically shaped spectra. The key question is whether preserving weak singular directions helps optimization or merely amplifies noise.

**Balancing spectral shaping**

I’m finding competing evidence: some studies favor suppressing noisy spectral tails, while others report benefits from preserving weak directions. I’m treating exact orthogonalization, fractional shaping, and adaptive clipping as context-dependent rather than universally superior.

**Connected spectral results and balanced spectral shaping**

**Searched 39 websites**

*Progress note:* A strong pattern is emerging, but it’s nuanced: **multiple 2026 papers independently reject “exactly flatten every singular value to 1” as the uniquely correct target.** At the same time, theory on imbalanced/heavy-tailed data says some weak directions really are important, so “just truncate the tail” is also too crude.

**Mapping the evidence**

I’m broadening the comparison to include low-rank projection, principal-spectrum regularization, fractional Muon powers, and tail-sensitive objectives. I’m separating pretraining, fine-tuning, and numerical evidence because their conclusions may differ.

**Mapped the evidence**

**Synthesizing the literature**

I’m now organizing the findings around a practical conclusion: full SVD is rarely necessary, weak modes can matter selectively, and adaptive spectral shaping appears more promising than fixed truncation.

**Weighing spectral evidence**

I’m finding a nuanced pattern: exact singular-value flattening isn’t uniquely optimal, yet blindly truncating weak modes is also risky. I’m comparing noise suppression, rare-signal preservation, adaptive shaping, and theory across training settings.

**Synthesized the literature**

**Searched 20 websites**

*Progress note:* A useful distinction from the broader search: there are **two different “truncations”** in this literature. Some papers truncate the *number of Newton–Schulz iterations* (a softer approximation to polar), while Polar Express explicitly truncates the *singular spectrum* by zeroing small modes. They answer different questions, and both independently suggest exact SVD is not the right optimization target.

**Refining the evidence**

I’m separating established results from newer preprints. The emerging view is that Muon’s benefit may come from stable update geometry, not literal unit singular values; exact SVD is often unnecessary, while adaptive tail control remains unresolved.

**Searched 5 websites**

</details>

Yes. I went broader than the two papers we were discussing. There is now enough work around Muon/spectral optimization that a fairly coherent picture emerges—although much of the most relevant work is from 2026 and several papers are still preprints.

The main conclusion I would draw is:

$$
\boxed{\text{Exact polar orthogonalization is almost certainly not the right fundamental objective.}}
$$

But the alternative is **not** simply “truncate all small singular values.” The literature increasingly points toward **selective, noise-aware spectral shaping**.

### What the literature is saying

| Work | What it changes | Main implication |
|---|---|---|
| **Polar Express** | Approximate vs exact polar; truncate/reverse small modes | Exact SVD gives no gain; very small modes can be changed drastically with almost no effect |
| **How Much Orthogonalization Does Muon Need?** | Number/type of polynomial iterations vs exact SVD | Independent evidence that more accurate orthogonalization does not monotonically improve training |
| **DynMuon** | $U\Sigma^pV^\top$, with $p$ changing during training | Optimal spectral shaping changes with training stage and noise |
| **Spectra** | Shrink large spike; leave tail largely alone | Tail has semantic signal **and** high relative noise |
| **HTMuon** | Preserve heavier-tailed spectrum instead of flat spectrum | Full whitening may overemphasize noisy directions |
| **Muon$^p$** | $0<p<1$, retain some singular-value information | Complete flattening can discard useful magnitude information |
| **Muon on imbalanced / tail data** | Studies why flattening helps | Weak directions can contain important rare/minority information |
| **GaLore / GoLore** | Low-rank gradient subspaces | Truncation can work, but biased low-rank projection can also fail under stochastic noise |

Here's how I think these pieces fit.

---

## 1. The exact-SVD result now has independent support

Polar Express's cleanest experiment is exactly the one we discussed. They run Muon using 2–30 Polar Express iterations and also compute the polar factor **exactly** using `torch.linalg.svd`.

Performance improves up through roughly 5–6 iterations, but after that additional numerical accuracy doesn't help—and exact SVD doesn't improve validation loss either. In their setup the SVD roughly doubles step time. ([arXiv](https://arxiv.org/html/2505.16932))

And importantly, five iterations have **not actually converged to the polar factor**. The authors explicitly observe that the approximate update remains meaningfully different from exact $UV^\top$ while optimizing just as well. ([arXiv](https://arxiv.org/html/2505.16932))

A subsequent 2026 paper, **How Much Orthogonalization Does Muon Need?**, repeats essentially this question. On GPT-2 Small it compares multiple truncated polynomial schemes with an explicit FP32 SVD polar factor, across three seeds. Its strongest approximate methods and exact SVD all land within about $0.003$ final validation loss; exact SVD is not best, and adding another orthogonalization iteration is not even monotonically beneficial. ([arXiv](https://arxiv.org/html/2606.00371))

So I now consider this reasonably robust:

$$
\boxed{
\operatorname{accuracy}(\widehat{UV^\top},UV^\top)
\quad\text{is not monotonically related to training quality.}
}
$$

That's a surprisingly important result.

---

## 2. Polar Express gives unusually strong evidence about the *extreme* tail

Their truncated/reverse experiment is stronger than an ordinary low-rank ablation.

They partition

$$
M=U_1\Sigma_1V_1^\top+U_2\Sigma_2V_2^\top
$$

at a relative singular-value threshold $\gamma\sigma_1$.

Then they test:

$$
\text{exact:}\quad
U_1V_1^\top+U_2V_2^\top,
$$

$$
\text{truncate:}\quad
U_1V_1^\top,
$$

and

$$
\text{reverse:}\quad
U_1V_1^\top-U_2V_2^\top.
$$

At a cutoff around

$$
\gamma\sim10^{-3},
$$

all three train well. In other words, you can approximately **zero or reverse the signs** of sufficiently small modes without much consequence in that GPT-2 experiment. ([arXiv](https://arxiv.org/html/2505.16932))

That's much stronger evidence than merely observing

> low-rank approximation retains most Frobenius energy.

If reversing the direction doesn't hurt, those modes apparently contribute very little useful descent information under that experimental regime.

It also explains why five Polar Express iterations work: five iterations have approximately dealt with modes above the useful threshold, without bothering to perfectly map the extreme tail to 1. ([arXiv](https://arxiv.org/html/2505.16932))

So finite NS may accidentally be implementing a useful **soft spectral filter**.

---

# 3. But DynMuon provides an extremely important counterpoint

This is perhaps the most interesting paper I found for our discussion.

DynMuon considers

$$
\boxed{
D_p=U\Sigma^pV^\top
}
$$

with

$$
p=1 \Rightarrow \text{ordinary gradient/momentum},
$$

$$
p=0 \Rightarrow \text{Muon},
$$

and

$$
p<0 \Rightarrow \text{even stronger amplification of small modes than Muon}.
$$

Their theory explicitly decomposes each mode into residual signal and stochastic noise. The key result is that decreasing $p$:

- makes progress faster in weak/flat modes,
- **but simultaneously amplifies their stochastic noise**. ([arXiv](https://arxiv.org/html/2605.17109))

So the right spectral exponent depends on

$$
\frac{\text{remaining signal}_i}
{\text{noise}_i}.
$$

Even more interestingly, they measure this during training.

Early in training, stronger/high-curvature modes contain more residual optimization signal, so **positive $p$** works better.

Later, those modes have largely been learned while significant residual signal remains in flatter modes, so mildly negative

$$
p\approx-0.1\text{ or }-0.25
$$

can actually outperform Muon's $p=0$. But going very negative becomes unstable because noise amplification dominates. ([arXiv](https://arxiv.org/html/2605.17109))

This is an important correction to any blanket conclusion from Polar Express:

$$
\boxed{\text{small modes are not intrinsically useless.}}
$$

Their usefulness depends on **training stage and SNR**.

---

## 4. DynMuon even does the batch-size experiment we wanted

This is particularly relevant to our discussion of the spectral-scaling paper.

They vary batch size from

$$
2\rightarrow128
$$

and find that the best late-stage spectral exponent changes systematically:

$$
B=2 \Rightarrow p\approx-0.1
$$

whereas at larger batches the optimum moves toward more aggressive small-mode amplification; at $B=128$, $p=-0.5$ performs best among their tested settings. ([arXiv](https://arxiv.org/html/2605.17109))

That's almost exactly the signal/noise story we hypothesized.

Larger batch:

$$
\operatorname{Var}(\epsilon)\downarrow
$$

so previously questionable weak modes become more trustworthy:

$$
\mathrm{SNR}_{\rm weak}\uparrow.
$$

Then it becomes worthwhile to amplify them more aggressively.

So **batch size genuinely can interact with the optimum degree of spectral flattening**.

That's a useful result when thinking about 2606.04058, where batch size wasn't reported.

---

# 5. Spectra measures something even closer to our proposed experiment

The new **Spectra** paper explicitly decomposes the gradient into a low-rank spike and spectral tail.

Their reported picture is:

$$
\boxed{
\text{small dominant spike}
+
\text{long semantic tail}.
}
$$

They estimate that roughly the top $1.5\%$ of directions form a pronounced spike, separated by one to two orders of magnitude from the tail. They associate the spike primarily with frequently recurring linguistic structure and the tail with finer/context-dependent semantic variation. ([arXiv](https://arxiv.org/html/2602.11185))

Crucially, however, they also measure stochastic reliability along the spectrum.

As singular values decrease:

$$
\boxed{
\frac{\text{variance}}
{\text{signal scale}^2}
\uparrow.
}
$$

So weaker directions increasingly carry **sparser signals with higher relative variance**. ([arXiv](https://arxiv.org/html/2602.11185))

That gives exactly the nuanced answer we were looking for:

> The tail contains information, but progressively deeper into the tail that information becomes less statistically reliable.

Spectra therefore does **not** do Muon:

$$
[\sigma_1,\ldots,\sigma_r]
\rightarrow
[1,\ldots,1].
$$

Nor does it simply truncate the tail.

Instead, it leaves the tail essentially at its natural scale and **shrinks the dominant spike toward the tail**. ([arXiv](https://arxiv.org/html/2602.11185))

Conceptually:

$$
\begin{array}{ccc}
\text{raw}&\text{Muon}&\text{Spectra}\\
100&&\\
10&&\\
1&1&1\\
0.5&1&0.5\\
0.1&1&0.1\\
0.01&1&0.01
\end{array}
$$

with the huge $100,10$ components brought down.

That's almost the inverse perspective:

$$
\boxed{\text{don't amplify the tail; suppress the spike.}}
$$

Their reported LLaMA-3 8B/50B-token results favor this approach over both AdamW and Muon, though Spectra is currently a very recent preprint, so I'd regard those precise performance claims as provisional. ([arXiv](https://arxiv.org/html/2602.11185))

---

# 6. HTMuon reaches a similar conclusion independently

HTMuon was published in **Findings of ACL 2026**, so it's somewhat stronger evidence than the newest preprints.

Its premise is explicitly that Muon's full flattening

$$
\Sigma\rightarrow I
$$

can **over-emphasize noise-dominated directions** and prevent the optimizer/weights from developing beneficial heavy-tailed spectral structure.

Their alternative deliberately keeps updates more heavy-tailed. They report improvements over Muon in LLM pretraining and vision, including up to $0.98$ perplexity reduction in their LLaMA/C4 setup. ([ACL Anthology](https://aclanthology.org/2026.findings-acl.1819/))

So both Spectra and HTMuon independently arrive at:

$$
\boxed{
\text{complete spectral flattening is too aggressive.}
}
$$

But they get there from somewhat different arguments.

---

# 7. Muon$^p$ gives another version of the same story

Muon$^p$ interpolates continuously:

$$
U\Sigma V^\top
\rightarrow
U\Sigma^pV^\top,
\qquad0<p<1.
$$

Thus instead of

$$
[100,10,1,.1,.01]
\rightarrow
[1,1,1,1,1],
$$

you might use $p=1/2$:

$$
[10,\sqrt{10},1,\sqrt{.1},.1].
$$

You reduce the condition number drastically while retaining some information about original singular magnitudes.

Their motivation is explicitly that **full flattening discards singular-value information that may matter**, and they find fractional powers particularly beneficial in billion-scale finetuning experiments. ([arXiv](https://arxiv.org/abs/2606.13867))

Another 2026 analysis, *Delving into Muon and Beyond*, also studies $U\Sigma^pV^\top$ and finds that $p=0$ isn't uniformly best across settings. ([arXiv](https://arxiv.org/abs/2602.04669))

So this is no longer one isolated result.

---

# 8. But there is strong evidence against simply dropping weak directions

Here's the other side.

Two ICLR 2026 papers give compelling reasons why **small/dominated directions can be precisely where important information lives**.

*How Muon's Spectral Design Benefits Generalization* studies imbalanced data. In its simplified models, ordinary GD learns the high-variance/dominant principal components first, whereas spectral gradient descent

$$
G=U\Sigma V^\top
\rightarrow UV^\top
$$

learns all components at more equal rates.

That improves balanced/minority performance—and they find that increasing network depth can actually amplify this advantage. ([arXiv](https://arxiv.org/abs/2510.22980))

Similarly, *Muon Outperforms Adam in Tail-End Associative Memory Learning* finds that Muon's advantage is concentrated particularly in VO and FFN associative-memory parameters and argues that isotropic spectral updates help learn rare/tail associations in heavy-tailed language data. ([ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3eec5006051d9544e717067de3220198-Abstract-Conference.html))

That gives us the key warning:

$$
\boxed{
\sigma_i\text{ small}
\not\Rightarrow
\text{mode }i\text{ unimportant}.
}
$$

A mode can be small precisely because its underlying event/class occurs rarely.

If you truncate it because of magnitude alone, you can systematically sacrifice rare knowledge.

---

# 9. GaLore gives a useful older analogue

GaLore showed empirically that substantial LLM training can occur in relatively low-dimensional gradient subspaces, which is evidence that **not every instantaneous gradient direction is necessary**. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v235/zhao24s.html))

But the follow-up convergence literature is instructive.

An ICML 2025 analysis shows that fixed/structured low-rank gradient projection such as GaLore can actually **fail to converge under ordinary stochastic gradients**. They prove convergence under conditions like sufficiently large batches or isotropic noise, and introduce randomized GoLore to restore convergence in more general stochastic settings. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v267/he25i.html))

Again:

$$
\boxed{
\text{low rank alone is not the issue; interaction with gradient noise matters}.
}
$$

And NeurIPS 2025 theory independently shows that approximate low-rank gradient structure can arise naturally from data/residual correlations rather than being a numerical accident. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/df7ac32d352db77aad1f355ad43206a9-Abstract-Conference.html))

---

# 10. So I think we can draw a fairly strong synthesis now

There seem to be **three spectral regimes**, rather than merely “large” and “small.”

Conceptually:

$$
\boxed{
\begin{array}{rcl}
\textbf{Spike} &&
\text{large, coherent, often redundant/frequent signal}\\[2mm]
\textbf{Middle} &&
\text{useful learnable signal}\\[2mm]
\textbf{Extreme tail} &&
\text{sparse signal mixed increasingly with noise}
\end{array}}
$$

Different optimizers mishandle different regions.

Adam can let the **spike dominate** its statistics and learning-rate restriction. Spectra provides direct evidence for this. ([arXiv](https://arxiv.org/html/2602.11185))

SGD leaves the huge dynamic range intact, so weak but useful modes progress slowly.

Muon solves both by doing

$$
\Sigma\rightarrow I,
$$

but that creates the opposite problem:

$$
\boxed{\text{it promotes the extreme noise floor to full-strength update directions}.}
$$

Polar Express, HTMuon, Spectra and DynMuon all provide different evidence for this.

---

# 11. This suggests a better picture of why approximate NS works

Originally one might say:

> NS is just a cheap approximation to the mathematically ideal exact polar factor.

I now think that framing is probably wrong.

A finite polynomial $f_k$ gives something roughly like

$$
\sigma
\longmapsto
f_k(\sigma):
$$

$$
\begin{cases}
\approx1,&\text{moderate/large }\sigma\\
\text{partially amplified},&\text{small }\sigma\\
\approx0,&\text{extremely tiny }\sigma.
\end{cases}
$$

That's not simply an inaccurate Muon.

It may be an **implicitly regularized Muon**.

Polar Express demonstrates that getting the tiniest modes closer to 1 doesn't improve its training experiment. ([arXiv](https://arxiv.org/html/2505.16932))

And Huang's later exact-SVD experiments independently support the same conclusion. ([arXiv](https://arxiv.org/html/2606.00371))

So the fact that practical Muon uses an imperfect Newton–Schulz approximation may partly be **why it works as well as it does**, rather than merely a computational compromise.

That's a substantially different interpretation.

---

# 12. What I think the missing optimizer is

After seeing this literature, I would modify the regularized-polar idea we discussed.

I wouldn't choose the transform solely as

$$
f(\sigma).
$$

I'd want

$$
\boxed{
f(\sigma_i,\mathrm{SNR}_i,t,\ell)
}
$$

depending on:

- singular magnitude;
- statistical reliability of the mode;
- training stage $t$;
- layer $\ell$.

The ideal behavior looks qualitatively like:

$$
f_i
\simeq
\begin{cases}
\text{shrink},&
\text{giant redundant spike}\\
1,&
\text{weak-but-reproducible useful mode}\\
<1,&
\text{noise-dominated extreme tail}.
\end{cases}
$$

And unlike my earlier speculation, **DynMuon and Spectra actually measure quantities very close to the SNR statistic we proposed**.

DynMuon estimates mini-batch variance of gradient projections along modes and compares residual signal against that noise. ([arXiv](https://arxiv.org/html/2605.17109))

Spectra measures relative variance across the spectral tail and finds it rises as singular values shrink. ([arXiv](https://arxiv.org/html/2602.11185))

So the SNR formulation isn't just hypothetical anymore.

---

## Where I think the evidence currently lands

I would assign fairly high confidence to:

$$
\boxed{\text{“More accurate polar decomposition” is not synonymous with “better Muon.”}}
$$

I'd also assign moderate-to-high confidence to:

$$
\boxed{\text{Full flattening unnecessarily amplifies some extreme-tail modes.}}
$$

But I would **not** endorse:

$$
\boxed{\text{“truncate all modes below }10^{-3}\sigma_{\max}\text{.”}}
$$

The $10^{-3}$ result is regime-specific, and DynMuon plus the imbalanced-data work shows why such a universal cutoff can fail: some weak modes have good SNR or encode rare but important structure. ([arXiv](https://arxiv.org/html/2605.17109))

The most defensible general principle emerging from the literature is instead:

$$
\boxed{
\textbf{Flatten based on reliability, not merely magnitude.}
}
$$

That also changes how I would read the original **Spectral Scaling Laws of Muon** result. Seeing late-layer median singular values shrink with scale does **not by itself tell us to use more NS iterations**. We first need to know whether those newly accessible late-layer modes have reproducible gradient signal. If they don't, recovering them more accurately could actually be counterproductive.

And given DynMuon's batch-size result, that answer may even change with the training batch.
