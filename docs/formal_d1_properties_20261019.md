# Internal appendix: D1 properties, conditional stability, and collision limits

Date: 2026-10-19. Status: internal appendix, closed to further theory expansion. Scope: the implemented external-hash D1 writer, with D0 as a comparison. The research focus remains forward D1 writes into persistent n-gram memory with the backbone frozen.

This note derives properties of the update actually executed by the repository. It does not prove factual correctness, semantic memory, generalization, or an improvement in language-model loss. The accompanying small CPU checks are in [`verify_d1_formal.py`](../scripts/verify_d1_formal.py); they check the algebra and its correspondence to the implementation, rather than estimate model performance.

## 1. Implementation boundary and assumptions

| Object | Implemented behavior | Source |
| --- | --- | --- |
| D1 update | Every event for a physical row uses one common old value; its increment is averaged over **all** non-padding events for that row, including zero gates. | [`functional_d1`](../src/delta_ngram/operators.py), lines 9-47 |
| Evidence features | Token hidden state, within-block right-context mean, block attention pool, and order embedding. Hidden states are detached. | [`ForwardWriter.events`](../src/delta_ngram/writer.py), lines 65-103 |
| Target | A linear-SiLU-linear network of evidence features; no output clipping, normalization, or bounded activation. | [`ForwardWriter`](../src/delta_ngram/writer.py), lines 47-48 and 103 |
| D1 gate | Sigmoid of a network receiving evidence features, old value, target minus old value, and `log1p(old_count)`. | [`ForwardWriter.events`](../src/delta_ngram/writer.py), lines 105-112 |
| D0 gate | Evidence features only; the same update equation as D1. | [`functional_d0`](../src/delta_ngram/operators.py), lines 147-149; [`writer.py`](../src/delta_ngram/writer.py), lines 105-106 |
| State | Missing rows and padding row zero read as zero. A touched row's count advances once per block, even if every gate is zero. | [`SparseMemoryState.lookup`](../src/delta_ngram/state.py), lines 76-93; [`operators.py`](../src/delta_ngram/operators.py), lines 44-47 |
| Reader | Sum of the addressed order-specific row values, divided by the square root of the number of orders. | [`ExternalHashAdapter.read_residual`](../src/delta_ngram/reader.py), lines 32-34 |
| Hash | Fixed deterministic polynomial arithmetic; each order slot has a disjoint row namespace. | [`ngram_rows`](../src/delta_ngram/ngram_lookup.py), lines 5-28 |
| Batched ownership | Physical rows are packed into disjoint owner namespaces. A batch represents independent streams at one logical time, not successive times of one stream. | [`BatchMemoryState`](../src/delta_ngram/batching.py), lines 28-36 and 179-228 |

The mathematical statements below use finite real arithmetic. The implementation casts event targets, gates, and updated values to FP32, so numerical tests use tolerances. Floating-point reductions need not be bitwise permutation invariant; rounding, overflow, NaNs, and sigmoid saturation are outside the exact-real proofs. A finite sigmoid logit gives a gate strictly between zero and one in real arithmetic, but FP32 can produce either endpoint.

The raw `functional_d1` function validates tensor shapes, devices, and row IDs; it does **not** reject arbitrary out-of-range or non-finite gates. Thus `0 <= beta <= 1` is a mathematical precondition, supplied by the normal writer path for finite outputs, not an unconditional guarantee of that low-level API. Likewise, valid nonnegative observation counts and finite activations are preconditions of the writer analysis.

Fix one evidence stream, its block schedule, model and writer parameters, and the token-derived addresses. In the intended path, evidence features come from the unpatched frozen backbone via [`encode_evidence`](../src/delta_ngram/reader.py), lines 95-118. Consequently an event target is independent of the old memory value when the evidence is held fixed. A D1 gate can depend on the old value and count of its own row. Other rows can share evidence features, but their values are not inputs to this gate. Statements comparing two memories always hold the evidence, event identities, parameters, and addresses fixed unless explicitly stated otherwise.

## 2. Exact common-snapshot update

Consider one non-padding physical row. Let its old value be `x` in R^d and its old observation count be `c`. Suppose the current block contributes `m > 0` events to that row, with targets `t_i` and gates `beta_i`. Define

\[
a = \frac{1}{m}\sum_{i=1}^{m}\beta_i,
\qquad
b = \frac{1}{m}\sum_{i=1}^{m}\beta_i t_i.
\]

The implementation is exactly

\[
\boxed{x' = x + \frac{1}{m}\sum_i\beta_i(t_i-x) = (1-a)x+b,\qquad c'=c+1.}
\tag{1}
\]

Every `beta_i` is computed before this update, from the same old snapshot. This is an average of event increments, not a sequential application of the events. A row with no events keeps both its value and count. Padding events are removed before computing `m`.

Zero gates remain in `m`. For example, with scalar `x=2`, an event `(t=6, beta=1/2)` gives `x'=4`. Adding a second non-padding event with `beta=0` gives `x'=3`, regardless of that second event's target. An all-zero-gate block leaves the value unchanged while advancing the count; that changed count can affect a later D1 gate.

In exact arithmetic, permuting the events within this common snapshot does not change (1). Splitting a block into sequential writes generally changes the result: later writes see updated values and counts, and their evidence features may also change when block boundaries change.

## 3. Convex-hull invariance and bounded values

**Proposition 1 (one-step convexity).** If every gate is in `[0,1]`, then

\[
x' = (1-a)x + \sum_i \frac{\beta_i}{m}t_i
\in \operatorname{conv}\{x,t_1,\ldots,t_m\}.
\tag{2}
\]

**Proof.** All displayed coefficients are nonnegative, and their sum is `1-a+a=1`. No independence between a gate and `x` is needed. If `a=0`, all gates are zero and `x'=x`. If `a>0`, setting `t_bar=b/a` also gives `x'=(1-a)x+a*t_bar`, where `t_bar` is a convex combination of targets. QED.

**Corollary 1 (any norm).** If `||t_i|| <= R` for every event of this row, then

\[
\|x'\| \le (1-a)\|x\| + \frac1m\sum_i\beta_i\|t_i\|
\le \max\{\|x\|,R\}.
\tag{3}
\]

Induction gives, for every finite trajectory,

\[
x_T\in\operatorname{conv}\bigl(\{x_0\}\cup\{\text{all targets observed by this row up to }T\}\bigr),
\qquad
\|x_T\|\le\max\{\|x_0\|,\sup_{k,i}\|t_{k,i}\|\}.
\tag{4}
\]

This proof permits adaptive gates and even adaptive targets, provided the asserted target bound holds on the realized trajectory. It also applies to every row separately, including newly allocated rows whose old value is zero. It bounds memory **values**, not the observation counts or the number of allocated rows.

**What is conditional here?** The current target MLP has no architecture-enforced uniform output bound. Small initial weights do not impose a bound throughout training. A finite trace with finite targets has a finite trace-specific maximum, but that is not a uniform guarantee over future evidence, learned parameters, or arbitrary stream lengths. A global value bound requires an explicit bound on targets, or assumptions sufficient to derive one, such as bounded features and fixed bounded network weights. This note does not assert that those assumptions have been established experimentally.

## 4. Exact path expansion and what “retention” means

For a fixed row, index block times by `k=1,...,T`. Set `a_k=b_k=0` at times when the row is untouched. Let

\[
\rho_{s:t}=\prod_{\ell=s}^{t}(1-a_\ell),\qquad \rho_{s:t}=1\text{ if }s>t.
\]

**Proposition 2 (pathwise expansion).** Repeated substitution in (1) yields

\[
\boxed{x_T=\rho_{1:T}x_0+\sum_{k=1}^{T}\rho_{k+1:T}b_k}
=\rho_{1:T}x_0+
\sum_{k:m_k>0}\sum_{i=1}^{m_k}
\rho_{k+1:T}\frac{\beta_{k,i}}{m_k}t_{k,i}.
\tag{5}
\]

**Proof.** The claim holds at one step. Substitute the expression for `x_(T-1)` into `x_T=(1-a_T)x_(T-1)+b_T` to obtain the induction step. The initial-state coefficient and all target coefficients are nonnegative, and their sum is

\[
\rho_{1:T}+\sum_k\rho_{k+1:T}a_k=1,
\tag{6}
\]

by the same induction, or by telescoping products. QED.

The coefficient `rho_(1:T)` is an exact **algebraic** retention factor along the realized path. If each of `N` touched steps has `a_k >= epsilon > 0`, then this coefficient is at most `(1-epsilon)^N`. There is no such uniform decay guarantee without a lower bound on the gates.

For state-dependent D1 gates, all the coefficients in (5) can themselves depend on `x_0`. Therefore (5) is not a causal attribution, an independent mixture model, or a bound on sensitivity to the initial state. In general,

\[
\frac{\partial x_T}{\partial x_0}=J_TJ_{T-1}\cdots J_1
\ne \rho_{1:T}I.
\tag{7}
\]

The equality to `rho*I` does hold for D0 value dynamics under the fixed-evidence comparison below.

## 5. D1 need not be contractive

### 5.1 A bounded, realizable counterexample

Let `d=m=1`, fix `t=0`, and use `beta(x)=sigmoid(-x)`. Then

\[
F(x)=(1-\operatorname{sigmoid}(-x))x,
\qquad
F'(x)=1-\beta(x)+x\beta(x)(1-\beta(x)).
\]

At `x=2`,

\[
\beta(2)\approx0.119202922,
\qquad F'(2)\approx1.090784249>1.
\tag{8}
\]

On `[0,3]`, every state remains in `[0,3]` and moves toward the bounded target zero. Nevertheless, sufficiently close states near two move farther apart after one update. Thus convex-hull invariance and sigmoid-range gates do not even imply nonexpansiveness, and cannot by themselves imply global contraction.

This gate is realizable by the existing architecture, not just an unrelated gate function: `SiLU(z)-SiLU(-z)=z`. Two hidden units that receive `x` and `-x`, with output weights `-1` and `+1`, produce logit `-x`; all other gate weights can be zero. Setting the target network to zero produces `t=0`. The verification script constructs this `ForwardWriter` and differentiates its actual update.

### 5.2 Correct Jacobian and a sufficient condition

Hold evidence and count `c` fixed. The targets are constant with respect to `x` in this implementation. Write `beta_i=beta_i(x,c)`. Differentiating (1) gives

\[
\boxed{J(x,c)=(1-a(x,c))I+
\frac1m\sum_i(t_i-x)\nabla_x\beta_i(x,c)^\top.}
\tag{9}
\]

The second term is essential. In any vector norm with its induced matrix norm and dual vector norm,

\[
\|J(x,c)\|\le 1-a(x,c)+
\frac1m\sum_i\|t_i-x\|\,\|\nabla_x\beta_i(x,c)\|_*.
\tag{10}
\]

For a differentiable map on a convex domain, `sup_x ||J(x,c)|| <= q` is sufficient for a `q`-Lipschitz bound, by integrating the Jacobian along the line segment between two points. In particular, if throughout that domain

\[
\frac1m\sum_i\|t_i-x\|\,\|\nabla_x\beta_i(x,c)\|_*
\le a(x,c)-\epsilon
\quad\text{for one fixed }\epsilon>0,
\tag{11}
\]

then the row value map contracts by at most `1-epsilon`. This is sufficient, not necessary: the triangle bound in (10) discards cancellations. Conversely, a differentiable `q`-Lipschitz map must have `||J|| <= q` at every interior point, so (8) rules out any `q <= 1` on a neighborhood of two.

The sigmoid range alone supplies no useful bound on `nabla beta`. If `beta_i=sigmoid(g_i)`, then `nabla beta_i=beta_i(1-beta_i)*nabla g_i`. For example, in the Euclidean norm, the additional assumptions

\[
\|x\|,\|t_i\|\le R,\quad \|\nabla g_i\|\le L_g,\quad
a\ge a_{\min}>0
\]

give the sufficient bound

\[
\|J\|\le 1-a_{\min}+\frac{RL_g}{2}.
\tag{12}
\]

Hence `R*L_g/2 < a_min` is one explicit sufficient contraction condition. Neither this inequality nor the necessary target and gate-gradient bounds are enforced by the current architecture or established by this note.

To connect (9) to the gate inputs, let the first gate layer's blocks multiplying `x` and `(t_i-x)` be `W_x` and `W_r`. Its derivative with respect to `x` contains `W_x-W_r`, not either block alone. The count input is fixed in (9). If targets were changed in a future implementation to depend on memory, an additional term `m^(-1)*sum_i beta_i*D_x t_i` would be required.

### 5.3 Counts and the complete memory

For a touched row, the augmented real-valued map `(x,c) -> (F(x,c),c+1)` has Jacobian

\[
\begin{pmatrix}
J & \frac1m\sum_i(t_i-x)\partial_c\beta_i\\
0 & 1
\end{pmatrix}.
\tag{13}
\]

It has eigenvalue one, so its induced operator norm cannot be below one. Equivalently, in a standard product norm, the count-coordinate difference is preserved. Claims of strict contraction must specify value dynamics at the same count, not the full value-and-count state. Real observation counts are discrete; this Jacobian uses the smooth extension of the implemented arithmetic, and the preserved-count-difference argument applies directly without that extension.

Likewise, an untouched physical row is an identity map on its value. Even if every touched value map contracts, the complete value map over a fixed row namespace is not strictly contractive in a standard product norm whenever some row is untouched. Contraction over a multi-block interval needs assumptions covering every row being compared. None of these results proves stability of training gradients through arbitrary trajectories.

## 6. D0: a precise conditional contraction statement

D0's gates and targets depend on evidence only. Compare two runs with the same evidence, parameters, addresses, and write schedule, but different initial values `x_0` and `y_0`. For one row they have identical `a_k,b_k`, and therefore

\[
x_T-y_T=\rho_{1:T}(x_0-y_0),
\qquad
\|x_T-y_T\|=\rho_{1:T}\|x_0-y_0\|.
\tag{14}
\]

This is exact for every norm. A touched step with `a>0` strictly contracts this row's values; an untouched or all-zero-gate step is the identity. If `a>=epsilon>0` on `N` touched steps, the factor is at most `(1-epsilon)^N`. Finite real sigmoid logits give `a>0` on nonempty rows, but do not provide a uniform positive lower bound across arbitrary evidence. Counts, untouched rows, finite precision, and comparisons that change the evidence still require the qualifications above.

D1 also obeys (14) if its gates are artificially held fixed while varying the initial value. That restricted comparison does not establish contraction of the actual state-dependent writer.

## 7. Deterministic collision interference bounds

These bounds concern events that already share one physical row. They do not assign a probability to the implemented deterministic hash, and they do not establish that the writer distinguishes semantically related and unrelated events.

Fix the same old `x,c` and the same per-event evidence features. Partition the events into `p>0` intended events `I` and `q>=0` colliding events `C`. Compute `F_I(x)` and, when `q>0`, `F_C(x)` by applying (1) to each subset separately using that same snapshot and those fixed features. The combined write satisfies

\[
F_{I\cup C}(x)=\frac{p}{p+q}F_I(x)+\frac{q}{p+q}F_C(x).
\tag{15}
\]

For `q=0` the interference is zero. Otherwise,

\[
\boxed{\|F_{I\cup C}(x)-F_I(x)\|
=\frac{q}{p+q}\|F_C(x)-F_I(x)\|
\le \frac{2Rq}{p+q}}
\tag{16}
\]

provided `||x|| <= R`, every target has norm at most `R`, and all gates are in `[0,1]`.

**Proof.** Separate the sum in (1) into its two subsets to get (15). Proposition 1 puts both subset outputs in the radius-`R` ball. Apply the triangle inequality to their difference. The bound is tight for a scalar example with every intended gate one and target `+R`, and every colliding gate one and target `-R`. QED.

This includes denominator interference. If all colliding gates are zero, `F_C(x)=x`, so

\[
F_{I\cup C}(x)-F_I(x)=-\frac{q}{p+q}\bigl(F_I(x)-x\bigr).
\tag{17}
\]

Zero gates prevent direct target injection but still dilute the intended write. The multiplicity is the event count, not the number of positive gates or distinct keys.

**Comparison boundary.** In the real writer, deleting colliding tokens from an evidence block can change right-context features, the pooled feature, and thus the remaining targets and gates. Equation (16) compares fixed event features at a shared snapshot; it is not an end-to-end guarantee for deleting or replacing input text. Multi-block collisions can change later values and gates. A one-step bound alone cannot establish that their influence remains small indefinitely; propagation requires a separate Lipschitz or stability argument such as (10).

## 8. What a random-hash assumption would buy

The current `ngram_rows` hash is deterministic and is not sampled from a proved universal family. For any fixed input stream its collisions are fixed. The following statements are therefore **conditional models**, not unconditional properties of the implementation.

Assume a hash family such that each distinct unrelated key collides with a fixed intended key with probability `1/B`, in one order slot with `B` buckets. For `n` distinct unrelated keys, let `Q` be the number that collide. Linearity of expectation and the union bound give

\[
\mathbb E[Q]=n/B,\qquad
\Pr(Q\ge1)\le\min(1,n/B).
\tag{18}
\]

These two results require the stated pairwise collision probabilities but not mutual independence of collision events. If the complete key hashes are independent and uniform, the sharper probability is `1-(1-1/B)^n`. The verification script enumerates this toy model; it does not validate the repository hash against that model.

For a conditional version of (16), assume additionally one event per unrelated key, a fixed positive number `p` of intended events, and the same uniform norm bound `R` for every hash realization. The expected one-step fixed-feature interference obeys

\[
\mathbb E\bigl[\|F_{I\cup C}-F_I\|\bigr]
\le 2R\,\mathbb E\left[\frac{Q}{p+Q}\right]
\le \frac{2R(n/B)}{p+n/B}.
\tag{19}
\]

The second inequality is Jensen's inequality for the concave function `z/(p+z)`, `z>=0`. If unrelated key `j` contributes a fixed multiplicity `w_j`, the colliding **event** count is a weighted sum, with expectation `sum_j(w_j)/B` under the same pairwise model. Repetition does not create independent keys. Data-dependent or adversarial keys and hash-dependent event selection require additional assumptions.

Order slots use disjoint physical namespaces. Keys from different order slots cannot collide in storage, even if their bucket indices match. This namespace separation does not imply statistically independent hashes or errors across orders.

## 9. Reader-level consequence and its limit

For a fixed query position with `K` configured orders, the external adapter reads

\[
r=\frac1{\sqrt K}\sum_{h=1}^{K}x_h.
\]

For two memories at the same query addresses,

\[
\|r-\widetilde r\|
\le\frac1{\sqrt K}\sum_h\|x_h-\widetilde x_h\|
\le\sqrt K\max_h\|x_h-\widetilde x_h\|.
\tag{20}
\]

The factor `1/sqrt(K)` does not give a worst-case constant bound independent of `K`; aligned row errors can add. Incomplete n-grams and padding contribute zero. Combining row-specific bounds from (16) with (20) is valid for the same fixed-feature, one-step comparison. No independence assumption is needed for (20).

The reader then casts the residual to the backbone activation dtype and injects it into a decoder layer. A bound on logit changes would additionally require a suitable bound on the downstream network's sensitivity and a treatment of rounding. A bound on factual accuracy, useful knowledge retention, language-model loss, or semantic generalization does not follow from (20).

## 10. Reproducible checks and claim ledger

Run from the repository root with its existing Python/PyTorch environment:

```powershell
python scripts/verify_d1_formal.py
```

Verified on 2026-10-19 with PyTorch 2.8.0 on CPU: all checks passed. The implemented counterexample derivative was `1.090784311`; the maximum multi-event D1 Jacobian discrepancy was `4.66e-10`, and the D0 discrepancy was zero. These are numerical implementation checks, not additional model-performance evidence.

The script uses CPU tensors only, no downloaded model, no corpus, and no evaluation split. It prints a JSON report with source-file hashes and assertions covering:

- exact rational convex combinations, including 2,000 scalar cases;
- adaptive-gate path expansion, coefficient normalization, and D0 sensitivity;
- the actual operator's shared snapshot, zero-gate denominator, padding exclusion, count advancement, and preservation of the input state;
- the actual writer's counterexample, a multi-event D1 Jacobian, and D0's Jacobian;
- the complete-state count derivative, deterministic collision identities and a tight bound;
- exhaustive independent uniform hashes in a small hypothetical model, and the implemented adapter's residual sum.

| Claim | Status |
| --- | --- |
| Common-snapshot affine identity and path expansion | Proved in finite real arithmetic; implementation correspondence checked numerically. |
| Convex-hull invariance | Proved assuming finite targets and gates in `[0,1]`, including state-dependent gates. |
| A uniform bound on all future memory values | Conditional on a uniform target bound; not enforced by the target MLP. |
| D1 contraction from sigmoid-range gates alone | False; an architecture-realizable counterexample is given. |
| D1 contraction under a uniform Jacobian bound | Proved sufficient condition; not certified for learned checkpoints. |
| D0 row-value sensitivity under fixed evidence | Exact contraction factor, conditional on gate mass and row touches. |
| Complete value-and-count state is strictly contractive | False for the stated standard norm/Jacobian formulation. |
| One-step collision bound | Proved under a norm bound and the fixed-feature comparison. |
| Random-hash collision rates for this deterministic hash | Not established; random-hash results are explicitly hypothetical. |
| Useful semantics, generalization, or improved downstream loss | Requires separate experiments; no claim is made here. |
