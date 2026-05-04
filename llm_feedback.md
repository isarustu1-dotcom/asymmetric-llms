I reviewed the uploaded nine-page PDF. The main issue is not just grammar; the paper currently reads like an internal report. The contribution is plausible, but the writing does not yet create a strong NeurIPS-style argument. It needs a sharper problem framing, more careful claims, clearer definitions, and much better figure/table presentation.

## Overall diagnosis

The paper’s likely core contribution is:

> A simple two-parameter, validation-tuned logit fusion between a strong LLM classifier and a smaller sidekick model improves calibration-versus-compute tradeoffs on MCQA tasks, and can be applied on top of existing calibration-aware methods.

That is a reasonable story. However, the current manuscript obscures it in several ways:

1. **The introduction under-sells and misframes the contribution.** It spends too much space on generic LLM/calibration motivation and too little on the specific compute-calibration gap the method addresses.
2. **The claims are sometimes stronger or vaguer than the evidence supports.** Phrases such as “state-of-the-art,” “firmly on the Pareto frontier,” “orthogonal,” and “composes additively” need tighter qualification.
3. **The paper does not anticipate the obvious reviewer objection:** this looks like two-model post-hoc logit stacking/temperature scaling. The paper must explicitly explain what is new, what is inherited from Asymmetric Duos, and why the LLM/MCQA setting is nontrivial.
4. **The experimental narrative is weak.** The results section mostly says “as shown in Table 1” and “highly competitive.” It should instead explain the tradeoff: large gains over the base model, modest gains over temperature scaling, and lower compute than heavier Bayesian/ensemble baselines.
5. **The presentation has visible LaTeX/PDF issues.** Every citation and cross-reference is boxed in green/red. This is distracting and should be removed with `\hypersetup{hidelinks}` or an equivalent clean hyperref setup.
6. **Several important definitions are missing or delayed.** The paper needs to define MCQA logit extraction, ECE computation, NLL computation, validation split usage, FLOP accounting, and baseline implementation details in the main text, not only in appendices.

------

# Section-by-section critique and fixes

## Title

Current title:

> Pushing the Calibration-Compute Frontier of LLM Classifiers via Asymmetric Duos

### Problems

The title is catchy but slightly overclaims. “Pushing the frontier” suggests a broadly new state-of-the-art frontier, but the experiments are on Qwen2 models, six MCQA datasets, and mainly inference-FLOP tradeoffs. The paper also shows that BLoB often achieves better absolute NLL/ECE, while Duo is cheaper. So the title should emphasize **cost-efficient calibration**, not necessarily “pushing the frontier” in a universal sense.

### Suggested alternatives

Better, more precise options:

> **Asymmetric Duos for Cost-Efficient Calibration of LLM Classifiers**

or

> **Cost-Efficient Calibration of LLM Classifiers via Asymmetric Logit Fusion**

or

> **A Small Sidekick Model Improves Calibration-Compute Tradeoffs for LLM Classifiers**

The second title is probably the safest because it explains the mechanism directly: asymmetric logit fusion.

------

## Abstract

### Main problems

The abstract has the right components, but it is too generic and contains several imprecise claims.

Problematic phrases:

> “The importance of reliable uncertainty estimation in large language models has become increasingly pronounced”

This is vague and sounds like a template sentence.

> “we address this by proposing an efficient method”

Too broad. The method is specifically a two-parameter post-hoc logit fusion between a base LLM and a smaller sidekick model.

> “placing our method firmly on the calibration-compute Pareto frontier”

This is too strong unless shown across all datasets and all relevant compute baselines. The paper mainly shows this clearly for CSQA in Figure 1.

> “acts orthogonal to existing calibration-aware fine-tuning”

Grammatically wrong. Use “is complementary to” or “can be applied on top of.”

> “the benefits of an asymmetric fusion successfully transfer to the language domain”

Too broad. The paper does not study the whole “language domain”; it studies MCQA-style LLM classification.

### Suggested abstract rewrite

Here is a stronger version:

> Reliable confidence estimates are critical for deploying LLM classifiers, but LoRA-fine-tuned LLMs can remain highly overconfident even when their accuracy is strong. We study a simple asymmetric logit-fusion approach for multiple-choice question answering: a Qwen2-7B base model is paired with a Qwen2-1.5B sidekick model, and two nonnegative validation-tuned weights combine their answer-option logits. Across six MCQA benchmarks, the resulting Duo substantially reduces the base model’s NLL and ECE while preserving accuracy and adding only the inference cost of the smaller sidekick. Compared with stronger Bayesian or ensemble-style calibration baselines, Duo is usually not the absolute best calibrator, but it provides a favorable low-compute tradeoff. We further show that the same fusion procedure can be applied on top of Laplace-LoRA, IB-EDL, and BLoB, improving their probabilistic predictions in most settings. A logit-level analysis suggests that the sidekick moderates extreme base-model margins, reducing overconfident predictions. These results identify asymmetric logit fusion as a simple, compute-efficient calibration tool for LLM classifiers in MCQA settings.

This version is more defensible. It avoids overstating generality, names the setting, states the method concretely, and frames the actual value: compute-efficient calibration, not universal dominance.

------

## 1. Introduction

### Main problems

The introduction is the weakest section. It underwhelms the contribution because it starts with generic statements and does not quickly establish the specific research gap.

Current opening:

> “With the recent developments and technological enhancements, Large Language Models (LLMs) have driven remarkable advancements...”

This is not NeurIPS-level writing. It is generic, wordy, and imprecise.

Other weak phrases:

> “predictive powers”
> “being overconfident”
> “these neural networks should also be able to indicate when they can be incorrect”
> “This paper will study...”
> “Our contributions are organized as follows”

These sound informal or student-report-like.

### Bigger narrative problem

The introduction should not merely say:

1. LLMs are useful.
2. LLMs are overconfident.
3. Asymmetric Duos exist in vision.
4. We try them on MCQA.

That is too linear and too weak.

The stronger story should be:

1. **LLM classifiers need calibrated probabilities, not only high accuracy.**
2. **Fine-tuning can worsen overconfidence.**
3. **Cheap methods such as temperature scaling are limited.**
4. **Richer uncertainty methods such as BLoB, Laplace-LoRA, deep ensembles, or Bayesian approximations are more expensive.**
5. **Asymmetric Duos offer a possible middle ground: one strong model plus one much smaller auxiliary model.**
6. **But transferring this idea to LLM MCQA is nontrivial because logits come from prompted answer-token scoring, option-token alignment, and calibration metrics over a finite answer space.**
7. **This paper evaluates whether such asymmetric fusion improves calibration-compute tradeoffs.**

### Suggested introduction structure

Use four paragraphs.

**Paragraph 1: problem.**

> LLM classifiers are increasingly used in settings where predictions must be accompanied by reliable confidence estimates. Accuracy alone is insufficient: a model that assigns near-unit probability to wrong answers can be difficult to monitor, abstain from, or safely deploy. This issue is especially pronounced after downstream adaptation, where fine-tuned models may become accurate but poorly calibrated.

**Paragraph 2: gap.**

> Existing calibration methods occupy different points on the compute-accuracy tradeoff. Temperature scaling is cheap but only rescales the base model’s logits. Deep ensembles and Bayesian LoRA-style methods can improve uncertainty estimates, but they require additional inference cost, posterior sampling, or multiple large models. This raises a practical question: can we improve the calibration of an LLM classifier without paying the cost of another large model?

**Paragraph 3: proposed idea.**

> We revisit Asymmetric Duos, originally studied in computer vision, for LLM-based MCQA. The key idea is to pair a strong base model with a much smaller sidekick model and learn two validation-tuned nonnegative logit scales. Unlike standard ensembling, the sidekick is deliberately much cheaper than the base. Unlike temperature scaling, the fusion can incorporate class-specific evidence from a second model rather than merely rescaling the base logits.

**Paragraph 4: contributions.**

Then list the contributions more concretely.

### Current contribution bullets need revision

Current contribution 1:

> “Cost-efficient calibration: We show that Asymmetric Duos improve LLM calibration on the cost–performance frontier, matching or surpassing state-of-the-art baselines often at a fraction of the inference FLOPs across all benchmarks.”

Problem: this overstates. Table 1 shows BLoB often has better absolute NLL/ECE. Duo’s strength is tradeoff, not absolute dominance.

Better:

> **Cost-efficient calibration tradeoff.** Across six MCQA benchmarks, Duo consistently improves the base model’s NLL and ECE while adding only the sidekick’s inference cost. Relative to high-compute calibration baselines, Duo provides a favorable low-FLOP operating point rather than universally dominating in absolute calibration.

Current contribution 2:

> “Compatibility with calibration-aware methods... compose additively...”

Problem: “compose additively” is not precise. Use “can be applied on top of” or “is complementary to.”

Better:

> **Complementarity with calibration-aware fine-tuning.** We apply the same fusion procedure to Laplace-LoRA, IB-EDL, and BLoB outputs and show that the sidekick fusion further improves NLL/ECE in most method-dataset pairs.

Current contribution 3:

> “Logit-space mechanism... reducing overconfidence by smoothing extreme predictions while preserving decision boundaries...”

Problem: “preserving decision boundaries” is stronger than shown. Accuracy is mostly preserved, but decision-boundary preservation is not directly analyzed.

Better:

> **Logit-level analysis.** We analyze how the learned fusion changes logit margins and predictive confidence, showing that Duo reduces extreme base-model margins and redistributes probability mass away from overconfident predictions.

------

## 2. Related Work and Background

### Main problems

The related work is too much like a general literature survey. It summarizes topics but does not build a positioning argument.

A strong related work section should repeatedly answer:

> What does prior work do, why is it insufficient for this paper’s goal, and how is this paper different?

The current section mostly says:

- calibration is important,
- MCQA is useful,
- LoRA exists,
- ensembles are expensive,
- Asymmetric Duos exist.

That is accurate but not argumentative.

### Section 2.1: Calibration and uncertainty

Problematic wording:

> “With the increasing usage of Large Language Models...”

Use:

> “As LLMs are adapted for classification tasks, calibration becomes important because confidence scores are often used for abstention, reranking, or risk estimation.”

Also, the paper loosely treats NLL and ECE as “uncertainty quantification metrics.” Be more precise: ECE measures calibration error; NLL is a proper scoring rule capturing both calibration and sharpness. Accuracy measures predictive performance.

Suggested addition:

> In this paper, we use ECE as the primary calibration diagnostic and NLL as a proper scoring-rule measure of probabilistic prediction quality.

### Section 2.2: MCQA

Current sentence:

> “Multiple-choice question answering (MCQA) is a prominent use case of LLMs for evaluating uncertainty since, it reduces free-form generation to a finite decision space.”

Problems: comma error, generic phrasing.

Better:

> “MCQA provides a controlled setting for studying LLM calibration because predictions can be reduced to a finite answer-option distribution.”

The section correctly mentions option ordering, selection bias, and tokenization issues, but it does not connect these issues to the method. Add a sentence such as:

> We therefore fix a shared prompt format and answer-option token mapping for both the base and sidekick models, ensuring that their logits are aligned before fusion.

If the method actually does this, state it explicitly. If not, the experimental protocol needs to be clarified.

### Section 2.3: Fine-tuning and calibration-aware training

The section is serviceable but too broad. The useful contrast is:

- LoRA adapts the model.
- Calibration-aware methods try to improve uncertainty.
- Duo is post-hoc and can be applied after adaptation.

Suggested closing sentence:

> Our method is not an alternative fine-tuning objective; it is a post-hoc fusion layer that can be applied after LoRA or after calibration-aware variants of LoRA.

### Section 2.4: Multi-model uncertainty and Asymmetric Duos

This is the most important related work subsection, but it should be sharper.

Add explicit contrasts:

> Unlike deep ensembles, Duo does not require multiple full-size models. Unlike temperature scaling, Duo incorporates an additional class-specific signal from a sidekick model. Unlike Bayesian LoRA methods, Duo does not require posterior approximation or sampling at inference time. Our work tests whether this asymmetric design transfers from vision models to prompted LLM classifiers.

This would make the contribution much clearer.

------

## 3. Asymmetric LLM Duos

This section is relatively stronger, but it still has important writing and presentation problems.

## 3.1 Problem formulation

### Problems

The formulation is too generic. It defines `x`, `y`, `f_base`, and `f_side`, but it does not explain how an LLM produces a `K`-dimensional MCQA logit vector.

For an LLM classifier, the reader needs to know:

- Are logits taken from answer-letter tokens such as A/B/C/D/E?
- Is the prompt fixed?
- Are answer options scored jointly or independently?
- Are multi-token answer options avoided by predicting only option labels?
- How are base and sidekick logits aligned?
- Are logits extracted before or after temperature scaling/baseline calibration?

The paper says later:

> “The target space is defined as tokens corresponding to different pre-determined options like A, B, C, D, and E.”

This belongs earlier, in the method.

### Suggested rewrite

Add something like:

> For each MCQA example, we construct a prompt containing the question and its answer options and score only the canonical option tokens (v_1,\ldots,v_K), corresponding to labels such as A, B, C, D, and E. For model (m \in {\mathrm{base}, \mathrm{side}}), (z^m_k(x)) denotes the pre-softmax logit assigned to option token (v_k). Both models use the same prompt template and option-token mapping, yielding aligned logits (z^m(x) \in \mathbb{R}^K).

That one paragraph would prevent many reviewer questions.

## 3.2 Asymmetric logit fusion

The formulation is fine, but the paper should explain why logit fusion is meaningful. A useful interpretation:

> The fused distribution satisfies (p_{\mathrm{duo}}(y|x) \propto \exp(\alpha z_{\mathrm{base},y}(x) + \beta z_{\mathrm{side},y}(x))), so the learned scales act both as inverse temperatures and as relative model weights.

This connects Section 3.2 to the later “implicit temperature scaling” discussion.

## 3.3 Learning the fusion weights

### Problems

The NLL objective is mostly clear, but the notation is somewhat sloppy. Equation (2) should make clear that the log-softmax is indexed by the gold label.

Current equation is readable but visually dense.

Suggested notation:

[
\mathcal{L}(\alpha,\beta)
= -\frac{1}{N}\sum_{i=1}^N
\log
\left[
\operatorname{softmax}
\left(
\alpha z_{\mathrm{base}}^{(i)}
+
\beta z_{\mathrm{side}}^{(i)}
\right)
\right]_{y_i}.
]

Also, state explicitly that (D_{\mathrm{val}}) is disjoint from the test set and is used only for calibration-weight fitting.

## Algorithm 1

### Presentation and correctness issues

The algorithm appears before some notation is fully introduced. It would be better after Sections 3.1-3.3.

More importantly, the inference part is incomplete:

Current:

> (z_{\mathrm{duo}}(x) \leftarrow \alpha^* z_{\mathrm{base}}(x) + \beta^* z_{\mathrm{side}}(x))

But the algorithm does not compute (z_{\mathrm{base}}(x)) and (z_{\mathrm{side}}(x)) inside the test loop.

Add:

> (z_{\mathrm{base}} \leftarrow f_{\mathrm{base}}(x))
> (z_{\mathrm{side}} \leftarrow f_{\mathrm{side}}(x))

Also, line 9:

> (\hat{y} \leftarrow \arg\max_{y \in Y} \operatorname{softmax}(z_{\mathrm{duo}}(x)))

Better:

> (\hat{y} \leftarrow \arg\max_{k \in {1,\ldots,K}} z_{\mathrm{duo},k}(x))

Softmax is unnecessary for the argmax. If probabilities are needed for NLL/ECE, state separately:

> (p_{\mathrm{duo}}(\cdot|x) = \operatorname{softmax}(z_{\mathrm{duo}}(x))).

## 3.4 Interpretation

### Problems

This section contains a useful idea, but the claims need to be more careful.

Current:

> “If the sidekick model provides little useful information, the optimization assigns it a small weight, effectively recovering the base model.”

This is intuitive, but not guaranteed in finite validation data. Better:

> “When the sidekick provides little validation-set benefit, the nonnegativity-constrained objective can reduce its influence by assigning a small (\beta), making the predictor close to a rescaled base model.”

Current:

> “We empirically validate this behavior in Table 1...”

Table 1 shows performance improvements, but it does not directly validate this behavior. To validate this behavior, the paper should report learned (\alpha,\beta) values, perhaps in a small table or appendix. Otherwise, rephrase:

> “The performance gains in Table 1 are consistent with this interpretation.”

## 3.5 Measuring computational cost

### Problems

This section is too vague for a paper whose title centers compute.

Current:

> “All FLOP estimates are obtained from standard model-level inference complexity calculations based on the model architecture and input sequence length.”

This is insufficient. A reviewer will ask:

- What exact formula is used?
- Is FLOP cost per prompt, per answer option, or per generated token?
- Does it include only forward inference?
- Does it include sidekick inference?
- Does it include calibration-weight optimization?
- Is BLoB using multiple samples? If so, how many?
- Are sequence lengths fixed or dataset-specific?
- Is the 1.22× ratio from 7B + 1.5B relative to 7B?

Suggested addition:

> We report inference FLOPs per example and do not include training or validation-time calibration-weight fitting. For Duo, the inference cost is (C_{\mathrm{base}} + C_{\mathrm{side}}). With Qwen2-7B as base and Qwen2-1.5B as sidekick under the same input length, this corresponds to approximately (1.22\times) the base-model inference cost. For sampling-based or ensemble baselines, we multiply the single-forward cost by the number of required forward passes.

If exact formulas are in the appendix, summarize the key one in the main text.

------

## 4. Experiments

## 4.1 Experimental setup

The section title should be singular:

> **Experimental setup**

### Problems

The writing is informal and underspecified.

Current:

> “After this section, these two models will be referred to as...”

Better:

> “We refer to Qwen2-7B as the base model (B) and Qwen2-1.5B as the sidekick model (S).”

Current:

> “These datasets are chosen for the experiments because they are widely used in other similar studies...”

Better:

> “We use these datasets because they cover science, commonsense, open-book, and reading-comprehension MCQA, and are standard benchmarks for evaluating LLM classifiers.”

There is also a small punctuation issue:

> ```
> [Li et al., 2025] .
> ```

Remove the extra space before the period.

### Missing details

The main text should include at least brief details on:

- train/validation/test split construction,
- prompt template,
- answer-token scoring procedure,
- LoRA rank and basic training setup,
- number of ECE bins,
- whether ECE is top-label ECE,
- whether NLL is computed on answer-option probabilities,
- number of seeds,
- whether all baselines use the same base model and splits.

It is acceptable to put full hyperparameters in the appendix, but the main text needs enough information for a reviewer to understand what was evaluated.

## 4.2 Cost-efficient uncertainty quantification

This subsection is currently much too weak.

Current:

> “As seen in Table 1, the Duo method is highly competitive in all metrics compared to the baselines.”

This is generic. It does not say what the reader should learn.

Current:

> “For the metric NLL, it is either the best-performing or the second-best-performing one across all datasets. This also holds for ECE.”

This is not the most compelling way to present the result. “Best or second-best” sounds like leaderboard reporting. The real contribution is the compute-calibration tradeoff.

### Better narrative

Use a three-comparison structure:

1. **Against the base model:** Duo substantially improves NLL and ECE.
2. **Against temperature scaling:** Duo usually improves calibration, but the gains are smaller; this is the key cheap-baseline comparison.
3. **Against BLoB / Laplace-LoRA / ensembles:** Duo is usually cheaper and competitive, though not always the best in absolute calibration.

Suggested rewrite:

> Table 1 shows that Duo consistently improves the uncalibrated base model while preserving accuracy. Relative to (B), Duo reduces NLL on every dataset and reduces ECE on every dataset; from the reported means, the average reductions are roughly 62% for NLL and 66% for ECE. Accuracy changes are small, with the largest difference below half a percentage point. Compared with temperature scaling, Duo usually provides lower NLL/ECE, although the margin is modest on some datasets. Compared with BLoB, Duo is often worse in absolute NLL/ECE but uses far less inference compute. Thus, the main result is not that Duo universally dominates all baselines, but that it provides a strong low-compute calibration point.

This is more honest and stronger.

### Figure 1 problems

Figure 1 is useful, but the presentation is weak.

Issues:

- It appears before the experiments are introduced, interrupting the introduction.
- It only shows CSQA, but the title/narrative imply a general frontier.
- The x-axis tick labels are awkward and visually crowded.
- The “better” arrow is informal.
- The legend is large relative to the plot.
- It is not obvious which methods are Pareto-optimal.
- The caption is too minimal.

Suggested fix:

Move Figure 1 into the results section, or explicitly call it a teaser figure in the introduction. Improve the caption:

> **Figure 1:** Inference FLOPs versus NLL/ECE on CSQA. Duo adds only the Qwen2-1.5B sidekick cost to the Qwen2-7B base model and improves over the base and temperature scaling, while remaining much cheaper than high-compute Bayesian/ensemble-style baselines.

Also consider using normalized FLOPs relative to the base model: (1.0\times), (1.22\times), (2.0\times), etc. That would be clearer than raw FLOPs.

### Table 1 problems

Table 1 is too dense. It is readable but not elegant.

Issues:

- Too many numbers are packed into one table.
- Bold and underline are visually subtle.
- The caption contains an argumentative claim: “The Duo method has either the best or second-best...”
- It does not separate the main story: base vs TS vs Duo vs high-compute baselines.
- It does not include compute, even though compute is central.

Suggested fix:

Use a smaller main table with:

- Base,
- Temperature Scaling,
- BLoB,
- Duo,
- FLOPs multiplier,
- average NLL/ECE/Acc across datasets,

and move the full per-dataset table to the appendix. Or keep the table but add a summary row or separate compute column.

## 4.3 Compatibility with calibration-aware methods

### Problems

The phrase “composes additively” is imprecise. The section should say “is complementary to” or “can be applied on top of.”

Current:

> “Bars are positive on nearly every cell, indicating that the gains from the duo procedure are largely orthogonal...”

This is too strong. Positive bars show empirical improvement in most cells, not true orthogonality. “Orthogonal” has a stronger methodological meaning.

Better:

> “The positive reductions in most cells suggest that the sidekick fusion provides improvements that are complementary to these calibration-aware training methods.”

Also, the section should define the relative reduction formula:

[
100 \times \frac{M(BX)-M(\mathrm{Duo},X)}{M(BX)}
]

for lower-is-better metrics (M).

### Figure 2 problems

- “B X” is ambiguous.
- The y-axis says “reduction over B X,” but this notation is not intuitive.
- Some tiny labels overlap bars.
- There is a negative cell, but the text does not discuss it.
- No error bars are shown.
- The caption does not state that higher bars are better because these are reductions in lower-is-better metrics.

Suggested caption:

> **Figure 2:** Relative NLL and ECE reductions from applying Duo on top of calibration-aware base methods. For each method family (X), we compare (\mathrm{Duo},X) against the corresponding base predictor (B X). Positive values indicate improved NLL/ECE after sidekick fusion.

## Remark on Duo TS

This remark is technically useful but interrupts the results flow. It probably belongs in the method section or appendix.

Also, phrase it more simply:

> Temperature scaling followed by Duo fusion is equivalent to plain Duo with reparameterized weights because the learned nonnegative scales can absorb the individual temperatures. We therefore do not report Duo TS separately.

## 4.4 Ablation: trivial Duos

### Major technical-writing problem

Current:

> “The two trivial settings produce identical accuracy because softmax is invariant under uniform logit scaling...”

This is incorrect. Softmax probabilities are **not** invariant under uniform scaling. The **argmax** is invariant under positive scaling.

Correct version:

> The two fixed-weight settings produce identical predictions, and therefore identical accuracy, because (z_{\mathrm{base}}+z_{\mathrm{side}}) is a positive scalar multiple of (0.5(z_{\mathrm{base}}+z_{\mathrm{side}})). Their probabilities differ, however, because softmax is not invariant to logit scaling; this explains why their NLL and ECE differ.

Also, “trivial Duos” sounds dismissive. Use:

> **Ablation: fixed-weight fusion**

or

> **Ablation: learned versus fixed fusion weights**

Suggested rewrite:

> Table 2 compares the learned Duo weights with two fixed-weight alternatives: logit averaging ((\alpha=\beta=0.5)) and logit summation ((\alpha=\beta=1)). Both fixed variants use the same decision rule up to a positive scalar multiple and therefore yield identical accuracy. However, their predicted probabilities differ, and both have substantially worse NLL/ECE than the validation-tuned Duo. This shows that learning the global logit scales is essential for calibration.

------

## 5. How does Duo improve calibration?

This section has good intent but weak execution.

### Main problems

1. **The section title is informal.**
    Better:

    > **Logit-Level Analysis**

    or

    > **Mechanistic Analysis of Duo Calibration**

2. **Definitions are repeated.**
    Lines 233-238 define the quantities in prose, then lines 239-248 define them again as a list. Delete one version.

3. **The “For clarity...” sentence is repeated.**
    It appears before and after the definitions.

4. **Figure references are confusing.**
    Figure 3 shows correct-class logits and logit spread, not logit margins. The text says both models output “large logit margins” while discussing Figure 3. That should be corrected.

5. **The causal interpretation is too strong.**
    The paper says the sidekick “acts as an implicit regularizer.” That may be plausible, but the analysis only shows distributional changes. Use softer language unless supported by an ablation or learned-weight analysis.

6. **Table 4 is referenced but not present in the uploaded main paper.**
    If Table 4 is in the appendix, the reference is fine, but the main text should not rely on a missing table for a core claim. If the appendix is not included, this is a serious issue.

7. **Single-seed analysis is weak.**
    The paper says the analysis focuses on CSQA using a single seed. That is acceptable as an illustrative diagnostic, but then the claims should be cautious.

### Suggested restructuring

Use this structure:

1. One paragraph explaining the question.
2. One short paragraph defining margin, confidence, and correct-class probability.
3. Figure 3: base vs sidekick logit scale.
4. Figure 4: base vs Duo margin/confidence shift.
5. Cautious interpretation.

### Suggested rewrite

> We next examine how Duo changes the base model’s probabilistic predictions. For each test example, we analyze the correct-class logit (z_{i,y_i}), the logit spread (\max_k z_{i,k}-\min_k z_{i,k}), the correct-class margin (z_{i,y_i}-\max_{k\neq y_i}z_{i,k}), the predictive confidence (\max_k p_{i,k}), and the probability assigned to the correct class (p_{i,y_i}). We report CSQA results for one representative seed; additional datasets are provided in Appendix A.2.

> Figure 3 shows that the base model produces substantially larger correct-class logits and wider logit spreads than the sidekick, indicating more extreme post-fine-tuning scores. Figure 4 compares the base model with Duo. Duo shifts the margin distribution toward smaller positive values and reduces the mass of predictions with near-unit confidence. This suggests that the sidekick fusion moderates extreme base-model logits rather than changing predictions wholesale. Consistent with this interpretation, Table 1 shows that Duo improves NLL and ECE while leaving accuracy nearly unchanged.

This is clearer and avoids overclaiming.

### Figure 3 and Figure 4 presentation problems

Figure 3:

- Caption says “Logit (left)” but should say “correct-class logit.”
- Caption should state the dataset and seed.
- The sidekick/base color overlap may be hard to read.
- The figure should mention whether logits are before or after learned fusion weights.

Figure 4:

- The legend overlaps the plot.
- The CDF plots are small and hard to interpret.
- The caption should say what the CDFs demonstrate.
- The learned (\alpha,\beta) values would help interpret the margin shift.

Suggested Figure 4 caption:

> **Figure 4:** Effect of Duo on CSQA logit margins and predictive probabilities. Duo reduces extreme positive margins and shifts confidence away from near-one values, consistent with lower overconfidence while maintaining similar accuracy.

------

## 6. Discussion and limitations

### Main problems

The discussion mostly repeats the results instead of interpreting them. It says again that Duo is best or second-best, that it is cost-efficient, and that it composes with other methods. A discussion section should answer:

- What is the main takeaway?
- When should someone use this method?
- When might it fail?
- What does this imply about calibration in LLM classifiers?
- What are the limitations of the evidence?

### Current problematic claim

> “BLoB, which often achieves better results but requires substantially higher compute.”

This is important and should appear earlier in the results framing. The paper should be honest that Duo is not always the best absolute calibrator.

Better:

> BLoB often achieves the strongest absolute NLL/ECE, but Duo offers a substantially cheaper operating point. The practical value of Duo is therefore its calibration-compute tradeoff, not universal dominance.

### Limitations are too generic

Current limitations:

- one model family,
- MCQA only,
- not current state-of-the-art large-scale models.

These are valid, but incomplete. Add calibration-specific and compute-specific limitations:

- The method requires a separate validation set for fitting (\alpha,\beta).
- Learned global weights may overfit small validation sets.
- ECE depends on binning choices.
- The method adds latency and memory because two models must run at inference time.
- FLOPs do not fully capture wall-clock latency, batching, memory bandwidth, or deployment constraints.
- The experiments are limited to answer-letter MCQA, not open-ended generation.
- The method is not evaluated under distribution shift, abstention, selective prediction, or OOD detection.
- The logit-level analysis is mostly illustrative and uses a single seed/dataset in the main text.

### Suggested discussion rewrite

> The main takeaway is that asymmetric sidekick fusion offers a useful middle ground between temperature scaling and heavier uncertainty-estimation methods. It is more expressive than temperature scaling because it incorporates class-specific information from a second model, but substantially cheaper than methods requiring multiple large models or repeated posterior sampling. This makes Duo attractive when a small fine-tuned sidekick is available and inference-time calibration matters.

> At the same time, Duo should not be presented as a universal replacement for stronger Bayesian or ensemble methods. BLoB often achieves better absolute NLL/ECE, and Duo adds nonzero latency and memory overhead relative to a single calibrated base model. Its effectiveness also depends on a representative validation split for fitting the fusion weights. Future work should evaluate whether the same tradeoff holds for other model families, larger models, open-ended generation, and distribution-shift or abstention settings.

That is more mature and reviewer-friendly.

------

# Global presentation problems

## 1. Visible citation and reference boxes

The rendered PDF shows green boxes around citations and red boxes around references. This is very distracting and looks unpolished.

Fix the LaTeX hyperref settings. For example:

```latex
\hypersetup{hidelinks}
```

or configure clean color links without boxes. The submitted PDF should not show rectangular link borders around every citation.

## 2. Inconsistent terminology

The paper alternates among:

- “Asymmetric Duos”
- “Asymmetric Duo”
- “Duo”
- “duo method”
- “duo framework”
- “duo procedure”
- “asymmetric fusion”

Choose a convention.

Suggested:

- Use **Asymmetric Duo** for the general method.
- Use **Duo** as the method label in tables.
- Use **duo fusion** for the operation.
- Avoid repeatedly saying “the Duo method.”

Also be consistent with:

- “base” vs “Base”
- “sidekick” vs “Sidekick”
- “Large Language Models” vs “LLMs”

## 3. Overuse of vague intensifiers

Avoid phrases like:

- “remarkable advancements”
- “increasingly pronounced”
- “highly competitive”
- “substantially higher”
- “state-of-the-art baselines”
- “firmly on the frontier”
- “broadly applicable”

Use numbers or constrained claims instead.

Example:

Instead of:

> Duo is highly competitive.

Write:

> Duo reduces the base model’s NLL and ECE on all six datasets while changing accuracy by less than 0.5 percentage points in the reported means.

## 4. Weak table/figure captions

Many captions are descriptive but not informative enough. A good caption should tell the reader what to conclude.

Weak:

> Figure 1: Pareto frontier of FLOPs vs. NLL and ECE for dataset CSQA.

Better:

> Figure 1: On CSQA, Duo improves over the base model and temperature scaling while adding only the sidekick’s inference cost. BLoB achieves lower absolute NLL/ECE but requires substantially more FLOPs, illustrating Duo’s low-compute tradeoff.

## 5. Missing or broken appendix/table references

The paper references:

- Appendix C,
- Appendix A.1,
- Appendix Table 3,
- Appendix A.2,
- Table 4.

These are not present in the uploaded nine pages. If the upload is only the main paper excerpt, that is fine. If this is the intended submission PDF, it is incomplete. At minimum, all references should compile and all cited appendix materials should exist.

## 6. The paper needs a stronger “reviewer objection” paragraph

A reviewer may ask:

> Is this just temperature scaling or logit stacking with a second model?

The paper should answer explicitly:

> Duo reduces to temperature scaling when the sidekick weight is zero, but differs from temperature scaling because (\beta z_{\mathrm{side}}) introduces class-dependent evidence from an independently trained smaller model. It differs from standard ensembles because the auxiliary model is deliberately much smaller, making the compute overhead close to (1.22\times) rather than (2\times) or higher.

This kind of paragraph would substantially improve the perceived contribution.

------

# Line-level style fixes

Representative fixes:

| Current wording                                              | Problem              | Suggested fix                                          |
| ------------------------------------------------------------ | -------------------- | ------------------------------------------------------ |
| “With the recent developments and technological enhancements...” | Generic and wordy    | “Recent advances in LLMs have...”                      |
| “predictive powers”                                          | Informal             | “predictive performance”                               |
| “being overconfident”                                        | Informal             | “overconfidence”                                       |
| “these neural networks”                                      | Repetitive/imprecise | “LLMs” or “fine-tuned classifiers”                     |
| “This paper will study...”                                   | Future tense, weak   | “We study...”                                          |
| “Our contributions are organized as follows”                 | Awkward              | “We make three contributions.”                         |
| “Experimental setups”                                        | Usually singular     | “Experimental setup”                                   |
| “as it can be seen in Table 4”                               | Ungrammatical        | “as shown in Table 4”                                  |
| “acts orthogonal”                                            | Incorrect            | “is complementary”                                     |
| “softmax is invariant under uniform logit scaling”           | Technically false    | “the argmax is invariant under positive logit scaling” |
| “Except the dataset RACE”                                    | Awkward              | “Except on RACE”                                       |
| “like A, B, C, D, and E”                                     | Informal             | “such as A, B, C, D, and E”                            |

------

# Highest-priority revision plan

Before submission, I would ask the student to do these in order:

1. **Rewrite the introduction completely** around the compute-calibration gap: temperature scaling is cheap but limited; Bayesian/ensemble methods are stronger but expensive; Duo is a middle ground.
2. **Rewrite the abstract** to state the method concretely and avoid broad “language domain” claims.
3. **Add a precise MCQA scoring definition** in the method section: prompt, option tokens, logit extraction, and logit alignment.
4. **Clarify FLOP accounting** with exact assumptions and normalized compute ratios.
5. **Rewrite the results narrative** around three comparisons: base, temperature scaling, and high-compute baselines.
6. **Fix the ablation explanation** because the current statement about softmax invariance is technically wrong.
7. **Clean all figures and tables**, especially Figure 1, Figure 2, and Table 1.
8. **Remove visible hyperlink boxes** from citations and cross-references.
9. **Delete duplicated definitions in Section 5** and make the mechanism analysis more concise and cautious.
10. **Make the discussion less repetitive** and add serious limitations about validation-set dependence, ECE binning, latency/memory, MCQA-only evaluation, and distribution shift.

The paper does not only need copyediting. It needs a narrative revision so that the reader understands the actual contribution: a very simple asymmetric logit-fusion calibrator that is not necessarily the strongest absolute calibration method, but may be a strong low-compute point for LLM MCQA calibration.