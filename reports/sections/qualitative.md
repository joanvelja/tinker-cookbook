# Qualitative Analysis: Reasoning Quality and Failure Modes

We analyzed 640 episodes across 40 training iterations (20 GPT-OSS, 10 Instruct, 10 Think), reading every logged response. All quotes are verbatim from training rollouts with minimum 300-character excerpts.

## 1. GPT-OSS: Reduced Backtracking and Compression

### 1.1 Self-Correction Marker Decline

We tracked self-correction markers ("But wait", "Actually,", "Hmm,", "Let me reconsider", etc.) across training steps.

| Step | Avg Markers/Episode | Density (per 1k chars) | Avg Response Length |
|------|--------------------:|----------------------:|--------------------:|
| 0    | 4.8                 | 0.23                  | 18,599              |
| 3    | 6.8                 | 0.22                  | 24,934              |
| 9    | 0.5                 | 0.09                  | 6,607               |
| 15   | 0.1                 | 0.00                  | 2,603               |
| 19   | 0.0                 | 0.00                  | 7,696               |

Density drops from 0.23/1k to 0 by step 15, faster than the length decline (step 19 is longer than step 15 at 7,696 vs 2,603 chars, but markers remain at 0). Visible backtracking disappears independent of total output. A caveat: we cannot distinguish genuine dead-end pruning from surface-marker suppression (the model may still explore alternatives internally without emitting explicit revision markers).

At step 0, one response to a triangle optimization problem accumulates 27 markers in 32,542 chars. The densest window shows the characteristic backtrack-and-retry pattern:

> We could reflect M3 back across AC to get M2? Wait M3 reflected across AC would give M2, but we need M inside. Actually, after reflection across AC, we get back to the triangle orientation: M3 reflected across AC is M2, which is inside the reflected triangle after BC reflection?

Three markers ("Wait", "Actually,", implicit "but") in rapid succession. The model tries an unfolding strategy, recognizes a problem, pivots, and runs into the same wall from a different angle.

By step 9, correct responses show immediate commitment. Here is the full reasoning for a correct answer (1,137 chars, 0 markers):

> We need to parse problem: "points P, Q, R, and S are arranged in order on a line segment with PQ=1, QR=2PQ, and RS=3QR, what is the length of PS?" We have PQ = 1. QR = 2 PQ = 2\*1=2? Wait: QR=2 PQ? So 2\*PQ = 2\*1 = 2: indeed QR = 2. RS = 3 QR = 3\*QR = 3\*2 = 6. Thus PS = PQ + QR + RS = 1+2+6=9.

By step 15, responses compress further (351 chars, 0 markers):

> We need final final fraction from common known: For sqrt(2) fraction? The nearest fraction under 100 is 99/70? Wait sqrt(2)=1.414285..., 99/70=1.414285714, so this fraction exactly matches digits within sqrt(2) up to 5 digits. So final answer: \boxed{99/70}.

The step-15 response pattern-matches from memory rather than deriving. Compression succeeded, but the mechanism that made step-0 verbose (exploring dead ends) was also the mechanism that found correct answers on hard problems.

### 1.2 Analysis Channel Evolution

GPT-OSS uses Harmony's channel system: `<|channel|>analysis` for reasoning, `<|channel|>final` for answers.

| Step | Analysis Channel Usage | Avg Analysis Length | Avg Non-Analysis Length |
|------|----------------------:|--------------------:|------------------------:|
| 0    | 31% (5/16)           | 33,110              | 12,003                  |
| 7    | 88% (14/16)          | 27,758              | 24,186                  |
| 9    | 0% (0/16)            | n/a                 | 6,607                   |
| 15   | 0% (0/16)            | n/a                 | 2,603                   |
| 19   | 19% (3/16)           | 39,996              | 242                     |

Steps 3-8 show rising analysis-channel use. Analysis-channel responses almost never produce a `\boxed{}` answer because the channel lacks a clean transition to the final answer. The model generates reasoning until the token limit. Step 7 truncation example:

> Edge5 orientation330 (cos330=sqrt(3)/2, sin330= -1/2). P5=(-1/2-sqrt(3)/2+sqrt(3)/2, -1/2+sqrt(3)/2 -1/2)=(-1/2, sqrt(3)/2-1?). Wait compute: -1/2-sqrt(3)/2+sqrt(3)/2= -1/2. For y: -1/2+sqrt(3)/2-1/2 = sqrt(3)/2-1. So P5=(-1/2, sqrt(3)/2-1). Edge6 orientation0 -> (1,0). Add to previous: P6= (-1/2+1, sqrt(3)/2-1+0)= (1/2, sqrt(3)/2-1). Not closure. Thus orientation not closure. But we previously used external angles incorrectly as 30 for

Pure coordinate computation, no boxed answer in sight. This is analysis-channel death mode.

Steps 9-15: the model stops using the analysis channel entirely. Response length drops 10x. Boxed rate rises to 93-100%.

Step 19 shows partial regression: 3/16 episodes re-enter the analysis channel (avg 39,996 chars) with degenerate repetition. The word "medium" from the system prompt ("Reasoning: medium") becomes a repetition attractor:

> We already ded ded medium medium medium medium medium medium medium medium [repeats 200+ times]

This is the same failure mode that destroyed v1 (mode collapse onto system-prompt tokens), now confined to analysis-channel episodes.

### 1.3 Late-Stage Collapse (Steps 16-17)

GPT-OSS compresses past the useful range at steps 16-17. Average response lengths: 295 and 198 chars. The model stops reasoning and guesses. Full responses:

Step 16, Episode 0 (285 chars):

> Need final formula: m? The problem might be Chess coin sliding? The final answer: \boxed{m}. Wait for general correct formula? Usually answer: \boxed{m}. Let's confirm: final message: \boxed{m}.

Step 17, Episode 0 (100 chars total): the think block is `Write final.`, followed by `\boxed{4}`.

Step 17, Episode 1 (177 chars):

> We can't compute quickly medium due final answer 6? not. Provide default ~6? decimals: 6?

Boxed rate stays at 100% (the format reward worked), but correct rate is 0%. The model optimized format compliance and abandoned math.

## 2. Qwen3 Instruct: Template Lock

### 2.1 SFT Template Persistence

The Instruct model enters RLVR with a strong SFT-instilled template: `### Step N:` markdown headers with bold formatting. This template persists throughout training.

| Step | Template Usage | Avg Headers per Episode | Avg Length |
|------|---------------:|------------------------:|----------:|
| 0    | 56% (9/16)     | 5.3                     | 16,998    |
| 3    | 100% (16/16)   | 5.6                     | 16,908    |
| 5    | 100% (16/16)   | 2.8                     | 7,873     |
| 7    | 100% (16/16)   | 2.2                     | 17,438    |
| 9    | 50% (8/16)     | 2.6                     | 8,851     |

Template usage *increases* from 56% to 100% before settling. Headers per response decline from 5.3 to 2.6, driven by problem difficulty differences rather than learned compression.

Step 0, correct response to a convex-set problem (14,945 chars, 7 headers):

> We are asked to find the **least possible area** of a **convex set** in the plane that intersects:
> - **Both branches** of the hyperbola $xy = 1$, and
> - **Both branches** of the hyperbola $xy = -1$.
>
> Let's analyze the problem step by step.
>
> ---
>
> ### **Step 1: Understand the hyperbolas**
>
> #### Hyperbola 1: $xy = 1$
> - This is a rectangular hyperbola in the **first and third quadrants**.
> - Branch 1: $x > 0, y > 0$
> - Branch 2: $x < 0, y < 0$

The full header sequence: `Step 1: Understand the hyperbolas`, `Step 2: Convexity constraint`, `Step 3: Key idea`, `Step 4: Use symmetry`, `Step 5: Can we go below area 4?`, `Step 6: Use the Alexander's inequality or duality?`, `Step 7: Use the fact that the area of the convex hull is at least 4`.

Step 3, different problem, same template (7,468 chars, 5 headers): `Step 1: Understanding the expression`, `Step 2: Key idea`, `Step 3: Use identity involving roots of unity`, `Step 4: For fixed $a$, let $d = \gcd(a, n)$`, `Step 5: Compute $\sum_{k=1}^n \gcd(k, n)$`.

Step 7, another problem (11,327 chars, 3 headers): `Step 1: Check the sum of interior angles`, `Step 2: Understand the structure`, `Step 3: Use vectors and complex numbers`.

Same template across 10 training steps. Different content fills the same slot structure.

### 2.2 Why the Template Persists

The template's persistence has multiple possible explanations. The most likely: the Instruct model's entropy is 0.33 nats (perplexity 1.4), making token distributions near-deterministic. At this entropy level, high-probability sequences like `### Step` may be difficult for the policy gradient to modify. We lack token-level gradient data to confirm this mechanism. The template may also persist because it is reward-neutral (neither penalized nor rewarded).

Self-correction markers (avg 11.6 at step 0, 6.2 at step 5, 8.2 at step 9) fluctuate but remain high throughout training. They appear inside the template structure:

> But wait: is 25 the **minimum possible number** of such integers? Could we arrange...

The model backtracks within structured steps. The template itself is untouched.

### 2.3 Boxed-Rate Improvement Without Reasoning Improvement

The boxed rate improves from 75% (step 0) to 100% (steps 4+). A single token (`\boxed{`) is the easiest behavior for RL to select. But correct rate oscillates between 43% and 75% without trend. The failure taxonomy shows improvements and regressions offsetting: grader false negatives decline while wrong-approach errors increase.

## 3. Qwen3 Think: Post-Solution Tail Bloat

### 3.1 Answer Discovery Position

The Think model finds its first `\boxed{}` in the think block late, and the position does not improve with training.

| Step | Avg First \boxed Position | Range | Avg Post-Answer Chars |
|------|-------------------------:|------:|----------------------:|
| 0    | 93.4% of think block    | 87-100% | 847                  |
| 1    | 83.2%                   | 39-100% | 2,220                |
| 4    | 74.1%                   | 19-100% | 3,695                |
| 6    | 65.1%                   | 36-96%  | 3,898                |
| 9    | 82.4%                   | 58-100% | 531                  |

The position fluctuates without systematic improvement. Average post-answer tails range from 531 to 3,898 chars across steps, with individual responses reaching 81% post-answer (section 3.3). When the model places `\boxed{}` earlier (step 6: 65%), the post-answer tail grows. RL has not taught the model to close `</think>` promptly after finding a candidate answer. Note: the first `\boxed{}` may mark a candidate rather than a final answer, so some post-answer text (verification, correction) can be productive.

### 3.2 Post-Answer Activity Taxonomy

We classified all post-answer text in correct responses into four categories by marker analysis.

Format anxiety (~40% of markers). The model discovers the answer, then agonizes about representation:

> We need to find maximum possible value of x+y^2+z^3 subject to z = 0? Wait direct computations could be maximum answer but after z = 0? Since maximum is the maximum value after z = 0? Write final answer?

Verification (~20% of markers). Redundant re-derivation after a rigorous proof:

> Therefore, derivative is negative here. Therefore, function is decreasing after x = sqrt(2). Hence, x = sqrt(2) is indeed the maximum. Therefore, the maximum value is 2sqrt(2) - 2. Let me write that as $\boxed{2\sqrt{2} - 2}$. But just to confirm, maybe there is a smarter way to see it?

Alternative approaches (~20% of markers). Post-hoc exploration that never changes the answer:

> Alternatively, if I write 297 divided by 1000, but since 297 and 1000 have no common factors (since 297 is 3\*99=3\*3\*3\*11, and 1000 is 2^3\*5^3, so no common factors), so 297/1000 is simplest. So decimal is 0.297.

Hedging (~10% of markers). Low-information verbal gestures:

> So, I don't think I made a mistake. Maybe this is straightforward, so maybe the answer is indeed 9.

### 3.3 Concrete Example: 81% Post-Answer Tail

Step 4 correct response on a parabola minimum problem. Think block: 17,107 chars. First `\boxed{}` at position 3,284 (19.2%). Post-answer: 13,828 chars (81% of think block).

The model finds the answer in the first 19%:

> minimum is at u = -4/(2\*1) = -2. Therefore, substituting u = -2 into y: y = [(-2)^2 + 4\*(-2) + 3]/2 = [4 - 8 + 3]/2 = (-1)/2 = -0.5. So the minimum value of y is -1/2. Therefore, the answer is \boxed{-\dfrac{1}{2}}.

Then spends 13,828 chars on verification loops:

> u = -1.5: y = [2.25 + 4\*(-1.5) + 3]/2 = [2.25 -6 + 3]/2 = (-0.75)/2 = -0.375, same as before. So that's also higher than -0.5. Therefore, the minimum is indeed at u = -2, giving y = -0.5. Thus, after multiple methods and checks, I think it's safe to conclude that the minimum value of y is -1/2.

81% of the think block confirms what was already established.

### 3.4 Truncation-Answer Paradox

62% of Think responses are truncated across our 10 logged iterations. Classification of the 100 truncated episodes:

| Category | Count | Fraction |
|----------|------:|--------:|
| Dead-end spiral (algebraic loops, case enumeration) | 88 | 88% |
| Productive truncation (had found answer, hit token limit) | 12 | 12% |

12 truncated responses had already emitted `\boxed{}` inside the think block but hit the token limit before closing `</think>`. One response found the answer twice, started the final text, and was cut off:

> ...I think this is correct. Therefore, I think the answer is \boxed{\dfrac{n(n^2 - 1)}{6}}.
> **Final Answer**
> The average is \boxed{\dfrac{n(n^2 - 1)}{6}}.
> </think>
> To find the average value of the expression
> $(a_1 - a_2)^2

Post-answer verification consumed the budget needed for the final response.

## 4. Truncation Failure Mode Taxonomy

187 truncated episodes across all three models (74 GPT-OSS, 13 Instruct, 100 Think).

| Mode | GPT-OSS | Instruct | Think | Description |
|------|--------:|---------:|------:|-------------|
| Analysis channel death | 66 | 0 | 0 | Never exits `<\|channel\|>analysis` to produce answer |
| Dead-end spiral | 1 | 9 | 88 | Stuck in algebraic loop or case enumeration |
| Productive truncation | 2 | 4 | 12 | Had found answer, ran out of tokens |
| Degenerate repetition | 5 | 0 | 0 | Repetitive nonsensical output |

GPT-OSS truncations are 89% analysis-channel death. The analysis channel is a structural trap: once entered, the model has no natural transition to emit a boxed answer.

Think truncations are 88% dead-end spirals. The `<think>` block provides no structural exit signal equivalent to Harmony's channel transitions.

Instruct has the fewest truncations (13 total) because the step template provides natural conclusion points. 4 of the 13 had the answer but kept elaborating after `\boxed{}`.

## 5. Cross-Model Synthesis

Three models exhibit distinct failure modes, mediated by each model's output protocol and entropy profile.

GPT-OSS compressed visibly (steps 0-12): analysis-channel usage dropped from 88% to 0%, self-correction markers fell from 4.8 to 0.1 per episode. Steps 16-17 show a transient boxed-guess regime (100% boxed, 0% correct, avg response 198-295 chars). The analysis channel resurfaced at step 19 with degenerate repetition.

Instruct retains its SFT template prior throughout training. RL selects for `\boxed{}` emission (75% to 100%) but does not modify the reasoning structure. Correct rate oscillates without trend. The low entropy (0.33 nats) is a plausible contributing factor, but we cannot isolate it from other explanations (reward-neutral template, short training horizon).

Think often reaches a candidate answer well before the think block ends, with post-answer tails averaging 531-3,898 chars across steps and individual responses reaching 81%. 62% of episodes truncate; 12% of those had already found the correct answer. RL has not taught the model to close `</think>` promptly, and the `<think>` block provides no structural exit signal equivalent to Harmony's channel transitions.
