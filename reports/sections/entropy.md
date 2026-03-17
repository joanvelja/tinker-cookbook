# Entropy Bottleneck: Why GPT-OSS Learns and Qwen Doesn't

## The Gap

All three v3b models see the same problems in the same order (seed=42, identical dataset split and batch ordering), train with the same PPO config (LR=1e-5, clip 0.2, KL=0, grad_clip=0.3), and use the same reward function. The outcomes diverge sharply.

| Model | Entropy (nats) | Perplexity | Eval correct (step 10) | Eval correct (latest) |
|-------|---------------|------------|----------------------|---------------------|
| GPT-OSS-20B | 0.84 | 2.32 | 54.8% | 53.2% (step 70, peaked 57.2%) |
| Qwen3 Instruct | 0.33 | 1.38 | 53.2% | 61.6% (step 50, still rising) |
| Qwen3 Think | 0.27 | 1.31 | 25.3% | 26.7% (step 20) |

GPT-OSS has 2.6x more entropy than Instruct and 3.1x more than Think at step 0.

## What Entropy Measures

Policy entropy is the average negative log-probability of the next token, averaged across all positions in the response. A perplexity of 2.32 means GPT-OSS is choosing between roughly 2.3 equiprobable tokens at each position. Qwen Instruct at perplexity 1.38 is nearly deterministic: the most probable token exceeds 70% probability at most positions.

The gap is architectural. GPT-OSS (Harmony architecture) maintains higher entropy because its analysis-channel/final-channel structure distributes probability mass across more tokens. Qwen's instruction-tuned models have been SFT'd to produce specific response templates (`### Step N:` headers, deterministic formatting), concentrating probability mass on the trained templates.

## Why Low Entropy Kills RL Learning

Policy gradient updates are proportional to the advantage times the log-probability gradient. When the policy is near-deterministic (entropy << 1 nat), the log-probability gradient of the chosen action approaches zero. The model is already assigning near-certain probability to its current behavior, so there is almost no gradient to push on.

At probability 0.95 per token (Qwen regime), the gradient from reinforcing an alternative token is small. At probability 0.50 (GPT-OSS regime), the gradient is substantially larger. The numbers confirm this on train metrics:

| Model | Train correct (step 0) | Train correct (latest) | pp/step |
|-------|----------------------|----------------------|---------|
| GPT-OSS | 39.6% | 52.1% (step 74) | +0.17 |
| Instruct | 35.1% | 38.9% (step 50) | +0.08 |
| Think | 36.4% | 40.2% (step 22) | +0.17 |

GPT-OSS and Think have similar per-step learning rates on train correctness (~0.17 pp/step), but Think's train improvement doesn't transfer to eval (25.3% to 26.7%). Instruct learns 2.4x slower on train metrics but transfers better to eval (53.2% to 61.6%).

## The Paradox: Instruct Wins on Eval

Qwen Instruct reaches the highest eval accuracy of all three models at step 50 (61.6% vs GPT-OSS's peak of 57.2% at step 40). The model with the least entropy headroom achieves the best held-out performance.

The pattern is consistent with greater policy plasticity under high entropy. High entropy lets RL modify the policy rapidly, but that same malleability allows it to overfit to training-set problem patterns. GPT-OSS eval peaks at step 40 then declines 4.1pp to step 70, losing 27 problems on the 664-problem eval set. Entropy drops from 0.84 to 0.66 nats over 75 steps (21% reduction), indicating progressive mode collapse.

Qwen Instruct, constrained by low entropy, changes slowly and sidesteps over-optimization. Its eval scores increase monotonically through all 5 checkpoints (53.2% to 61.6%) with no sign of peaking. The stronger pretraining/SFT prior appears to function as implicit regularization. The policy gradient cannot overcome the sharp template distribution, which limits drift from pretrained behavior. Other factors (hyperparameters better suited to Instruct, model-family differences beyond entropy, single-seed variance) may also contribute.

## Entropy Trajectories

| Step | GPT-OSS | Instruct | Think |
|------|---------|----------|-------|
| 0 | 0.841 | 0.325 | 0.270 |
| 10 | 0.765 | 0.305 | 0.255* |
| 20 | 0.763 | 0.307 | 0.248* |
| 30 | 0.729 | 0.300 | |
| 40 | 0.707 | 0.291 | |
| 50 | 0.698 | 0.292 | |
| 60 | 0.717 | | |
| 70 | 0.681 | | |

(*Think model only ran 23 steps)

GPT-OSS entropy drops 21% (0.841 to 0.662 at step 74). Instruct drops 10% (0.325 to 0.292). Think drops 17% (0.270 to 0.225 at step 22). All models show PPO concentrating probability mass on rewarded behaviors, but the absolute levels remain qualitatively different throughout training.

## Implications for Model Selection in RLVR

High entropy is a double-edged property. GPT-OSS learns faster per step but over-optimizes. Instruct learns slower but generalizes better. The optimal entropy level for RLVR lies between these regimes: enough headroom for meaningful gradient signal, low enough that the policy doesn't drift into overfit territory in fewer than 50 steps.

Qwen Instruct's low entropy comes from instruction tuning, which implicitly constrains the RL policy. This is functionally similar to a strong KL penalty toward the base policy, achieved without explicitly computing KL. For high-entropy models like GPT-OSS, explicit KL regularization or early stopping becomes correspondingly more important.

The think model's entropy problem is qualitatively different. Qwen Think has the lowest entropy (0.27 nats) and the lowest eval accuracy (25-27%). Its bimodal behavior (structured think blocks vs raw think text) means most probability mass is locked into format decisions rather than reasoning content. Token-level entropy masks the structured/unstructured mode split, so the headline number alone does not diagnose the failure.

## Possible Interventions

For low-entropy models: raise training temperature to 1.2-1.5 to broaden the sampling distribution; increase LoRA LR to 2-3e-5 to compensate for small gradients with larger steps; or expand LoRA rank to 64-128 for more trainable directions. For Qwen Instruct specifically, a single concise math exemplar in the system prompt could break the verbose markdown template prior, achieving in one prompt change what 100 RL steps cannot.

For high-entropy models: apply early stopping at the eval peak (step 40 for GPT-OSS), add an explicit KL penalty (0.01-0.1 coef) to slow policy drift, or use a larger eval set to detect over-optimization earlier with lower noise.
