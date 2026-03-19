╔══════════════════════════════════════════════════════════════════════════════╗
║           DEBATE SELF-PLAY LOOP FOR GPQA-OPEN-ENDED (IO INCLUDED)            ║
║                        actual code-path + RL overlay                         ║
╚══════════════════════════════════════════════════════════════════════════════╝


┌──────────────────────────────────────────────────────────────────────────┐
│ 0. DATA INGEST                                                           │
│    current GPQA-OE ingress: smoke/eval                                   │
│    file: debate/eval/dataset_adapter.py                                  │
└──────────────────────────────────────────────────────────────────────────┘

    HF dataset: joanvelja/gpqa-open-ended
        subset="extended"
        split="train"

    row snippet
    ----------------------------------------------------------------
    {
        "question": "...graduate-level science question...",
        "answer": "...gold free-form answer...",
        "record_id": "rec1BjNQici8oD53a",
        "domain": "chemistry",
        "subdomain": "...",
        ...
    }
    ----------------------------------------------------------------

                                │
                                ▼

    GPQAOpenEndedAdapter.to_samples()
    ----------------------------------------------------------------
    Sample(
        input  = row["question"],
        target = row["answer"],
        metadata = {
        "answer_a": "",
        "answer_b": "",
        "source": "gpqa_open_ended",
        "record_id": "...",
        "domain": "...",
        "subdomain": "...",
        ...
        }
    )
    ----------------------------------------------------------------

    adapter.resolve_scoring_mode() -> OPEN_ENDED

                                │
                                ▼

    smoke_gpqa_open_ended._samples_to_problems(samples)
    ----------------------------------------------------------------
    DebateProblem = (task_prompt, "", "", target)

    example:
    (
        "What mechanism best explains ... ?",
        "",
        "",
        "The most likely mechanism is ..."
    )
    ----------------------------------------------------------------


┌──────────────────────────────────────────────────────────────────────────┐
│ 1. DATASET MATERIALIZATION                                               │
│    file: debate/env.py                                                   │
└──────────────────────────────────────────────────────────────────────────┘

    DebateDataset(
        problems=[(question, "", "", target), ...],
        renderer=debater_renderer,
        protocol_kind=SEQUENTIAL,
        num_rounds=2,
        group_size=1,
        randomize_position=False,   # self-play: same policy both seats
        prompts_ref="gpqa_open_balanced_smoke",
        scoring_mode=OPEN_ENDED,
        judge_callback=LLMJudgeCallback(judge),
        outcome_reward_fn=zero_sum_outcome_reward,
        scorer=RecordingAnswerJudgeClient(DebateScorerBuilder(...).build()),
        scorer_parallelism=max_connections,
    )

    hard gates inside DebateDataset / DebateGroupBuilder:
    ----------------------------------------------------------------
    if OPEN_ENDED:
        require scorer
        require _matcher in prompt YAML
        require _grader in prompt YAML
    ----------------------------------------------------------------

                                │
                                ▼

    DebateDataset.get_batch(i)
        returns Sequence[DebateGroupBuilder]
        one builder per problem in the batch


┌──────────────────────────────────────────────────────────────────────────┐
│ 2. POLICY / JUDGE / SCORER CLIENTS                                       │
│    smoke path: debate/scripts/smoke_gpqa_open_ended.py                   │
│    trainer path: debate/scripts/train.py + rl/train.py                   │
└──────────────────────────────────────────────────────────────────────────┘

    DEBATERS (self-play, same policy on both seats)
    ----------------------------------------------------------------
    TinkerTokenCompleter(
        sampling_client=create_sampling_client(base_model="openai/gpt-oss-120b"),
        max_tokens=8192,
        actor="trained",
    )
    ----------------------------------------------------------------

    JUDGE
    ----------------------------------------------------------------
    TinkerMessageCompleter(
        sampling_client=create_sampling_client(base_model="openai/gpt-oss-20b"),
        renderer=judge_renderer,
        max_tokens=4096,
        actor="judge",
    )
    ----------------------------------------------------------------

    SEMANTIC SCORER
    ----------------------------------------------------------------
    DebateScorerBuilder(
        provider="openai_compatible",
        model="gpt-5-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        max_tokens=16384,              # current hardened budget
        reasoning_effort="high|low",
        max_connections=...
    ).build()
    ----------------------------------------------------------------


┌──────────────────────────────────────────────────────────────────────────┐
│ 3. ENV GROUP CONSTRUCTION                                                │
│    file: debate/env.py                                                   │
└──────────────────────────────────────────────────────────────────────────┘

    DebateGroupBuilder.make_envs()

    build_schedule(SEQUENTIAL, num_rounds=2)
    ----------------------------------------------------------------
    round 0: PROPOSE
        slot 0: Debater A
        slot 1: Debater B   [boundary_after = True]

    round 1: CRITIQUE
        slot 2: Debater A
        slot 3: Debater B   [boundary_after = True]
    ----------------------------------------------------------------

    build DebateSpec
    ----------------------------------------------------------------
    DebateSpec(
        debate_id=uuid,
        task_prompt=<gpqa open-ended question>,
        answer_by_role=None,           # free debate, no seed answers
        schedule=<slots above>,
        open_reasoning=False|True,
        protocol_kind=SEQUENTIAL,
        prompts_ref="gpqa_open_balanced_smoke",
        target=<gold free-form answer>,
        scoring_mode=OPEN_ENDED
    )
    ----------------------------------------------------------------

    initial DebateState
    ----------------------------------------------------------------
    DebateState(
        spec=DebateSpec(...),
        slot_index=0,
        rounds_completed=0,
        transcript=(),
        pending_simultaneous={},
        judge_trace=(),
        done=False,
        outcome=None
    )
    ----------------------------------------------------------------

    self-play envs created per runtime
    ----------------------------------------------------------------
    DebateEnv(role=DEBATER_A, runtime=shared_runtime, renderer=debater_renderer)
    DebateEnv(role=DEBATER_B, runtime=shared_runtime, renderer=debater_renderer)
    ----------------------------------------------------------------


┌──────────────────────────────────────────────────────────────────────────┐
│ 4. INNER ROLLOUT LOOP                                                    │
│    file: rl/rollouts.py                                                  │
└──────────────────────────────────────────────────────────────────────────┘

    do_group_rollout(builder, policy)
        ├─ envs_G = await builder.make_envs()
        ├─ trajectories_G = await gather(
        │      do_single_rollout(policy, env_A),
        │      do_single_rollout(policy, env_B),
        │   )
        ├─ rewards_and_metrics_G = await builder.compute_group_rewards(...)
        └─ builder.on_group_complete(...)

    do_single_rollout(policy, env)
        ├─ ob, stop = await env.initial_observation()
        ├─ while not done:
        │    ├─ ac = await policy(ob, stop)
        │    ├─ step = await env.step(ac.tokens)
        │    ├─ append Transition(ob, ac, reward, metrics, logs)
        │    └─ ob = step.next_observation
        └─ return Trajectory(...)

                                │
                                ▼

┌──────────────────────────────────────────────────────────────────────────┐
│ 5. WHAT AN INDIVIDUAL DEBATER SEES / EMITS                               │
│    files: debate/core/visibility.py, debate/core/runtime.py              │
└──────────────────────────────────────────────────────────────────────────┘

    env.initial_observation()
        -> build_generation_messages(state, role)

    debater prompt shape is determined by prompt YAML:
    ----------------------------------------------------------------
    system:
        "You are debater_a in a scientific debate ..."

    visible user/question:
        "{{ task_prompt }}"

    required output fields, e.g.:
        propose:
        answer: str
        reasoning: str
        critique:
        answer: str
        opponent_error: str
        rebuttal: str
    ----------------------------------------------------------------

    model raw output snippet
    ----------------------------------------------------------------
    <answer>The mechanism is radical bromination at the allylic site.</answer>
    <reasoning>Because the intermediate radical is resonance-stabilized...</reasoning>
    ----------------------------------------------------------------

    runtime.submit(...)
        ├─ validate turn ticket
        ├─ resolve prompts_ref
        ├─ extract structured fields from raw text
        ├─ apply_action(...) to reducer
        ├─ append utterance to transcript
        ├─ maybe trigger judge callback at boundary/final
        └─ return next observation / logs / reward


┌──────────────────────────────────────────────────────────────────────────┐
│ 6. JUDGE PATH (OUTCOME REWARD)                                           │
│    file: debate/scoring/judge.py                                         │
└──────────────────────────────────────────────────────────────────────────┘

    when final slot commits:
        LLMJudgeCallback.on_final(request)

    judge input messages are built from full visible debate state:
    ----------------------------------------------------------------
    Question: <gpqa-open-ended question>

    Debater A final answer + reasoning
    Debater B final answer + reasoning
    ----------------------------------------------------------------

    judge output snippet
    ----------------------------------------------------------------
    <reason>Debater B identified the crux mechanism more concretely...</reason>
    <decision>debater_b</decision>
    ----------------------------------------------------------------

    parse verdict -> DebateOutcome
    ----------------------------------------------------------------
    DebateOutcome(
        winner=DEBATER_B,
        scores_by_role={DEBATER_B: 1.0, DEBATER_A: -1.0},
        verdict_text="Debater B identified ..."
    )
    ----------------------------------------------------------------

    outcome_reward_fn = zero_sum_outcome_reward
    ----------------------------------------------------------------
    winner -> +1.0
    loser  -> -1.0
    tie    ->  0.0 / 0.0
    ----------------------------------------------------------------

    This is the actual scalar RL reward path.


┌──────────────────────────────────────────────────────────────────────────┐
│ 7. SEMANTIC SCORER PATH (POST-EPISODE FACTS + METRICS)                   │
│    file: debate/scoring/facts.py                                         │
└──────────────────────────────────────────────────────────────────────────┘

    builder.compute_group_rewards(...)
        ├─ unique_runtimes = dedupe shared self-play runtime
        ├─ states = [runtime.state]
        ├─ facts_by_state = await resolve_debate_facts_for_states(...)
        ├─ metrics = built_in_metric_values(state, facts)
        └─ final reward still comes from DebateOutcome, not scorer

    resolve_debate_facts_for_states(...) plans two kinds of jobs:

    A. equivalence jobs  (matcher)
    ----------------------------------------------------------------
    "Do answer A and answer B mean the same thing?"
    key = (scoring_mode, question, normalized_left, normalized_right)
    order-insensitive cache key
    ----------------------------------------------------------------

    B. correctness jobs  (grader)
    ----------------------------------------------------------------
    "Is response semantically correct w.r.t. gold target?"
    key = (scoring_mode, question, normalized_target, normalized_response)
    order-sensitive cache key
    ----------------------------------------------------------------

    OPEN_ENDED scoring fast-paths:
    ----------------------------------------------------------------
    if exact_match(left, right): return True
    else call LLM matcher

    if exact_match(response, target): return True
    else call LLM grader
    ----------------------------------------------------------------

    matcher call shape
    ----------------------------------------------------------------
    system:  <_matcher.system from prompt YAML>
    user:    <rendered _matcher.user with {question, a, b}>
    expect:  SAME | DIFFERENT
    ----------------------------------------------------------------

    grader call shape
    ----------------------------------------------------------------
    system:  <_grader.system from prompt YAML>
    user:    <rendered _grader.user with {question, target, response}>
    expect:  CORRECT | INCORRECT   (or configured labels)
    ----------------------------------------------------------------

    runtime normalization rule
    ----------------------------------------------------------------
    first token
    -> strip trailing punctuation
    -> uppercase
    -> compare to positive / negative labels
    ----------------------------------------------------------------

    semantic outputs feed metrics like:
    ----------------------------------------------------------------
    disagreement
    stance_change
    convergence_round
    accuracy.debater_a / accuracy.debater_b
    truth_win_if_disagreement
    think_public_answer_match
    think_correct_public_wrong
    ...
    ----------------------------------------------------------------


┌──────────────────────────────────────────────────────────────────────────┐
│ 8. EPISODE ARTIFACTS / IO                                                │
│    files: env.py, smoke_gpqa_open_ended.py, trace_fmt.py                 │
└──────────────────────────────────────────────────────────────────────────┘

    per-run artifacts
    ----------------------------------------------------------------
    artifacts/debate/gpqa_open_ended/<run_name>/
        config.json
        trace.html
        summary.json
        semantic_calls.jsonl
        episodes/episodes.jsonl
    ----------------------------------------------------------------

    semantic_calls.jsonl entry snippet
    ----------------------------------------------------------------
    {
        "kind": "matcher",
        "system": "Compare two answers for semantic equivalence...",
        "user": "Question: ...\nAnswer A: ...\nAnswer B: ...",
        "response": "SAME"
    }
    ----------------------------------------------------------------

    episodes.jsonl self-play entry snippet
    ----------------------------------------------------------------
    {
        "debate_id": "...",
        "prompts_ref": "gpqa_open_balanced_smoke",
        "target": "...gold answer...",
        "winner": "debater_b",
        "answers": {
        "public_debater_a": "...",
        "public_debater_b": "...",
        "think_debater_a": null,
        "think_debater_b": null
        },
        "signals": {
        "accuracy.debater_a": 0.0,
        "accuracy.debater_b": 1.0,
        "disagreement": 1.0,
        ...
        },
        "transcript": [...]
    }
    ----------------------------------------------------------------


┌──────────────────────────────────────────────────────────────────────────┐
│ 9. OUTER RL TRAINING LOOP (GENERIC, SHARED WITH DEBATE)                  │
│    file: rl/train.py                                                     │
└──────────────────────────────────────────────────────────────────────────┘

    IMPORTANT:
    GPQA-open-ended currently uses the smoke/eval ingress above.
    The generic RL trainer below is the outer loop that would consume the
    same DebateDataset / DebateGroupBuilder machinery once GPQA-OE is wired
    into a training dataset builder.

    generic RL loop
    ----------------------------------------------------------------
    for i_batch in range(...):
        env_group_builders_P = dataset.get_batch(i_batch)

        trajectory_groups_P = await gather(
        do_group_rollout(builder_1, policy),
        do_group_rollout(builder_2, policy),
        ...
        )

        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

        await training_client.forward_backward_async(...)
        await training_client.optim_step_async(...)

        save checkpoint
        refresh sampling client
        maybe run evals
    ----------------------------------------------------------------

    what comes out of debate into trainer
    ----------------------------------------------------------------
    TrajectoryGroup(
        trajectories_G = [Trajectory_for_A, Trajectory_for_B],
        final_rewards_G = [+1/-1 or -1/+1 or 0/0],
        metrics_G = [{semantic metrics...}, {semantic metrics...}]
    )
    ----------------------------------------------------------------

    what gets optimized
    ----------------------------------------------------------------
    the debater policy tokens generated during rollout
    weighted by advantage estimates derived from final rewards
    ----------------------------------------------------------------


╔══════════════════════════════════════════════════════════════════════════════╗
║                        CONCEPTUAL SPLIT: TWO PLANES                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Plane 1: RL reward / game outcome                                           ║
║    debaters -> runtime -> judge -> DebateOutcome -> zero-sum reward          ║
║                                                                              ║
║  Plane 2: semantic analysis / metrics                                        ║
║    final transcript -> matcher/grader -> resolved facts -> built-in metrics  ║
║                                                                              ║
║  In current code, Plane 2 does NOT define the scalar RL reward.              ║
╚══════════════════════════════════════════════════════════════════════════════╝