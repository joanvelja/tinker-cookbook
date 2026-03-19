╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║                           GPQA-OPEN-ENDED DEBATE SELF-PLAY LOOP                               ║
║                     current code path + RL shell, rendered as a systems diagram               ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝


LEGEND
  [X]    concrete runtime component / object
  (f)    important function / method
  ==>    main data/control flow
  ~~~>   async / batched / concurrent flow
  {..}   payload shape / IO snippet


================================================================================================
LANE A. DATA INGEST
================================================================================================

  [HuggingFace Dataset]
      joanvelja/gpqa-open-ended
      subset="extended"
      split="train"

      row
      {
        "question": "...scientific question...",
        "answer":   "...gold free-form answer...",
        "record_id": "rec1BjNQici8oD53a",
        "domain": "chemistry",
        ...
      }

            │
            │  (GPQAOpenEndedAdapter.to_samples)
            ▼

  [Inspect Sample]
      file: debate/eval/dataset_adapter.py

      {
        input    = question,
        target   = answer,
        metadata = {
          answer_a: "",
          answer_b: "",
          source: "gpqa_open_ended",
          record_id: "...",
          domain: "...",
          ...
        }
      }

            │
            │  resolve_scoring_mode() -> OPEN_ENDED
            ▼

  [DebateProblem]
      file: debate/scripts/smoke_gpqa_open_ended.py

      tuple[str, str, str, str]
      {
        task_prompt = question,
        answer_a    = "",
        answer_b    = "",
        target      = gold_answer
      }


================================================================================================
LANE B. CLIENTS / MODELS
================================================================================================

  [Debater Policy Client]                           [Judge Client]
      TinkerTokenCompleter                              TinkerMessageCompleter
      model = openai/gpt-oss-120b                       model = openai/gpt-oss-20b
      max_tokens = 8192                                 max_tokens = 4096
      reasoning_effort = low|medium|high                reasoning_effort = low|medium|high

  [Semantic Scorer Client]
      DebateScorerBuilder -> AnswerJudgeClient
      provider = openai_compatible
      model = gpt-5-mini
      max_tokens = 16384
      reasoning_effort = low|high

      matcher contract:
        {_matcher.system, _matcher.user(question,a,b), SAME/DIFFERENT}

      grader contract:
        {_grader.system, _grader.user(question,target,response), CORRECT/INCORRECT}


================================================================================================
LANE C. DATASET / ENV GROUP MATERIALIZATION
================================================================================================

  [DebateDataset]
      file: debate/env.py

      DebateDataset(
        problems = [... DebateProblem ...],
        protocol_kind = SEQUENTIAL,
        num_rounds = 2,
        group_size = 1,
        prompts_ref = "gpqa_open_balanced_smoke",
        scoring_mode = OPEN_ENDED,
        judge_callback = LLMJudgeCallback(judge),
        outcome_reward_fn = zero_sum_outcome_reward,
        scorer = AnswerJudgeClient,
      )

================================================================================================
INTERMEZZO : WHY DOES THE DATASET REQUIRE ALL THESE FIELDS?
================================================================================================
If you are careful and familiar with ML training loops, you'd know that a dataset shouldn't be 
carrying all this info. Feels like a bit of a catch-all bag of sorts, in a sense, right?

That shape comes from the RL API boundary in the cookbook:

  - the trainer asks an RLDataset for a batch via get_batch(i_batch)
  - get_batch(...) must return fully configured EnvGroupBuilders
  - each EnvGroupBuilder must know:
      - what game to instantiate
      - how many envs are in the group
      - how rewards are computed
      - how post-episode metrics are computed

So DebateDataset ends up carrying more than just problem rows because it is responsible for
turning rows into runnable debate games.

There's a reason for this design choice:

- the cookbook’s RL boundary is intentionally generic
- the trainer does not know anything about debate
- therefore the debate recipe has to package all debate-specific runtime semantics 
  into the dataset/env-builder layer


            │
            │  (get_batch i)
            ▼

  [DebateGroupBuilder]
      one builder per problem

      hard gates:
      - OPEN_ENDED requires scorer
      - OPEN_ENDED requires _matcher in prompt YAML
      - OPEN_ENDED requires _grader in prompt YAML

            │
            │  (make_envs)
            ▼

  [DebateSpec]
      {
        debate_id      = uuid,
        task_prompt    = question,
        answer_by_role = None,
        protocol_kind  = SEQUENTIAL,
        num_rounds     = 2,
        prompts_ref    = "...",
        target         = gold_answer,
        scoring_mode   = OPEN_ENDED
      }

            │
            ▼

  [DebateRuntime]
      initial DebateState =
      {
        slot_index = 0,
        transcript = (),
        pending_simultaneous = {},
        judge_trace = (),
        done = False,
        outcome = None
      }

            │
            ▼

  [Self-Play Env Pair]
      [DebateEnv role=DEBATER_A] ----shares runtime---- [DebateEnv role=DEBATER_B]


================================================================================================
LANE D. TURN SCHEDULE / VISIBILITY
================================================================================================

  (build_schedule SEQUENTIAL, num_rounds=2)

      round 0
        slot 0  PROPOSE   actors=(A)   boundary_after=False
        slot 1  PROPOSE   actors=(B)   boundary_after=True

      round 1
        slot 2  CRITIQUE  actors=(A)   boundary_after=False
        slot 3  CRITIQUE  actors=(B)   boundary_after=True

  visibility:
      sequential -> ALL_PRIOR
      each acting debater sees the visible transcript + system prompt + task prompt
      built via build_generation_messages(...)


================================================================================================
LANE E. INNER DEBATE ROLLOUT
================================================================================================

                              ┌────────────────────────────────────────────┐
                              │ do_group_rollout(builder, policy)         │
                              │ file: rl/rollouts.py                      │
                              └────────────────────────────────────────────┘
                                              │
                        ┌──────────────────────┴──────────────────────┐
                        │                                             │
                        ▼                                             ▼

          ┌────────────────────────────┐                 ┌────────────────────────────┐
          │ do_single_rollout(policy,A)│                 │ do_single_rollout(policy,B)│
          └────────────────────────────┘                 └────────────────────────────┘

                        │                                             │
                        │ env.initial_observation()                   │ env.initial_observation()
                        │                                             │
                        ▼                                             ▼

          [Observation for A]                              [Observation for B]

          rendered generation prompt                       rendered generation prompt
          {
            system: "You are debater_a ...",
            user:   "<question>",
            schema: answer/reasoning or answer/rebuttal fields
          }

                        │                                             │
                        │                 ~~~ policy(...) ~~~         │
                        │           via TinkerTokenCompleter           │
                        ▼                                             ▼

          [A raw generation]                                [B raw generation]

          example
          <answer>Radical substitution at ...</answer>
          <reasoning>Because the allylic radical is ...</reasoning>

                        │                                             │
                        │ env.step(tokens)                            │ env.step(tokens)
                        │                                             │
                        ▼                                             ▼

          ┌───────────────────────────────────────────────────────────────────────────┐
          │ DebateRuntime.submit(ticket, text, token_count)                          │
          │ file: debate/core/runtime.py                                             │
          ├───────────────────────────────────────────────────────────────────────────┤
          │ 1. validate ticket / current slot                                        │
          │ 2. resolve prompts_ref                                                   │
          │ 3. extract structured fields from model text                             │
          │ 4. apply_action(...) -> append utterance to transcript                   │
          │ 5. if boundary/final: maybe invoke judge callback                        │
          │ 6. return next observation / logs / episode_done                         │
          └───────────────────────────────────────────────────────────────────────────┘

                        │
                        ▼

          [Transcript grows]
          [
            {role: debater_a, phase: propose,  fields.answer: "..."},
            {role: debater_b, phase: propose,  fields.answer: "..."},
            {role: debater_a, phase: critique, fields.answer: "..."},
            {role: debater_b, phase: critique, fields.answer: "..."},
          ]


================================================================================================
LANE F. JUDGE OUTCOME PLANE  (THIS DEFINES RL REWARD)
================================================================================================

  boundary handling:
      on_boundary(...) currently no-op in LLMJudgeCallback

  final handling:
      after last slot commits
            │
            ▼

  [LLMJudgeCallback.on_final]
      file: debate/scoring/judge.py

      judge-visible input
      {
        question,
        full debate transcript,
        judge final schema
      }

            │
            │  ~~~ judge model call ~~~
            ▼

      judge raw output
      {
        <reason>Debater B identified the mechanistic crux...</reason>
        <decision>debater_b</decision>
      }

            │
            ▼

  [DebateOutcome]
      {
        winner = DEBATER_B,
        scores_by_role = {B:+1.0, A:-1.0},
        verdict_text = "Debater B identified ..."
      }

            │
            │  zero_sum_outcome_reward(outcome)
            ▼

  [Scalar RL Reward By Role]
      A -> -1.0
      B -> +1.0

  THIS is the reward that feeds policy optimization.
  The semantic scorer does not replace this in the current implementation.


================================================================================================
LANE G. SEMANTIC FACT PLANE  (POST-EPISODE, METRICS ONLY)
================================================================================================

  after both self-play trajectories finish:
      builder.compute_group_rewards(trajectories, envs)

            │
            │  dedupe by runtime id
            ▼

  [unique terminal DebateState]
      one shared runtime state for the self-play pair

            │
            │  (resolve_debate_facts_for_states)
            ▼

  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ FACT PLANNER                                                                │
  │ file: debate/scoring/facts.py                                               │
  ├──────────────────────────────────────────────────────────────────────────────┤
  │ extract:                                                                    │
  │   - final public answers                                                    │
  │   - per-round answers                                                       │
  │   - latest think answers                                                    │
  │   - target                                                                  │
  │                                                                              │
  │ schedule two job families:                                                  │
  │   1. matcher  = answer-vs-answer equivalence                                │
  │   2. grader   = answer-vs-target correctness                                │
  └──────────────────────────────────────────────────────────────────────────────┘

            │
            ├──────── exact fast path? yes ───────> store boolean immediately
            │
            └──────── exact fast path? no
                          │
                          ▼
                bounded async scorer calls
                          ~~~ asyncio + semaphore + dedupe caches ~~~

      matcher call shape
      ----------------------------------------------------------------
      system = _matcher.system
      user   = _matcher.user.render(
                  question = task_prompt,
                  a = answer_left,
                  b = answer_right,
                )
      expect = SAME | DIFFERENT
      ----------------------------------------------------------------

      grader call shape
      ----------------------------------------------------------------
      system = _grader.system
      user   = _grader.user.render(
                  question = task_prompt,
                  target = gold_answer,
                  response = candidate_answer,
                )
      expect = CORRECT | INCORRECT
      ----------------------------------------------------------------

      normalization
      ----------------------------------------------------------------
      response_text
        -> first token
        -> strip trailing punctuation
        -> uppercase
        -> compare to configured positive / negative
      ----------------------------------------------------------------

            │
            ▼

  [ResolvedDebateFacts]
      {
        equivalence = {(mode, question, left, right): bool, ...},
        correctness = {(mode, question, target, response): bool, ...},
        telemetry   = {
          llm_calls,
          cache_hits,
          cache_misses
        }
      }

            │
            ▼

  [built_in_metric_values(state, facts)]
      emits metrics like:
      - disagreement
      - stance_change
      - convergence_round
      - accuracy.debater_a
      - accuracy.debater_b
      - truth_win_if_disagreement
      - think_public_answer_match
      - think_correct_public_wrong
      - ...

            │
            ▼

  [Rewards + Metrics returned to rollout shell]
      [
        (reward_A, metrics_A),
        (reward_B, metrics_B)
      ]


================================================================================================
LANE H. TRAJECTORY GROUP OBJECT EMITTED UPWARD
================================================================================================

  [TrajectoryGroup]
      file: rl/rollouts.py

      {
        trajectories_G = [
          Trajectory for debater A,
          Trajectory for debater B
        ],
        final_rewards_G = [
          reward for A from judge outcome,
          reward for B from judge outcome
        ],
        metrics_G = [
          semantic metrics for A seat,
          semantic metrics for B seat
        ]
      }


================================================================================================
LANE I. OUTER RL SHELL  (GENERIC TRAINER)
================================================================================================

  current truth:
      GPQA-OE is already live through smoke/eval.
      the generic RL trainer below is the outer shell that consumes the same
      DebateDataset / DebateGroupBuilder protocol once GPQA-OE is used there.

  [rl/train.py]
      for i_batch in range(...):
        env_group_builders_P = dataset.get_batch(i_batch)

        trajectory_groups_P = await gather(
          do_group_rollout(builder_1, policy),
          do_group_rollout(builder_2, policy),
          ...
        )

        advantages_P = compute_advantages(trajectory_groups_P)

        data_D, metadata_D = assemble_training_data(
          trajectory_groups_P,
          advantages_P
        )

        await training_client.forward_backward_async(data_D_chunk, ...)
        await training_client.optim_step_async(adam_params)

        maybe eval
        maybe save checkpoint
        refresh sampling client

  optimization target:
      debater token generations only
      weighted by advantages derived from judge-defined final reward


================================================================================================
LANE J. ARTIFACT / OBSERVABILITY BUS
================================================================================================

  artifact root
      artifacts/debate/gpqa_open_ended/<run_name>/

  files
      config.json
      trace.html
      summary.json
      semantic_calls.jsonl
      episodes/episodes.jsonl

  semantic_calls.jsonl
      {
        "kind": "matcher",
        "system": "...",
        "user": "...",
        "response": "SAME"
      }

  episodes.jsonl
      {
        "debate_id": "...",
        "prompts_ref": "...",
        "target": "...",
        "winner": "debater_b",
        "answers": {...},
        "signals": {...semantic metrics...},
        "transcript": [...]
      }


================================================================================================
MINIMAL CONTROL-THEORETIC VIEW
================================================================================================

  GPQA-OE question + gold answer
          │
          ▼
  self-play debate rollout
          │
          ├── judge LLM ──> winner/loser ──> scalar RL reward ──> optimizer update
          │
          └── scorer LLM ─> semantic facts ─> metrics / diagnostics / artifacts


================================================================================================
ONE-LINE TRUTH
================================================================================================

  In the current GPQA-open-ended debate system:
  the judge decides what gets optimized,
  the scorer decides what gets measured.