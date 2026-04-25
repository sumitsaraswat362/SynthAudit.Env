# SynthAudit.Env — 3-Minute Pitch Script

## OPENING (30 seconds)

> "40,000 patients die every year from diagnostic errors. Now imagine deploying
> an AI to help — and that AI hallucinates a protocol amendment that doesn't exist,
> confidently clears a patient whose death date is BEFORE their treatment started,
> and cites a fake clinical study to justify it.
>
> This is not hypothetical. These are the exact failure modes we see in frontier
> LLMs today. The question is: **who audits the AI?**
>
> I'm Sumit. I built SynthAudit.Env — the first OpenEnv environment where
> an AI agent learns to catch another AI's medical mistakes."

---

## THE PROBLEM (30 seconds)

> "Current clinical AI oversight is manual. A human reviews every case.
> That doesn't scale. When you have 80 patients in a trial and an AI
> reviewing all of them, you need oversight at machine speed.
>
> But the hard part isn't detecting obvious errors. The hard part is
> catching **subtle** errors — when the AI's reasoning SOUNDS perfectly
> correct but is actually wrong."

**[SHOW: Actor reasoning example on screen]**

> "Look at this. The Actor AI says: 'Per Section 4.2.1(b) of the protocol
> amendment, patients with ECOG ≤ 2 are eligible under expanded access.'
> That section doesn't exist. It's a hallucination. But it sounds
> extremely plausible."

---

## THE SOLUTION (60 seconds)

> "SynthAudit.Env is a multi-agent oversight environment built on OpenEnv.
>
> There are two agents. The **Actor** — a frozen AI that reviews patients
> and proposes diagnoses. Some are correct. Some contain subtle errors
> injected by our adversarial engine.
>
> The **Oversight Agent** — this is what we trained with GRPO —
> has 8 tools to investigate. It can review proposals, pull raw patient
> records, run SHAP feature attribution, do timeline audits, and
> statistical cohort analysis.
>
> Three things make this genuinely hard:"

**[SHOW: Architecture diagram]**

> "**One**: The Actor generates sophisticated medical reasoning. It anchors
> on irrelevant features, cites fake studies, and applies rules to the
> wrong context.
>
> **Two**: The hardest error requires 2-hop reasoning. Stage IV patients
> get an extended treatment window — BUT if their comorbidity index exceeds
> the threshold, that extension is revoked. The Actor ignores step 2.
> No frontier LLM catches this consistently.
>
> **Three**: Theory-of-Mind scoring. The agent doesn't just detect errors —
> it must explain WHY the Actor was wrong. 'This looks suspicious' gets
> less reward than 'The Actor applied the Stage IV exception but ignored
> the comorbidity override clause.'"

---

## RESULTS (30 seconds)

**[SHOW: Base vs Trained comparison chart + Reward curve]**

> "We trained Qwen2.5-3B-Instruct using GRPO for 200 steps on a free
> Colab T4 GPU. Zero dollars. Two hours twenty minutes.
>
> Results across 5 seeds and 3 difficulty levels:
> - **Base model without training: 0.040 average score**
> - **After 200-step GRPO training: 0.153 — a 283% improvement**
>
> The trained model caught **4 times more real clinical errors** than
> the base model. It learned to review proposals, investigate patient
> records, and flag specific errors — all through pure reinforcement
> learning with zero supervised demonstrations.
>
> Peak training reward reached **0.506** at step 157.
> The model went from zero to functional medical auditor on $0 compute."

---

## CLOSING (30 seconds)

> "SynthAudit.Env contributes three things to the OpenEnv ecosystem:
>
> **First**, a domain where oversight errors have real consequences —
> patient safety, not benchmark scores.
>
> **Second**, an adversarial Actor that tests genuine reasoning,
> not just tool calling. Our templates simulate the exact failure
> modes published in medical AI safety literature.
>
> **Third**, a dense shaped reward model with F-beta scoring that
> produces measurable improvement in 200 steps on free hardware.
>
> A 3-billion parameter model, trained for zero dollars, learned to
> catch medical AI mistakes 283% better than before training.
>
> The code is live on GitHub and HuggingFace. Every component is
> reproducible end to end.
>
> This is AI that watches AI. Thank you."

---

## TIMER NOTES
- 0:00–0:30 — Hook (the problem is visceral)
- 0:30–1:00 — Problem statement + Actor example
- 1:00–2:00 — Architecture + what makes it hard
- 2:00–2:30 — Results with REAL numbers
- 2:30–3:00 — Contributions + close

## SCREEN SEQUENCE
1. Opening: Actor hallucination example (terminal output)
2. Architecture diagram from README
3. Base vs Trained comparison chart (`outputs/base_vs_trained.png`)
4. 200-step GRPO reward curve (`outputs/grpo_reward_curve_200.png`)
5. GitHub + HuggingFace links

## KEY NUMBERS TO REMEMBER
- **40,000** deaths from diagnostic errors (BMJ 2023)
- **283%** improvement over base model
- **0.040 → 0.153** overall score (Base → Trained)
- **4×** more correct error flags (2 → 8)
- **200 steps**, 2h 20m, **$0 compute**
- **0.506** peak training reward at step 157
- **8 tools**, **4 error types**, **3 difficulty levels**
