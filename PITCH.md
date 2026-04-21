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
> The **Oversight Agent** — this is what we're training with GRPO —
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

**[SHOW: Evaluation table + Reward curve]**

> "Baseline results across 5 seeds:
> - No-op agent: 0.01 average score
> - Random agent: 0.05
> - Smart heuristic with all 8 tools: 0.17
>
> After GRPO training with Llama 3.2 3B:
> The reward curve rises from 0.28 to 0.71 over 20 episodes.
>
> The gap between the heuristic and training ceiling shows exactly
> what reinforcement learning adds. Raw pattern matching can't
> solve 2-hop reasoning — you need genuine agentic capability."

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
> trains 10x faster than sparse rewards — critical for the 24-hour
> hackathon format.
>
> The code is live on GitHub and HuggingFace. Every component is
> built on TRL with Llama 3.2 — Meta-native, end to end.
>
> This is AI that watches AI. Thank you."

---

## TIMER NOTES
- 0:00–0:30 — Hook (the problem is visceral)
- 0:30–1:00 — Problem statement
- 1:00–2:00 — Architecture + what makes it hard
- 2:00–2:30 — Results with numbers
- 2:30–3:00 — Contributions + close

## SCREEN SEQUENCE
1. Opening: Actor hallucination example (terminal output)
2. Architecture diagram from README
3. Evaluation table (No-Op vs Random vs Heuristic)
4. Reward curve (outputs/reward_curve.png)
5. HuggingFace demo URL
