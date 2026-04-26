# SynthAudit.Env — 2-Minute Pitch

## HOOK (15 seconds)

> "A patient dies. Not because the doctor was wrong — because the AI
> that screened them hallucinated a protocol amendment, cleared them
> for treatment, and cited a fake clinical study to justify it.
>
> **40,000 patients die every year from diagnostic errors.**
> I built the first environment where AI learns to catch AI mistakes."

---

## WHAT IT IS (30 seconds)

> "SynthAudit.Env. Two agents. One **Actor** — a frozen AI that
> reviews patients and makes mistakes. Subtle ones. It cites fake
> study sections. It ignores age violations. It applies the wrong rules.
>
> The **Oversight Agent** — trained with GRPO — gets 8 tools to
> investigate and catch those mistakes. Review, investigate, flag, approve.
>
> The hardest test? **2-hop reasoning.** Stage IV patients get an
> extended treatment window — BUT if comorbidity exceeds the threshold,
> that extension is revoked. The Actor ignores step 2.
> **No frontier LLM catches this consistently.**"

---

## WHY IT'S HARD (20 seconds)

> "Our environment is **intentionally brutal**. Hard mode has 14-17
> hidden errors per episode. Even GPT-4 class models struggle.
>
> This isn't a toy benchmark with 90% accuracy. This is adversarial
> clinical reasoning where a base model scores **0.04 out of 1.0**.
> That's the point — if it was easy, you wouldn't need GRPO."

---

## RESULTS (30 seconds)

**[SHOW: Comparison chart]**

> "Free Colab T4. Zero dollars. Two hours.
>
> **283% improvement** over the untrained model.
> **4 times more** clinical errors correctly caught.
> Error detection jumped from **0.13 per episode to 0.53**.
>
> On a 3-billion parameter model. Intentionally small.
> Because if a 3B model can learn clinical oversight on free hardware,
> imagine what this environment teaches a 70B.
>
> **The environment is the contribution. The model proves it works.**"

---

## CLOSE (15 seconds)

> "SynthAudit.Env: 8 tools, 4 adversarial error types,
> Theory-of-Mind scoring, dense shaped rewards, adaptive curriculum.
>
> **AI that watches AI. Zero dollars. Lives saved.**
>
> The code is on GitHub and HuggingFace. Thank you."

---

## CHEAT SHEET (memorize these)

| Number | What |
|--------|------|
| **40,000** | Deaths from diagnostic errors/year |
| **283%** | Improvement over base model |
| **4×** | More errors caught (2 → 8) |
| **$0** | Compute cost |
| **0.04 → 0.153** | Base → Trained score |
| **0.506** | Peak training reward (step 157) |
| **3B** | Model size (intentionally small) |
| **200** | GRPO training steps |

## SCREEN ORDER
1. Hook → blank screen or logo
2. Architecture diagram
3. Base vs Trained comparison chart
4. GRPO reward curve
5. GitHub + HF links
