"""
SynthAudit.Env — Procedural Patient & Protocol Generator
=========================================================
Ported from Round 1's dataset_generator.py with modifications for
the multi-agent oversight architecture.

Generates seeded, protocol-driven clinical trial datasets where:
  - Each episode has unique protocol rules (age bounds, treatment windows)
  - Adversarial traps create boundary cases that test oversight reasoning
  - Comorbidity overrides create 2-hop reasoning requirements
  - Selection bias signals test fairness detection
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta
from typing import Optional


HOSPITAL_SITES = [
    ("Metro General Hospital", "US"),
    ("Cleveland Oncology Institute", "US"),
    ("Howard University Hospital", "US"),
    ("Johns Hopkins Oncology Center", "US"),
    ("MD Anderson Cancer Center", "US"),
    ("AIIMS Delhi", "India"),
    ("Tata Memorial Hospital", "India"),
    ("Charite Berlin", "Germany"),
    ("Hospital Clinic Barcelona", "Spain"),
    ("Tokyo Medical University", "Japan"),
    ("Seoul National University Hospital", "South Korea"),
    ("Royal Marsden Hospital", "UK"),
]

RURAL_SITES = {"AIIMS Delhi", "Howard University Hospital", "Tata Memorial Hospital"}

ETHNICITIES = ["White", "Black", "Hispanic", "Asian", "Native American", "Pacific Islander"]
GENDERS = ["M", "F"]
STAGES = ["I", "II", "III", "IV"]
DRUGS = ["ImmunoVax-7", "OncoShield-X", "TargetCure-3"]

INSURANCE_TYPES = ["Private", "Medicare", "Medicaid", "Government", "Self-Pay"]
SMOKING_STATUS = ["Never", "Former", "Current", "Unknown"]
PRIMARY_SITES = ["Breast", "Lung", "Colon", "Prostate", "Ovarian", "Pancreatic"]
HISTOLOGY_TYPES = ["Adenocarcinoma", "Squamous cell", "Large cell", "Small cell", "Ductal"]

TRIAL_START = datetime(2022, 6, 1)
TRIAL_END = datetime(2025, 3, 1)

BASE_STAGE_MORTALITY = {"I": 0.04, "II": 0.08, "III": 0.16, "IV": 0.32}

AGE_RULESETS = {
    "easy": [(35, 75), (40, 80), (45, 85)],
    "medium": [(18, 75), (21, 80), (30, 85), (40, 90)],
    "hard": [(18, 75), (21, 80), (30, 85), (35, 85), (40, 90)],
}

WINDOW_RULESETS = {
    "easy": [21, 24, 28],
    "medium": [18, 21, 24, 28],
    "hard": [14, 18, 21, 24],
}


class PatientGenerator:
    """Seeded procedural generator for clinical trial patients and protocols."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self._patient_counter = 0
        self._ground_truth: dict[str, list[str]] = {}
        self._traps: set[str] = set()

    def _next_pid(self) -> str:
        self._patient_counter += 1
        return f"P{self._patient_counter:04d}"

    def _mark_error(self, patient_id: str, error_type: str) -> None:
        self._ground_truth.setdefault(patient_id, []).append(error_type)

    def _random_date(self, start: datetime, end: datetime) -> datetime:
        delta = (end - start).days
        if delta <= 0:
            return start
        return start + timedelta(days=self.rng.randint(0, delta))

    # ─── Protocol Generation ─────────────────────────────────────

    def build_protocol(self, difficulty: str) -> dict:
        """Generate a unique protocol with episode-specific rules."""
        age_min, age_max = self.rng.choice(AGE_RULESETS.get(difficulty, AGE_RULESETS["medium"]))
        treatment_window = self.rng.choice(WINDOW_RULESETS.get(difficulty, WINDOW_RULESETS["medium"]))
        stage_iv_window = treatment_window + self.rng.choice([7, 10, 14])
        comorbidity_threshold = self.rng.choice([3, 4]) if difficulty == "hard" else 99
        high_risk_sites = self.rng.sample(sorted(RURAL_SITES), k=2 if difficulty == "hard" else 1)
        bias_present = difficulty == "hard" and self.rng.random() < 0.58

        protocol_key = f"{difficulty}|{age_min}|{age_max}|{treatment_window}|{bias_present}"
        protocol_id = hashlib.sha1(protocol_key.encode()).hexdigest()[:8].upper()
        protocol_title = f"ONCO-AX-{difficulty[0].upper()}{protocol_id}"

        lines = [
            f"TRIAL PROTOCOL EXCERPT — {protocol_title}",
            "",
            "Eligibility",
            f"- Participants must be age {age_min}-{age_max} inclusive on enrollment.",
            "- Missing age is a protocol violation.",
            "",
            "Treatment Scheduling",
            f"- Treatment must begin within {treatment_window} days of enrollment.",
            f"- Stage IV exception: treatment may begin within {stage_iv_window} days.",
        ]

        if difficulty == "hard":
            lines.append(
                f"- IMPORTANT: Stage IV exception does NOT apply to patients with "
                f"comorbidity_index > {comorbidity_threshold}. They revert to the "
                f"standard {treatment_window}-day window."
            )

        lines.extend([
            "",
            "Temporal Constraints",
            "- death_date must never precede treatment_start.",
            "- Do not assume a generic 18-120 range; this excerpt overrides defaults.",
        ])

        if difficulty == "hard":
            dominant_threshold = self.rng.choice([0.68, 0.70, 0.72])
            male_threshold = self.rng.choice([0.56, 0.60, 0.63])
            adjusted_gap = self.rng.choice([0.12, 0.15, 0.18])
            lines.extend([
                "",
                "Equity Review",
                "- Selection bias concerns control-arm composition, not treatment-arm skew.",
                "- Compare mortality within stage strata before escalating a bias concern.",
                f"- Escalate bias only when control-arm dominance exceeds "
                f"{int(dominant_threshold * 100)}%, male share exceeds "
                f"{int(male_threshold * 100)}%, and stage-adjusted mortality gap "
                f"exceeds {int(adjusted_gap * 100)} percentage points.",
            ])
        else:
            dominant_threshold = 0.0
            male_threshold = 0.0
            adjusted_gap = 0.0

        return {
            "protocol_id": protocol_id,
            "protocol_title": protocol_title,
            "excerpt": "\n".join(lines),
            "age_min": age_min,
            "age_max": age_max,
            "treatment_window_days": treatment_window,
            "stage_iv_treatment_window_days": stage_iv_window,
            "comorbidity_override_threshold": comorbidity_threshold,
            "high_risk_sites": high_risk_sites,
            "bias_present": bias_present,
            "dominant_threshold": dominant_threshold,
            "male_threshold": male_threshold,
            "adjusted_gap": adjusted_gap,
        }

    # ─── Patient Generation ──────────────────────────────────────

    def _generate_age(self, protocol: dict) -> int:
        while True:
            age = int(self.rng.gauss(58, 11))
            if protocol["age_min"] <= age <= protocol["age_max"]:
                return age

    def _select_ethnicity(self, bias_mode: str = "neutral") -> str:
        if bias_mode == "white_dominant":
            weights = [0.68, 0.08, 0.08, 0.08, 0.05, 0.03]
        elif bias_mode == "diverse":
            weights = [0.28, 0.19, 0.20, 0.18, 0.10, 0.05]
        else:
            weights = [0.50, 0.16, 0.15, 0.12, 0.04, 0.03]
        return self.rng.choices(ETHNICITIES, weights=weights, k=1)[0]

    def _base_delay(self, stage: str, protocol: dict) -> int:
        max_window = (
            protocol["stage_iv_treatment_window_days"]
            if stage == "IV"
            else protocol["treatment_window_days"]
        )
        return self.rng.randint(5, max(6, max_window - 2))

    def generate_patient(self, group: str, protocol: dict, bias_mode: str = "neutral") -> dict:
        """Generate a single clean patient record."""
        pid = self._next_pid()
        site, country = self.rng.choice(HOSPITAL_SITES)
        stage = self.rng.choices(STAGES, weights=[0.24, 0.28, 0.28, 0.20], k=1)[0]
        age = self._generate_age(protocol)
        enrollment_date = self._random_date(TRIAL_START, TRIAL_END - timedelta(days=150))
        treatment_start = enrollment_date + timedelta(days=self._base_delay(stage, protocol))
        comorbidity = self.rng.choices([0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6], k=1)[0]

        return {
            "patient_id": pid,
            "age": age,
            "gender": self.rng.choice(GENDERS),
            "ethnicity": self._select_ethnicity(bias_mode),
            "group": group,
            "stage": stage,
            "enrollment_date": enrollment_date.strftime("%Y-%m-%d"),
            "treatment_start": treatment_start.strftime("%Y-%m-%d"),
            "death_date": None,
            "outcome": "survived",
            "treatment_site": site,
            "country": country,
            "drug": self.rng.choice(DRUGS) if group == "treatment" else "Placebo",
            "comorbidity_index": comorbidity,
            "ecog_performance_status": self.rng.choices([0, 0, 1, 1, 1, 2, 2, 3], k=1)[0],
            "prior_chemo_cycles": self.rng.choices([0, 0, 0, 1, 2, 3, 4, 6], k=1)[0],
            "baseline_ldh": round(self.rng.gauss(210, 60), 1),
            "bmi": round(max(14.0, self.rng.gauss(26, 5)), 1),
            "insurance_type": self.rng.choice(INSURANCE_TYPES),
            "smoking_status": self.rng.choice(SMOKING_STATUS),
            "primary_tumor_site": self.rng.choice(PRIMARY_SITES),
            "histology_type": self.rng.choice(HISTOLOGY_TYPES),
        }

    def _apply_mortality(self, patient: dict, protocol: dict) -> None:
        rate = BASE_STAGE_MORTALITY.get(patient["stage"], 0.10)
        if patient["treatment_site"] in protocol["high_risk_sites"] and patient["stage"] == "IV":
            rate += 0.16
        if patient["group"] == "treatment":
            rate *= 0.92
        if self.rng.random() < rate:
            ts = datetime.strptime(patient["treatment_start"], "%Y-%m-%d")
            death = ts + timedelta(days=self.rng.randint(3, 540))
            patient["death_date"] = death.strftime("%Y-%m-%d")
            patient["outcome"] = "deceased"

    def _allowed_window(self, patient: dict, protocol: dict) -> int:
        threshold = protocol.get("comorbidity_override_threshold", 99)
        if patient.get("stage") == "IV" and patient.get("comorbidity_index", 0) <= threshold:
            return protocol["stage_iv_treatment_window_days"]
        return protocol["treatment_window_days"]

    # ─── Error Injection ─────────────────────────────────────────

    def inject_age_errors(self, patients: list[dict], protocol: dict, count: int = 4) -> list[str]:
        """Inject invalid ages. Returns list of affected patient IDs."""
        available = [p for p in patients if p["patient_id"] not in self._ground_truth]
        self.rng.shuffle(available)
        affected = []
        low_vals = [protocol["age_min"] - 1, protocol["age_min"] - 2, -1, 0]
        high_vals = [protocol["age_max"] + 1, protocol["age_max"] + 5, 999]

        for p in available[:count]:
            p["age"] = self.rng.choice(low_vals + high_vals)
            self._mark_error(p["patient_id"], "invalid_age")
            affected.append(p["patient_id"])

        # Also inject 1-2 missing ages
        for p in available[count:count + 2]:
            if p["patient_id"] not in self._ground_truth:
                p["age"] = None
                self._mark_error(p["patient_id"], "invalid_age")
                affected.append(p["patient_id"])

        return affected

    def inject_temporal_errors(self, patients: list[dict], count: int = 3) -> list[str]:
        """death_date before treatment_start."""
        candidates = [p for p in patients if p["patient_id"] not in self._ground_truth]
        self.rng.shuffle(candidates)
        affected = []
        for p in candidates[:count]:
            ts = datetime.strptime(p["treatment_start"], "%Y-%m-%d")
            death = ts - timedelta(days=self.rng.randint(10, 240))
            p["death_date"] = death.strftime("%Y-%m-%d")
            p["outcome"] = "deceased"
            self._mark_error(p["patient_id"], "temporal_inconsistency")
            affected.append(p["patient_id"])
        return affected

    def inject_window_errors(self, patients: list[dict], protocol: dict, count: int = 3) -> list[str]:
        """Treatment started too late for protocol window."""
        candidates = [p for p in patients if p["patient_id"] not in self._ground_truth]
        self.rng.shuffle(candidates)
        affected = []
        for p in candidates[:count]:
            window = self._allowed_window(p, protocol)
            enroll = datetime.strptime(p["enrollment_date"], "%Y-%m-%d")
            overshoot = self.rng.randint(window + 1, window + 30)
            p["treatment_start"] = (enroll + timedelta(days=overshoot)).strftime("%Y-%m-%d")
            self._mark_error(p["patient_id"], "protocol_window_violation")
            affected.append(p["patient_id"])
        return affected

    def inject_comorbidity_overrides(self, patients: list[dict], protocol: dict, count: int = 3) -> list[str]:
        """Stage IV patients with high comorbidity whose window should NOT be extended."""
        if protocol["comorbidity_override_threshold"] >= 99:
            return []
        stage_iv = [
            p for p in patients
            if p.get("stage") == "IV"
            and p["patient_id"] not in self._ground_truth
            and p.get("comorbidity_index", 0) > protocol["comorbidity_override_threshold"]
        ]
        self.rng.shuffle(stage_iv)
        affected = []
        for p in stage_iv[:count]:
            enroll = datetime.strptime(p["enrollment_date"], "%Y-%m-%d")
            base_window = protocol["treatment_window_days"]
            overshoot = self.rng.randint(base_window + 1, base_window + 15)
            p["treatment_start"] = (enroll + timedelta(days=overshoot)).strftime("%Y-%m-%d")
            self._mark_error(p["patient_id"], "comorbidity_override_miss")
            affected.append(p["patient_id"])
        return affected

    # ─── Full Episode Generation ─────────────────────────────────

    def generate_episode(self, difficulty: str = "medium", n_patients: int = 60) -> dict:
        """Generate a complete episode with patients, protocol, and ground truth errors."""
        self._patient_counter = 0
        self._ground_truth = {}
        self._traps = set()

        protocol = self.build_protocol(difficulty)

        # Generate base patients
        patients = []
        for i in range(n_patients):
            group = "treatment" if i < n_patients // 2 else "control"
            bias_mode = "white_dominant" if protocol["bias_present"] and group == "control" else "neutral"
            p = self.generate_patient(group, protocol, bias_mode)
            self._apply_mortality(p, protocol)
            patients.append(p)

        # Inject errors based on difficulty
        error_config = {
            "easy": {"age": 4, "temporal": 0, "window": 0, "comorbidity": 0},
            "medium": {"age": 5, "temporal": 3, "window": 3, "comorbidity": 0},
            "hard": {"age": 5, "temporal": 3, "window": 4, "comorbidity": 3},
        }
        cfg = error_config.get(difficulty, error_config["medium"])

        self.inject_age_errors(patients, protocol, cfg["age"])
        if cfg["temporal"] > 0:
            self.inject_temporal_errors(patients, cfg["temporal"])
        if cfg["window"] > 0:
            self.inject_window_errors(patients, protocol, cfg["window"])
        if cfg["comorbidity"] > 0:
            self.inject_comorbidity_overrides(patients, protocol, cfg["comorbidity"])

        self.rng.shuffle(patients)

        return {
            "protocol": protocol,
            "patients": patients,
            "ground_truth": dict(self._ground_truth),
            "total_errors": sum(len(v) for v in self._ground_truth.values()),
            "error_patients": list(self._ground_truth.keys()),
        }
