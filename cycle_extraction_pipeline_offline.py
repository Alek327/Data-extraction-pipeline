#!/usr/bin/env python
"""
cycle_extraction_pipeline_offline_v1_0_3.py

OFFLINE (0-API) qualitative coding + summarization for public health / intervention trials.

v1.0.3 (your requested final patches on top of v1.0.2):
- strict vs lenient mode (already)
- CSV columns: exclude_reason + is_excluded (already)
- Public-health / intervention-trial tuning (already)
- FIXES:
  A) Exclude theory-only framing from IMPACTS in strict mode (e.g., "we theorized...", TPB/TTM).
  B) Exclude outcome-measures-only blocks from IMPACTS in strict mode unless results language present.
  C) Prevent METHODS/BASELINE paragraphs from becoming FREQUENCIES in strict mode unless frequency-outcome language present.

Usage:
  python cycle_extraction_pipeline_offline_v1_0_3.py Bernshtein2017.pdf outputs\\segments.csv --markdown outputs\\summary.md --mode strict
  python cycle_extraction_pipeline_offline_v1_0_3.py Bernshtein2017.pdf outputs\\segments.csv --markdown outputs\\summary.md --mode lenient
"""

import os
import re
import argparse
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter

import pandas as pd
from tqdm import tqdm
from pypdf import PdfReader


# -----------------------------
# CONFIG
# -----------------------------

BATCH_SIZE = 50
MAX_SEGMENTS_PER_CATEGORY = 90
MAX_THEMES_PER_CATEGORY = 7

CATEGORIES = [
    "experiences",
    "perceptions",
    "purposes",
    "frequencies",
    "barriers",
    "enablers",
    "correlates_determinants",
    "interventions",
    "impacts",
    "none",
]

# Optional embeddings (offline). If unavailable, script falls back to rules-only.
USE_EMBEDDINGS_IF_AVAILABLE = True
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_ONLY_FOR_UNCERTAIN = True
EMBEDDING_MIN_SIMILARITY = 0.20

RULE_MIN_SCORE_TO_ASSIGN_BASE = 2.0
RULE_MARGIN_FOR_CONFIDENT_BASE = 1.0

EVIDENCE_MAX_CHARS = 420


# -----------------------------
# PUBLIC HEALTH / INTERVENTION-TRIAL TUNING
# -----------------------------

CYCLING_RELEVANCE_PATTERNS = [
    r"\b(utility cycling|transport cycling|cycling for transport|cycling to work|cycling to school)\b",
    r"\b(active (travel|mobility|transport))\b",
    r"\b(cycl(e|ing)|bicyc(le|ling)|bike|biking|cyclist(s)?)\b",
    r"\b(bike lane|cycle lane|cycle track|bike parking|bikeshare|bike share|e-?bike|pedelec)\b",
]

METHODS_HEAVY_PATTERNS = [
    r"\b(methods?|methodology|study design|protocol)\b",
    r"\b(we (used|use)|data (were|was) collected|analysis|model|regression|adjust(ed)? for)\b",
    r"\b(sample|participants? were recruited|randomi[sz]ed|controlled trial|pilot study)\b",
    r"\b(ethics approval|irb|consent|questionnaire|survey instrument|scale|measure(s)?|ipaq)\b",
    r"\b(baseline|follow-?up|timepoint|weeks?|months?)\b",
]

STRUCTURAL_NOISE_PATTERNS = [
    r"\b(references?|bibliography)\b",
    r"\b(doi\s*[:=]|doi:)\b",
    r"http[s]?://",
    r"\b(accessed\s+\w+)\b",
    r"\b(author affiliations?|affiliations?|corresponding author)\b",
    r"\b(acknowledg(e)?ments?|competing interests|grant information|fund(ing|ers))\b",
    r"\b(copyright|creative commons|cc-by)\b",
    r"\b(page\s+\d+\s+of\s+\d+)\b",
    r"\b(issn|publisher full text|pubmed abstract)\b",
]

# Eligibility exclusion ONLY if framed as eligibility/screening AND contains criteria.
ELIGIBILITY_FRAMING_PATTERNS = [
    r"\b(eligible participants?|eligibility)\b",
    r"\b(inclusion criteria|exclusion criteria)\b",
    r"\b(were eligible|were included if|were excluded if)\b",
    r"\b(screened for|screening)\b",
    r"\b(consort|participant flow|randomi[sz]ed)\b",
]
ELIGIBILITY_CRITERIA_PATTERNS = [
    r"\b(years old|aged\s+\d+)\b",
    r"\b(body mass index|bmi)\b",
    r"\b(physically inactive|inactive adults?)\b",
    r"\b(pregnan(t|cy))\b",
    r"\b(planning to relocate|planning to become pregnant)\b",
    r"\b(clearance|par-q|readiness questionnaire)\b",
]

BACKGROUND_PATTERNS = [
    r"\b(background|introduction)\b",
    r"\b(is inversely related|has been found to|research suggests|prior research)\b",
    r"\b(nationally|rates are increasing)\b",
]

IMPACT_HARD_RESULT_PATTERNS = [
    r"\b(p\s*[<=>]\s*0\.\d+)\b",
    r"\b(95%\s*ci|confidence interval)\b",
    r"\b(mean difference|odds ratio|relative risk|risk ratio|hazard ratio)\b",
    r"\b(baseline|follow-?up|week\s*\d+|\d+\s*weeks|\d+\s*months)\b",
    r"\b(intervention group|control group|compared to|versus|vs\.)\b",
    r"\b(increased significantly|decreased significantly|declined significantly)\b",
]

# Allow conclusions to count as impacts even without p-values/CI (strict mode) when explicitly stating effects.
IMPACT_CONCLUSION_ALLOW_PATTERNS = [
    r"\b(conclusion|conclusions|key findings|findings)\b",
    r"\b(this (trial|study|intervention) (increased|decreased|reduced|improved))\b",
    r"\b(decreased perceived barriers|increased bicycling|increased cycling)\b",
    r"\b(support(s)? a larger(-|\s)?scale study|supports further research)\b",
]

FREQUENCY_RESULT_PATTERNS = [
    r"\b(time spent|minutes/day|min/d|min/day|hours/week|times per week|days per week)\b",
    r"\b(bicycl(e|ing)|bike|biking)\b",
    r"\b(increased|decreased|change(d)?)\b",
    r"\b(compared to|versus|vs\.)\b",
    r"\b(p\s*[<=>]\s*0\.\d+|95%\s*ci)\b",
]

# v1.0.3: theory-only framing exclusion (strict)
THEORY_ONLY_PATTERNS = [
    r"\b(we theorized)\b",
    r"\b(theory of planned behavior)\b",
    r"\b(transtheoretical model)\b",
]

# v1.0.3: outcome-measures-only exclusion (strict)
OUTCOME_MEASURES_PATTERNS = [
    r"\boutcome measures?\b",
    r"\bdata were collected\b",
    r"\bthe following outcomes were collected\b",
    r"\boutcome measures were collected\b",
]


# -----------------------------
# CATEGORY PATTERNS
# -----------------------------

CATEGORY_PATTERNS: Dict[str, List[Tuple[str, float]]] = {
    "experiences": [
        (r"\b(experience(d)?|describ(e|ed)|reported|narrative|accounts?)\b", 1.6),
        (r"\b(near-?miss|crash|collision|injur(y|ies))\b", 2.2),
        (r"\b(harass(ment)?|police harassment|discrimination)\b", 1.8),
        (r"\b(while cycling|when cycling|during cycling)\b", 1.4),
    ],
    "perceptions": [
        (r"\b(perceiv(e|ed)|belief(s)?|attitude(s)?|view(s)?|think|felt|feel)\b", 1.8),
        (r"\b(safe|unsafe|dangerous|risk(y)?)\b", 1.7),
        (r"\b(stigma|status|low social status|embarrass|pride|empower)\b", 1.8),
        (r"\b(confiden(ce|t)|self-efficacy|fear)\b", 1.6),
        (r"\b(norms?|social norms?)\b", 1.2),
    ],
    "purposes": [
        (r"\b(commut(e|ing)|to work|for work)\b", 2.0),
        (r"\b(to school|university|college|education)\b", 1.6),
        (r"\b(shopping|grocer(y|ies)|errand(s)?|market)\b", 1.5),
        (r"\b(appointments?|clinic|hospital|healthcare)\b", 1.4),
        (r"\b(transport|travel|utility cycling|utilitarian)\b", 1.8),
        (r"\b(access(ing)? destinations?|access to)\b", 1.3),
    ],
    "frequencies": [
        (r"\b(how often|frequency|times per|days per|hours per|minutes per)\b", 2.1),
        (r"\b(time spent|min/d|min/day|minutes/day|hours/week|trips per)\b", 2.1),
        (r"\b(mode share|percent|%|proportion)\b", 1.8),
        (r"\b(baseline|follow-?up)\b", 1.0),
        (r"\b(bicycl(e|ing)|bike|biking)\b", 1.2),
    ],
    "barriers": [
        (r"\b(barrier(s)?|obstacle(s)?|constraint(s)?|prevent(ed)?|hinder(ed)?)\b", 2.1),
        (r"\b(traffic|speed(ing)?|close pass|driver aggression)\b", 2.0),
        (r"\b(safety from traffic|not feeling safe)\b", 2.1),
        (r"\b(crime|violence|harass(ment)?|assault|personal security)\b", 2.0),
        (r"\b(cost|expensive|afford|price)\b", 1.7),
        (r"\b(distance|too far|travel time)\b", 1.4),
        (r"\b(weather|rain|snow|heat)\b", 1.1),
        (r"\b(theft|stolen|bike theft)\b", 1.6),
        (r"\b(lack of (a )?bicycle|no bicycle)\b", 1.6),
        (r"\b(not feeling healthy enough|physically uncomfortable)\b", 1.3),
    ],
    "enablers": [
        (r"\b(enabl(e|ed)|facilitat(e|ed)|support(ed)?|help(ed)?|encourag(e|ed))\b", 2.0),
        (r"\b(protected lane|cycle track|bike lane|infrastructure|secure parking)\b", 1.8),
        (r"\b(program(me)?|scheme|initiative|subsid(y|ies)|voucher|loan)\b", 1.7),
        (r"\b(provided (a )?bicycle|helmets?|locks?)\b", 1.7),
        (r"\b(training|instruction|skills?)\b", 1.4),
        (r"\b(community|peer|group|social support)\b", 1.3),
        (r"\b(affordable|low cost)\b", 1.2),
    ],
    "correlates_determinants": [
        (r"\b(associated with|predict(ed|or)?|correlat(e|ion)|determinant(s)?)\b", 2.0),
        (r"\b(age|gender|sex|income|education|ethnic(ity)?|race|deprivation)\b", 1.8),
        (r"\b(neighborhood|area-level|urban|rural|density)\b", 1.3),
        (r"\b(infrastructure|access|built environment)\b", 1.2),
        (r"\b(disadvantage|low income|underserved|equity|inequit)\w*\b", 1.4),
    ],
    "interventions": [
        (r"\b(intervention|program(me)?|initiative|scheme|policy)\b", 2.0),
        (r"\b(training|education|promotion|campaign)\b", 1.6),
        (r"\b(bike loan|bikeshare|bike share|e-?bike)\b", 1.5),
        (r"\b(group session(s)?|instructor|curriculum)\b", 1.6),
        (r"\b(provided bicycles?|helmets?|locks?)\b", 1.7),
        (r"\b(implementation|delivered|fidelity)\b", 1.2),
    ],
    "impacts": [
        (r"\b(impact(s)?|effect(s)?|outcome(s)?|result(s)?)\b", 1.6),
        (r"\b(increased?|decreased?|reduced?|improved?)\b", 1.6),
        (r"\b(health|fitness|wellbeing|mental health)\b", 1.5),
        (r"\b(physical activity|active travel)\b", 1.2),
        (r"\b(equity|inclusion|access|justice)\b", 1.3),
        (r"\b(safety|injury|risk)\b", 1.2),
        (r"\b(p\s*[<=>]\s*0\.\d+|95%\s*ci)\b", 1.2),
    ],
}


# -----------------------------
# DATA STRUCTURE
# -----------------------------

@dataclass
class SegmentClassification:
    source_file: str
    segment_id: int
    category: str
    rationale: str
    exclude_reason: str
    text: str


# -----------------------------
# UTIL: safer overwrites on Windows
# -----------------------------

def safe_remove(path: Optional[str]) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError as e:
        raise PermissionError(
            f"Cannot write to '{path}'. It may be open in Excel or locked by another program. "
            f"Close it or choose a different output name/location."
        ) from e


# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    all_text = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            all_text.append(t)
    return "\n\n".join(all_text)


# -----------------------------
# SEGMENTATION
# -----------------------------

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_segments(text: str, min_len: int = 120, max_len: int = 900) -> List[str]:
    raw_segments = [s.strip() for s in text.split("\n\n") if s.strip()]

    segments: List[str] = []
    buffer = ""
    for seg in raw_segments:
        if not buffer:
            buffer = seg
        else:
            if len(buffer) < min_len:
                buffer = buffer + " " + seg
            else:
                segments.append(buffer.strip())
                buffer = seg
    if buffer:
        segments.append(buffer.strip())

    final_segments: List[str] = []
    for seg in segments:
        if len(seg) <= max_len:
            final_segments.append(seg)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", seg)
        current = ""
        for s in sentences:
            if not s.strip():
                continue
            if len(current) + len(s) < max_len:
                current += (" " if current else "") + s.strip()
            else:
                if current:
                    final_segments.append(current.strip())
                current = s.strip()
        if current:
            final_segments.append(current.strip())

    return [s for s in final_segments if len(s) >= 40]


# -----------------------------
# CLASSIFICATION HELPERS
# -----------------------------

def _regex_score(text: str, patterns: List[Tuple[str, float]]) -> Tuple[float, List[str]]:
    score = 0.0
    hits: List[str] = []
    for pat, w in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            score += w
            hits.append(pat)
    return score, hits


def _any_match(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def is_structural_noise(text: str) -> bool:
    return _any_match(text, STRUCTURAL_NOISE_PATTERNS)


def is_methods_heavy(text: str) -> bool:
    return _any_match(text, METHODS_HEAVY_PATTERNS)


def is_cycling_relevant(text: str) -> bool:
    return _any_match(text, CYCLING_RELEVANCE_PATTERNS)


def is_background_framing(text: str) -> bool:
    return _any_match(text, BACKGROUND_PATTERNS)


def has_hard_result_language(text: str) -> bool:
    return _any_match(text, IMPACT_HARD_RESULT_PATTERNS)


def has_conclusion_impact_language(text: str) -> bool:
    t = text.lower()
    has_conc = _any_match(t, IMPACT_CONCLUSION_ALLOW_PATTERNS)
    has_effect = bool(re.search(r"\b(increased|decreased|reduced|improved|declined)\b", t))
    has_intervention_context = bool(re.search(r"\b(intervention|program|trial|study)\b", t))
    return has_conc and has_effect and has_intervention_context


def has_frequency_result_language(text: str) -> bool:
    return _any_match(text, FREQUENCY_RESULT_PATTERNS)


def is_theoretical_framing(text: str) -> bool:
    return _any_match(text, THEORY_ONLY_PATTERNS)


def is_outcome_measures_only(text: str) -> bool:
    return _any_match(text, OUTCOME_MEASURES_PATTERNS)


def is_eligibility_text(text: str) -> bool:
    t = text.lower()
    framing = _any_match(t, ELIGIBILITY_FRAMING_PATTERNS)
    criteria = _any_match(t, ELIGIBILITY_CRITERIA_PATTERNS)
    return framing and criteria


def looks_like_table(text: str) -> bool:
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    if len(lines) < 4:
        return False

    col_like = sum(1 for l in lines if re.search(r"\S\s{2,}\S", l))
    col_ratio = col_like / max(1, len(lines))

    short_digit = sum(1 for l in lines if len(l) <= 80 and re.search(r"\d", l))
    short_digit_ratio = short_digit / max(1, len(lines))

    has_table_word = bool(re.search(r"\b(table|figure)\s*\d+\b", text, flags=re.IGNORECASE))
    has_many_percent_rows = sum(1 for l in lines if re.search(r"%", l)) >= 3

    narrative_markers = bool(
        re.search(
            r"\b(results?|conclusion|discussion|we found|declined significantly|increased significantly)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    if narrative_markers and col_ratio < 0.40:
        return False

    return (col_ratio >= 0.45) or (short_digit_ratio >= 0.70) or (has_table_word and col_ratio >= 0.25) or has_many_percent_rows


# -----------------------------
# RULE-BASED CLASSIFICATION
# -----------------------------

def classify_segment_rules(text: str, mode: str) -> Tuple[str, str, Dict[str, float], str]:
    """
    Returns (label, rationale, score_map, exclude_reason).
    exclude_reason is non-empty when label == "none".
    """

    if mode == "strict":
        rule_min_score = RULE_MIN_SCORE_TO_ASSIGN_BASE + 0.4
        rule_margin = RULE_MARGIN_FOR_CONFIDENT_BASE + 0.2
        exclude_tables = True
        exclude_eligibility = True
        require_hard_results_for_impacts = True
        require_result_for_frequencies = True
    else:
        rule_min_score = RULE_MIN_SCORE_TO_ASSIGN_BASE
        rule_margin = RULE_MARGIN_FOR_CONFIDENT_BASE
        exclude_tables = False
        exclude_eligibility = False
        require_hard_results_for_impacts = False
        require_result_for_frequencies = False

    # 1) Hard exclusions FIRST
    if is_structural_noise(text):
        return "none", "Excluded structural noise.", {}, "structural_noise"
    if exclude_eligibility and is_eligibility_text(text):
        return "none", "Excluded eligibility/inclusion criteria segment.", {}, "eligibility_or_inclusion_criteria"
    if exclude_tables and looks_like_table(text):
        return "none", "Excluded layout table/statistics block.", {}, "table_or_statistics_block"

    # 2) Relevance gate
    if not is_cycling_relevant(text):
        return "none", "No cycling/active travel relevance detected.", {}, "not_cycling_relevant"

    methods_heavy = is_methods_heavy(text)

    # 3) Score categories
    scores: Dict[str, float] = {}
    for cat in CATEGORIES:
        if cat == "none":
            continue
        s, _ = _regex_score(text, CATEGORY_PATTERNS.get(cat, []))
        scores[cat] = s

    sorted_cats = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_cat, top_score = sorted_cats[0]
    second_score = sorted_cats[1][1] if len(sorted_cats) > 1 else 0.0

    # 4) Weak evidence => none
    if top_score < rule_min_score:
        if methods_heavy and mode == "strict":
            return "none", "Methods/background without extractable findings (strict).", scores, "methods_or_background"
        return "none", "Cycling mentioned but no strong extractable cues.", scores, "weak_evidence"

    # 5) STRICT guards
    if mode == "strict":
        # A) Exclude theory-only framing from impacts
        if top_cat == "impacts" and is_theoretical_framing(text) and not has_hard_result_language(text):
            return "none", "Theoretical framing without observed outcomes (strict).", scores, "theoretical_framing"

        # Drop BACKGROUND framing from being called impacts
        if top_cat == "impacts" and is_background_framing(text):
            return "none", "Background/framing excluded (strict).", scores, "background_framing"

        # B) Exclude outcome-measures-only blocks from impacts unless there are results
        if top_cat == "impacts" and is_outcome_measures_only(text) and not has_hard_result_language(text):
            return "none", "Outcome measures description without results (strict).", scores, "outcome_measures_only"

        # Require trial-style results for impacts, BUT allow conclusion-style impact statements.
        if top_cat == "impacts" and require_hard_results_for_impacts:
            if not (has_hard_result_language(text) or has_conclusion_impact_language(text)):
                return "none", "No trial-style outcome evidence (strict).", scores, "impact_without_hard_results"

        # Frequencies must look like cycling time-use/frequency outcomes.
        if top_cat == "frequencies" and require_result_for_frequencies and not has_frequency_result_language(text):
            return "none", "Frequency label without cycling time-use evidence (strict).", scores, "frequency_without_timeuse_or_result_language"

        # C) Prevent methods/baseline paragraphs from becoming frequencies unless frequency-outcome language exists
        if top_cat == "frequencies" and methods_heavy and not has_frequency_result_language(text):
            return "none", "Methods/baseline statistics without frequency outcomes (strict).", scores, "methods_without_frequency_outcomes"

        # Methods-heavy gate: do NOT exclude if clearly results/intervention categories.
        allowed_even_if_methods_heavy = {"interventions", "impacts", "barriers", "enablers", "frequencies"}
        if methods_heavy and top_cat not in allowed_even_if_methods_heavy:
            return "none", "Methods/study-design segment excluded (strict).", scores, "methods_or_study_design"

    # 6) Ambiguity marking (embeddings can arbitrate if installed)
    margin = top_score - second_score
    if margin < rule_margin:
        return top_cat, f"Uncertain (close scores). Matched cues for {top_cat}.", scores, ""

    return top_cat, f"Matched cues for {top_cat} (score={top_score:.1f}).", scores, ""


# -----------------------------
# OPTIONAL OFFLINE EMBEDDINGS
# -----------------------------

def try_load_embedding_model():
    if not USE_EMBEDDINGS_IF_AVAILABLE:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception:
        return None


def build_category_prototypes() -> Dict[str, List[str]]:
    return {
        "experiences": [
            "Participants described experiences cycling in their neighborhood, including interactions with traffic and other road users.",
            "People reported day-to-day challenges cycling, including harassment or near-misses.",
        ],
        "perceptions": [
            "Cycling was perceived as unsafe or risky due to car traffic or personal security.",
            "Participants described stigma or social norms affecting cycling.",
        ],
        "purposes": [
            "Cycling was used for transport, including commuting and accessing essential destinations.",
            "Participants cycled for errands, appointments, and visiting friends or family.",
        ],
        "frequencies": [
            "Cycling frequency and time spent biking were reported, often comparing baseline and follow-up.",
            "Reported trips per week, minutes per day, or mode share outcomes for cycling.",
        ],
        "barriers": [
            "Barriers included traffic danger, lack of safe infrastructure, cost, theft, and personal security concerns.",
            "People avoided cycling due to distance, discomfort, and cultural perceptions.",
        ],
        "enablers": [
            "Enablers included provision of bicycles, training, social support, protected infrastructure, and secure parking.",
            "Programs reduced access barriers and increased confidence or skills.",
        ],
        "correlates_determinants": [
            "Cycling was associated with socioeconomic disadvantage, demographics, neighborhood context, and infrastructure access.",
            "Determinants included education, income, deprivation, and built environment factors.",
        ],
        "interventions": [
            "Interventions included training and promotion programs, bike loan schemes, provision of equipment, and infrastructure initiatives.",
            "Programs delivered sessions, instruction, and resources to increase transport cycling.",
        ],
        "impacts": [
            "Impacts included increased cycling time, reduced perceived barriers, improved wellbeing, and changes compared with controls.",
            "Reported outcomes often include statistically significant changes and follow-up comparisons.",
        ],
    }


def classify_segment_embeddings(model, text: str, proto_embeds: Dict[str, Any]) -> Tuple[str, str]:
    import numpy as np
    seg_emb = model.encode([text], normalize_embeddings=True)[0]
    sims: Dict[str, float] = {}
    for cat, emb_list in proto_embeds.items():
        sim = float(np.max(np.dot(emb_list, seg_emb)))
        sims[cat] = sim
    best_cat, best_sim = max(sims.items(), key=lambda kv: kv[1])
    if best_sim < EMBEDDING_MIN_SIMILARITY:
        return "none", f"Low semantic similarity (max={best_sim:.2f})."
    return best_cat, f"Semantic match to {best_cat} (sim={best_sim:.2f})."


def classify_batch_offline(
    batch_items: List[Dict[str, Any]],
    mode: str,
    emb_model=None,
    proto_embeds=None,
) -> List[Dict[str, Any]]:
    results = []
    for item in batch_items:
        seg_id = int(item["segment_id"])
        text = str(item["text"])

        rule_label, rule_rat, _scores, exclude_reason = classify_segment_rules(text, mode=mode)

        label = rule_label
        rationale = rule_rat

        if emb_model is not None and proto_embeds is not None:
            is_uncertain = rationale.lower().startswith("uncertain")
            hard_exclusions = {
                "structural_noise",
                "eligibility_or_inclusion_criteria",
                "table_or_statistics_block",
                "not_cycling_relevant",
                "methods_or_study_design",
                "methods_or_background",
                "background_framing",
                "theoretical_framing",
                "outcome_measures_only",
                "methods_without_frequency_outcomes",
            }
            eligible_for_embed = (not EMBEDDINGS_ONLY_FOR_UNCERTAIN) or is_uncertain
            if exclude_reason in hard_exclusions:
                eligible_for_embed = False

            if eligible_for_embed:
                emb_label, emb_rat = classify_segment_embeddings(emb_model, text, proto_embeds)
                if emb_label != "none":
                    label = emb_label
                    rationale = f"{rule_rat} | {emb_rat}"
                    exclude_reason = ""

        if label not in CATEGORIES:
            label = "none"
            if not exclude_reason:
                exclude_reason = "invalid_label"

        results.append(
            {
                "segment_id": seg_id,
                "label": label,
                "rationale": rationale,
                "exclude_reason": exclude_reason if label == "none" else "",
            }
        )

    return results


def classify_all_segments(segments: List[str], source_name: str, mode: str) -> List["SegmentClassification"]:
    records: List[SegmentClassification] = []
    indexed = [{"segment_id": i + 1, "text": seg} for i, seg in enumerate(segments)]

    emb_model = try_load_embedding_model()
    proto_embeds = None
    if emb_model is not None:
        prototypes = build_category_prototypes()
        proto_embeds = {cat: emb_model.encode(examples, normalize_embeddings=True) for cat, examples in prototypes.items()}
        print(f"Embeddings enabled (offline): {EMBEDDING_MODEL_NAME}")
    else:
        print("Embeddings disabled: running rules-only classifier (offline).")

    for i in tqdm(range(0, len(indexed), BATCH_SIZE), desc=f"Classifying (offline, {mode})"):
        batch = indexed[i:i + BATCH_SIZE]
        results = classify_batch_offline(batch, mode=mode, emb_model=emb_model, proto_embeds=proto_embeds)

        text_lookup = {item["segment_id"]: item["text"] for item in batch}
        for res in results:
            seg_id = int(res["segment_id"])
            label = str(res["label"])
            rationale = str(res["rationale"])
            exclude_reason = str(res.get("exclude_reason", ""))

            records.append(
                SegmentClassification(
                    source_file=source_name,
                    segment_id=seg_id,
                    category=label,
                    rationale=rationale,
                    exclude_reason=exclude_reason,
                    text=text_lookup.get(seg_id, ""),
                )
            )

    records.sort(key=lambda r: r.segment_id)
    return records


# -----------------------------
# OFFLINE SUMMARISATION
# -----------------------------

STOPWORDS = set("""
a an and are as at be by for from has have he her his i in is it its of on or that the their there they this to was were will with
""".split())

GENERIC_THEME_BLOCKLIST = set([
    "physical activity", "control group", "intervention group", "participant characteristics",
    "study design", "data collection", "research methods", "original research",
    "author affiliations", "competing interests", "grant information",
])

THEME_PREFERENCE_TOKENS = [
    "increase", "increased", "decrease", "decreased", "reduce", "reduced", "improve", "improved",
    "significant", "compared", "versus", "difference",
    "utility", "transport", "commute", "access", "destinations",
    "traffic", "safe", "unsafe", "theft", "security", "harassment", "cost", "infrastructure",
    "training", "program", "sessions", "bicycle", "helmet", "lock", "parking",
    "time spent", "minutes", "days per", "times per",
]


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) >= 3]


def _phrase_has_preference(phrase: str) -> bool:
    p = phrase.lower()
    return any(tok in p for tok in THEME_PREFERENCE_TOKENS)


def _top_phrases(texts: List[str], ngram: int = 2, topk: int = 20) -> List[str]:
    counts = Counter()
    for t in texts:
        toks = _tokenize(t)
        if len(toks) < ngram:
            continue
        for i in range(len(toks) - ngram + 1):
            phrase = " ".join(toks[i:i + ngram])
            counts[phrase] += 1

    preferred = []
    general = []
    for phrase, _c in counts.most_common(topk * 10):
        if phrase in GENERIC_THEME_BLOCKLIST:
            continue
        if any(x in phrase for x in ["study", "participants", "data", "table", "figure", "copyright", "journal", "page"]):
            continue
        if _phrase_has_preference(phrase):
            preferred.append(phrase)
        else:
            general.append(phrase)

    out = preferred[:topk]
    if len(out) < topk:
        out.extend([p for p in general if p not in out][: (topk - len(out))])
    return out[:topk]


def summarise_all_categories(df: pd.DataFrame, mode: str) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {cat: [] for cat in CATEGORIES if cat != "none"}

    for cat in out.keys():
        subset = df[(df["category"] == cat)].sort_values("segment_id")
        if subset.empty:
            out[cat] = []
            continue

        rows = subset.head(MAX_SEGMENTS_PER_CATEGORY)
        texts = [str(x) for x in rows["text"].tolist()]

        phrases = _top_phrases(texts, ngram=2, topk=MAX_THEMES_PER_CATEGORY * 3)
        if not phrases:
            kw = Counter([w for t in texts for w in _tokenize(t)]).most_common(MAX_THEMES_PER_CATEGORY * 4)
            phrases = [w for w, _ in kw if w not in STOPWORDS]

        themes_added = 0
        used_seg_ids = set()

        for ph in phrases:
            if themes_added >= MAX_THEMES_PER_CATEGORY:
                break

            ph_lc = ph.lower().strip()
            if ph_lc in GENERIC_THEME_BLOCKLIST:
                continue

            best_row = None
            for _, r in rows.iterrows():
                txt = str(r["text"])
                if len(txt.strip()) < 90:
                    continue
                if re.search(rf"\b{re.escape(ph)}\b", txt, flags=re.IGNORECASE):
                    sid = int(r["segment_id"])
                    if sid not in used_seg_ids:
                        best_row = r
                        break

            if best_row is None:
                continue

            used_seg_ids.add(int(best_row["segment_id"]))
            theme = ph.strip().capitalize()

            evidence = str(best_row["text"]).strip()
            if len(evidence) > EVIDENCE_MAX_CHARS:
                evidence = evidence[:EVIDENCE_MAX_CHARS - 1].rsplit(" ", 1)[0] + "…"

            if is_structural_noise(evidence):
                continue
            if mode == "strict" and looks_like_table(evidence):
                continue

            out[cat].append({"theme": theme, "evidence": evidence})
            themes_added += 1

        if not out[cat]:
            r0 = rows.iloc[0]
            ev = str(r0["text"]).strip()
            if len(ev) > EVIDENCE_MAX_CHARS:
                ev = ev[:EVIDENCE_MAX_CHARS - 1].rsplit(" ", 1)[0] + "…"
            if not is_structural_noise(ev):
                out[cat] = [{"theme": "Representative evidence", "evidence": ev}]
            else:
                out[cat] = []

    return out


# -----------------------------
# MARKDOWN OUTPUT
# -----------------------------

def write_markdown_matrix(source_name: str, summaries_by_cat: Dict[str, Any], markdown_path: str, mode: str) -> None:
    lines: List[str] = []
    lines.append(f"# Summarized evidence matrix for {source_name}")
    lines.append("")
    lines.append(f"> Generated offline (no external API). Mode: **{mode}**.")
    lines.append("")

    for cat in CATEGORIES:
        if cat == "none":
            continue
        entries = summaries_by_cat.get(cat, [])
        if not entries:
            continue

        pretty_name = cat.capitalize().replace("_", " / ")
        lines.append(f"## {pretty_name}")
        lines.append("")
        lines.append("| Theme | Evidence from Study |")
        lines.append("|-------|---------------------|")
        for item in entries:
            theme = str(item.get("theme", "")).replace("|", "\\|").strip()
            evidence = str(item.get("evidence", "")).replace("|", "\\|").strip()
            if not theme and not evidence:
                continue
            lines.append(f"| {theme} | {evidence} |")
        lines.append("")

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# PIPELINE
# -----------------------------

def process_pdf(pdf_path: str, output_csv: str, markdown_path: Optional[str], mode: str) -> None:
    raw_text = extract_text_from_pdf(pdf_path)
    raw_text = normalize_whitespace(raw_text)

    segments = split_into_segments(raw_text)
    source_name = os.path.basename(pdf_path)
    print(f"Extracted {len(segments)} segments from {source_name}")

    records = classify_all_segments(segments, source_name, mode=mode)
    df = pd.DataFrame([asdict(r) for r in records])

    df["is_excluded"] = df["category"].eq("none")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    safe_remove(output_csv)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved segment-level classifications to {output_csv}")

    if markdown_path:
        os.makedirs(os.path.dirname(markdown_path) or ".", exist_ok=True)
        safe_remove(markdown_path)
        included_df = df[df["category"] != "none"].copy()
        summaries_by_cat = summarise_all_categories(included_df, mode=mode)
        write_markdown_matrix(source_name, summaries_by_cat, markdown_path, mode=mode)
        print(f"Saved summarized Markdown matrix to {markdown_path}")

    exc_counts = df["exclude_reason"].value_counts(dropna=False)
    if not exc_counts.empty:
        print("\nExclusion reasons (top):")
        print(exc_counts.head(12).to_string())


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OFFLINE qualitative coding + summarisation for public health cycling trials (v1.0.3, 0 API)."
    )
    parser.add_argument("pdf_path", help="Input PDF file")
    parser.add_argument("output_csv", help="Output CSV file for coded segments")
    parser.add_argument("--markdown", default=None, help="Optional output Markdown file with summarized category tables")
    parser.add_argument(
        "--mode",
        choices=["strict", "lenient"],
        default="strict",
        help="strict = publication-ready (more exclusions), lenient = early mapping (fewer exclusions)",
    )
    args = parser.parse_args()
    process_pdf(args.pdf_path, args.output_csv, markdown_path=args.markdown, mode=args.mode)


if __name__ == "__main__":
    main()
