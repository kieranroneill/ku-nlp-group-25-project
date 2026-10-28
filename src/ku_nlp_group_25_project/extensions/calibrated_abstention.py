
"""
abstention.py -- Week 43 individual extension: calibrated abstention for
multilingual extractive QA (TyDi QA / XOR-RC, questions: ar/ko/te, context: English).

numpy + pandas for all decision-rule and metric logic, so the analysis cells
rerun in seconds on CPU after the (expensive) score cache has been built once
with `score_examples`.

Conventions
-----------
For every validation example we store two numbers from ONE forward pass of the
same frozen Week 40/41 multilingual QA checkpoint:

    s_null      = start_logits[0] + end_logits[0]        (no-answer / CLS span)
    s_best      = max_{0 <= i <= j < n_ctx, j-i < MAX_ANS} start[i] + end[j]
    margin      = s_null - s_best                          (larger => more "no answer")

Decision rule shared by every condition:  ABSTAIN iff margin > t.

Every condition only differs in *how t is chosen* (or in what the score is
before the threshold is applied), so all differences are attributable to the
decision rule.
"""

from __future__ import annotations

import json
import math
import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LANGS = ("ar", "ko", "te")


# --------------------------------------------------------------------------- #
# Text / metric primitives (SQuAD-v2 conventions)
# --------------------------------------------------------------------------- #
def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD v2 style)."""
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def squad_em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def squad_f1(pred: str, gold: str) -> float:
    p_tok, g_tok = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p_tok or not g_tok:
        return float(p_tok == g_tok)
    common = Counter(p_tok) & Counter(g_tok)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision, recall = n_same / len(p_tok), n_same / len(g_tok)
    return 2 * precision * recall / (precision + recall)


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(np.asarray(a, dtype=float)),
                     where=(np.asarray(b) != 0))


# --------------------------------------------------------------------------- #
# Threshold search
# --------------------------------------------------------------------------- #
def threshold_sweep(margins: np.ndarray, labels: np.ndarray):
    """Exact O(n log n) sweep of every achievable decision boundary.

    Returns (thresholds, macro_f1) of length n+1, where thresholds[k] means
    "answerable iff margin <= thresholds[k]" (k examples answered).
    """
    margins = np.asarray(margins, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(margins)
    order = np.argsort(margins, kind="mergesort")
    m_sorted, y_sorted = margins[order], labels[order]

    k = np.arange(n + 1)
    tp = np.concatenate([[0], np.cumsum(y_sorted)]).astype(float)
    pos = y_sorted.sum()
    fn = pos - tp
    fp = k - tp
    tn = n - pos - fp

    f1_ans = _safe_div(2 * tp, 2 * tp + fp + fn)
    f1_un = _safe_div(2 * tn, 2 * tn + fn + fp)
    macro = 0.5 * (f1_ans + f1_un)

    thresholds = np.concatenate([[m_sorted[0] - 1.0],
                                 (m_sorted[:-1] + m_sorted[1:]) / 2.0,
                                 [m_sorted[-1] + 1.0]])
    return thresholds, macro


def best_threshold(margins: np.ndarray, labels: np.ndarray,
                   tie_break: str = "median") -> Tuple[float, float]:
    """Threshold maximising answerability macro-F1.

    Ties are broken at the median of the tied plateau, not at an arbitrary edge,
    because plateaus are common on small per-language validation subsets.
    """
    thresholds, macro = threshold_sweep(margins, labels)
    best = macro.max()
    tied = np.flatnonzero(np.isclose(macro, best, atol=1e-12))
    k = {"median": int(np.median(tied)), "conservative": int(tied.min()),
         "liberal": int(tied.max())}[tie_break]
    return float(thresholds[k]), float(macro[k])


def apply_threshold(margins: np.ndarray, threshold: float) -> np.ndarray:
    """Return boolean `answerable_pred` (True = the system answers)."""
    return np.asarray(margins, dtype=float) <= float(threshold)


def per_language_thresholds(df: pd.DataFrame, margin_col: str = "margin",
                            fallback: Optional[float] = None) -> Dict[str, float]:
    """One threshold per language; pooled fallback if a language has one class."""
    if fallback is None:
        fallback, _ = best_threshold(df[margin_col].to_numpy(), df["answerable"].to_numpy())
    out = {"global": fallback}
    for lang, sub in df.groupby("lang"):
        y = sub["answerable"].to_numpy()
        out[lang], _ = (best_threshold(sub[margin_col].to_numpy(), y)
                        if len(np.unique(y)) > 1 else (fallback, np.nan))
    return out


def apply_lang_thresholds(df: pd.DataFrame, thresholds: Dict[str, float],
                          margin_col: str = "margin") -> np.ndarray:
    t = df["lang"].map(lambda l: thresholds.get(l, thresholds["global"])).to_numpy(dtype=float)
    return df[margin_col].to_numpy(dtype=float) <= t


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(df: pd.DataFrame, answerable_pred: np.ndarray) -> Dict[str, float]:
    """Answerability PRF/macro-F1 + SQuAD-v2 EM/F1 + coverage."""
    y = df["answerable"].to_numpy().astype(int)
    p = np.asarray(answerable_pred).astype(int)
    em_span = df["em"].to_numpy(dtype=float)
    f1_span = df["f1"].to_numpy(dtype=float)

    em = np.where(p == 1, em_span, 0.0)
    f1 = np.where(p == 1, f1_span, 0.0)
    out = _prf(y, p)
    out.update({
        "n": float(len(df)),
        "EM": float(em.mean()) if len(df) else 0.0,
        "F1": float(f1.mean()) if len(df) else 0.0,
        "coverage_answerable": float(p[y == 1].mean()) if (y == 1).any() else float("nan"),
        "abstention_recall": float(1 - p[y == 0].mean()) if (y == 0).any() else float("nan"),
        "answered_precision": float(y[p == 1].mean()) if (p == 1).any() else float("nan"),
    })
    return out


def _prf(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    tp = float(((y == 1) & (p == 1)).sum())
    fp = float(((y == 0) & (p == 1)).sum())
    fn = float(((y == 1) & (p == 0)).sum())
    tn = float(((y == 0) & (p == 0)).sum())
    prec_a = tp / (tp + fp) if (tp + fp) else 0.0
    rec_a = tp / (tp + fn) if (tp + fn) else 0.0
    f1_a = 2 * prec_a * rec_a / (prec_a + rec_a) if (prec_a + rec_a) else 0.0
    prec_u = tn / (tn + fn) if (tn + fn) else 0.0
    rec_u = tn / (tn + fp) if (tn + fp) else 0.0
    f1_u = 2 * prec_u * rec_u / (prec_u + rec_u) if (prec_u + rec_u) else 0.0
    return {
        "macro_f1": 0.5 * (f1_a + f1_u),
        "answerable_precision": prec_a, "answerable_recall": rec_a, "answerable_f1": f1_a,
        "unanswerable_precision": prec_u, "unanswerable_recall": rec_u, "unanswerable_f1": f1_u,
        "accuracy": (tp + tn) / max(len(y), 1),
    }


def evaluate_by_language(df: pd.DataFrame, answerable_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for lang in sorted(df["lang"].unique()):
        mask = (df["lang"] == lang).to_numpy()
        row = {"lang": lang}
        row.update(evaluate(df[mask].reset_index(drop=True), np.asarray(answerable_pred)[mask]))
        rows.append(row)
    flat = {"lang": "ALL"}
    flat.update(evaluate(df, answerable_pred))
    rows.append(flat)
    return pd.DataFrame(rows)


def transition_counts(df: pd.DataFrame, pred_a: np.ndarray, pred_b: np.ndarray) -> pd.DataFrame:
    """Paired per-example changes between two decision rules (rescues vs breaks)."""
    y = df["answerable"].to_numpy().astype(int)
    a, b = np.asarray(pred_a).astype(int), np.asarray(pred_b).astype(int)
    kind = np.where((a == 0) & (b == 1) & (y == 1), "rescued_answerable",
            np.where((a == 0) & (b == 1) & (y == 0), "newly_wrong_answer",
            np.where((a == 1) & (b == 0) & (y == 0), "rescued_abstention",
            np.where((a == 1) & (b == 0) & (y == 1), "newly_wrong_abstention", "unchanged"))))
    out = pd.DataFrame({"lang": df["lang"].to_numpy(), "kind": kind})
    tab = out.groupby(["lang", "kind"]).size().unstack(fill_value=0)
    tab.loc["ALL"] = tab.sum()
    for col in ["rescued_abstention", "newly_wrong_abstention",
                "rescued_answerable", "newly_wrong_answer", "unchanged"]:
        if col not in tab:
            tab[col] = 0
    return tab[["rescued_abstention", "newly_wrong_abstention",
                "rescued_answerable", "newly_wrong_answer", "unchanged"]]


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def answer_prob(margins: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Two-way softmax over {no-answer, best span} with a single temperature."""
    return 1.0 / (1.0 + np.exp(np.asarray(margins, dtype=float) / temperature))


def nll(margins: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    p = np.clip(answer_prob(margins, temperature), 1e-12, 1 - 1e-12)
    y = np.asarray(labels, dtype=float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_temperature(margins: np.ndarray, labels: np.ndarray,
                    lo: float = 0.05, hi: float = 50.0) -> float:
    """Golden-section search over log T minimising validation NLL."""
    a, b = math.log(lo), math.log(hi)
    gr = (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(80):
        if nll(margins, labels, math.exp(c)) < nll(margins, labels, math.exp(d)):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return math.exp((a + b) / 2)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - probs[m].mean())
    return float(total)


def reliability_table(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
                         "confidence": float(probs[m].mean()), "accuracy": float(y[m].mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Feature-based post-hoc abstention (logistic regression, out-of-fold)
# --------------------------------------------------------------------------- #
FEATURES = ["margin", "log_ctx_tokens", "log_q_tokens", "overlap_frac", "span_len"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_ctx_tokens"] = np.log1p(df["n_ctx_tokens"].astype(float))
    df["log_q_tokens"] = np.log1p(df["n_q_tokens"].astype(float))
    return df


def oof_logreg(df: pd.DataFrame, n_splits: int = 5, seed: int = 13,
               features: Sequence[str] = FEATURES) -> np.ndarray:
    """Out-of-fold P(answerable); cross-fitting keeps the rule from seeing the
    label of the example it scores, so the threshold search stays honest."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = df[list(features)].to_numpy(dtype=float)
    y = df["answerable"].to_numpy(dtype=int)
    oof = np.zeros(len(df))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, class_weight="balanced"))
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof


# --------------------------------------------------------------------------- #
# Alternative decision criterion: constrained operating point
# --------------------------------------------------------------------------- #
def threshold_at_abstention_recall(margins: np.ndarray, labels: np.ndarray,
                                   target_recall: float = 0.90) -> float:
    """Largest threshold whose unanswerable-class recall reaches `target_recall`.

    A different criterion, not just a re-tuned scalar: the operating point is
    fixed by a stated reliability constraint.
    """
    thresholds, _ = threshold_sweep(margins, labels)
    y = np.asarray(labels, dtype=int)
    neg = margins[y == 0]
    if len(neg) == 0:
        return float(thresholds[-1])
    cand = thresholds[thresholds <= np.quantile(neg, 1 - target_recall)]
    return float(cand.max() if len(cand) else thresholds[0])


def risk_coverage(df: pd.DataFrame, answerable_pred: np.ndarray,
                  confidence: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Risk (1 - EM) of answered examples as coverage grows."""
    if confidence is None:
        confidence = -df["margin"].to_numpy(dtype=float)
    em = df["em"].to_numpy(dtype=float)
    order = np.argsort(-np.asarray(confidence, dtype=float))
    hits = np.cumsum(em[order]) / np.arange(1, len(df) + 1)
    rows = []
    for cov in np.linspace(0.05, 1.0, 20):
        k = max(int(round(cov * len(df))), 1)
        rows.append({"coverage": k / len(df), "risk": float(1 - hits[k - 1])})
    return pd.DataFrame(rows)


def aurc(curve: pd.DataFrame) -> float:
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(curve["risk"].to_numpy(), curve["coverage"].to_numpy()))


# --------------------------------------------------------------------------- #
# Scoring a frozen QA checkpoint (GPU section; run once, cache to disk)
# --------------------------------------------------------------------------- #
@dataclass
class ScoringConfig:
    model_name: str = "deepset/xlm-roberta-base-squad2"
    max_length: int = 384
    stride: int = 128
    max_answer_len: int = 30
    batch_size: int = 16
    max_examples_per_lang: Optional[int] = None
    seed: int = 13


def select_model_inputs(question: str, context: str, tokenizer, cfg: ScoringConfig):
    return tokenizer(
        question, context,
        truncation="only_second",
        max_length=cfg.max_length,
        stride=cfg.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )


def score_examples(examples: Iterable[dict], model, tokenizer, cfg: ScoringConfig,
                   device: str = "cuda") -> pd.DataFrame:
    """Run the frozen model; one row per example with s_null / s_best / margin.

    Windowing follows the standard SQuAD-2 recipe: the no-answer score is the MIN
    over windows, the answer score the MAX over valid spans inside the (English)
    context, answers capped at `max_answer_len` tokens.
    """
    import torch

    model.eval().to(device)
    rows = []
    for ex in examples:
        feats = select_model_inputs(ex["question"], ex["context"], tokenizer, cfg)
        offset = feats.pop("offset_mapping")
        feats.pop("overflow_to_sample_mapping", None)   # must not reach the model
        feats = {k: v.to(device) for k, v in feats.items()}
        with torch.no_grad():
            out = model(**feats)
        s_start = out.start_logits.cpu().numpy()
        s_end = out.end_logits.cpu().numpy()
        seq_ids = feats["input_ids"].cpu().numpy()

        null_scores, best = [], (-1e9, 0, 0)
        for w in range(s_start.shape[0]):
            off = offset[w].numpy()
            ctx_mask = _context_mask(seq_ids[w], off, tokenizer)
            if not ctx_mask.any():
                continue
            null_scores.append(float(s_start[w, 0] + s_end[w, 0]))
            idx = np.flatnonzero(ctx_mask)
            for i in idx:
                js = np.arange(i, min(i + cfg.max_answer_len, idx.max() + 1))
                tot = s_start[w, i] + s_end[w, js]
                k = int(np.argmax(tot))
                if tot[k] > best[0]:
                    best = (float(tot[k]), int(off[i, 0]), int(off[js[k], 1]))
        s_null = min(null_scores) if null_scores else 0.0
        s_best, cs, ce = best
        pred = ex["context"][cs:ce] if s_best > -1e8 else ""
        gold = ex.get("answer", "")
        is_ans = ex.get("answerable", 0)
        em = squad_em(pred, gold) if is_ans else float(pred == "")
        f1 = squad_f1(pred, gold) if is_ans else float(pred == "")
        rows.append({
            "id": ex.get("id"), "lang": ex["lang"], "answerable": int(bool(is_ans)),
            "pairid": ex.get("pairid"), "question": ex["question"],
            "gold_answer": gold, "pred_span": pred,
            "answer_start": ex.get("answerstart", ex.get("answer_start", -1)),
            "s_null": s_null, "s_best": s_best, "margin": s_null - s_best,
            "n_ctx_tokens": int(len(tokenizer(ex["context"], add_special_tokens=False)["input_ids"])),
            "n_q_tokens": int(len(tokenizer(ex["question"], add_special_tokens=False)["input_ids"])),
            "overlap_frac": _overlap_frac(ex["question"], ex["context"]),
            "span_len": int(len(pred.split())), "em": em, "f1": f1,
        })
    return pd.DataFrame(rows)


def _context_mask(seq_ids, offsets, tokenizer) -> np.ndarray:
    """True for tokens of the second sequence (the English context)."""
    mask = np.zeros(len(seq_ids), dtype=bool)
    sep = tokenizer.sep_token_id
    sep_pos = np.flatnonzero(seq_ids == sep)
    if len(sep_pos) >= 1:
        mask[int(sep_pos[0]) + 1:] = True
    mask &= (offsets[:, 0] < offsets[:, 1])       # drops specials and padding
    return mask


def _overlap_frac(question: str, context: str) -> float:
    q = set(normalize_answer(question).split())
    c = set(normalize_answer(context).split())
    return float(len(q & c) / len(q)) if q else 0.0


# --------------------------------------------------------------------------- #
# Orchestration: all conditions on one cached score table
# --------------------------------------------------------------------------- #
def run_conditions(df: pd.DataFrame, target_recall: float = 0.90, seed: int = 13) -> dict:
    """Conditions A-F. Everything is tuned on `df` (validation) only."""
    df = add_features(df)
    y = df["answerable"].to_numpy()
    m = df["margin"].to_numpy(dtype=float)
    res = {"conditions": {}, "thresholds": {}, "extra": {}}

    # A: single pooled threshold (core baseline)
    t_glob, _ = best_threshold(m, y)
    pred_a = df["margin"].to_numpy() <= t_glob
    res["thresholds"]["A_classic"] = {"global": t_glob}

    # B: per-language thresholds
    th_b = per_language_thresholds(df, fallback=t_glob)
    pred_b = apply_lang_thresholds(df, th_b)
    res["thresholds"]["B_per_language"] = th_b

    # C: temperature scaling, then one pooled threshold on the calibrated p
    T = fit_temperature(m, y)
    p_raw = answer_prob(m, 1.0)
    p_cal = answer_prob(m, T)
    t_p, _ = best_threshold(-p_cal, y)
    pred_c = (-p_cal) <= t_p
    res["thresholds"]["C_temperature"] = {"T": T, "neg_p_threshold": t_p}
    res["extra"]["ece_raw"] = ece(p_raw, y)
    res["extra"]["ece_temperature"] = ece(p_cal, y)
    res["extra"]["temperature"] = T

    # D: constrained operating point (abstention recall >= target), per language
    th_d = {"global": threshold_at_abstention_recall(m, y, target_recall)}
    for lang, sub in df.groupby("lang"):
        th_d[lang] = threshold_at_abstention_recall(
            sub["margin"].to_numpy(dtype=float), sub["answerable"].to_numpy(), target_recall)
    pred_d = apply_lang_thresholds(df, th_d)
    res["thresholds"]["D_constrained"] = th_d

    # E: feature-based post-hoc combiner, out-of-fold
    oof = oof_logreg(df, seed=seed)
    t_e, _ = best_threshold(-oof, y)
    pred_e = (-oof) <= t_e
    res["extra"]["logreg_auc"] = _auc(oof, y)
    res["thresholds"]["E_features"] = {"neg_p_threshold": t_e}

    # F: fixed 0.5 cut on the calibrated probability (NOT tuned on validation;
    # a deployment ablation, never used for model selection)
    pred_f = p_cal >= 0.5

    preds = {"A_classic": pred_a, "B_per_language": pred_b,
             "C_temperature": pred_c, "D_constrained": pred_d,
             "E_features": pred_e, "F_fixed_prob": pred_f}
    for name, pred in preds.items():
        res["conditions"][name] = evaluate(df, pred)
        res.setdefault("by_language", {})[name] = evaluate_by_language(df, pred)

    res["curve_A"] = risk_coverage(df, pred_a, confidence=-m)
    res["curve_B"] = risk_coverage(df, pred_b, confidence=-m)
    res["extra"]["aurc_A"] = aurc(res["curve_A"])
    res["extra"]["aurc_B"] = aurc(res["curve_B"])
    res["extra"]["aurc_by_lang"] = {
        lang: aurc(risk_coverage(sub.reset_index(drop=True),
                                 pred_b[(df["lang"] == lang).to_numpy()],
                                 confidence=-sub["margin"].to_numpy(dtype=float)))
        for lang, sub in df.groupby("lang")}
    res["bootstrap_AB"] = paired_bootstrap(df, pred_a, pred_b, seed=seed)
    res["transitions_AB"] = transition_counts(df, pred_a, pred_b)
    res["transitions_AD"] = transition_counts(df, pred_a, pred_d)
    res["reliability_raw"] = reliability_table(p_raw, y)
    res["reliability_cal"] = reliability_table(p_cal, y)
    res["predictions"] = df.assign(
        **{f"pred_{k}": v.astype(int) for k, v in preds.items()},
        p_raw=p_raw, p_cal=p_cal, p_features=oof)
    return res


def paired_bootstrap(df: pd.DataFrame, pred_a: np.ndarray, pred_b: np.ndarray,
                     n_boot: int = 2000, seed: int = 13) -> dict:
    """Paired bootstrap over examples of the macro-F1 difference (B - A).

    Reported as a paired uncertainty interval, not as a significance test.
    """
    rng = np.random.default_rng(seed)
    y = df["answerable"].to_numpy().astype(int)
    a, b = np.asarray(pred_a).astype(int), np.asarray(pred_b).astype(int)
    idx = np.arange(len(df))
    obs = _prf(y, b)["macro_f1"] - _prf(y, a)["macro_f1"]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        diffs[i] = _prf(y[s], b[s])["macro_f1"] - _prf(y[s], a[s])["macro_f1"]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta_macro_f1": float(obs), "ci_low": float(lo), "ci_high": float(hi),
            "frac_replicates_not_better": float((diffs <= 0).mean())}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    y = np.asarray(labels)[order]
    ranks = np.arange(1, len(y) + 1)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
