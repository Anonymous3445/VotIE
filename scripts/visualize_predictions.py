#!/usr/bin/env python3
"""
VotIE Prediction Explorer — interactive Streamlit app.

Usage:
    pip install streamlit
    streamlit run scripts/visualize_predictions.py
"""

import sys
import json
import html as _html
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import streamlit as st
except ImportError:
    sys.exit(
        "Streamlit not installed.\n"
        "  pip install streamlit\n"
        "  streamlit run scripts/visualize_predictions.py"
    )

import pandas as pd

from scripts.evaluate import load_predictions as _norm_load, bio_to_spans
from scripts.error_classification import ErrorClassifier, Entity, ErrorCase
from src.llm_extraction.shared.data_utils import load_jsonl

# ── Paths ─────────────────────────────────────────────────────────────────────
PRED_DIR = REPO_ROOT / "predictions"
EVAL_DIR = REPO_ROOT / "evaluation"
DATA_DIR = REPO_ROOT / "data"

GOLD_OPTIONS: Dict[str, Path] = {
    "v6/test.jsonl":                  DATA_DIR / "citilink-votie" / "test.jsonl",
    "v6/test_gold_with_sub_subjects": DATA_DIR / "citilink-votie" / "test_gold_with_sub_subjects.jsonl",
}

# ── Styling ───────────────────────────────────────────────────────────────────
ENTITY_COLORS: Dict[str, str] = {
    "VOTER-FAVOR":        "#1b5e20",
    "VOTER-AGAINST":      "#b71c1c",
    "VOTER-ABSTENTION":   "#e65100",
    "VOTER-ABSENT":       "#37474f",
    "VOTING":             "#0d47a1",
    "SUBJECT":            "#4a148c",
    "COUNTING-UNANIMITY": "#004d40",
    "COUNTING-MAJORITY":  "#3e2723",
    "COUNT-FAVOR":        "#33691e",
    "COUNT-BLANK":        "#546e7a",
    "VOTING-METHOD":      "#880e4f",
    "SUB-SUBJECT":        "#7b1fa2",
}
_DEF_COLOR = "#616161"

ABBREVS: Dict[str, str] = {
    "VOTER-FAVOR": "VF", "VOTER-AGAINST": "VA",
    "VOTER-ABSTENTION": "VAb", "VOTER-ABSENT": "VAs",
    "VOTING": "VOT", "SUBJECT": "SUB",
    "COUNTING-UNANIMITY": "CU", "COUNTING-MAJORITY": "CM",
    "COUNT-FAVOR": "CF", "COUNT-BLANK": "CB",
    "VOTING-METHOD": "VM", "SUB-SUBJECT": "SS",
}

ERROR_LABELS = {
    "CORRECT": "Correct",
    "MIS": "Missed (FN)",
    "SPU": "Spurious (FP)",
    "INC_BOUNDARY": "Boundary error",
    "INC_TYPE": "Type error",
}

VOTER_TYPES   = {"VOTER-FAVOR", "VOTER-AGAINST", "VOTER-ABSTENTION", "VOTER-ABSENT"}
COUNTING_TYPES = {"COUNTING-UNANIMITY", "COUNTING-MAJORITY"}


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class GoldSpan:
    text: str
    label: str
    start: int
    end: int
    event_id: Optional[int] = None


@dataclass
class PredSpan:
    text: str
    label: str
    start: int
    end: int


@dataclass
class Segment:
    id: str
    text: str
    municipality: str
    gold_spans: List[GoldSpan]
    pred_spans: List[PredSpan]
    errors: List[ErrorCase]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def gold_events(self) -> Dict[int, List[GoldSpan]]:
        d: Dict[int, List[GoldSpan]] = defaultdict(list)
        for s in self.gold_spans:
            if s.event_id is not None:
                d[s.event_id].append(s)
        return dict(d)


@dataclass
class ModelEntry:
    name: str
    pred_path: Path
    eval_path: Optional[Path]
    gold_path: Path


# ── Discovery ─────────────────────────────────────────────────────────────────
def discover_models() -> List[ModelEntry]:
    entries = []
    for p in sorted(PRED_DIR.glob("*.jsonl")):
        stem = p.stem
        ev   = EVAL_DIR / f"{stem}.json"
        gp   = _infer_gold(stem)
        entries.append(ModelEntry(stem, p, ev if ev.exists() else None, gp))
    return entries


def _infer_gold(stem: str) -> Path:
    v6  = DATA_DIR / "citilink-votie" / "test.jsonl"
    sub = DATA_DIR / "citilink-votie" / "test_gold_with_sub_subjects.jsonl"
    if "sub_subjects" in stem:
        return sub if sub.exists() else v6
    return v6


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading predictions and classifying errors…")
def load_segments(pred_str: str, gold_str: str) -> List[Segment]:
    raw_preds  = _norm_load(pred_str)
    pred_by_id = {p["id"]: p for p in raw_preds}

    gold_data  = load_jsonl(gold_str)
    gold_by_id = {ex["id"]: ex for ex in gold_data}

    clf = ErrorClassifier()
    gold_ent: Dict[str, List[Entity]] = {}
    pred_ent: Dict[str, List[Entity]] = {}
    texts:    Dict[str, str]          = {}

    for doc_id, gex in gold_by_id.items():
        texts[doc_id] = gex["text"]
        gold_ent[doc_id] = [
            Entity(
                sp["text"],
                sp.get("label", sp.get("type", "UNKNOWN")),
                sp["start"], sp["end"], doc_id,
            )
            for sp in gex.get("spans", [])
        ]
        pr = pred_by_id.get(doc_id)
        if pr is None:
            pred_ent[doc_id] = []
            continue
        raw_spans = bio_to_spans(
            pr["tokens"], pr["pred_labels"],
            token_offsets=pr.get("token_offsets"),
            text=pr.get("text"),
        )
        pred_ent[doc_id] = [
            Entity(s["text"], s["type"], s["start"], s["end"], doc_id)
            for s in raw_spans
        ]

    all_errors = clf.classify_errors(gold_ent, pred_ent, texts)
    err_by_doc: Dict[str, List[ErrorCase]] = defaultdict(list)
    for e in all_errors:
        err_by_doc[e.doc_id].append(e)

    segments = []
    for gex in gold_data:
        doc_id = gex["id"]
        segments.append(Segment(
            id=doc_id,
            text=gex["text"],
            municipality=gex.get("municipality", ""),
            gold_spans=[
                GoldSpan(
                    sp["text"],
                    sp.get("label", sp.get("type", "UNKNOWN")),
                    sp["start"], sp["end"],
                    sp.get("event_id"),
                )
                for sp in gex.get("spans", [])
            ],
            pred_spans=[
                PredSpan(e.text, e.type, e.start, e.end)
                for e in pred_ent.get(doc_id, [])
            ],
            errors=err_by_doc.get(doc_id, []),
        ))
    return segments


@st.cache_data(show_spinner=False)
def load_eval_json(path_str: str) -> Optional[Dict]:
    p = Path(path_str)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ── HTML rendering ────────────────────────────────────────────────────────────
def _esc(t: str) -> str:
    return _html.escape(t).replace("\n", "<br>")


def _badge(label: str, color: str) -> str:
    ab = ABBREVS.get(label, label[:3])
    return (
        f'<sup style="font-size:0.65em;background:{color};color:#fff;'
        f'padding:1px 4px;border-radius:3px;margin-right:2px">{ab}</sup>'
    )


def _span_style(cat: str, color: str) -> str:
    if cat == "CORRECT":
        return f"background:{color}22;border-bottom:2px solid {color};padding:0 1px;"
    if cat == "MIS":
        return "background:#ef9a9a33;text-decoration:line-through red;border-bottom:2px dashed #c62828;padding:0 1px;"
    if cat == "SPU":
        return "background:#fff17644;border-bottom:2px dotted #f57f17;padding:0 1px;"
    if cat == "INC_BOUNDARY":
        return f"background:{color}18;border-bottom:3px dashed {color};padding:0 1px;"
    if cat == "INC_TYPE":
        return "background:#ce93d833;border-bottom:2px solid #6a1b9a;padding:0 1px;"
    return ""


def render_segment_html(seg: Segment, active_cats: set, active_labels: set) -> str:
    # Map (start, end, type) -> error category for gold spans
    err_map: Dict[Tuple, str] = {}
    for e in seg.errors:
        if e.gold_entity:
            key = (e.gold_entity.start, e.gold_entity.end, e.gold_entity.type)
            # If multiple errors for same span, prefer INC_TYPE > INC_BOUNDARY > MIS
            existing = err_map.get(key)
            priority = {"INC_TYPE": 3, "INC_BOUNDARY": 2, "MIS": 1, "SPU": 0}
            if existing is None or priority.get(e.category, 0) > priority.get(existing, 0):
                err_map[key] = e.category

    annotations: List[Tuple] = []  # (start, end, cat, label, is_gold, color)

    for gs in seg.gold_spans:
        if gs.label not in active_labels:
            continue
        cat   = err_map.get((gs.start, gs.end, gs.label), "CORRECT")
        color = ENTITY_COLORS.get(gs.label, _DEF_COLOR)
        if cat != "CORRECT" and cat not in active_cats:
            continue
        annotations.append((gs.start, gs.end, cat, gs.label, True, color))

    if "SPU" in active_cats:
        for e in seg.errors:
            if e.category == "SPU" and e.pred_entity and e.entity_type in active_labels:
                pe    = e.pred_entity
                color = ENTITY_COLORS.get(pe.type, _DEF_COLOR)
                annotations.append((pe.start, pe.end, "SPU", pe.type, False, color))

    annotations.sort(key=lambda a: (a[0], not a[4]))  # gold before pred on ties

    text = seg.text
    pts  = sorted(set([0, len(text)] + [a[0] for a in annotations] + [a[1] for a in annotations]))

    parts = []
    for i in range(len(pts) - 1):
        s, e   = pts[i], pts[i + 1]
        chunk  = _esc(text[s:e])
        active = [a for a in annotations if a[0] <= s and a[1] >= e]
        if not active:
            parts.append(chunk)
            continue
        a_start, a_end, cat, label, is_gold, color = active[0]
        style = _span_style(cat, color)
        tip   = _html.escape(f"{ERROR_LABELS.get(cat, cat)} | {label} [{a_start}:{a_end}]")
        badge = _badge(label, color) if s == a_start else ""
        parts.append(f'<span style="{style}" title="{tip}">{badge}{chunk}</span>')

    return "".join(parts)


# ── Aggregate stats ───────────────────────────────────────────────────────────
def compute_error_agg(segments: List[Segment]):
    cat_counts  = Counter()
    type_counts: Dict[str, Counter] = defaultdict(Counter)
    type_conf   = Counter()  # (gold_type, pred_type) for INC_TYPE

    for seg in segments:
        for e in seg.errors:
            cat_counts[e.category] += 1
            type_counts[e.entity_type][e.category] += 1
            if e.category == "INC_TYPE" and e.gold_entity and e.pred_entity:
                type_conf[(e.gold_entity.type, e.pred_entity.type)] += 1

    return cat_counts, type_counts, type_conf


def _component_role(label: str) -> str:
    if label in VOTER_TYPES:    return "Voter"
    if label == "VOTING":       return "Voting"
    if label == "SUBJECT":      return "Subject"
    if label in COUNTING_TYPES: return "Counting"
    return "Other"


def compute_event_agg(segments: List[Segment]):
    total_events = 0
    full_correct = 0
    comp_stats: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for seg in segments:
        err_map: Dict[Tuple, str] = {}
        for e in seg.errors:
            if e.gold_entity:
                key = (e.gold_entity.start, e.gold_entity.end, e.gold_entity.type)
                err_map[key] = e.category

        for ev_id, spans in seg.gold_events.items():
            total_events += 1
            ev_ok = True
            for sp in spans:
                cat  = err_map.get((sp.start, sp.end, sp.label), "CORRECT")
                role = _component_role(sp.label)
                comp_stats[role]["total"] += 1
                if cat == "CORRECT":
                    comp_stats[role]["correct"] += 1
                else:
                    ev_ok = False
            if ev_ok:
                full_correct += 1

    return total_events, full_correct, dict(comp_stats)


def event_status_rows(seg: Segment) -> List[Dict]:
    err_map: Dict[Tuple, str] = {}
    for e in seg.errors:
        if e.gold_entity:
            err_map[(e.gold_entity.start, e.gold_entity.end, e.gold_entity.type)] = e.category

    rows = []
    for ev_id, spans in sorted(seg.gold_events.items()):
        for sp in spans:
            cat = err_map.get((sp.start, sp.end, sp.label), "CORRECT")
            rows.append({
                "Event": ev_id,
                "Role":   _component_role(sp.label),
                "Label":  sp.label,
                "Text":   sp.text[:70] + ("…" if len(sp.text) > 70 else ""),
                "Status": ERROR_LABELS.get(cat, cat),
            })
    return rows


def _color_status_row(row):
    val = row.get("Status", "")
    if   "Correct"  in val: bg = "#c8e6c9"
    elif "Missed"   in val: bg = "#ffcdd2"
    elif "Boundary" in val: bg = "#ffe0b2"
    elif "Type"     in val: bg = "#e1bee7"
    elif "Spurious" in val: bg = "#fff9c4"
    else: bg = ""
    return ["background-color:" + bg if col == "Status" else "" for col in row.index]


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="VotIE Prediction Explorer",
        page_icon="🗳️",
        layout="wide",
    )
    st.title("🗳️ VotIE — Prediction Explorer & Error Analysis")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Model")
        models = discover_models()
        if not models:
            st.error(f"No `.jsonl` files found in `{PRED_DIR}`")
            st.stop()

        model_names = [m.name for m in models]
        sel_name    = st.selectbox("Prediction file", model_names)
        model       = next(m for m in models if m.name == sel_name)

        st.divider()
        st.header("Gold reference")
        inferred_key = next(
            (k for k, v in GOLD_OPTIONS.items() if v == model.gold_path),
            list(GOLD_OPTIONS.keys())[0],
        )
        gold_keys = [k for k, v in GOLD_OPTIONS.items() if v.exists()]
        if not gold_keys:
            gold_keys = list(GOLD_OPTIONS.keys())
        default_idx = gold_keys.index(inferred_key) if inferred_key in gold_keys else 0
        gold_sel  = st.selectbox("Gold test file", gold_keys, index=default_idx)
        gold_path = GOLD_OPTIONS[gold_sel]

        if not gold_path.exists():
            st.error(f"Gold file not found:\n`{gold_path}`")
            st.stop()

        st.divider()
        if model.eval_path:
            st.success(f"Eval JSON: `{model.eval_path.name}`")
        else:
            st.warning("No paired evaluation JSON")

    # ── Load ──────────────────────────────────────────────────────────────────
    with st.spinner("Loading data…"):
        segments  = load_segments(str(model.pred_path), str(gold_path))
        eval_json = load_eval_json(str(model.eval_path)) if model.eval_path else None

    all_labels = sorted({sp.label for seg in segments for sp in seg.gold_spans})

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_ov, tab_seg, tab_ev = st.tabs(
        ["📊 Overview", "🔍 Segment browser", "🗃️ Events"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Overview
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ov:
        em = (eval_json or {}).get("entity_level_metrics", {})
        f1 = em.get("entity_f1")
        p  = em.get("entity_precision")
        r  = em.get("entity_recall")
        ac = em.get("entity_accuracy")

        if f1 is None:
            st.info(
                "No paired evaluation JSON found. "
                "Run `scripts/evaluate.py` to generate one, or recompute below."
            )
            if st.button("Recompute metrics from loaded data"):
                from src.evaluation.entity_metrics import EntityLevelEvaluator
                from scripts.evaluate import extract_labels
                raw_preds = _norm_load(str(model.pred_path))
                pred_lbl, gold_lbl, _ = extract_labels(raw_preds)
                has_gold = any(p != ["O"] * len(p) for p in gold_lbl)
                if has_gold:
                    metrics = EntityLevelEvaluator().compute_metrics(pred_lbl, gold_lbl)
                    st.json(metrics)
                else:
                    st.warning("Prediction file contains no gold labels; cannot recompute.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Macro F1",        f"{f1:.4f}")
            c2.metric("Macro Precision",  f"{p:.4f}"  if p  is not None else "—")
            c3.metric("Macro Recall",     f"{r:.4f}"  if r  is not None else "—")
            c4.metric("Token Accuracy",   f"{ac:.4f}" if ac is not None else "—")

        pt = em.get("per_type_metrics", {})
        if pt:
            st.subheader("Per-entity-type metrics")
            df_pt = pd.DataFrame([
                {
                    "Type":      etype,
                    "F1":        round(m.get("f1_score",  0), 4),
                    "Precision": round(m.get("precision", 0), 4),
                    "Recall":    round(m.get("recall",    0), 4),
                    "Support":   int(m.get("support",     0)),
                }
                for etype, m in sorted(pt.items())
            ]).set_index("Type")

            st.dataframe(
                df_pt.style.background_gradient(
                    subset=["F1"], cmap="RdYlGn", vmin=0, vmax=1
                ),
                use_container_width=True,
            )
            st.bar_chart(df_pt[["F1", "Precision", "Recall"]])

        st.divider()
        st.subheader("Error category breakdown")
        cat_counts, type_counts, type_conf = compute_error_agg(segments)

        col_a, col_b = st.columns(2)
        with col_a:
            if cat_counts:
                df_cat = pd.DataFrame(
                    [{"Category": ERROR_LABELS.get(k, k), "Count": v}
                     for k, v in cat_counts.most_common()]
                ).set_index("Category")
                st.bar_chart(df_cat)
            else:
                st.success("No errors found!")

        with col_b:
            if type_counts:
                all_err_types = sorted(type_counts.keys())
                cats          = ["MIS", "SPU", "INC_BOUNDARY", "INC_TYPE"]
                df_hm = pd.DataFrame(
                    {cat: [type_counts[t].get(cat, 0) for t in all_err_types]
                     for cat in cats},
                    index=all_err_types,
                )
                df_hm.index.name = "Entity type"
                st.dataframe(
                    df_hm.style.background_gradient(cmap="YlOrRd"),
                    use_container_width=True,
                )

        if type_conf:
            st.subheader("Type confusion — INC_TYPE errors")
            all_gt = sorted({g for g, _ in type_conf})
            all_pt = sorted({p for _, p in type_conf})
            mat    = {g: {p: 0 for p in all_pt} for g in all_gt}
            for (g, p_), cnt in type_conf.items():
                mat[g][p_] = cnt
            df_conf = pd.DataFrame(mat).T.fillna(0).astype(int)
            df_conf.index.name   = "Gold type"
            df_conf.columns.name = "Pred type"
            st.dataframe(
                df_conf.style.background_gradient(cmap="Blues"),
                use_container_width=True,
            )

        rb = (eval_json or {}).get("relaxed_boundary_metrics")
        if rb:
            st.divider()
            st.subheader("Relaxed boundary metrics")
            r1, r2, r3 = st.columns(3)
            r1.metric("Relaxed F1",       f"{rb.get('f1', 0):.4f}")
            r2.metric("Relaxed Precision", f"{rb.get('precision', 0):.4f}")
            r3.metric("Relaxed Recall",    f"{rb.get('recall', 0):.4f}")

        fe = (eval_json or {}).get("full_event")
        if fe:
            st.divider()
            st.subheader("Full-event metrics (from evaluation file)")
            e1, e2, e3 = st.columns(3)
            e1.metric("Full-event F1",       f"{fe.get('f1', 0):.4f}")
            e2.metric("Full-event Precision", f"{fe.get('precision', 0):.4f}")
            e3.metric("Full-event Recall",    f"{fe.get('recall', 0):.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Segment browser
    # ══════════════════════════════════════════════════════════════════════════
    with tab_seg:
        filt_col, main_col = st.columns([1, 3])

        with filt_col:
            st.subheader("Filters")
            active_cats = set(st.multiselect(
                "Error categories",
                ["MIS", "SPU", "INC_BOUNDARY", "INC_TYPE"],
                default=["MIS", "SPU", "INC_BOUNDARY", "INC_TYPE"],
                format_func=lambda x: ERROR_LABELS[x],
            ))
            active_labels = set(st.multiselect(
                "Entity types", all_labels, default=all_labels,
            ))
            search = st.text_input("Search segment text", "")
            sort_by = st.radio(
                "Sort segments",
                ["Original order", "Error count ↓", "Segment ID"],
            )
            only_errors = st.checkbox("Only show segments with errors", value=False)

        # Filter + sort
        filtered = [
            seg for seg in segments
            if (not search or search.lower() in seg.text.lower())
            and (not only_errors or seg.error_count > 0)
        ]
        if sort_by == "Error count ↓":
            filtered = sorted(filtered, key=lambda s: s.error_count, reverse=True)
        elif sort_by == "Segment ID":
            filtered = sorted(filtered, key=lambda s: s.id)

        with main_col:
            if not filtered:
                st.warning("No segments match the current filters.")
            else:
                # ── Navigation ────────────────────────────────────────────────
                if "cur_seg_id" not in st.session_state or \
                        st.session_state.cur_seg_id not in {s.id for s in filtered}:
                    st.session_state.cur_seg_id = filtered[0].id

                cur_idx = next(
                    (i for i, s in enumerate(filtered)
                     if s.id == st.session_state.cur_seg_id), 0
                )

                nav_l, nav_c, nav_r = st.columns([1, 10, 1])
                with nav_l:
                    if st.button("◀", key="prev_seg") and cur_idx > 0:
                        st.session_state.cur_seg_id = filtered[cur_idx - 1].id
                        st.rerun()
                with nav_r:
                    if st.button("▶", key="next_seg") and cur_idx < len(filtered) - 1:
                        st.session_state.cur_seg_id = filtered[cur_idx + 1].id
                        st.rerun()
                with nav_c:
                    chosen = st.selectbox(
                        f"Segment  {cur_idx + 1} / {len(filtered)}",
                        options=[s.id for s in filtered],
                        index=cur_idx,
                    )
                    if chosen != st.session_state.cur_seg_id:
                        st.session_state.cur_seg_id = chosen
                        cur_idx = next(i for i, s in enumerate(filtered) if s.id == chosen)

                seg = filtered[cur_idx]

                # ── Info cards ────────────────────────────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Gold spans", len(seg.gold_spans))
                m2.metric("Pred spans", len(seg.pred_spans))
                m3.metric("Total errors", seg.error_count)
                m4.metric("Municipality", seg.municipality or "—")

                # ── Legend ────────────────────────────────────────────────────
                type_badges = " ".join(
                    f'<span style="background:{ENTITY_COLORS.get(lb, _DEF_COLOR)}22;'
                    f'border-bottom:2px solid {ENTITY_COLORS.get(lb, _DEF_COLOR)};'
                    f'padding:2px 5px;border-radius:3px;font-size:0.75em;margin:2px">'
                    f'<b>{ABBREVS.get(lb, lb[:3])}</b> {lb}</span>'
                    for lb in all_labels if lb in active_labels
                )
                err_legend = (
                    '<span style="text-decoration:line-through red" title="Missed (FN)">MIS</span>&nbsp;&nbsp;'
                    '<span style="border-bottom:2px dotted #f57f17" title="Spurious (FP)">SPU</span>&nbsp;&nbsp;'
                    '<span style="border-bottom:3px dashed #e65100" title="Boundary error">INC_BND</span>&nbsp;&nbsp;'
                    '<span style="background:#ce93d833;border-bottom:2px solid #6a1b9a" title="Type error">INC_TYPE</span>'
                )
                st.markdown(
                    f'<div style="font-size:0.78em;margin-bottom:6px">'
                    f'{type_badges}</div>'
                    f'<div style="font-size:0.78em;margin-bottom:10px">'
                    f'<b>Error styles:</b> {err_legend}</div>',
                    unsafe_allow_html=True,
                )

                # ── Highlighted text ──────────────────────────────────────────
                text_html = render_segment_html(seg, active_cats, active_labels)
                st.markdown(
                    f'<div style="max-height:440px;overflow-y:auto;border:1px solid #ddd;'
                    f'border-radius:6px;padding:14px;font-size:0.92em;line-height:1.9;'
                    f'font-family:sans-serif;background:transparent;color:inherit;word-break:break-word">'
                    f'{text_html}</div>',
                    unsafe_allow_html=True,
                )

                # ── Error table ───────────────────────────────────────────────
                visible_errors = [
                    e for e in seg.errors
                    if e.category in active_cats and e.entity_type in active_labels
                ]
                if visible_errors:
                    st.subheader(f"Errors ({len(visible_errors)})")
                    err_rows = [
                        {
                            "Category":    ERROR_LABELS.get(e.category, e.category),
                            "Type":        e.entity_type,
                            "Gold text":   e.gold_entity.text if e.gold_entity else "—",
                            "Gold span":   f"{e.gold_entity.start}:{e.gold_entity.end}" if e.gold_entity else "—",
                            "Pred text":   e.pred_entity.text if e.pred_entity else "—",
                            "Pred span":   f"{e.pred_entity.start}:{e.pred_entity.end}" if e.pred_entity else "—",
                        }
                        for e in visible_errors
                    ]
                    st.dataframe(
                        pd.DataFrame(err_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                elif seg.error_count == 0:
                    st.success("This segment has no errors — perfect prediction!")
                else:
                    st.info("No errors visible with current filters.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Events
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ev:
        st.subheader("Event-level analysis")

        total_evs, full_correct, comp_stats = compute_event_agg(segments)

        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Total gold events", total_evs)
        ec2.metric("Fully correct events", full_correct)
        if total_evs:
            ec3.metric("Full-event recall", f"{full_correct / total_evs:.2%}")

        if comp_stats:
            st.subheader("Per-component recall")
            comp_rows = [
                {
                    "Role":    role,
                    "Correct": s["correct"],
                    "Total":   s["total"],
                    "Recall":  f"{s['correct'] / s['total']:.2%}" if s["total"] else "—",
                }
                for role, s in sorted(comp_stats.items())
            ]
            st.dataframe(
                pd.DataFrame(comp_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()
        st.subheader("Worst segments by event-level errors")
        seg_ev_rows = []
        for seg in segments:
            em_seg: Dict[Tuple, str] = {}
            for e in seg.errors:
                if e.gold_entity:
                    em_seg[(e.gold_entity.start, e.gold_entity.end, e.gold_entity.type)] = e.category
            ev_total = ev_err = 0
            for spans in seg.gold_events.values():
                for sp in spans:
                    ev_total += 1
                    if em_seg.get((sp.start, sp.end, sp.label), "CORRECT") != "CORRECT":
                        ev_err += 1
            if ev_total:
                seg_ev_rows.append({
                    "Segment":      seg.id,
                    "Municipality": seg.municipality,
                    "Ev. spans":    ev_total,
                    "Ev. errors":   ev_err,
                    "Error rate":   f"{ev_err / ev_total:.0%}",
                })
        seg_ev_rows.sort(key=lambda x: x["Ev. errors"], reverse=True)
        if seg_ev_rows:
            st.dataframe(
                pd.DataFrame(seg_ev_rows[:25]),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()
        st.subheader("Events in selected segment")

        seg_ids   = [s.id for s in segments]
        ev_sel_id = st.session_state.get("cur_seg_id", seg_ids[0] if seg_ids else "")
        if ev_sel_id not in seg_ids:
            ev_sel_id = seg_ids[0] if seg_ids else ""

        ev_seg_id = st.selectbox(
            "Segment",
            seg_ids,
            index=seg_ids.index(ev_sel_id) if ev_sel_id in seg_ids else 0,
            key="ev_seg_sel",
        )
        ev_seg = next((s for s in segments if s.id == ev_seg_id), None)
        if ev_seg:
            rows = event_status_rows(ev_seg)
            if rows:
                df_ev = pd.DataFrame(rows)
                st.dataframe(
                    df_ev.style.apply(_color_status_row, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

                n_ev    = len(ev_seg.gold_events)
                n_full  = sum(
                    1 for ev_id, spans in ev_seg.gold_events.items()
                    if all(
                        next(
                            (e.category for e in ev_seg.errors
                             if e.gold_entity and
                             e.gold_entity.start == sp.start and
                             e.gold_entity.end   == sp.end and
                             e.gold_entity.type  == sp.label),
                            "CORRECT",
                        ) == "CORRECT"
                        for sp in spans
                    )
                )
                st.caption(f"{n_full}/{n_ev} voting events fully correct in this segment.")
            else:
                st.info("No events (with `event_id`) found in this segment.")


if __name__ == "__main__":
    main()
