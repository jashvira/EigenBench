"""Run the OASST BM25 router audit for local anchors.

This script treats BM25 as an open lexical gate, not as a final classifier.
For each OASST scenario, it retrieves criteria from two corpora:

- compressed criteria cards
- raw criterion text

Any positive lexical match is allowed through as a candidate. Rows with no
positive BM25 match are marked for embedding fallback, because BM25 cannot catch
synonyms or paraphrases that do not share surface words.
"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scenario_classifier.core.io import DEFAULT_CARD_CORPUS, DEFAULT_RAW_CORPUS, DEFAULT_SCENARIOS
from scenario_classifier.core.io import OUTPUT_ROOT, load_jsonl, load_scenarios
from scenario_classifier.retrieval.retrievers import BM25Retriever
from scenario_classifier.retrieval.retrievers import tokenize


DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "router_audit"
OUTPUT_PREFIX = "oasst_bm25_local_anchors"
TOP_K = 8


def margin(top: list[dict[str, Any]]) -> float:
    """Return the score gap between the top two candidates."""
    if not top:
        return 0.0
    if len(top) == 1:
        return float(top[0]["score"])
    return round(float(top[0]["score"]) - float(top[1]["score"]), 6)


def flag(card_top: list[dict[str, Any]], raw_top: list[dict[str, Any]]) -> str:
    """Compare the top card hit and top raw-criterion hit for audit triage."""
    card_score = float(card_top[0]["score"]) if card_top else 0.0
    raw_score = float(raw_top[0]["score"]) if raw_top else 0.0
    if card_score == 0 and raw_score == 0:
        return "no_match"
    if card_score > 0 and raw_score == 0:
        return "card_only"
    if card_score == 0 and raw_score > 0:
        return "raw_only"
    if card_top[0]["constitution"] == raw_top[0]["constitution"]:
        return "both_agree"
    return "disagree"


def positive_candidates(*ranked_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge positive BM25 hits from card and raw corpora without duplicates."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ranked in ranked_lists:
        for item in ranked:
            if float(item["score"]) <= 0:
                continue
            criterion_id = str(item["criterion_id"])
            if criterion_id in seen:
                continue
            seen.add(criterion_id)
            candidates.append(item)
    return candidates


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate routing outcomes for the audit summary."""
    flag_counts = Counter(row["flag"] for row in rows)
    gate_counts = Counter(row["gate_decision"] for row in rows)
    card_counts = Counter(row["card_top_constitution"] for row in rows if row["card_top_score"] > 0)
    raw_counts = Counter(row["raw_top_constitution"] for row in rows if row["raw_top_score"] > 0)
    candidate_counts = [row["candidate_count"] for row in rows]
    return {
        "rows": len(rows),
        "flag_counts": dict(sorted(flag_counts.items())),
        "gate_decision_counts": dict(sorted(gate_counts.items())),
        "avg_candidate_count": round(sum(candidate_counts) / max(len(candidate_counts), 1), 3),
        "max_candidate_count": max(candidate_counts) if candidate_counts else 0,
        "card_top_constitution_counts": dict(sorted(card_counts.items())),
        "raw_top_constitution_counts": dict(sorted(raw_counts.items())),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write audit rows as compact JSONL."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def compact_top(top: list[dict[str, Any]]) -> str:
    """Compact ranked hits for the CSV export."""
    return " | ".join(
        f"{item['rank']}. {item['criterion_id']} ({item['score']})"
        for item in top
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the audit table in spreadsheet-friendly form."""
    fields = [
        "scenario_index",
        "scenario",
        "gate_decision",
        "candidate_count",
        "candidate_constitutions",
        "flag",
        "card_top_constitution",
        "card_top_score",
        "card_margin",
        "card_top",
        "raw_top_constitution",
        "raw_top_score",
        "raw_margin",
        "raw_top",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scenario_index": row["scenario_index"],
                    "scenario": row["scenario"],
                    "gate_decision": row["gate_decision"],
                    "candidate_count": row["candidate_count"],
                    "candidate_constitutions": ";".join(row["candidate_constitutions"]),
                    "flag": row["flag"],
                    "card_top_constitution": row["card_top_constitution"],
                    "card_top_score": row["card_top_score"],
                    "card_margin": row["card_margin"],
                    "card_top": compact_top(row["card_top"]),
                    "raw_top_constitution": row["raw_top_constitution"],
                    "raw_top_score": row["raw_top_score"],
                    "raw_margin": row["raw_margin"],
                    "raw_top": compact_top(row["raw_top"]),
                }
            )


def render_top(items: list[dict[str, Any]]) -> str:
    """Render ranked candidates for the static HTML viewer."""
    parts = []
    for item in items:
        terms = ", ".join(item["matched_terms"][:12])
        parts.append(
            "<li>"
            f"<strong>{html.escape(str(item['criterion_id']))}</strong> "
            f"<span>{html.escape(str(item['constitution']))}</span> "
            f"<code>{item['score']}</code>"
            f"<p>{html.escape(str(item['text']))}</p>"
            f"<small>{html.escape(terms)}</small>"
            "</li>"
        )
    return "<ol>" + "".join(parts) + "</ol>"


def write_html(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], *, title: str) -> None:
    """Write a static HTML audit view."""
    sample_rows = rows[:500]
    flag_options = "".join(f"<option value=\"{html.escape(flag)}\">{html.escape(flag)}</option>" for flag in sorted(summary["flag_counts"]))
    cards = []
    for row in sample_rows:
        cards.append(
            "<article class=\"row\" data-flag=\"{flag}\" data-gate=\"{gate}\">"
            "<div class=\"scenario\"><span>#{idx}</span><p>{scenario}</p></div>"
            "<div><div class=\"badge {flag}\">{flag}</div><div class=\"gate\">{gate}<br>{candidate_count} candidates</div></div>"
            "<section><h3>Cards: {card_const} <code>{card_score}</code></h3>{card_top}</section>"
            "<section><h3>Raw: {raw_const} <code>{raw_score}</code></h3>{raw_top}</section>"
            "</article>".format(
                flag=html.escape(row["flag"]),
                gate=html.escape(row["gate_decision"]),
                candidate_count=row["candidate_count"],
                idx=row["scenario_index"],
                scenario=html.escape(row["scenario"]),
                card_const=html.escape(row["card_top_constitution"]),
                card_score=row["card_top_score"],
                card_top=render_top(row["card_top"]),
                raw_const=html.escape(row["raw_top_constitution"]),
                raw_score=row["raw_top_score"],
                raw_top=render_top(row["raw_top"]),
            )
        )

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; background: #f7f7f4; color: #202124; font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 24px; }}
    header {{ display: flex; justify-content: space-between; gap: 16px; align-items: end; border-bottom: 1px solid #ddd; padding-bottom: 16px; }}
    h1 {{ margin: 0; font-size: 26px; }}
    .stats {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .pill {{ border: 1px solid #ddd; background: white; border-radius: 999px; padding: 5px 9px; }}
    .controls {{ margin: 16px 0; display: flex; gap: 10px; flex-wrap: wrap; }}
    input, select {{ border: 1px solid #ddd; background: white; border-radius: 8px; padding: 9px 10px; font: inherit; }}
    input {{ min-width: 320px; flex: 1; }}
    .row {{ display: grid; grid-template-columns: minmax(260px, 1fr) 100px minmax(260px, 1fr) minmax(260px, 1fr); gap: 12px; background: white; border: 1px solid #ddd; border-radius: 8px; padding: 13px; margin: 12px 0; }}
    .scenario span {{ color: #62665f; font-size: 12px; }}
    .scenario p {{ margin: 5px 0 0; }}
    h3 {{ margin: 0 0 8px; font-size: 13px; color: #62665f; }}
    ol {{ margin: 0; padding-left: 19px; }}
    li + li {{ margin-top: 9px; }}
    li p {{ margin: 3px 0; color: #343633; }}
    small {{ color: #62665f; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .badge {{ align-self: start; justify-self: start; border-radius: 999px; padding: 5px 9px; font-size: 12px; font-weight: 650; background: #eef2ff; color: #3730a3; }}
    .gate {{ margin-top: 8px; color: #62665f; font-size: 12px; }}
    .disagree {{ background: #fff7ed; color: #9a3412; }}
    .no_match {{ background: #f1f5f9; color: #475569; }}
    .both_agree {{ background: #ecfdf5; color: #166534; }}
    .card_only, .raw_only {{ background: #fefce8; color: #854d0e; }}
    @media (max-width: 960px) {{ .row {{ grid-template-columns: 1fr; }} header {{ display: block; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>{html.escape(title)}</h1>
        <p>Showing first {len(sample_rows)} of {len(rows)} OASST scenarios. BM25 is an open candidate gate: positives pass through; zero-overlap rows are marked for embedding fallback.</p>
      </div>
      <div class="stats">
        {''.join(f'<span class="pill">{html.escape(k)}: <strong>{v}</strong></span>' for k, v in summary['flag_counts'].items())}
        {''.join(f'<span class="pill">{html.escape(k)}: <strong>{v}</strong></span>' for k, v in summary['gate_decision_counts'].items())}
      </div>
    </header>
    <div class="controls">
      <input id="q" type="search" placeholder="Search scenario or criterion text">
      <select id="flag"><option value="">All flags</option>{flag_options}</select>
    </div>
    <section id="rows">{''.join(cards)}</section>
  </main>
  <script>
    const q = document.getElementById('q');
    const flag = document.getElementById('flag');
    const rows = [...document.querySelectorAll('.row')];
    function apply() {{
      const query = q.value.toLowerCase();
      const wanted = flag.value;
      rows.forEach(row => {{
        const okFlag = !wanted || row.dataset.flag === wanted;
        const okQuery = !query || row.textContent.toLowerCase().includes(query);
        row.style.display = okFlag && okQuery ? '' : 'none';
      }});
    }}
    q.addEventListener('input', apply);
    flag.addEventListener('change', apply);
  </script>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def main() -> None:
    """Run the OASST audit and write JSONL, CSV, summary JSON, and HTML."""
    output_dir = DEFAULT_OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(DEFAULT_SCENARIOS)
    card_docs = load_jsonl(DEFAULT_CARD_CORPUS)
    raw_docs = load_jsonl(DEFAULT_RAW_CORPUS)
    card_index = BM25Retriever(card_docs)
    raw_index = BM25Retriever(raw_docs)

    rows: list[dict[str, Any]] = []
    for idx, scenario in enumerate(scenarios):
        card_top = card_index.top_k(scenario, TOP_K)
        raw_top = raw_index.top_k(scenario, TOP_K)
        row_flag = flag(card_top, raw_top)
        candidates = positive_candidates(card_top, raw_top)
        candidate_constitutions = sorted({str(item["constitution"]) for item in candidates})
        # Open gate: any positive BM25 hit passes; zero lexical overlap falls back.
        rows.append(
            {
                "scenario_index": idx,
                "scenario": scenario,
                "flag": row_flag,
                "gate_decision": "pass_bm25" if candidates else "embedding_fallback",
                "candidate_count": len(candidates),
                "candidate_constitutions": candidate_constitutions,
                "candidate_criteria": [item["criterion_id"] for item in candidates],
                "card_top_constitution": card_top[0]["constitution"] if card_top else "",
                "card_top_score": card_top[0]["score"] if card_top else 0.0,
                "card_margin": margin(card_top),
                "card_top": card_top,
                "raw_top_constitution": raw_top[0]["constitution"] if raw_top else "",
                "raw_top_score": raw_top[0]["score"] if raw_top else 0.0,
                "raw_margin": margin(raw_top),
                "raw_top": raw_top,
            }
        )

    summary = summarize(rows)
    jsonl_path = output_dir / f"{OUTPUT_PREFIX}.jsonl"
    csv_path = output_dir / f"{OUTPUT_PREFIX}.csv"
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"
    html_path = output_dir / f"{OUTPUT_PREFIX}.html"

    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_html(html_path, rows, summary, title="OASST BM25 Local Anchors Audit")

    print(f"scenarios: {len(scenarios)}")
    print(f"card_docs: {len(card_docs)}")
    print(f"raw_docs: {len(raw_docs)}")
    print(f"flag_counts: {summary['flag_counts']}")
    print(f"gate_decision_counts: {summary['gate_decision_counts']}")
    print(f"avg_candidate_count: {summary['avg_candidate_count']}")
    print(f"wrote: {jsonl_path}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {summary_path}")
    print(f"wrote: {html_path}")


if __name__ == "__main__":
    main()
