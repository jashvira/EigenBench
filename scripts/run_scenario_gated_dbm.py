"""Run a scenario-heldout Scenario-Gated DBM probe on raw ValueArena logs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Iterable

import numpy as np
import requests
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    "37_model_run",
    "8_models/conservatism",
    "8_models/deep_ecology",
    "8_models/kindness",
]
ERROR_RE = re.compile(r"Error in \w+ API call")
CHOICE_RE = re.compile(
    r"<criterion_(\d+)_choice>\s*(.*?)\s*</criterion_\1_choice>",
    flags=re.DOTALL,
)
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_APP_REFERER = "https://github.com/Constitutional-Evals/"
OPENROUTER_APP_TITLE = "EigenBench Scenario-Gated DBM"


@dataclass
class Data:
    """Criterion-level trits with scenario-heldout row indexes."""

    judge: np.ndarray
    eval1: np.ndarray
    eval2: np.ndarray
    criterion: np.ndarray
    scenario: np.ndarray
    label: np.ndarray
    scenario_texts: list[str]
    model_names: list[str]
    criterion_names: list[str]
    run_counts: dict[str, int]
    raw_record_counts: dict[str, int]
    consistency_summary: dict[str, int]
    split: dict[str, np.ndarray]


def stable_hash(text: str) -> str:
    """Stable scenario identifier for split metadata."""

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def corpus_hash(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def slugify_model(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "__", model).strip("_")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            os.environ.setdefault("OPENROUTER_API_KEY", line.split("=", 1)[1].strip())


def contiguous_prefix(valid_scores: dict[int, int]) -> int:
    idx = 1
    while idx in valid_scores:
        idx += 1
    return idx - 1


def load_meta(run_id: str) -> dict:
    return json.loads((ROOT / "data/output/valuearena/raw/runs" / run_id / "meta.json").read_text())


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def response_has_error(item: dict) -> bool:
    fields = [
        "judge response",
        "eval1 response",
        "eval2 response",
        "eval1 reflection",
        "eval2 reflection",
    ]
    for field in fields:
        value = item.get(field)
        if value is None or ERROR_RE.search(value):
            return True
    return False


def get_model_name(item: dict, key: str, meta_models_by_index: dict[int, str]) -> str:
    name = item.get(f"{key}_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    idx = item[key]
    return meta_models_by_index.get(idx, f"{key}:{idx}")


def apply_repo_consistency_policy(
    judges: list[int],
    eval1s: list[int],
    eval2s: list[int],
    criteria: list[int],
    scenarios: list[int],
    labels: list[int],
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int], dict[str, int]]:
    """Match the repo training policy for transposed pair inconsistencies.

    If both response orders give the same strict winner label, the pair is
    treated as an inconsistent tie. Groups with duplicate or otherwise overfull
    transposes are dropped, matching `handle_inconsistencies_with_ties_criteria`.
    """

    groups: dict[tuple[int, int, int, int, int], list[int]] = {}
    rows = zip(judges, eval1s, eval2s, criteria, scenarios)
    for idx, (judge, eval1, eval2, criterion, scenario) in enumerate(rows):
        lo, hi = sorted((eval1, eval2))
        key = (criterion, scenario, judge, lo, hi)
        groups.setdefault(key, []).append(idx)

    keep = np.ones(len(labels), dtype=bool)
    new_labels = list(labels)
    converted = 0
    overfull_groups = 0
    dropped = 0

    for indexes in groups.values():
        if len(indexes) == 1:
            continue
        if len(indexes) != 2:
            overfull_groups += 1
            dropped += len(indexes)
            keep[indexes] = False
            continue

        first, second = indexes
        if labels[first] != 0 and labels[first] == labels[second]:
            new_labels[first] = 0
            new_labels[second] = 0
            converted += 2

    kept = np.flatnonzero(keep)
    summary = {
        "input_rows": len(labels),
        "output_rows": int(len(kept)),
        "converted_strict_to_tie": converted,
        "dropped_rows": dropped,
        "overfull_groups": overfull_groups,
    }
    return (
        [judges[i] for i in kept],
        [eval1s[i] for i in kept],
        [eval2s[i] for i in kept],
        [criteria[i] for i in kept],
        [scenarios[i] for i in kept],
        [new_labels[i] for i in kept],
        summary,
    )


def parse_logs(seed: int, handle_inconsistencies: bool) -> Data:
    """Expand raw ValueArena records into `(judge, eval1, eval2, C, s, trit)` rows."""

    model_to_id: dict[str, int] = {}
    criterion_to_id: dict[str, int] = {}
    scenario_to_id: dict[str, int] = {}
    scenario_texts: list[str] = []
    model_names: list[str] = []
    criterion_names: list[str] = []

    judges: list[int] = []
    eval1s: list[int] = []
    eval2s: list[int] = []
    criteria: list[int] = []
    scenarios: list[int] = []
    labels: list[int] = []
    run_counts: dict[str, int] = {}
    raw_record_counts: dict[str, int] = {}

    def intern(mapping: dict[str, int], names: list[str], key: str) -> int:
        if key not in mapping:
            mapping[key] = len(names)
            names.append(key)
        return mapping[key]

    for run_id in RUNS:
        meta = load_meta(run_id)
        num_criteria = int(meta["constitution"]["num_criteria"])
        constitution_name = Path(meta["constitution"]["path"]).stem
        meta_models_by_index = {idx: name for idx, name in enumerate(meta["models"].keys())}
        path = ROOT / "data/output/valuearena/raw/runs" / run_id / "evaluations.jsonl"

        raw_count = 0
        trit_count = 0
        for item in read_jsonl(path):
            raw_count += 1
            if response_has_error(item):
                continue

            response = item["judge response"]
            valid_scores: dict[int, int] = {}
            for match in CHOICE_RE.finditer(response):
                criterion_idx = int(match.group(1))
                if criterion_idx > num_criteria or criterion_idx in valid_scores:
                    continue
                try:
                    score = int(match.group(2).strip())
                except Exception:
                    continue
                if score in (0, 1, 2):
                    valid_scores[criterion_idx] = score

            prefix = min(contiguous_prefix(valid_scores), num_criteria)
            if prefix == 0:
                continue

            judge_name = get_model_name(item, "judge", meta_models_by_index)
            eval1_name = get_model_name(item, "eval1", meta_models_by_index)
            eval2_name = get_model_name(item, "eval2", meta_models_by_index)
            judge_id = intern(model_to_id, model_names, judge_name)
            eval1_id = intern(model_to_id, model_names, eval1_name)
            eval2_id = intern(model_to_id, model_names, eval2_name)

            scenario_text = str(item["scenario"]).strip()
            scenario_key = stable_hash(scenario_text)
            if scenario_key not in scenario_to_id:
                scenario_to_id[scenario_key] = len(scenario_texts)
                scenario_texts.append(scenario_text)
            scenario_id = scenario_to_id[scenario_key]

            for criterion_idx in range(1, prefix + 1):
                criterion_name = f"{constitution_name}/criterion_{criterion_idx:02d}"
                criterion_id = intern(criterion_to_id, criterion_names, criterion_name)
                judges.append(judge_id)
                eval1s.append(eval1_id)
                eval2s.append(eval2_id)
                criteria.append(criterion_id)
                scenarios.append(scenario_id)
                labels.append(valid_scores[criterion_idx])
                trit_count += 1

        raw_record_counts[run_id] = raw_count
        run_counts[run_id] = trit_count

    if handle_inconsistencies:
        judges, eval1s, eval2s, criteria, scenarios, labels, consistency_summary = apply_repo_consistency_policy(
            judges,
            eval1s,
            eval2s,
            criteria,
            scenarios,
            labels,
        )
    else:
        consistency_summary = {
            "input_rows": len(labels),
            "output_rows": len(labels),
            "converted_strict_to_tie": 0,
            "dropped_rows": 0,
            "overfull_groups": 0,
        }

    rng = np.random.default_rng(seed)
    scenario_ids = np.arange(len(scenario_texts))
    rng.shuffle(scenario_ids)
    n = len(scenario_ids)
    train_scenarios = set(scenario_ids[: int(0.8 * n)])
    dev_scenarios = set(scenario_ids[int(0.8 * n) : int(0.9 * n)])
    test_scenarios = set(scenario_ids[int(0.9 * n) :])

    scenario_arr = np.asarray(scenarios, dtype=np.int32)
    split = {
        "train": np.flatnonzero(np.isin(scenario_arr, list(train_scenarios))),
        "dev": np.flatnonzero(np.isin(scenario_arr, list(dev_scenarios))),
        "test": np.flatnonzero(np.isin(scenario_arr, list(test_scenarios))),
    }

    return Data(
        judge=np.asarray(judges, dtype=np.int32),
        eval1=np.asarray(eval1s, dtype=np.int32),
        eval2=np.asarray(eval2s, dtype=np.int32),
        criterion=np.asarray(criteria, dtype=np.int32),
        scenario=scenario_arr,
        label=np.asarray(labels, dtype=np.int32),
        scenario_texts=scenario_texts,
        model_names=model_names,
        criterion_names=criterion_names,
        run_counts=run_counts,
        raw_record_counts=raw_record_counts,
        consistency_summary=consistency_summary,
        split=split,
    )


def softmax_logits(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def logits_base(params: dict[str, np.ndarray], data: Data, idx: np.ndarray) -> np.ndarray:
    u = params["u"][data.judge[idx]]
    v1 = params["v"][data.eval1[idx]]
    v2 = params["v"][data.eval2[idx]]
    phi = params["phi"][data.criterion[idx]]
    x = u * phi
    s1 = np.sum(x * v1, axis=1)
    s2 = np.sum(x * v2, axis=1)
    tie = params["ell"][data.judge[idx]] + 0.5 * (s1 + s2)
    return np.stack([tie, s1, s2], axis=1)


def logits_gate(
    base: dict[str, np.ndarray],
    gate: dict[str, np.ndarray],
    z: np.ndarray,
    data: Data,
    idx: np.ndarray,
) -> np.ndarray:
    u = base["u"][data.judge[idx]]
    v1 = base["v"][data.eval1[idx]]
    v2 = base["v"][data.eval2[idx]]
    phi = base["phi"][data.criterion[idx]]
    z_batch = z[data.scenario[idx]]
    q = np.clip(z_batch @ gate["W"].T, -4.0, 4.0)
    g = np.exp(q)
    x = u * phi * g
    s1 = np.sum(x * v1, axis=1)
    s2 = np.sum(x * v2, axis=1)
    tie = base["ell"][data.judge[idx]] + z_batch @ gate["wb"] + 0.5 * (s1 + s2)
    return np.stack([tie, s1, s2], axis=1)


def nll_from_logits(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    probs = softmax_logits(logits)
    row_nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))
    return float(row_nll.mean()), row_nll, probs


def calibration_ece(prob: np.ndarray, obs: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if np.any(mask):
            ece += mask.mean() * abs(float(prob[mask].mean()) - float(obs[mask].mean()))
    return float(ece)


def evaluate_logits(logits: np.ndarray, labels: np.ndarray) -> dict:
    nll, row_nll, probs = nll_from_logits(logits, labels)
    pred = np.argmax(probs, axis=1)
    tie_obs = labels == 0
    strict = labels != 0
    if np.any(strict):
        strict_prob = probs[strict, 1] / np.clip(probs[strict, 1] + probs[strict, 2], 1e-12, None)
        strict_obs = labels[strict] == 1
        winloss_ece = calibration_ece(strict_prob, strict_obs)
    else:
        winloss_ece = float("nan")
    return {
        "nll": nll,
        "accuracy": float(np.mean(pred == labels)),
        "tie_ece": calibration_ece(probs[:, 0], tie_obs),
        "winloss_ece": winloss_ece,
        "tie_rate_observed": float(np.mean(tie_obs)),
        "tie_rate_predicted": float(np.mean(probs[:, 0])),
        "row_nll": row_nll,
        "probs": probs,
    }


def adam_update(params: dict[str, np.ndarray], grads: dict[str, np.ndarray], state: dict, lr: float, step: int) -> None:
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for key, param in params.items():
        if key not in state:
            state[key] = {
                "m": np.zeros_like(param),
                "v": np.zeros_like(param),
            }
        st = state[key]
        grad = grads[key]
        st["m"] = beta1 * st["m"] + (1.0 - beta1) * grad
        st["v"] = beta2 * st["v"] + (1.0 - beta2) * (grad * grad)
        m_hat = st["m"] / (1.0 - beta1**step)
        v_hat = st["v"] / (1.0 - beta2**step)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_base(
    data: Data,
    dim: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Fit the base Davidson bilinear model on train scenarios."""

    rng = np.random.default_rng(seed)
    num_models = len(data.model_names)
    num_criteria = len(data.criterion_names)
    params = {
        "u": rng.normal(0.0, 0.1, size=(num_models, dim)),
        "v": rng.normal(0.0, 0.1, size=(num_models, dim)),
        "phi": 1.0 + rng.normal(0.0, 0.01, size=(num_criteria, dim)),
        "ell": np.zeros(num_models),
    }
    state: dict = {}
    best = {k: v.copy() for k, v in params.items()}
    best_dev = float("inf")
    bad_epochs = 0
    history: list[dict] = []
    step = 0
    train_idx = data.split["train"]

    for epoch in range(1, epochs + 1):
        shuffled = rng.permutation(train_idx)
        for start in range(0, len(shuffled), batch_size):
            idx = shuffled[start : start + batch_size]
            labels = data.label[idx]
            u = params["u"][data.judge[idx]]
            v1 = params["v"][data.eval1[idx]]
            v2 = params["v"][data.eval2[idx]]
            phi = params["phi"][data.criterion[idx]]
            x = u * phi
            s1 = np.sum(x * v1, axis=1)
            s2 = np.sum(x * v2, axis=1)
            tie = params["ell"][data.judge[idx]] + 0.5 * (s1 + s2)
            logits = np.stack([tie, s1, s2], axis=1)
            probs = softmax_logits(logits)
            glogits = probs.copy()
            glogits[np.arange(len(labels)), labels] -= 1.0
            glogits /= len(labels)
            ds1 = 0.5 * glogits[:, 0] + glogits[:, 1]
            ds2 = 0.5 * glogits[:, 0] + glogits[:, 2]
            common = ds1[:, None] * v1 + ds2[:, None] * v2

            grads = {k: weight_decay * v for k, v in params.items()}
            np.add.at(grads["u"], data.judge[idx], common * phi)
            np.add.at(grads["v"], data.eval1[idx], ds1[:, None] * x)
            np.add.at(grads["v"], data.eval2[idx], ds2[:, None] * x)
            np.add.at(grads["phi"], data.criterion[idx], common * u)
            np.add.at(grads["ell"], data.judge[idx], glogits[:, 0])

            step += 1
            adam_update(params, grads, state, lr, step)

        train_eval = evaluate_logits(
            logits_base(params, data, data.split["train"]),
            data.label[data.split["train"]],
        )
        dev_eval = evaluate_logits(
            logits_base(params, data, data.split["dev"]),
            data.label[data.split["dev"]],
        )
        row = {"epoch": epoch, "train_nll": train_eval["nll"], "dev_nll": dev_eval["nll"]}
        history.append(row)
        print(
            f"base epoch {epoch:03d} train_nll={row['train_nll']:.4f} "
            f"dev_nll={row['dev_nll']:.4f}",
            flush=True,
        )
        if dev_eval["nll"] < best_dev - 1e-5:
            best_dev = dev_eval["nll"]
            best = {k: v.copy() for k, v in params.items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    return best, history


def build_text_embeddings(data: Data, dim: int, seed: int) -> np.ndarray:
    """Local lexical baseline: train-only TF-IDF followed by SVD."""

    train_scenarios = np.unique(data.scenario[data.split["train"]])
    train_texts = [data.scenario_texts[i] for i in train_scenarios]
    vectorizer = TfidfVectorizer(
        max_features=4096,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
    )
    train_x = vectorizer.fit_transform(train_texts)
    n_components = min(dim, max(2, train_x.shape[1] - 1), max(2, len(train_texts) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    svd.fit(train_x)
    all_x = vectorizer.transform(data.scenario_texts)
    z = svd.transform(all_x)
    if z.shape[1] < dim:
        z = np.pad(z, ((0, 0), (0, dim - z.shape[1])))

    train_z = z[train_scenarios]
    mean = train_z.mean(axis=0, keepdims=True)
    std = train_z.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((z - mean) / std).astype(np.float64)


def openrouter_embed_batch(model: str, texts: list[str], api_key: str) -> list[list[float]]:
    """Embed one batch through OpenRouter's embeddings endpoint."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_APP_REFERER,
        "X-Title": OPENROUTER_APP_TITLE,
    }
    payload = {"model": model, "input": texts}

    for attempt in range(5):
        response = requests.post(
            OPENROUTER_EMBEDDINGS_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2**attempt)
            continue
        if not response.ok:
            raise RuntimeError(
                f"OpenRouter embeddings failed: {response.status_code} {response.text[:500]}"
            )
        data = response.json().get("data")
        if not isinstance(data, list):
            raise RuntimeError(f"OpenRouter embeddings returned no data: {response.text[:500]}")
        return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]

    raise RuntimeError("OpenRouter embeddings failed after retries")


def load_or_create_openrouter_embeddings(
    texts: list[str],
    model: str,
    batch_size: int,
    cache_dir: Path,
    force: bool,
) -> np.ndarray:
    """Load cached scenario vectors or create them with OpenRouter."""

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY")

    model_dir = cache_dir / slugify_model(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = model_dir / "scenario_embeddings.npy"
    manifest_path = model_dir / "scenario_embeddings_manifest.json"
    manifest = {"model": model, "count": len(texts), "sha256": corpus_hash(texts)}

    if not force and vectors_path.exists() and manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) == manifest:
            print(f"{model}: cached scenario embeddings", flush=True)
            return np.load(vectors_path)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(openrouter_embed_batch(model, batch, api_key))
        print(f"{model}: embedded {min(start + len(batch), len(texts))}/{len(texts)} scenarios", flush=True)

    array = np.asarray(vectors, dtype=np.float32)
    np.save(vectors_path, array)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return array


def reduce_and_standardize_embeddings(data: Data, vectors: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """Fit dimensionality reduction and scaling on train scenarios only."""

    train_scenarios = np.unique(data.scenario[data.split["train"]])
    if dim < vectors.shape[1]:
        n_components = min(dim, vectors.shape[1] - 1, len(train_scenarios) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=seed)
        svd.fit(vectors[train_scenarios])
        z = svd.transform(vectors)
        if z.shape[1] < dim:
            z = np.pad(z, ((0, 0), (0, dim - z.shape[1])))
    else:
        z = vectors.astype(np.float64, copy=True)
        if z.shape[1] < dim:
            z = np.pad(z, ((0, 0), (0, dim - z.shape[1])))

    train_z = z[train_scenarios]
    mean = train_z.mean(axis=0, keepdims=True)
    std = train_z.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((z - mean) / std).astype(np.float64)


def build_scenario_embeddings(
    data: Data,
    source: str,
    dim: int,
    seed: int,
    model: str,
    batch_size: int,
    cache_dir: Path,
    force: bool,
) -> np.ndarray:
    """Return scenario embeddings used by the real gate."""

    if source == "tfidf":
        return build_text_embeddings(data, dim, seed)
    vectors = load_or_create_openrouter_embeddings(
        data.scenario_texts,
        model=model,
        batch_size=batch_size,
        cache_dir=cache_dir,
        force=force,
    )
    return reduce_and_standardize_embeddings(data, vectors, dim, seed)


def train_gate(
    name: str,
    base: dict[str, np.ndarray],
    z: np.ndarray,
    data: Data,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Fit only the scenario gate `W` and ambiguity head `wb`."""

    rng = np.random.default_rng(seed)
    dim = base["u"].shape[1]
    embed_dim = z.shape[1]
    params = {
        "W": np.zeros((dim, embed_dim), dtype=np.float64),
        "wb": np.zeros(embed_dim, dtype=np.float64),
    }
    state: dict = {}
    best = {k: v.copy() for k, v in params.items()}
    best_dev = float("inf")
    bad_epochs = 0
    history: list[dict] = []
    train_idx = data.split["train"]
    step = 0

    for epoch in range(1, epochs + 1):
        shuffled = rng.permutation(train_idx)
        for start in range(0, len(shuffled), batch_size):
            idx = shuffled[start : start + batch_size]
            labels = data.label[idx]
            u = base["u"][data.judge[idx]]
            v1 = base["v"][data.eval1[idx]]
            v2 = base["v"][data.eval2[idx]]
            phi = base["phi"][data.criterion[idx]]
            z_batch = z[data.scenario[idx]]
            q_raw = z_batch @ params["W"].T
            q = np.clip(q_raw, -4.0, 4.0)
            g = np.exp(q)
            x = u * phi * g
            s1 = np.sum(x * v1, axis=1)
            s2 = np.sum(x * v2, axis=1)
            tie = base["ell"][data.judge[idx]] + z_batch @ params["wb"] + 0.5 * (s1 + s2)
            logits = np.stack([tie, s1, s2], axis=1)
            probs = softmax_logits(logits)
            glogits = probs.copy()
            glogits[np.arange(len(labels)), labels] -= 1.0
            glogits /= len(labels)
            ds1 = 0.5 * glogits[:, 0] + glogits[:, 1]
            ds2 = 0.5 * glogits[:, 0] + glogits[:, 2]
            dg = (u * phi) * (ds1[:, None] * v1 + ds2[:, None] * v2)
            dq = dg * g
            dq[(q_raw < -4.0) | (q_raw > 4.0)] = 0.0
            grads = {
                "W": dq.T @ z_batch + weight_decay * params["W"],
                "wb": glogits[:, 0] @ z_batch + weight_decay * params["wb"],
            }
            step += 1
            adam_update(params, grads, state, lr, step)

        train_eval = evaluate_logits(
            logits_gate(base, params, z, data, data.split["train"]),
            data.label[data.split["train"]],
        )
        dev_eval = evaluate_logits(
            logits_gate(base, params, z, data, data.split["dev"]),
            data.label[data.split["dev"]],
        )
        row = {"epoch": epoch, "train_nll": train_eval["nll"], "dev_nll": dev_eval["nll"]}
        history.append(row)
        print(
            f"{name} epoch {epoch:03d} train_nll={row['train_nll']:.4f} "
            f"dev_nll={row['dev_nll']:.4f}",
            flush=True,
        )
        if dev_eval["nll"] < best_dev - 1e-5:
            best_dev = dev_eval["nll"]
            best = {k: v.copy() for k, v in params.items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    return best, history


def strip_eval(eval_result: dict) -> dict:
    return {k: v for k, v in eval_result.items() if k not in ("row_nll", "probs")}


def save_arrays(
    out_dir: Path,
    data: Data,
    base: dict[str, np.ndarray],
    gates: dict[str, tuple[dict[str, np.ndarray], np.ndarray]],
) -> Path:
    """Persist fitted params and split ids for exact reruns/audits."""

    payload = {
        "base_u": base["u"],
        "base_v": base["v"],
        "base_phi": base["phi"],
        "base_ell": base["ell"],
        "train_row_idx": data.split["train"],
        "dev_row_idx": data.split["dev"],
        "test_row_idx": data.split["test"],
        "train_scenario_ids": np.unique(data.scenario[data.split["train"]]),
        "dev_scenario_ids": np.unique(data.scenario[data.split["dev"]]),
        "test_scenario_ids": np.unique(data.scenario[data.split["test"]]),
    }
    for name, (gate, z) in gates.items():
        payload[f"{name}_W"] = gate["W"]
        payload[f"{name}_wb"] = gate["wb"]
        payload[f"{name}_z"] = z

    path = out_dir / "scenario_gated_dbm_arrays.npz"
    np.savez_compressed(path, **payload)
    return path


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the probe runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--embedding-source", choices=("tfidf", "openrouter"), default="tfidf")
    parser.add_argument("--embedding-model", default="google/gemini-embedding-2")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=ROOT / "data/output/scenario_gated_dbm/embeddings",
    )
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-epochs", type=int, default=60)
    parser.add_argument("--gate-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--base-lr", type=float, default=0.01)
    parser.add_argument("--gate-lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--handle-inconsistencies", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/output/scenario_gated_dbm")
    return parser.parse_args()


def build_result(args: argparse.Namespace, data: Data, metrics: dict, histories: dict, arrays_path: Path) -> dict:
    """Assemble the JSON artifact without embedding large arrays inline."""

    config = vars(args).copy()
    config["out_dir"] = str(args.out_dir)
    config["embedding_cache_dir"] = str(args.embedding_cache_dir)
    return {
        "config": config,
        "data": {
            "rows": int(len(data.label)),
            "raw_record_counts": data.raw_record_counts,
            "run_counts": data.run_counts,
            "num_scenarios": len(data.scenario_texts),
            "num_models": len(data.model_names),
            "num_criteria": len(data.criterion_names),
            "consistency_summary": data.consistency_summary,
            "split_rows": {k: int(len(v)) for k, v in data.split.items()},
            "split_scenario_ids": {
                k: np.unique(data.scenario[v]).astype(int).tolist()
                for k, v in data.split.items()
            },
            "scenario_hashes": [stable_hash(text) for text in data.scenario_texts],
            "scenario_corpus_sha256": corpus_hash(data.scenario_texts),
            "model_names": data.model_names,
            "criterion_names": data.criterion_names,
        },
        "metrics": metrics,
        "histories": histories,
        "arrays_path": str(arrays_path),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = parse_logs(args.seed, args.handle_inconsistencies)
    print(
        f"parsed rows={len(data.label)} scenarios={len(data.scenario_texts)} "
        f"models={len(data.model_names)} criteria={len(data.criterion_names)} "
        f"train/dev/test={len(data.split['train'])}/{len(data.split['dev'])}/{len(data.split['test'])}",
        flush=True,
    )
    print(f"run_counts={data.run_counts}", flush=True)
    print(f"consistency={data.consistency_summary}", flush=True)

    base, base_history = train_base(
        data,
        dim=args.dim,
        seed=args.seed,
        epochs=args.base_epochs,
        batch_size=args.batch_size,
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )
    z_real = build_scenario_embeddings(
        data,
        source=args.embedding_source,
        dim=args.embedding_dim,
        seed=args.seed,
        model=args.embedding_model,
        batch_size=args.embedding_batch_size,
        cache_dir=args.embedding_cache_dir,
        force=args.force_embeddings,
    )
    rng = np.random.default_rng(args.seed)
    z_by_name = {
        "real_gate": z_real,
        "shuffled_gate": z_real[rng.permutation(len(z_real))],
        "random_id_gate": rng.normal(0.0, 1.0, size=z_real.shape),
    }

    gates = {}
    histories = {"base": base_history}
    for name, z in z_by_name.items():
        gate, history = train_gate(
            name,
            base,
            z,
            data,
            seed=args.seed + len(histories),
            epochs=args.gate_epochs,
            batch_size=args.batch_size,
            lr=args.gate_lr,
            weight_decay=args.gate_weight_decay,
            patience=args.patience,
        )
        gates[name] = (gate, z)
        histories[name] = history

    metrics = {}
    base_logits = {
        split: logits_base(base, data, data.split[split])
        for split in ("train", "dev", "test")
    }
    for split in ("train", "dev", "test"):
        base_eval = evaluate_logits(base_logits[split], data.label[data.split[split]])
        metrics.setdefault("base", {})[split] = strip_eval(base_eval)

    base_test_eval = evaluate_logits(base_logits["test"], data.label[data.split["test"]])
    for name, (gate, z) in gates.items():
        for split in ("train", "dev", "test"):
            eval_result = evaluate_logits(
                logits_gate(base, gate, z, data, data.split[split]),
                data.label[data.split[split]],
            )
            metrics.setdefault(name, {})[split] = strip_eval(eval_result)
        test_eval = evaluate_logits(
            logits_gate(base, gate, z, data, data.split["test"]),
            data.label[data.split["test"]],
        )
        shrinkage = base_test_eval["row_nll"] - test_eval["row_nll"]
        metrics[name]["test"]["residual_shrinkage_mean"] = float(np.mean(shrinkage))
        metrics[name]["test"]["residual_shrinkage_median"] = float(np.median(shrinkage))
        metrics[name]["test"]["residual_shrinkage_positive_frac"] = float(np.mean(shrinkage > 0))

    metrics["geometry"] = {
        "frozen_backbone": True,
        "v_phi_procrustes_drift": 0.0,
    }
    arrays_path = save_arrays(args.out_dir, data, base, gates)
    result = build_result(args, data, metrics, histories, arrays_path)

    out_path = args.out_dir / "scenario_gated_dbm_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}", flush=True)
    print(
        json.dumps(
            {
                "test_metrics": {
                    k: v["test"]
                    for k, v in metrics.items()
                    if k != "geometry"
                },
                "geometry": metrics["geometry"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
