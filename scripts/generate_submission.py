#!/usr/bin/env python3
"""Build submission.jsonl: one composed message per official test pair.

    python scripts/generate_submission.py            # LLM used only if a key is configured
    python scripts/generate_submission.py --no-llm   # deterministic templates only

Regenerates the expanded dataset (dataset/generate_dataset.py) into a temp dir, loads
test_pairs.json, composes every pair with app.composer.compose and writes one JSON object
per line with keys: test_id, body, cta, send_as, suppression_key, rationale.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYS = ("test_id", "body", "cta", "send_as", "suppression_key", "rationale")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def _load_dir(path: Path, key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(path.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        out[str(data.get(key) or f.stem)] = data
    return out


def expand_dataset(seed_dir: Path, out_dir: Path) -> None:
    script = seed_dir / "generate_dataset.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--seed-dir", str(seed_dir), "--out", str(out_dir)],
        capture_output=True, text=True, cwd=str(seed_dir),
    )
    if proc.returncode != 0:
        raise SystemExit(f"dataset generation failed:\n{proc.stdout}\n{proc.stderr}")


def load_expanded(data_dir: Path) -> dict:
    return {
        "categories": _load_dir(data_dir / "categories", "slug"),
        "merchants": _load_dir(data_dir / "merchants", "merchant_id"),
        "customers": _load_dir(data_dir / "customers", "customer_id"),
        "triggers": _load_dir(data_dir / "triggers", "id"),
        "pairs": json.loads((data_dir / "test_pairs.json").read_text(encoding="utf-8")).get("pairs", []),
    }


def compose_pairs(data: dict, *, use_llm: bool | None, now: str | None) -> list[tuple[dict, dict]]:
    from app import composer  # imported late so --no-llm can set the environment first

    rows = []
    for pair in data["pairs"]:
        trigger = data["triggers"].get(pair["trigger_id"])
        if trigger is None:
            raise SystemExit(f"{pair['test_id']}: trigger {pair['trigger_id']} not found")
        merchant_id = pair.get("merchant_id") or trigger.get("merchant_id")
        merchant = data["merchants"].get(merchant_id)
        if merchant is None:
            raise SystemExit(f"{pair['test_id']}: merchant {merchant_id} not found")
        category = data["categories"].get(merchant.get("category_slug") or "", {})
        customer_id = pair.get("customer_id") or trigger.get("customer_id")
        customer = data["customers"].get(customer_id) if customer_id else None
        started = time.perf_counter()
        out = composer.compose(category, merchant, trigger, customer, now=now, use_llm=use_llm)
        out = dict(out)
        out.setdefault("meta", {})
        out["meta"] = {**(out.get("meta") or {}), "elapsed_ms": int((time.perf_counter() - started) * 1000)}
        rows.append((pair, out))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "submission.jsonl"), help="output path (default: ./submission.jsonl)")
    ap.add_argument("--no-llm", action="store_true", help="force deterministic templates (no LLM calls)")
    ap.add_argument("--seed-dir", default=str(ROOT / "dataset"), help="directory with the seed files")
    ap.add_argument("--data-dir", default="", help="use an already-expanded dataset instead of regenerating")
    ap.add_argument("--now", default=None, help="simulated 'now' ISO timestamp passed to the composer")
    args = ap.parse_args(argv)

    if args.no_llm:
        os.environ["LLM_DISABLED"] = "1"
    sys.path.insert(0, str(ROOT))

    if args.data_dir:
        data = load_expanded(Path(args.data_dir))
    else:
        with tempfile.TemporaryDirectory(prefix="vera_expanded_") as tmp:
            expand_dataset(Path(args.seed_dir).resolve(), Path(tmp))
            data = load_expanded(Path(tmp))

    rows = compose_pairs(data, use_llm=False if args.no_llm else None, now=args.now)

    problems = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for pair, out in rows:
            record = {"test_id": pair["test_id"], **{k: out.get(k) for k in KEYS[1:]}}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{'test':<5} {'kind':<26} {'send_as':<18} {'cta':<22} {'src':<8} {'chars':>5}  body")
    print("-" * 120)
    for pair, out in rows:
        meta = out.get("meta") or {}
        body = str(out.get("body") or "")
        flags = []
        if not body.strip():
            flags.append("EMPTY")
        if URL_RE.search(body):
            flags.append("URL")
        if any(not out.get(k) for k in ("cta", "send_as", "suppression_key", "rationale")):
            flags.append("MISSING_FIELD")
        problems += bool(flags)
        preview = body.replace("\n", " ")[:48]
        print(f"{pair['test_id']:<5} {str(meta.get('kind') or '')[:26]:<26} {str(out.get('send_as')):<18} "
              f"{str(out.get('cta')):<22} {str(meta.get('source') or ''):<8} {len(body):>5}  {preview}"
              f"{'  [' + ','.join(flags) + ']' if flags else ''}")
    sources = [str((out.get("meta") or {}).get("source") or "?") for _p, out in rows]
    print("-" * 120)
    print(f"wrote {len(rows)} line(s) to {out_path}  |  llm: {sources.count('llm')}  template: "
          f"{sources.count('template')}  |  problems: {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
