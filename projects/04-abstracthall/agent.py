#!/usr/bin/env python3
"""
agent.py — Repair loop agent.

Ties solver + checker together into an agentic loop:
  1. Build initial prompt from abstract
  2. Solve (call LLM)
  3. Check reply for hallucinations
  4. If hallucinated, feed repair_prompt back into solver and retry
  5. Repeat up to --max-rounds times or until reward >= --reward-threshold
  6. Log every round to a JSONL file

Checker is swappable via --checker sumcheck | checkerprompts
"""

import argparse
import hashlib
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import solver
import sumcheck
import checkerprompts


# ---------------------------------------------------------------------------
# Checker registry — swap with --checker flag
# ---------------------------------------------------------------------------
CHECKERS = {
    "sumcheck":       sumcheck,
    "checkerprompts": checkerprompts,
}


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

def run_item(item, arm, checker, max_rounds=3, reward_threshold=0.9, samples=1):
    """
    Run the best-of-N repair loop for a single abstract.

    Each round generates `samples` replies in parallel, grades all of them,
    picks the one with the highest reward as the "best", and uses its
    repair_prompt for the next round.

    Parameters
    ----------
    item             : dict   one abstract from abstracts.jsonl
    arm              : Arm    solver generation settings
    checker          : module sumcheck or checkerprompts
    max_rounds       : int    max attempts including the first one
    reward_threshold : float  stop early if best reward >= this value
    samples          : int    number of parallel samples per round (1–4)

    Yields one record dict per sample per round (ready to write as JSONL).
    """
    prompt = "Summarise: " + item["text"]   # baseline prompt for round 1
    best = None  # best record across the current round

    for round_num in range(1, max_rounds + 1):
        replies = solver.solve(prompt, arm, samples)
        round_records = []

        for sample_idx, reply in enumerate(replies, start=1):
            record = dict(
                item_id=item["id"],
                round=round_num,
                sample=sample_idx,
                samples_per_round=samples,
                checker=checker.__name__,
                prompt=prompt,
                reply_text=reply.text,
                reply_status=reply.status,
                reply_latency_s=reply.latency_s,
                reply_error=reply.error,
                input_tokens=reply.input_tokens,
                recorded_at=datetime.now(timezone.utc).isoformat(),
                source_sha256=hashlib.sha256(item["text"].encode()).hexdigest(),
            )

            # Skip grading if solver failed
            if reply.status in ("error", "input_too_long", "empty", "truncated"):
                record.update(reward=None, h_score=None, label=reply.status,
                              repair_prompt=None, repair_method=None,
                              done=False, is_best_in_round=False)
                round_records.append(record)
                yield record
                continue

            # Grade the reply
            result = checker.check(item=item, reply=reply.text)
            record.update(
                reward=result["reward"],
                h_score=result["h_score"],
                total_claims=result.get("total_claims"),
                supported=result.get("supported"),
                contradicted=result.get("contradicted"),
                not_established=result.get("not_established"),
                label=result["label"],
                feedback=result["feedback"],
                repair_prompt=result.get("repair_prompt"),
                repair_method=result.get("repair_method"),
                done=result["reward"] >= reward_threshold,
                is_best_in_round=False,   # updated below after all samples graded
            )
            round_records.append(record)

        # --- Best-of-N selection ---
        # Pick the sample with the highest reward among graded records
        graded = [r for r in round_records if r.get("reward") is not None]
        if graded:
            best = max(graded, key=lambda r: r["reward"])
            best["is_best_in_round"] = True

        # Yield graded records now that is_best_in_round is marked
        # (error records were already yielded above)
        for r in round_records:
            if r.get("reward") is not None:
                yield r

        if best is None:
            # All samples failed at solver level
            print(f"  [{item['id']}] round {round_num}: all samples failed, stopping")
            break

        print(f"  [{item['id']}] round {round_num}: "
              f"best reward={best['reward']} (sample {best['sample']}/{samples}) "
              f"h={best.get('h_score')} label={best['label']}"
              + (" ✓ done" if best["done"] else
                 f" → retrying (method={best.get('repair_method', 'plain')})"))

        if best["done"]:
            break

        if best["repair_prompt"] is None:
            # Checker says faithful but below threshold — nothing to fix
            break

        # Next round uses the best sample's repair prompt
        prompt = best["repair_prompt"]

    else:
        print(f"  [{item['id']}] reached max_rounds={max_rounds}, "
              f"best reward={best['reward'] if best else 'n/a'}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def report(results, checker_name, reward_threshold):
    """
    Print an end-of-run summary table.

    results is a list of dicts, one per item, with keys:
      item_id, round_1_reward, final_reward, rounds_used, done, label
    """
    if not results:
        return

    n = len(results)
    done        = sum(1 for r in results if r["done"])
    improved    = sum(1 for r in results if r["final_reward"] > r["round_1_reward"])
    max_hit     = sum(1 for r in results if not r["done"])
    avg_r1      = sum(r["round_1_reward"] for r in results) / n
    avg_final   = sum(r["final_reward"]   for r in results) / n
    avg_rounds  = sum(r["rounds_used"]    for r in results) / n

    # label breakdown on final round
    from collections import Counter
    labels = Counter(r["label"] for r in results)

    print("\n" + "═" * 52)
    print(f"  Run summary — checker: {checker_name}")
    print("═" * 52)
    print(f"  Items              : {n}")
    print(f"  Threshold          : reward >= {reward_threshold}")
    print(f"  Reached threshold  : {done}/{n}")
    print(f"  Improved over r1   : {improved}/{n}")
    print(f"  Max rounds hit     : {max_hit}/{n}")
    print(f"  Avg reward round-1 : {avg_r1:.3f}")
    print(f"  Avg reward final   : {avg_final:.3f}")
    print(f"  Avg rounds used    : {avg_rounds:.1f}")
    print("\n  Final label breakdown:")
    for label, count in labels.most_common():
        print(f"    {label:<32} {count}")
    print("═" * 52)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",    type=Path, default=root / "data/abstracts.jsonl")
    parser.add_argument("--id",      help="Run only this abstract ID")
    parser.add_argument("--checker", choices=CHECKERS.keys(), default="sumcheck",
                        help="Which checker to use (default: sumcheck)")
    parser.add_argument("--max-rounds",        type=int,   default=3)
    parser.add_argument("--reward-threshold",  type=float, default=0.9,
                        help="Stop retrying when reward >= this value (default: 0.9)")
    parser.add_argument("--samples",  type=int, choices=range(1, 5), default=1)
    parser.add_argument("--max-tokens",  type=int,   default=300)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--timeout",     type=float, default=900)
    parser.add_argument("--input-limit", type=int,   default=8100)
    parser.add_argument("--base",  default=(os.getenv("CR_BASE_URL") or
                                            os.getenv("KERNEL_AGENT_BASE_URL") or
                                            os.getenv("HEATROD_BASE_URL")))
    parser.add_argument("--model", default=(os.getenv("CR_MODEL") or
                                            os.getenv("KERNEL_AGENT_MODEL") or
                                            "Qwen/Qwen3-8B"))
    parser.add_argument("--no-qwen-template", action="store_true")
    parser.add_argument("--out", type=Path,
                        help="Output JSONL file (default: runs/agent-<checker>-<uuid>.jsonl)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of abstracts to process in parallel (default: 1)")
    args = parser.parse_args()

    if not args.base:
        parser.error("No API URL: set CR_BASE_URL or pass --base http://localhost:8000/v1")

    items = solver.load_items(args.data)
    if args.id:
        items = [i for i in items if i["id"] == args.id]
        if not items:
            parser.error(f"Abstract ID '{args.id}' not found in dataset")

    arm = solver.Arm(
        base_url=args.base,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        qwen_template=not args.no_qwen_template,
        input_limit=args.input_limit,
    )

    checker = CHECKERS[args.checker]
    run_id  = uuid.uuid4().hex
    output  = args.out or root / "runs" / f"agent-{args.checker}-{run_id}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Checker : {args.checker}")
    print(f"Rounds  : up to {args.max_rounds}")
    print(f"Threshold: reward >= {args.reward_threshold}")
    print(f"Output  : {output}")

    run_results = []   # collects one dict per item for the summary
    with output.open("x", encoding="utf-8") as log:
        def process_item(item):
            print(f"\n[{item['id']}]")
            item_round1_reward = None
            item_final_reward  = None
            item_final_label   = None
            item_rounds_used   = 0
            item_done          = False
            records = []

            for record in run_item(item, arm, checker,
                                   max_rounds=args.max_rounds,
                                   reward_threshold=args.reward_threshold,
                                   samples=args.samples):
                record["run_id"] = run_id
                records.append(record)

                if record.get("reward") is not None and record.get("is_best_in_round"):
                    if record["round"] == 1:
                        item_round1_reward = record["reward"]
                    item_final_reward = record["reward"]
                    item_final_label  = record["label"]
                    item_rounds_used  = record["round"]
                    item_done         = record["done"]

            summary = None
            if item_round1_reward is not None:
                summary = dict(
                    item_id=item["id"],
                    round_1_reward=item_round1_reward,
                    final_reward=item_final_reward,
                    rounds_used=item_rounds_used,
                    done=item_done,
                    label=item_final_label,
                )
            return records, summary

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(process_item, item): item for item in items}
            for future in as_completed(futures):
                records, summary = future.result()
                for record in records:
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                if summary:
                    run_results.append(summary)

    report(run_results, args.checker, args.reward_threshold)
    print(f"\nFull results in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
