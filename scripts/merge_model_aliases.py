#!/usr/bin/env python3
"""Merge model identifiers that name the same model, and recompute ELO.

Why a replay and not a rename: ELO is path-dependent. Each match updates both
players by an amount derived from their ratings *at that moment*, so collapsing
two identifiers into one changes not only the merged model's trajectory but also
every opponent's. The only coherent result comes from replaying the whole
comparison log through the project's own `ELORegistry.record_comparison`.

Safety properties:

* Refuses to run while a benchmark process holds the registry. `ELORegistry.save`
  writes the full in-memory state via temp+rename, so a concurrent writer does
  not corrupt the file — it silently reverts our merge and resurrects the old
  identifiers. Atomic write is not concurrent-safe write.
* `--verify` replays with an EMPTY rename map. The result must reproduce the
  current registry; if it does not, this harness computes differently from the
  benchmark and its merged output cannot be trusted. Run this before `--apply`.
* `--apply` refuses without a fresh backup directory.

Original match timestamps are restored after the replay: `record_comparison`
stamps its own `now`, which would rewrite the audit trail's history.

Usage:
    python scripts/merge_model_aliases.py --verify
    python scripts/merge_model_aliases.py --dry-run
    python scripts/merge_model_aliases.py --apply --backup data/_merge_backup_20260814
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ukrqualbench.core.elo_registry import ELORegistry

REGISTRY = Path("data/elo_registry.json")
LOG = Path("data/elo_registry.log.json")

# Identifiers that denote the same model. Chosen by the repository owner:
# canonical `gpt-5.2` — neutral name, no date and no provider prefix.
RENAMES: dict[str, str] = {
    "gpt-5.2-2025-12-11": "gpt-5.2",
    "openrouter:openai/gpt-5.2": "gpt-5.2",
}

# Tolerance for the verify pass. Ratings are float sums of the same operations
# in the same order, so agreement should be near-exact; this only absorbs JSON
# round-tripping of the stored values.
EPSILON = 1e-6

# The out-of-band registry edits this data is known to carry, derived from the
# log's own rating chain (see chain_breaks) before any replay was written.
#
# Asserting the exact set is what gives the verify pass teeth. The repair
# mechanism resynchronises the replay whenever it disagrees with the log, so a
# harness that computed ELO incorrectly would still end up matching the stored
# ratings — it would just need a repair at nearly every match. Pinning the set
# to these four turns "we agree in the end" into "we agree throughout, except
# exactly here".
KNOWN_OUT_OF_BAND_EDITS: set[tuple[str, str]] = {
    ("2026-05-19T23:20:22", "openrouter:openai/gpt-5.2"),
    ("2026-05-19T23:21:19", "openrouter:google/gemma-4-31b-it"),
    ("2026-05-19T23:21:54", "openrouter:deepseek/deepseek-v4-flash"),
    ("2026-05-19T23:22:04", "gemini-3.5-flash"),
}


def assert_no_writers() -> None:
    """Abort if a benchmark process could overwrite our result."""
    # The [h] keeps the pattern from matching this script's own command line.
    proc = subprocess.run(
        ["pgrep", "-f", "venv/bin/ukrqualbenc[h]"], capture_output=True, text=True
    )
    pids = [p for p in proc.stdout.split() if p]
    if pids:
        sys.exit(
            f"ОТКАЗ: работают процессы ukrqualbench (PID {', '.join(pids)}).\n"
            "Они держат реестр в памяти и перезапишут результат слияния "
            "своим состоянием со старыми идентификаторами. Дождитесь завершения."
        )


def replay(
    log_entries: list[dict],
    renames: dict[str, str],
    *,
    repairs: list[tuple[str, str, float, float]] | None = None,
) -> ELORegistry:
    """Replay the comparison log into a fresh registry, applying `renames`.

    When `repairs` is a list, the replay re-applies out-of-band registry edits:
    before each match, any player whose computed rating disagrees with the
    rating the log says it had is forced to the logged value, and the event is
    appended to `repairs`. That makes the replay track the recorded history
    exactly, so reproducing the registry becomes a real assertion about the
    arithmetic in between — and any *unknown* edit shows up as an extra repair.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="elo_replay_"))
    reg = ELORegistry(registry_path=tmpdir / "elo_registry.json")

    def canon(model_id: str) -> str:
        return renames.get(model_id, model_id)

    for rec in log_entries:
        if repairs is not None:
            for side in ("a", "b"):
                model = canon(str(rec[f"model_{side}"]))
                logged = rec[f"old_rating_{side}"]
                entry = reg.models.get(model)
                have = entry.rating if entry else 1500.0
                if abs(have - logged) > EPSILON:
                    repairs.append((rec["timestamp"][:19], model, round(have, 2), round(logged, 2)))
                    reg.register_model(model).rating = logged

        judge = rec.get("judge_id")
        reg.record_comparison(
            model_a=canon(str(rec["model_a"])),
            model_b=canon(str(rec["model_b"])),
            winner=rec["winner"],
            prompt_id=rec.get("prompt_id"),
            judge_id=canon(judge) if judge else judge,
            run_id=rec.get("run_id"),
            benchmark_version=rec.get("benchmark_version"),
        )

    # record_comparison stamps `now`; restore the real history. Order is
    # preserved because we replayed the log sequentially.
    for produced, original in zip(reg._comparison_log, log_entries, strict=True):
        produced.timestamp = original["timestamp"]

    return reg


def ratings_of(reg: ELORegistry) -> dict[str, float]:
    return {mid: e.rating for mid, e in reg.models.items()}


def current_ratings() -> dict[str, float]:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {e["model_id"]: e["rating"] for e in data["models"]}


def chain_breaks(log_entries: list[dict]) -> dict[str, list[str]]:
    """Find models whose rating chain is discontinuous in the log.

    Each record stores the rating both players had before and after the match,
    so for any model the previous record's `new_rating` must equal the next
    record's `old_rating`. A break means the registry was changed outside the
    log — a manual edit, or a concurrent writer whose `save()` overwrote another
    process's state. Those models cannot be reproduced by replaying the log.

    This derivation is independent of the replay: it reads only the log's own
    recorded values. That independence is what lets `--apply` accept a known
    discrepancy without weakening the check into a rubber stamp.
    """
    last: dict[str, float] = {}
    breaks: dict[str, list[str]] = {}
    for rec in log_entries:
        for side in ("a", "b"):
            model = rec[f"model_{side}"]
            old = rec[f"old_rating_{side}"]
            prev = last.get(model)
            broken = abs(old - 1500.0) > EPSILON if prev is None else abs(prev - old) > EPSILON
            if broken:
                breaks.setdefault(model, []).append(
                    f"{rec['timestamp'][:19]}: {'—' if prev is None else round(prev, 2)} -> {round(old, 2)}"
                )
            last[model] = rec[f"new_rating_{side}"]
    return breaks


def cmd_verify(log_entries: list[dict]) -> int:
    """Prove the harness reproduces the registry once known edits are re-applied.

    A plain replay cannot match the stored ratings, because those ratings carry
    changes the log never recorded as matches (a manual rename and a concurrent
    -writer incident, both on 2026-05-19). Re-applying exactly those edits at the
    points the log itself marks them, and then requiring an exact match, tests
    the part that must be right — the arithmetic between the edits — instead of
    excusing the difference.
    """
    expected = current_ratings()
    repairs: list[tuple[str, str, float, float]] = []
    got = ratings_of(replay(log_entries, {}, repairs=repairs))

    print(f"Внелоговых правок переприменено: {len(repairs)}")
    for ts, model, have, logged in repairs:
        print(f"  {ts}  {model:42} {have} -> {logged}")

    seen = {(ts, model) for ts, model, _, _ in repairs}
    if seen != KNOWN_OUT_OF_BAND_EDITS:
        print("САМОПРОВЕРКА ПРОВАЛЕНА — состав внелоговых правок не тот, что ожидался.")
        for ts, model in sorted(seen - KNOWN_OUT_OF_BAND_EDITS):
            print(f"  НОВАЯ правка (не была известна): {ts}  {model}")
        for ts, model in sorted(KNOWN_OUT_OF_BAND_EDITS - seen):
            print(f"  ОЖИДАЛАСЬ, но не найдена:        {ts}  {model}")
        print(
            "  Много новых правок = харнесс считает ELO не так, как бенчмарк.\n"
            "  Единичная новая правка = данные изменились с момента анализа."
        )
        return 1

    missing = set(expected) - set(got)
    extra = set(got) - set(expected)
    drift = {
        m: (expected[m], got[m])
        for m in set(expected) & set(got)
        if abs(expected[m] - got[m]) > EPSILON
    }

    # `gpt-5.2-2025-12-11` has matches in the log but was deleted from the
    # registry by the May rename, so the replay legitimately produces it.
    extra -= set(RENAMES)

    if missing or extra or drift:
        print("САМОПРОВЕРКА ПРОВАЛЕНА — расхождение не объясняется известными правками.")
        if missing:
            print("  нет в реплее:", sorted(missing))
        if extra:
            print("  лишние в реплее:", sorted(extra))
        for m, (exp, act) in sorted(drift.items()):
            print(f"  {m:45} реестр={exp:9.2f}  реплей={act:9.2f}  Δ={act - exp:+.2f}")
        return 1

    print(
        f"САМОПРОВЕРКА ПРОЙДЕНА: реплей воспроизвёл {len(expected)} рейтингов "
        f"из {len(log_entries)} матчей при {len(repairs)} переприменённых правках."
    )
    return 0


def cmd_preview(log_entries: list[dict]) -> int:
    """Show what the merge would change, writing nothing."""
    before = current_ratings()
    after = ratings_of(replay(log_entries, RENAMES))

    print(f"{'МОДЕЛЬ':45} {'БЫЛО':>9} {'СТАНЕТ':>9} {'Δ':>8}")
    for m in sorted(set(before) | set(after), key=lambda x: -after.get(x, 0)):
        b = before.get(m)
        a = after.get(m)
        if b is None:
            print(f"{m:45} {'—':>9} {a:9.1f} {'НОВАЯ':>8}")
        elif a is None:
            print(f"{m:45} {b:9.1f} {'—':>9} {'СЛИТА':>8}")
        else:
            print(f"{m:45} {b:9.1f} {a:9.1f} {a - b:+8.1f}")
    return 0


def cmd_apply(log_entries: list[dict], backup: Path) -> int:
    if not backup.is_dir():
        sys.exit(f"ОТКАЗ: каталог бэкапа не найден: {backup}")

    if cmd_verify(log_entries) != 0:
        sys.exit("ОТКАЗ: слияние не применяется, пока самопроверка не пройдена.")

    breaks = chain_breaks(log_entries)
    print(f"\nРазрывов цепочки рейтингов в логе: {sum(len(v) for v in breaks.values())}")
    for model, events in sorted(breaks.items()):
        for ev in events:
            print(f"  {model:42} {ev}")
    print("Пересчёт отменяет эти внелоговые правки — таково подтверждённое решение владельца.\n")

    # The merged replay runs WITHOUT repairs: the point of the recompute is a
    # history derived from the matches alone.
    reg = replay(log_entries, RENAMES)
    reg.registry_path = REGISTRY
    reg.log_path = LOG
    reg.save()
    print(f"Применено. Реестр: {REGISTRY}, лог: {LOG}. Бэкап: {backup}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true", help="самопроверка харнесса")
    g.add_argument("--dry-run", action="store_true", help="показать эффект слияния")
    g.add_argument("--apply", action="store_true", help="применить слияние")
    ap.add_argument("--backup", type=Path, help="каталог бэкапа (обязателен для --apply)")
    args = ap.parse_args()

    assert_no_writers()
    log_entries = json.loads(LOG.read_text(encoding="utf-8"))

    if args.verify:
        return cmd_verify(log_entries)
    if args.dry_run:
        return cmd_preview(log_entries)
    return cmd_apply(log_entries, args.backup or Path("/nonexistent"))


if __name__ == "__main__":
    raise SystemExit(main())
