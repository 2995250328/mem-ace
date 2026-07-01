#!/usr/bin/env python3
"""Safely rename local Codex conversation titles in state_*.sqlite files.

This edits the `threads.title` and `threads.preview` columns used by the local
Codex resume UI. It does not modify transcript JSONL files or first_user_message.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from typing import Iterable


def _default_backup_root() -> Path:
    preferred = Path("/data/xwh/.tmp/codex_title_backups")
    if preferred.parent.exists():
        return preferred
    return Path("/tmp/codex_title_backups")


def _existing_account_roots(home: Path, codex_home: str | None) -> list[Path]:
    roots: set[Path] = set()
    if codex_home:
        roots.add(Path(codex_home).expanduser().resolve())
    for candidate in [home / ".codex", *sorted(home.glob(".codex-acc*"))]:
        if candidate.is_dir():
            roots.add(candidate.resolve())
    return sorted(roots)


def _state_dbs(account_roots: Iterable[Path]) -> list[Path]:
    dbs: list[Path] = []
    for root in account_roots:
        dbs.extend(sorted(root.glob("state_*.sqlite")))
    return sorted(dict.fromkeys(p.resolve() for p in dbs))


def _has_threads_table(con: sqlite3.Connection) -> bool:
    row = con.execute(
        "select name from sqlite_master where type='table' and name='threads'"
    ).fetchone()
    return row is not None


def _backup_db(db: Path, backup_dir: Path) -> list[str]:
    copied: list[str] = []
    backup_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{db.parent.name}_{db.name}"
    for suffix in ("", "-wal", "-shm"):
        src = Path(str(db) + suffix)
        if src.exists():
            dst = backup_dir / f"{stem}{suffix}"
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def _select_thread(con: sqlite3.Connection, thread_id: str) -> list[tuple]:
    return con.execute(
        "select id, title, preview, first_user_message, rollout_path "
        "from threads where id=?",
        (thread_id,),
    ).fetchall()


def _make_backup_dir(root: Path, thread_id: str) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"{stamp}_{thread_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename local Codex resume conversation titles safely."
    )
    parser.add_argument(
        "--thread-id",
        default=os.environ.get("CODEX_THREAD_ID"),
        help="Codex thread id. Defaults to CODEX_THREAD_ID.",
    )
    parser.add_argument(
        "--title",
        help="New title to write into threads.title and threads.preview.",
    )
    parser.add_argument(
        "--account-root",
        action="append",
        type=Path,
        help="Codex account root to scan, e.g. /home/xwh/.codex-acc3. "
        "May be passed multiple times. Defaults to ~/.codex and ~/.codex-acc*.",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=_default_backup_root(),
        help="Directory where SQLite backups are created before applying.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list matching title rows; no write.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes. Without this flag, --title is a dry-run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.thread_id:
        print("ERROR: no --thread-id and CODEX_THREAD_ID is unset", file=sys.stderr)
        return 2
    if not args.list and not args.title:
        print("ERROR: provide --title or --list", file=sys.stderr)
        return 2

    home = Path.home()
    if args.account_root:
        account_roots = [p.expanduser().resolve() for p in args.account_root]
    else:
        account_roots = _existing_account_roots(home, os.environ.get("CODEX_HOME"))

    dbs = _state_dbs(account_roots)
    backup_dir = _make_backup_dir(args.backup_root.expanduser(), args.thread_id)
    results: list[dict] = []
    matched = 0

    for db in dbs:
        try:
            con = sqlite3.connect(db)
        except sqlite3.Error as exc:
            results.append({"db": str(db), "error": f"open failed: {exc}"})
            continue
        try:
            if not _has_threads_table(con):
                continue
            rows = _select_thread(con, args.thread_id)
            if not rows:
                continue
            matched += len(rows)
            before = [
                {
                    "id": row[0],
                    "title": row[1],
                    "preview": row[2],
                    "first_user_message_prefix": (row[3] or "")[:160],
                    "rollout_path": row[4],
                }
                for row in rows
            ]
            result = {
                "account_root": str(db.parent),
                "db": str(db),
                "before": before,
                "mode": "list" if args.list else ("apply" if args.apply else "dry-run"),
            }
            if args.title and args.apply:
                copied = _backup_db(db, backup_dir)
                con.execute(
                    "update threads set title=?, preview=? where id=?",
                    (args.title, args.title, args.thread_id),
                )
                con.commit()
                result["backup_files"] = copied
                result["after"] = [
                    {"id": row[0], "title": row[1], "preview": row[2]}
                    for row in con.execute(
                        "select id, title, preview from threads where id=?",
                        (args.thread_id,),
                    ).fetchall()
                ]
            elif args.title:
                result["would_write"] = {"title": args.title, "preview": args.title}
            results.append(result)
        except sqlite3.Error as exc:
            results.append({"db": str(db), "error": str(exc)})
        finally:
            con.close()

    status = {
        "thread_id": args.thread_id,
        "account_roots": [str(p) for p in account_roots],
        "matched_rows": matched,
        "backup_dir": str(backup_dir) if args.apply and args.title and matched else None,
        "results": results,
    }

    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print(f"thread_id: {args.thread_id}")
        print(f"matched_rows: {matched}")
        if args.apply and args.title and matched:
            print(f"backup_dir: {backup_dir}")
        for item in results:
            if "error" in item:
                print(f"[ERROR] {item.get('db')}: {item['error']}")
                continue
            print(f"\n[{item['mode']}] {item['db']}")
            for row in item["before"]:
                print(f"  before.title: {row['title']}")
                print(f"  before.preview: {row['preview']}")
            if "would_write" in item:
                print(f"  would.title: {item['would_write']['title']}")
                print(f"  would.preview: {item['would_write']['preview']}")
            if "after" in item:
                for row in item["after"]:
                    print(f"  after.title: {row['title']}")
                    print(f"  after.preview: {row['preview']}")

    if matched == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
