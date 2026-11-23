"""Command line: ``grameen <command>``. Exit codes: 0 ok, 1 error, 2 blocked (data missing or unfit)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

from . import bids, paths, schema, synthetic, workflow
from .fitting import BlockedError, fit_dataset

OK, ERROR, BLOCKED = 0, 1, 2
ACCESS_MESSAGE = (
    "No borrower-level data supplied. The paper's data are available on request from its corresponding "
    "author (see docs/ACCESS_ORIGINAL_DATA.md); this repository never contacts anyone on your behalf.\n"
    "Once you have data: grameen import-data --source your.csv --meta your.meta.yaml, then grameen fit-models.\n"
    "To see the code run on fictional rows: grameen run-synthetic")


def _root_out(*parts: str) -> Path:
    return paths.repo_root().joinpath(*parts)


def cmd_reproduce(args: argparse.Namespace) -> int:
    written = workflow.reproduce_paper(Path(args.out) if args.out else _root_out("results", "published"))
    print(f"reproduce-paper: wrote {len(written)} files (dataset type: published_aggregate)")
    return OK


def cmd_run_synthetic(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else _root_out("results", "synthetic")
    summary, written = workflow.run_synthetic(out, seed=args.seed, n=args.n, autonomy_type=args.autonomy_type)
    fitted = int((summary.status["status"] == "fitted").sum())
    print(f"run-synthetic: SYNTHETIC data, {fitted} outcomes fitted, {len(written)} files in {out}")
    return OK


def cmd_validate(args: argparse.Namespace) -> int:
    if args.data:
        if not args.meta:
            print("validate-data: --meta is required with --data", file=sys.stderr)
            return ERROR
        rep = schema.validate_microdata(pd.read_csv(args.data), schema.load_meta(Path(args.meta)))
        print(rep.render())
        return OK if rep.ok else BLOCKED
    results = workflow.validate_bundled()
    bad = False
    for name, problems in results.items():
        print(f"{name}: {'ok' if not problems else 'PROBLEMS'}")
        for p in problems:
            print(f"  - {p}")
        bad = bad or bool(problems)
    return BLOCKED if bad else OK


def cmd_import_data(args: argparse.Namespace) -> int:
    df, meta = pd.read_csv(args.source), schema.load_meta(Path(args.meta))
    if args.column_map:
        mapping = yaml.safe_load(Path(args.column_map).read_text(encoding="utf-8"))
        df = df.rename(columns=mapping)
    rep = schema.validate_microdata(df, meta)
    print(rep.render())
    if not rep.ok:
        return BLOCKED
    csv_path, meta_path = paths.original_data()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, lineterminator="\n")
    shutil.copyfile(args.meta, meta_path)
    print(f"imported {len(df)} rows to {csv_path} (git-ignored; never commit row-level data)")
    return OK


def cmd_fit(args: argparse.Namespace) -> int:
    default_csv, default_meta = paths.original_data()
    csv_path, meta_path = Path(args.data or default_csv), Path(args.meta or default_meta)
    if not csv_path.exists() or not meta_path.exists():
        print(ACCESS_MESSAGE, file=sys.stderr)
        return BLOCKED
    meta = schema.load_meta(meta_path)
    dtype = meta.get("dataset_type", "original_microdata")
    folder = {"synthetic": "synthetic", "independent_bids": "bids"}.get(dtype, "original")
    out = Path(args.out) if args.out else _root_out("results", folder)
    try:
        summary = fit_dataset(pd.read_csv(csv_path), meta, out, outcomes=args.outcomes or None,
                              allow_unweighted=args.allow_unweighted)
    except BlockedError as exc:
        print(exc, file=sys.stderr)
        return BLOCKED
    print(summary.status[["outcome", "status", "message"]].to_string(index=False))
    print(f"fit-models: dataset type {dtype}; wrote {len(summary.files)} tables to {out}")
    return OK


def cmd_import_bids(args: argparse.Namespace) -> int:
    try:
        mapping = bids.load_mapping(Path(args.mapping))
        df = bids.import_bids(Path(args.input), mapping)
    except (bids.ConfigIncomplete, FileNotFoundError, ValueError) as exc:
        print(f"import-bids blocked: {exc}", file=sys.stderr)
        return BLOCKED
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")
    print(f"import-bids: {len(df)} rows -> {out}. Unsupported measures: "
          + ", ".join(r["target"] for r in bids.FIELD_MAP if r["status"] == "unsupported"))
    return OK


def cmd_report(args: argparse.Namespace) -> int:
    written = workflow.build_report(Path(args.out) if args.out else _root_out("reports"), refresh=not args.no_refresh)
    print(f"build-report: wrote {len(written)} files; report at {written[-1]}")
    return OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="grameen", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("reproduce-paper", help="rebuild Figures 1-4 from the transcribed published values")
    s.add_argument("--out")
    s.set_defaults(func=cmd_reproduce)
    s = sub.add_parser("run-synthetic", help="seeded fictional fixture + regression report (SYNTHETIC)")
    s.add_argument("--out")
    s.add_argument("--seed", type=int, default=synthetic.DEFAULT_SEED)
    s.add_argument("--n", type=int, default=synthetic.DEFAULT_N)
    s.add_argument("--autonomy-type", default="continuous", choices=["continuous", "binary", "ordinal"])
    s.set_defaults(func=cmd_run_synthetic)
    s = sub.add_parser("validate-data", help="validate bundled files, or --data/--meta row-level data")
    s.add_argument("--data")
    s.add_argument("--meta")
    s.set_defaults(func=cmd_validate)
    s = sub.add_parser("import-data", help="validate and copy your row-level data into data/original (git-ignored)")
    s.add_argument("--source", required=True)
    s.add_argument("--meta", required=True)
    s.add_argument("--column-map")
    s.set_defaults(func=cmd_import_data)
    s = sub.add_parser("fit-models", help="fit the Table 3 models; blocked without row-level data")
    s.add_argument("--data")
    s.add_argument("--meta")
    s.add_argument("--out")
    s.add_argument("--outcomes", nargs="*")
    s.add_argument("--allow-unweighted", action="store_true")
    s.set_defaults(func=cmd_fit)
    s = sub.add_parser("import-bids", help="import supported fields from a local World Bank/BIDS download")
    s.add_argument("--input", required=True)
    s.add_argument("--mapping", default=str(paths.config_dir() / "bids_mapping.yaml"))
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_import_bids)
    s = sub.add_parser("build-report", help="assemble reports/report.md")
    s.add_argument("--out")
    s.add_argument("--no-refresh", action="store_true")
    s.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return ERROR


if __name__ == "__main__":
    sys.exit(main())
