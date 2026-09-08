from __future__ import annotations

import argparse
from pathlib import Path

from akm_hrp.data.wrds_crsp import CRSPQueryConfig, fetch_crsp_sector_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download point-in-time CRSP UES/ICB/SIC sector history."
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--output",
        default="wrds_full_clean/crsp_sector_history.csv.gz",
    )
    parser.add_argument("--library", default=None)
    parser.add_argument("--wrds-username", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - optional remote dependency
        raise RuntimeError(
            "Install the 'wrds' package before using this command."
        ) from exc

    connection = wrds.Connection(wrds_username=args.wrds_username)
    try:
        history = fetch_crsp_sector_history(
            connection,
            args.start,
            args.end,
            CRSPQueryConfig(library=args.library),
        )
    finally:
        connection.close()

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(destination, index=False)
    print(f"rows: {len(history):,}")
    print(f"PERMNOs: {history['permno'].nunique():,}")
    print(f"sectors: {history['sector'].nunique():,}")
    print(f"saved: {destination.resolve()}")


if __name__ == "__main__":
    main()
