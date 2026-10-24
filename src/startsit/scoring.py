"""Scoring formats: the one place that knows how PPR, half-PPR and standard differ.

The formats differ only in points per reception, so any PPR value converts to another
format by removing the per-catch points it doesn't award. `from_ppr` is plain
arithmetic, so it works on floats and on polars expressions alike.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args, overload

if TYPE_CHECKING:
    import polars as pl

Scoring = Literal["ppr", "half", "std"]
SCORINGS: tuple[Scoring, ...] = get_args(Scoring)
LABELS: dict[Scoring, str] = {"ppr": "PPR", "half": "half-PPR", "std": "standard"}
POINTS_PER_RECEPTION: dict[Scoring, float] = {"ppr": 1.0, "half": 0.5, "std": 0.0}


def reception_discount(scoring: Scoring) -> float:
    """Points per catch that PPR awards and `scoring` doesn't (0 for PPR itself)."""
    return 1.0 - POINTS_PER_RECEPTION[scoring]


@overload
def from_ppr(ppr_points: float, receptions: float, scoring: Scoring) -> float: ...
@overload
def from_ppr(ppr_points: pl.Expr, receptions: pl.Expr, scoring: Scoring) -> pl.Expr: ...
def from_ppr(ppr_points: Any, receptions: Any, scoring: Scoring) -> Any:
    """Convert PPR points to `scoring` given the receptions behind them (floats or polars expressions)."""
    return ppr_points - reception_discount(scoring) * receptions


def column(prefix: str, scoring: Scoring, suffix: str = "") -> str:
    """Name of a per-format column, e.g. column("xfp", "half", "_l3") -> "xfp_half_l3"."""
    return f"{prefix}_{scoring}{suffix}"


# Per-format point columns stored for every player-week: actual (fp_*) and expected (xfp_*).
POINT_COLUMNS = [column(prefix, s) for prefix in ("fp", "xfp") for s in SCORINGS]
