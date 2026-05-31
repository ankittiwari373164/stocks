"""Central configuration, loaded from environment / .env."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv optional
    pass


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Config:
    # auth
    access_token: str = field(default_factory=lambda: _get("GROWW_ACCESS_TOKEN"))
    api_key: str = field(default_factory=lambda: _get("GROWW_API_KEY"))
    api_secret: str = field(default_factory=lambda: _get("GROWW_API_SECRET"))
    totp_secret: str = field(default_factory=lambda: _get("GROWW_TOTP_SECRET"))

    # prediction
    rank_metric: str = field(default_factory=lambda: _get("RANK_METRIC", "turnover"))
    open_from: str = field(default_factory=lambda: _get("OPEN_FROM", "09:15"))
    open_to: str = field(default_factory=lambda: _get("OPEN_TO", "09:25"))
    universe: str = field(default_factory=lambda: _get("UNIVERSE", "FNO"))
    custom_symbols: list[str] = field(
        default_factory=lambda: [s for s in _get("CUSTOM_SYMBOLS").split(",") if s.strip()]
    )
    model_mode: str = field(default_factory=lambda: _get("MODEL_MODE", "auto"))
    history_days: int = field(default_factory=lambda: int(_get("HISTORY_DAYS", "60") or 60))

    data_dir: Path = field(default_factory=lambda: Path(_get("DATA_DIR", "./data")))

    # constants
    base_url: str = "https://api.groww.in"
    instruments_url: str = "https://growwapi-assets.groww.in/instruments/instrument.csv"
    api_version: str = "1.0"

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rank_metric = self.rank_metric.lower()
        assert self.rank_metric in ("turnover", "volume"), "RANK_METRIC must be turnover|volume"

    @property
    def model_path(self) -> Path:
        return self.data_dir / "ranker_lgbm.txt"

    @property
    def shares_path(self) -> Path:
        return self.data_dir / "opening_shares.pkl"

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / "training_dataset.pkl"

    @property
    def universe_path(self) -> Path:
        return self.data_dir / "universe.csv"

    @property
    def predictions_path(self) -> Path:
        return self.data_dir / "predictions.json"

    @property
    def best_model_path(self) -> Path:
        return self.data_dir / "best_model.json"


CFG = Config()