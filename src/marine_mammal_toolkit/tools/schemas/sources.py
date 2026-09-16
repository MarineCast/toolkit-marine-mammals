"""Portable acquisition settings, independent of species selection."""

from __future__ import annotations
from datetime import date
from pathlib import Path
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo
from pydantic import Field, model_validator, field_validator
from marine_mammal_toolkit.tools._core.config.models import StrictConfig


class BBox(StrictConfig):
    min_lon: float = Field(ge=-180, le=180)
    min_lat: float = Field(ge=-90, le=90)
    max_lon: float = Field(ge=-180, le=180)
    max_lat: float = Field(ge=-90, le=90)

    def tuple(self) -> tuple[float, float, float, float]:
        return self.min_lon, self.min_lat, self.max_lon, self.max_lat

    @model_validator(mode="after")
    def validate_order(self) -> "BBox":
        if self.min_lon >= self.max_lon or self.min_lat >= self.max_lat:
            raise ValueError("Bounding-box minimums must be smaller than maximums")
        return self


class GbifDatasetSettings(StrictConfig):
    key: str
    title: str
    event_id_is_encounter: bool = False
    happywhale_exact_fallback: bool = False
    use_class: Literal["REDISTRIBUTABLE", "INTERNAL_ONLY"]

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        UUID(value)
        return value


class CwrAtlistMapSettings(StrictConfig):
    map_id: str
    page_url: str

    @field_validator("map_id")
    @classmethod
    def validate_map_id(cls, value: str) -> str:
        UUID(value)
        return value


class SourceSettings(StrictConfig):
    taxon_id: int | None = None
    enabled: bool = True
    url: str | None = None
    credential_env: str | None = None
    local_path: Path | None = None
    timezone: str = "UTC"
    timeout_seconds: int = Field(default=60, gt=0)
    max_retries: int = Field(default=3, ge=0)
    created_is_event_time: bool = False
    date_formats: tuple[str, ...] = ("%Y-%m-%d",)
    bbox: BBox | None = None
    min_date: str | None = None
    verified_coverage_start: str | None = None
    verified_coverage_through: str | None = None
    dataset_metadata_url: str | None = None
    taxon_key: str | None = None
    checklist_key: str | None = None
    basis_of_record: str | None = None
    occurrence_status: str | None = None
    require_no_geospatial_issue: bool = False
    page_size: int = Field(default=300, gt=0, le=300)
    max_search_results: int = Field(default=100_000, gt=0, le=100_000)
    max_coordinate_uncertainty_m: float | None = Field(default=None, gt=0)
    max_event_spread_km: float = Field(default=5.0, gt=0)
    dataset_allowlist: tuple[GbifDatasetSettings, ...] = ()
    excluded_dataset_keys: tuple[str, ...] = ()
    archive_index_url: str | None = None
    archive_years: tuple[int, ...] = ()
    archive_year_url_template: str | None = None
    archive_fetch_workers: int = Field(default=8, gt=0, le=32)
    coordinate_bounds: BBox | None = None
    atlist_api_root: str | None = None
    atlist_maps: dict[int, CwrAtlistMapSettings] = Field(default_factory=dict)
    source_license: str | None = None
    source_use_class: Literal["REDISTRIBUTABLE", "INTERNAL_ONLY"] | None = None
    source_license_terms_url: str | None = None
    source_attribution: str | None = None
    source_license_reviewed_at: str | None = None
    source_license_version: str | None = None
    source_license_jurisdiction: str | None = None

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        ZoneInfo(value)
        return value

    @field_validator("excluded_dataset_keys")
    @classmethod
    def validate_excluded_dataset_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for key in value:
            UUID(key)
        if len(value) != len(set(value)):
            raise ValueError("excluded_dataset_keys must be unique")
        return value

    @field_validator("min_date", "verified_coverage_start", "verified_coverage_through")
    @classmethod
    def validate_source_min_date(cls, value: str | None) -> str | None:
        if value is not None:
            date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def validate_verified_coverage(self) -> "SourceSettings":
        values = (self.verified_coverage_start, self.verified_coverage_through)
        if (values[0] is None) != (values[1] is None):
            raise ValueError(
                "verified_coverage_start and verified_coverage_through must be set together"
            )
        if (
            values[0] is not None
            and values[1] is not None
            and date.fromisoformat(values[0]) > date.fromisoformat(values[1])
        ):
            raise ValueError(
                "verified source coverage start cannot follow coverage through"
            )
        return self
