"""Strict configuration base shared by toolkit workflows."""

from pydantic import BaseModel, ConfigDict


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
