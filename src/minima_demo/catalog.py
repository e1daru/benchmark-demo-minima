"""Resolve the routing candidate pools against the live hosted catalog.

The hosted ``api.minima.sh`` catalog is the routing universe — Minima can only rank models it
knows. We resolve two pools off it:

- **live pool**    — every catalog model whose provider we hold an API key for (so we can run it).
- **dataset pool** — the LLMRouterBench candidate ids that map onto a real catalog id
                     (:data:`constants.DATASET_TO_CATALOG_ALIAS`), so the dataset track can route.
"""

from __future__ import annotations

from dataclasses import dataclass

from minima_client import MinimaClient

from . import constants
from .config import Settings


@dataclass(frozen=True)
class CatalogModel:
    model_id: str
    provider: str
    input_cost_per_mtok: float
    output_cost_per_mtok: float
    context_window: int

    @property
    def price(self) -> tuple[float, float]:
        return (self.input_cost_per_mtok, self.output_cost_per_mtok)


def fetch_catalog(client: MinimaClient) -> list[CatalogModel]:
    resp = client.models()
    return [
        CatalogModel(
            model_id=m.model_id,
            provider=m.provider,
            input_cost_per_mtok=m.input_cost_per_mtok,
            output_cost_per_mtok=m.output_cost_per_mtok,
            context_window=m.context_window,
        )
        for m in resp.models
    ]


def price_map(catalog: list[CatalogModel]) -> dict[str, tuple[float, float]]:
    return {m.model_id: m.price for m in catalog}


def resolve_live_pool(catalog: list[CatalogModel], settings: Settings) -> list[CatalogModel]:
    """Catalog models we can both route to AND call (provider key present)."""
    return [m for m in catalog if settings.has_provider(m.provider)]


def resolve_dataset_pool(catalog: list[CatalogModel]) -> list[tuple[str, str]]:
    """(dataset_model_id, catalog_model_id) pairs that resolve onto the live catalog."""
    have = {m.model_id for m in catalog}
    return [
        (ds_id, cat_id)
        for ds_id, cat_id in constants.DATASET_TO_CATALOG_ALIAS.items()
        if cat_id in have
    ]


def cost_spread(catalog: list[CatalogModel], model_ids: list[str]) -> float:
    """Ratio of priciest to cheapest output price across a pool (1.0 = no spread)."""
    outs = [dict((m.model_id, m.output_cost_per_mtok) for m in catalog)[mid] for mid in model_ids]
    outs = [o for o in outs if o > 0]
    return (max(outs) / min(outs)) if outs else 1.0
