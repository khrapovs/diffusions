"""Tests for UI components module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from affidiff.models.registry import MODEL_REGISTRY
from affidiff.ui import components
from affidiff.ui.simulation import extract_simulation_data, get_model_defaults, get_series_names, run_simulation

if TYPE_CHECKING:
    import plotly.graph_objects as go


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY))
def test_render_chart_plots_series_on_separate_subplots(*, model_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    figures: list[go.Figure] = []
    monkeypatch.setattr(components.st, "plotly_chart", lambda fig, **_: figures.append(fig))
    params = {key: value for key, value in get_model_defaults(model_name).items() if key != "measure"}
    path = extract_simulation_data(run_simulation(model_name=model_name, params=params, nobs=20, seed=1))
    series_names = get_series_names(model_name)

    components.render_chart(simulated_path=path, series_names=series_names, seed=1)

    (fig,) = figures
    assert [trace.name for trace in fig.data] == list(series_names)
    assert len({trace.yaxis for trace in fig.data}) == len(series_names)
    np.testing.assert_array_equal(np.column_stack([trace.y for trace in fig.data]), path.reshape(20, -1))
