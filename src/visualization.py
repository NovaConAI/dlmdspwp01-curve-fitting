"""Bokeh-based visualisation of training, ideal, and test data.

The single public class :class:`BokehPlotter` creates a grid of four
figures, one per training function and its assigned ideal function, and
overlays all test-point assignments belonging to that ideal function.
The result is written to an HTML file that can be opened in any browser.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, output_file, save

from .analysis import IdealMatch, TestAssignment


class BokehPlotter:
    """Produces a four-panel interactive Bokeh figure of the analysis."""

    def __init__(
        self,
        training: pd.DataFrame,
        ideal: pd.DataFrame,
        matches: list[IdealMatch],
        assignments: list[TestAssignment],
    ) -> None:
        """Bind the input data needed to render all panels."""
        self.training: pd.DataFrame = training
        self.ideal: pd.DataFrame = ideal
        self.matches: list[IdealMatch] = matches
        self.assignments: list[TestAssignment] = assignments

    def render(self, output_path: Path) -> Path:
        """Render the four-panel grid and save it to ``output_path``."""
        output_file(filename=str(output_path), title="DLMDSPWP01 — Curve Fitting")
        panels = [self._panel_for(match) for match in self.matches]
        grid = gridplot(panels, ncols=2, sizing_mode="stretch_width")
        save(grid)
        return output_path

    def _panel_for(self, match: IdealMatch):
        x = self.ideal["x"]
        fig = figure(
            title=f"{match.training_name} vs. {match.ideal_name}",
            x_axis_label="x",
            y_axis_label="y",
            height=350,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        fig.line(x, self.ideal[match.ideal_name], legend_label="ideal", line_width=2)
        fig.scatter(
            x,
            self.training[match.training_name],
            size=4,
            legend_label="training",
            color="#444444",
            alpha=0.7,
        )

        matched_assignments = [a for a in self.assignments if a.ideal_name == match.ideal_name]
        if matched_assignments:
            src = ColumnDataSource(
                data={
                    "x": [a.x for a in matched_assignments],
                    "y": [a.y for a in matched_assignments],
                    "delta": [a.delta_y for a in matched_assignments],
                }
            )
            r = fig.scatter(
                x="x",
                y="y",
                source=src,
                size=8,
                color="crimson",
                legend_label="test (assigned)",
            )
            fig.add_tools(
                HoverTool(tooltips=[("x", "@x"), ("y", "@y"), ("Δy", "@delta")], renderers=[r])
            )

        fig.legend.location = "top_left"
        fig.legend.click_policy = "hide"
        return fig
