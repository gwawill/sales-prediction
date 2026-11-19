"""The dashboard's two charts, built with Altair (the chart library Streamlit ships with).

Colour rules followed here:
- each region always has the same colour, in a fixed order, whatever filter is chosen;
- light and dark mode each have their own colour steps, checked for colour-blind safety;
- text (labels, axes) uses the text colours, never the series colours;
- every chart has a hover tooltip, and the page offers a table view of its data.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from sales_poc.dashboard.data import REGIONS

THEMES = {
    "light": {"series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
              "text": "#0b0b0b", "muted": "#52514e", "grid": "#e8e7e3"},
    "dark": {"series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
             "text": "#f0efec", "muted": "#c3c2b7", "grid": "#383835"},
}
MONEY = "$,.3~s"          # 12.9M, 450k
MONEY_EXACT = "$,.0f"     # in tooltips


def _theme(name: str) -> dict:
    return THEMES["dark" if name == "dark" else "light"]


def _style(chart: alt.Chart, theme: dict) -> alt.Chart:
    return (chart
            .configure(background="transparent", font="sans-serif")
            .configure_view(stroke=None)
            .configure_axis(labelColor=theme["muted"], titleColor=theme["muted"], gridColor=theme["grid"],
                            domainColor=theme["grid"], tickColor=theme["grid"], labelFontSize=12, titleFontSize=12)
            .configure_legend(labelColor=theme["text"], titleColor=theme["muted"], orient="top",
                              labelFontSize=12, symbolStrokeWidth=2))


def monthly_chart(monthly: pd.DataFrame, theme_name: str = "light") -> alt.Chart:
    """Monthly revenue, one line per region, labelled at the line ends."""
    theme = _theme(theme_name)
    colour = alt.Color("region_id:N", title="Region",
                       scale=alt.Scale(domain=REGIONS, range=theme["series"]))
    x = alt.X("month:T", title=None, axis=alt.Axis(format="%b %y", labelAngle=0, grid=False))
    y = alt.Y("revenue:Q", title="Revenue", axis=alt.Axis(format=MONEY, tickCount=5))
    hover = alt.selection_point(fields=["month"], nearest=True, on="pointerover", empty=False, clear="pointerout")

    base = alt.Chart(monthly).encode(x=x)
    lines = base.mark_line(strokeWidth=2).encode(y=y, color=colour)
    points = base.mark_point(filled=True, size=70).encode(
        y=y, color=colour,
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("month:T", title="Month", format="%B %Y"),
                 alt.Tooltip("region_id:N", title="Region"),
                 alt.Tooltip("revenue:Q", title="Revenue", format=MONEY_EXACT),
                 alt.Tooltip("orders:Q", title="Orders", format=",")],
    ).add_params(hover)
    rule = base.mark_rule(color=theme["muted"], strokeWidth=1).encode(
        opacity=alt.condition(hover, alt.value(0.5), alt.value(0))).transform_filter(hover)
    last = monthly[monthly["month"] == monthly["month"].max()] if not monthly.empty else monthly
    labels = alt.Chart(last).mark_text(align="left", dx=8, fontSize=12, color=theme["text"]).encode(
        x=x, y=y, text="region_id:N")
    return _style((lines + rule + points + labels).properties(height=320, padding={"right": 48}), theme)


def forecast_chart(chart_data: pd.DataFrame, theme_name: str = "light") -> alt.Chart:
    """Actual weekly revenue (solid) followed by the forecast (dashed), for one series."""
    theme = _theme(theme_name)
    # Ticks on real week starts (every second Monday, counted back from the last week), so a
    # label never names a day that is not the start of a week.
    weeks = sorted(pd.to_datetime(chart_data["week_start"]).dt.date.unique())
    ticks = [alt.DateTime(year=w.year, month=w.month, date=w.day) for w in weeks[::-1][::2][::-1]]
    x = alt.X("week_start:T", title="Week starting",
              axis=alt.Axis(format="%d %b", labelAngle=0, grid=False, values=ticks))
    y = alt.Y("revenue:Q", title="Revenue", axis=alt.Axis(format=MONEY, tickCount=5),
              scale=alt.Scale(zero=True))
    dash = alt.StrokeDash("kind:N", title=None, scale=alt.Scale(domain=["Actual", "Forecast"],
                                                                range=[[1, 0], [6, 4]]))
    hover = alt.selection_point(fields=["week_start"], nearest=True, on="pointerover", empty=False,
                                clear="pointerout")
    base = alt.Chart(chart_data).encode(x=x, y=y)
    line = base.mark_line(strokeWidth=2, color=theme["series"][0]).encode(strokeDash=dash)
    points = base.mark_point(filled=True, size=70, color=theme["series"][0]).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("week_start:T", title="Week starting", format="%d %b %Y"),
                 alt.Tooltip("kind:N", title="Type"),
                 alt.Tooltip("revenue:Q", title="Revenue", format=MONEY_EXACT)],
    ).add_params(hover)
    return _style((line + points).properties(height=280), theme)
