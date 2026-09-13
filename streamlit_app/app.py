"""
Sprint 7 — Streamlit dashboard.

Run with:
    streamlit run streamlit_app/app.py

Sections (per the spec): Overview, Genre Analysis, Theme Analysis,
Developer Analysis, Franchise Analysis, Replay Analysis, Recommendation
Center, Explainability.

NOTE ON TESTING: this file was written without ever running `streamlit
run` — the sandbox this was built in has neither streamlit nor a
browser available. All the actual logic (DB queries, pandas
transforms) lives in data_access.py and IS unit-tested; this file is
kept as thin as possible on purpose, using only long-stable streamlit
APIs (st.dataframe, st.bar_chart, st.metric, st.sidebar.radio,
st.columns) rather than newer or version-sensitive ones. Please run it
yourself and treat the first launch as the real first test of this
file specifically.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.storage.db import get_connection, init_schema
from streamlit_app.data_access import (
    load_completion_table,
    load_overview,
    load_recommendations,
    load_replay_table,
    load_tag_table,
    load_taste_drift,
    regenerate_recommendations,
)

DEFAULT_DB_PATH = "data/processed/glip.db"
DEFAULT_MODELS_DIR = "data/models"

st.set_page_config(page_title="Gaming Library Intelligence Platform", layout="wide")


@st.cache_resource
def _connect(db_path: str):
    conn = get_connection(db_path)
    init_schema(conn)
    return conn


def _no_data_notice(what: str, script: str) -> None:
    st.info(f"No {what} yet. Run `{script}` first, then reload this page.")


# ---------------------------------------------------------------------------
# Page renderers — each one just calls data_access.py and displays the result.
# ---------------------------------------------------------------------------

def render_overview(conn):
    st.header("Library Overview")
    overview = load_overview(conn)

    if overview["total_games"] == 0:
        _no_data_notice("games", "python scripts/run_ingestion.py <export.json>")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total games", overview["total_games"])
    col2.metric("Total hours logged", f"{overview['total_hours_logged']:,.0f}")
    col3.metric("Rated games", overview["rated_games"])
    avg_rating = overview["average_rating"]
    col4.metric("Average rating", f"{avg_rating:.1f}/10" if avg_rating is not None else "-")

    st.subheader("By status")
    st.bar_chart(pd.Series(overview["status_counts"]))

    st.subheader("By signal category")
    st.bar_chart(pd.Series(overview["signal_category_counts"]))


def render_tag_analysis(conn, category: str, label: str):
    st.header(f"{label} Analysis")
    table = load_tag_table(conn, category)

    if table.empty:
        _no_data_notice(
            f"{label.lower()} data",
            "python scripts/run_enrichment.py (Sprint 2 must run before tag breakdowns exist)",
        )
        return

    st.caption("Hours-weighted average rating - tags on fewer than 2 games are excluded.")
    st.dataframe(table, use_container_width=True, hide_index=True)

    top_20 = table.head(20).set_index("tag")["hours_weighted_rating"]
    st.subheader("Top 20 by hours-weighted rating")
    st.bar_chart(top_20)

    st.subheader("Completion rate")
    completion = load_completion_table(conn, category)
    if not completion.empty:
        st.dataframe(completion, use_container_width=True, hide_index=True)


def render_replay_analysis(conn):
    st.header("Replay Analysis")
    tab_labels = ["genres", "themes", "developers", "publishers"]
    tabs = st.tabs([t.capitalize() for t in tab_labels])
    for tab, category in zip(tabs, tab_labels):
        with tab:
            table = load_replay_table(conn, category)
            if table.empty:
                st.info("No replayed games found for this category yet.")
                continue
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.bar_chart(table.head(15).set_index("tag")["total_replays"])


def render_taste_drift(conn):
    st.header("Taste Drift (Genre Evolution)")
    top_genres_by_year, similarity_df = load_taste_drift(conn)

    if not top_genres_by_year:
        _no_data_notice(
            "playthrough history with parseable dates",
            "python scripts/run_ingestion.py <export.json>",
        )
        return

    st.subheader("Top genres by year (hours played)")
    for year in sorted(top_genres_by_year):
        entries = top_genres_by_year[year]
        if not entries:
            continue
        st.write(f"**{year}**: " + ", ".join(f"{tag} ({hours:.0f}h)" for tag, hours in entries))

    if similarity_df is not None and len(similarity_df) > 1:
        st.subheader("Year-over-year taste similarity")
        st.caption(
            "Cosine similarity between each year's genre-hours vector - 1.0 means "
            "identical genre mix, 0 means no overlap. Needs Sprint 2 (enrichment)."
        )
        st.dataframe(similarity_df.style.format("{:.2f}"), use_container_width=True)
    else:
        st.caption(
            "Similarity matrix needs Sprint 2 (enrichment) to have run - "
            "genres aren't available for un-enriched games."
        )


def render_recommendation_center(conn):
    st.header("Recommendation Center")

    if st.button("Regenerate recommendations"):
        with st.spinner("Scoring your backlog..."):
            try:
                regenerate_recommendations(conn, models_dir=DEFAULT_MODELS_DIR)
                st.success("Recommendations updated.")
            except ValueError as e:
                st.error(str(e))

    table = load_recommendations(conn)
    if table.empty:
        _no_data_notice(
            "recommendations",
            "python scripts/run_ml_training.py && python scripts/run_recommendations.py",
        )
        return

    display_cols = [
        "title", "recommendation_score", "predicted_rating",
        "predicted_completion_prob", "predicted_replay_prob", "predicted_engagement",
    ]
    st.dataframe(
        table[[c for c in display_cols if c in table.columns]],
        use_container_width=True,
        hide_index=True,
    )

    st.session_state["recommendation_table"] = table


def render_explainability(conn):
    st.header("Explainability")

    table = st.session_state.get("recommendation_table")
    if table is None:
        table = load_recommendations(conn)

    if table.empty:
        _no_data_notice(
            "recommendations to explain",
            "python scripts/run_ml_training.py && python scripts/run_recommendations.py",
        )
        return

    title = st.selectbox("Pick a recommended game", table["title"].tolist())
    row = table.loc[table["title"] == title].iloc[0]
    explanation = row["explanation"]
    method = explanation.get("method", "unavailable")
    factors = explanation.get("top_factors", [])

    st.metric("Recommendation score", f"{row['recommendation_score']:.1f}/100")

    if method == "shap":
        st.caption("SHAP values for this specific game.")
    elif method == "global_feature_importance":
        st.caption(
            "shap isn't installed, so this shows the model's overall feature "
            "importance (same for every game) rather than a per-game explanation. "
            "Install shap and re-run scripts/run_ml_training.py for real per-game explanations."
        )
    else:
        st.warning("No explanation is available for this recommendation.")
        return

    if factors:
        factor_df = pd.DataFrame(factors).set_index("feature")
        st.bar_chart(factor_df["contribution"])
    else:
        st.info("No contributing factors were recorded for this recommendation.")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def main():
    st.sidebar.title("GLIP")
    db_path = st.sidebar.text_input("Database path", value=DEFAULT_DB_PATH)
    conn = _connect(db_path)

    page = st.sidebar.radio(
        "Section",
        [
            "Overview",
            "Genre Analysis",
            "Theme Analysis",
            "Developer Analysis",
            "Franchise Analysis",
            "Replay Analysis",
            "Taste Drift",
            "Recommendation Center",
            "Explainability",
        ],
    )

    if page == "Overview":
        render_overview(conn)
    elif page == "Genre Analysis":
        render_tag_analysis(conn, "genres", "Genre")
    elif page == "Theme Analysis":
        render_tag_analysis(conn, "themes", "Theme")
    elif page == "Developer Analysis":
        render_tag_analysis(conn, "developers", "Developer")
    elif page == "Franchise Analysis":
        render_tag_analysis(conn, "franchises", "Franchise")
    elif page == "Replay Analysis":
        render_replay_analysis(conn)
    elif page == "Taste Drift":
        render_taste_drift(conn)
    elif page == "Recommendation Center":
        render_recommendation_center(conn)
    elif page == "Explainability":
        render_explainability(conn)


if __name__ == "__main__":
    main()
