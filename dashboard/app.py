# dashboard/app.py

import sys
sys.path.append('../src')

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WEC BoP Analyzer",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#f7f6f2',
    'axes.facecolor':   '#f7f6f2',
    'axes.grid':        True,
    'grid.color':       '#dcd9d5',
    'grid.linewidth':   0.6,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.family':      'sans-serif',
})

LABEL_COLORS = {
    'strong_penalty': '#a12c7b',
    'mild_penalty':   '#da7101',
    'no_change':      '#7a7974',
    'mild_relief':    '#006494',
    'strong_relief':  '#01696f',
}

HG_COLORS = {
    'LMH':  '#01696f',
    'LMDh': '#da7101',
    'GT3':  '#006494',
}

DB_PATH = '../data/processed/wec_bop.db'

# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    recs        = pd.read_sql("SELECT * FROM bop_recommendations", conn)
    event_model = pd.read_sql("SELECT * FROM event_model_features", conn)
    stints      = pd.read_sql("SELECT * FROM stint_features", conn)
    conn.close()
    return recs, event_model, stints

recs, event_model, stints = load_data()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏁 WEC BoP Analyzer")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Model Deep Dive", "📋 Recommendations", "📈 Trends"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Filters**")

    selected_class = st.selectbox(
        "Class",
        ["HYPERCAR", "GT3"],
    )

    available_years = sorted(
        event_model[event_model['class_normalized'] == selected_class]['year'].unique(),
        reverse=True
    )
    selected_year = st.selectbox("Season", available_years)

    available_hg = sorted(
        event_model[
            (event_model['class_normalized'] == selected_class) &
            (event_model['year'] == selected_year)
        ]['homologation_group'].unique()
    )
    selected_hg = st.multiselect(
        "Homologation Group",
        available_hg,
        default=available_hg
    )

    st.markdown("---")
    st.caption("Data: WEC 2021–2025\nModel: weighted score\nv1.0")

# ─── Page: Overview ──────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title(f"📊 Balance of Performance — {selected_class} {selected_year}")

    subset_recs = recs[
        (recs['class_normalized'] == selected_class) &
        (recs['year'] == selected_year) &
        (recs['homologation_group'].isin(selected_hg))
    ]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models in class", len(subset_recs))
    col2.metric(
        "Avg |Baseline Delta|",
        f"{subset_recs['avg_delta'].abs().mean():.3f}s"
    )
    col3.metric(
        "Require intervention",
        len(subset_recs[subset_recs['recommendation_label'].isin(
            ['strong_penalty', 'strong_relief']
        )])
    )
    col4.metric(
        "In balance",
        len(subset_recs[subset_recs['recommendation_label'] == 'no_change'])
    )

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Ranking by Baseline Delta")
        fig, ax = plt.subplots(figsize=(8, max(4, len(subset_recs) * 0.55)))
        s = subset_recs.sort_values('avg_delta')
        colors = [LABEL_COLORS[l] for l in s['recommendation_label']]
        bars = ax.barh(s['car_model_key'], s['avg_delta'], color=colors, height=0.6)
        ax.axvline(0, color='#28251d', linewidth=1.2, linestyle='--', alpha=0.5)
        for bar, val in zip(bars, s['avg_delta']):
            offset = 0.02 if val >= 0 else -0.02
            ha = 'left' if val >= 0 else 'right'
            ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                    f'{val:+.3f}s', va='center', ha=ha, fontsize=9)
        ax.set_xlabel('Avg Baseline Delta (s) — negative = faster than baseline')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Label Distribution")
        label_counts = subset_recs['recommendation_label'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        colors_pie = [LABEL_COLORS.get(l, '#ccc') for l in label_counts.index]
        ax2.pie(
            label_counts.values,
            labels=label_counts.index,
            colors=colors_pie,
            autopct='%1.0f%%',
            startangle=90,
            textprops={'fontsize': 9}
        )
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.subheader("Heatmap: Baseline Delta by Event")
    em_subset = event_model[
        (event_model['class_normalized'] == selected_class) &
        (event_model['year'] == selected_year) &
        (event_model['homologation_group'].isin(selected_hg))
    ]

    if not em_subset.empty:
        pivot = em_subset.pivot_table(
            index='car_model_key', columns='event',
            values='baseline_delta', aggfunc='mean'
        ).round(2)
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

        fig3, ax3 = plt.subplots(figsize=(14, max(4, len(pivot) * 0.55)))
        sns.heatmap(
            pivot, ax=ax3, cmap='RdYlGn_r', center=0,
            annot=True, fmt='.2f', linewidths=0.4,
            linecolor='#dcd9d5',
            cbar_kws={'label': 'Baseline Delta (s)', 'shrink': 0.6},
            annot_kws={'size': 9}
        )
        ax3.set_title(f'{selected_class} {selected_year} — Baseline Delta Heatmap')
        ax3.tick_params(axis='x', rotation=30)
        ax3.tick_params(axis='y', rotation=0)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close()

# ─── Page: Model Deep Dive ────────────────────────────────────────────────────
elif page == "🔍 Model Deep Dive":
    st.title("🔍 Model Deep Dive")

    em_filtered = event_model[
        (event_model['class_normalized'] == selected_class) &
        (event_model['homologation_group'].isin(selected_hg))
    ]

    models = sorted(em_filtered['car_model_key'].unique())
    selected_model = st.selectbox("Select model", models)

    model_data = em_filtered[em_filtered['car_model_key'] == selected_model]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", model_data['event'].nunique())
    c2.metric("Avg Delta", f"{model_data['baseline_delta'].mean():+.3f}s")
    c3.metric("Avg Consistency", f"{model_data['consistency_score'].mean():.3f}")
    c4.metric("Avg Long Run", f"{model_data['long_run_score'].median():+.4f}s/lap")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Baseline Delta by Season")
        fig, ax = plt.subplots(figsize=(7, 4))
        yearly = model_data.groupby('year')['baseline_delta'].mean()
        ax.bar(yearly.index.astype(str), yearly.values,
               color=['#437a22' if v < 0 else '#a12c7b' for v in yearly.values],
               alpha=0.8)
        ax.axhline(0, color='#28251d', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_xlabel('Season')
        ax.set_ylabel('Avg Baseline Delta (s)')
        for i, (yr, val) in enumerate(yearly.items()):
            ax.text(i, val + (0.02 if val >= 0 else -0.05),
                    f'{val:+.3f}', ha='center', fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Baseline Delta by Event")
        fig, ax = plt.subplots(figsize=(7, 4))
        by_event = model_data.groupby('event')['baseline_delta'].mean().sort_values()
        ax.barh(by_event.index, by_event.values,
                color=['#437a22' if v < 0 else '#a12c7b' for v in by_event.values],
                alpha=0.8)
        ax.axvline(0, color='#28251d', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_xlabel('Avg Baseline Delta (s)')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Stint Degradation Distribution")
    model_stints = stints[
        (stints['car_model_key'] == selected_model) &
        (stints['valid_laps_count'] >= 8)
    ]
    if not model_stints.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.hist(model_stints['stint_degradation'].dropna(),
                bins=40, color=HG_COLORS.get(selected_class, '#7a7974'),
                alpha=0.75, edgecolor='white')
        ax.axvline(model_stints['stint_degradation'].median(),
                   color='#28251d', linewidth=1.5, linestyle='--',
                   label=f"median = {model_stints['stint_degradation'].median():.3f}")
        ax.axvline(0, color='#7a7974', linewidth=1, linestyle=':')
        ax.set_xlabel('Stint Degradation (s/lap)')
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Not enough stints (≥8 laps) for this model")

    with st.expander("📄 Show raw event_model data"):
        st.dataframe(
            model_data[['year', 'event', 'baseline_delta',
                        'consistency_score', 'long_run_score',
                        'track_balance_score', 'clean_laps_count']]
            .sort_values(['year', 'event'])
            .reset_index(drop=True),
            use_container_width=True
        )

# ─── Page: Recommendations ───────────────────────────────────────────────────
elif page == "📋 Recommendations":
    st.title("📋 BoP Recommendations")

    recs_filtered = recs[
        (recs['class_normalized'] == selected_class) &
        (recs['homologation_group'].isin(selected_hg))
    ].sort_values(['year', 'recommendation_points'], ascending=[False, False])

    def color_label(val):
        colors_map = {
            'strong_penalty': 'background-color: #f0d0e8; color: #a12c7b',
            'mild_penalty':   'background-color: #fde8cc; color: #da7101',
            'no_change':      'background-color: #ebebeb; color: #7a7974',
            'mild_relief':    'background-color: #cce0f0; color: #006494',
            'strong_relief':  'background-color: #cce8e6; color: #01696f',
        }
        return colors_map.get(val, '')

    display_cols = [
        'year', 'car_model_key', 'homologation_group',
        'avg_delta', 'avg_consistency', 'avg_long_run',
        'recommendation_points', 'recommendation_label', 'confidence_score'
    ]

    styled = (
        recs_filtered[display_cols]
        .reset_index(drop=True)
        .style
        .map(color_label, subset=['recommendation_label'])
        .format({
            'avg_delta': '{:+.3f}',
            'avg_consistency': '{:.3f}',
            'avg_long_run': '{:+.4f}',
            'recommendation_points': '{:+.1f}',
            'confidence_score': '{:.2f}',
        })
    )
    st.dataframe(styled, use_container_width=True, height=500)

    st.markdown("### 💬 Recommendation Explanations")
    selected_model_rec = st.selectbox(
        "Select model",
        recs_filtered['car_model_key'].unique(),
        key='rec_model'
    )
    model_recs = recs_filtered[recs_filtered['car_model_key'] == selected_model_rec]
    for _, row in model_recs.iterrows():
        color = LABEL_COLORS.get(row['recommendation_label'], '#ccc')
        st.markdown(
            f"**{row['year']}** — "
            f"<span style='color:{color}; font-weight:bold'>{row['recommendation_label']}</span> "
            f"({row['recommendation_points']:+.1f} pts, confidence: {row['confidence_score']:.2f})<br>"
            f"<small>{row['explanation_text']}</small>",
            unsafe_allow_html=True
        )

# ─── Page: Trends ────────────────────────────────────────────────────────────
elif page == "📈 Trends":
    st.title("📈 Trends — Season-over-Season Dynamics")

    em_filtered = event_model[
        (event_model['class_normalized'] == selected_class) &
        (event_model['homologation_group'].isin(selected_hg))
    ]

    st.subheader("Baseline Delta Trend by Season")
    yearly_trend = (
        em_filtered.groupby(['year', 'car_model_key'])['baseline_delta']
        .mean().reset_index()
    )
    models_in_data = yearly_trend['car_model_key'].unique()

    fig, ax = plt.subplots(figsize=(12, 5))
    for model in models_in_data:
        mdata = yearly_trend[yearly_trend['car_model_key'] == model].sort_values('year')
        if len(mdata) >= 2:
            ax.plot(mdata['year'], mdata['baseline_delta'],
                    marker='o', linewidth=1.8, markersize=5, label=model)
        else:
            ax.scatter(mdata['year'], mdata['baseline_delta'], s=50, label=model)

    ax.axhline(0, color='#28251d', linewidth=1, linestyle='--', alpha=0.4)
    ax.set_xlabel('Season')
    ax.set_ylabel('Avg Baseline Delta (s)')
    ax.legend(fontsize=8, ncol=3, loc='upper right', framealpha=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Field Spread by Season (Std of Baseline Delta)")
    spread = (
        em_filtered.groupby(['year', 'homologation_group'])['baseline_delta']
        .std().reset_index()
        .rename(columns={'baseline_delta': 'field_std'})
    )
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for hg in spread['homologation_group'].unique():
        hdata = spread[spread['homologation_group'] == hg].sort_values('year')
        ax2.plot(hdata['year'], hdata['field_std'],
                 marker='s', linewidth=2, markersize=7,
                 color=HG_COLORS.get(hg, '#7a7974'), label=hg)
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Std of Baseline Delta (s)\n(lower = tighter field)')
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close()