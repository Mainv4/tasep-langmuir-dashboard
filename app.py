#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit dashboard - Interactive scatter plots for Langmuir simulations.

Single panel for exploring correlations between any pair of observables.
Choose X, Y, color, and size variables freely.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Langmuir Scatter Explorer",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Langmuir Simulations - Scatter Explorer")


@st.cache_data
def load_data():
    """Load the compiled simulation data."""
    csv_path = Path(__file__).parent / "data_compiled.csv"
    if not csv_path.exists():
        st.error(f"Data file not found: {csv_path}")
        st.info("Run `python3 compile_data.py` first to generate the data file.")
        st.stop()
    df = pd.read_csv(csv_path)
    # Add couple column for identification
    df['couple'] = df.apply(lambda r: f"α_in={r['alpha_in']:.1f}, α_out={r['alpha_out']:.1f}", axis=1)
    return df


# Load data
df = load_data()

# Variable definitions with nice labels
labels = {
    # System parameters
    'segment': 'Segment position',
    'Omega': 'Ω (Langmuir rate)',
    'log_Omega': 'log₁₀(Ω)',
    # Global (chain-level)
    'Rg_chain': 'Rg_chain (global gyration)',
    'D_eff': 'D_chain (COM diffusion)',
    'rho_mean': 'ρ_act_chain (global TASEP)',
    # Local (segment-level)
    'rho_local': 'ρ_act (TASEP occupation)',
    'D_local': 'D_local (segment diffusion)',
    'Rg_local': 'Rg_local (segment gyration)',
    'n_neighbors_3s': 'ρ_spatial (neighbors r<3σ)',
    'MSD_t20': 'MSD(t=20)',
    'MSD_t100': 'MSD(t=100)',
    'MSD_t200': 'MSD(t=200)',
    'MSD_t1000': 'MSD(t=1000)',
    'MSD_t2000': 'MSD(t=2000)',
    'MSD_t6000': 'MSD(t=6000)',
    # Fluctuations
    'rho_local_std': 'σ_ρ_act (measured)',
    'sigma_rho_theory': 'σ_ρ_theory (TASEP-Langmuir)',
    'Rg_local_std': 'σ_Rg_local',
    'n_neighbors_3s_std': 'σ_ρ_spatial',
}

# Ordered column groups for dropdown menus
COLUMN_GROUPS = [
    ('System Parameters', ['segment', 'Omega', 'log_Omega']),
    ('Global (chain)', ['Rg_chain', 'D_eff', 'rho_mean']),
    ('Local (segment)', ['rho_local', 'D_local', 'Rg_local', 'n_neighbors_3s',
                         'MSD_t20', 'MSD_t100', 'MSD_t200', 'MSD_t1000', 'MSD_t2000', 'MSD_t6000']),
    ('Fluctuations', ['rho_local_std', 'sigma_rho_theory', 'Rg_local_std', 'n_neighbors_3s_std'])
]

# Build selectable_cols in order, filtering for columns present in data
selectable_cols = []
for group_name, cols in COLUMN_GROUPS:
    selectable_cols.extend([c for c in cols if c in df.columns])

# Variable selection
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    x_var = st.selectbox(
        "X-axis",
        options=selectable_cols,
        index=selectable_cols.index('segment') if 'segment' in selectable_cols else 0,
        format_func=lambda x: labels.get(x, x)
    )

with col2:
    y_var = st.selectbox(
        "Y-axis",
        options=selectable_cols,
        index=selectable_cols.index('rho_local') if 'rho_local' in selectable_cols else 1,
        format_func=lambda x: labels.get(x, x)
    )

with col3:
    color_options = ["None"] + selectable_cols
    color_var = st.selectbox(
        "Color by",
        options=color_options,
        index=color_options.index('log_Omega') if 'log_Omega' in color_options else 0,
        format_func=lambda x: labels.get(x, x) if x != "None" else "None"
    )

with col4:
    # Size must be positive - filter columns with min > 0
    size_cols = [col for col in selectable_cols if df[col].min() > 0]
    size_options = ["None"] + size_cols
    size_var = st.selectbox(
        "Size by",
        options=size_options,
        index=size_options.index('segment') if 'segment' in size_options else 0,
        format_func=lambda x: labels.get(x, x) if x != "None" else "None"
    )

# Plot options
col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)

with col_opt1:
    log_x = st.checkbox("Log X", value=False)

with col_opt2:
    log_y = st.checkbox("Log Y", value=False)

with col_opt3:
    colorscale = st.selectbox(
        "Colorscale",
        options=["Viridis", "Plasma", "Turbo", "Jet", "Cividis", "Inferno"],
        index=0
    )

with col_opt4:
    display_mode = st.radio(
        "Display",
        options=["All couples", "Separate figures"],
        index=1,
        horizontal=True
    )

st.markdown("---")

# Get unique couples
couples = sorted(df['couple'].unique())

# Prepare plot data
plot_df = df.dropna(subset=[x_var, y_var])

if len(plot_df) == 0:
    st.warning(f"No valid data for {labels.get(x_var, x_var)} vs {labels.get(y_var, y_var)}")
else:
    # Build hover data
    hover_cols = ['couple', 'Omega', 'segment']
    if x_var not in hover_cols:
        hover_cols.append(x_var)
    if y_var not in hover_cols:
        hover_cols.append(y_var)

    if display_mode == "All couples":
        # Separate TASEP (Omega=0) from Langmuir (Omega>0) when coloring by log_Omega
        if color_var == 'log_Omega':
            df_langmuir = plot_df[plot_df['Omega'] > 0]
            df_tasep = plot_df[plot_df['Omega'] == 0]
        else:
            df_langmuir = plot_df
            df_tasep = pd.DataFrame()

        # Single plot with Langmuir data (colorbar)
        fig = px.scatter(
            df_langmuir,
            x=x_var,
            y=y_var,
            color=color_var if color_var != "None" else None,
            size=size_var if size_var != "None" else None,
            symbol='couple',
            color_continuous_scale=colorscale.lower(),
            hover_data=[c for c in hover_cols if c in df_langmuir.columns],
            labels=labels,
            log_x=log_x,
            log_y=log_y
        )

        # Add TASEP points in black if present
        if len(df_tasep) > 0:
            for couple in df_tasep['couple'].unique():
                couple_data = df_tasep[df_tasep['couple'] == couple]
                # Calculate size for TASEP points (same logic as Langmuir)
                if size_var != "None" and size_var in couple_data.columns:
                    size_values = couple_data[size_var]
                    size_min, size_max = plot_df[size_var].min(), plot_df[size_var].max()
                    if size_max > size_min:
                        tasep_size = 5 + 15 * (size_values - size_min) / (size_max - size_min)
                    else:
                        tasep_size = 10
                else:
                    tasep_size = 10
                fig.add_trace(go.Scatter(
                    x=couple_data[x_var],
                    y=couple_data[y_var],
                    mode='markers',
                    marker=dict(color='black', size=tasep_size, symbol='star', line=dict(width=0.5, color='white')),
                    name=f'TASEP (Ω=0) - {couple}',
                    hovertemplate=f'couple={couple}<br>Ω=0<br>{x_var}=%{{x:.4f}}<br>{y_var}=%{{y:.4f}}<extra></extra>'
                ))

        fig.update_traces(
            marker=dict(
                line=dict(width=0.5, color='white'),
                opacity=0.8
            ),
            selector=dict(mode='markers')
        )

        fig.update_layout(
            height=600,
            template="plotly_white",
            font=dict(size=14),
            xaxis_title=labels.get(x_var, x_var),
            yaxis_title=labels.get(y_var, y_var),
            coloraxis_colorbar_title=labels.get(color_var, color_var) if color_var != "None" else None,
            legend_title="Couple (α_in, α_out)"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        corr = plot_df[x_var].corr(plot_df[y_var])
        col1, col2, col3 = st.columns(3)
        col1.metric("Correlation (R)", f"{corr:.4f}")
        col2.metric("R²", f"{corr**2:.4f}")
        col3.metric("Data points", len(plot_df))

    else:
        # Separate figures with shared Y-axis and single colorbar
        fig = make_subplots(
            rows=1, cols=len(couples),
            shared_yaxes=True,
            subplot_titles=couples,
            horizontal_spacing=0.03
        )

        # Separate TASEP from Langmuir when coloring by log_Omega
        if color_var == 'log_Omega':
            df_langmuir = plot_df[plot_df['Omega'] > 0]
            df_tasep = plot_df[plot_df['Omega'] == 0]
        else:
            df_langmuir = plot_df
            df_tasep = pd.DataFrame()

        # Get color range for consistent colorbar (only from Langmuir data)
        if color_var != "None" and len(df_langmuir) > 0:
            cmin = df_langmuir[color_var].min()
            cmax = df_langmuir[color_var].max()
        else:
            cmin, cmax = 0, 1

        # Add Langmuir traces for each couple
        for i, couple in enumerate(couples):
            couple_df = df_langmuir[df_langmuir['couple'] == couple]

            if len(couple_df) == 0:
                continue

            # Build hover text
            hover_text = [
                f"Ω={row['Omega']:.3f}<br>seg={row['segment']}<br>{x_var}={row[x_var]:.4f}<br>{y_var}={row[y_var]:.4f}"
                for _, row in couple_df.iterrows()
            ]

            # Calculate marker size
            if size_var != "None":
                size_values = couple_df[size_var]
                # Normalize to 5-20 pixel range
                size_min, size_max = plot_df[size_var].min(), plot_df[size_var].max()
                if size_max > size_min:
                    normalized_size = 5 + 15 * (size_values - size_min) / (size_max - size_min)
                else:
                    normalized_size = 10
            else:
                normalized_size = 8

            marker_dict = dict(
                size=normalized_size,
                line=dict(width=0.5, color='white'),
                opacity=0.8
            )

            if color_var != "None":
                marker_dict.update(
                    color=couple_df[color_var],
                    colorscale=colorscale.lower(),
                    cmin=cmin,
                    cmax=cmax,
                    showscale=(i == len(couples) - 1),  # Only show colorbar on last subplot
                    colorbar=dict(
                        title=labels.get(color_var, color_var),
                        len=0.9
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=couple_df[x_var],
                    y=couple_df[y_var],
                    mode='markers',
                    marker=marker_dict,
                    hovertext=hover_text,
                    hoverinfo='text',
                    showlegend=False
                ),
                row=1, col=i+1
            )

        # Add TASEP traces in black for each couple
        if len(df_tasep) > 0:
            for i, couple in enumerate(couples):
                tasep_couple = df_tasep[df_tasep['couple'] == couple]
                if len(tasep_couple) > 0:
                    hover_text_tasep = [
                        f"Ω=0 (TASEP)<br>seg={row['segment']}<br>{x_var}={row[x_var]:.4f}<br>{y_var}={row[y_var]:.4f}"
                        for _, row in tasep_couple.iterrows()
                    ]
                    # Calculate size for TASEP points (same logic as Langmuir)
                    if size_var != "None" and size_var in tasep_couple.columns:
                        size_values = tasep_couple[size_var]
                        size_min, size_max = plot_df[size_var].min(), plot_df[size_var].max()
                        if size_max > size_min:
                            tasep_size = 5 + 15 * (size_values - size_min) / (size_max - size_min)
                        else:
                            tasep_size = 10
                    else:
                        tasep_size = 10
                    fig.add_trace(
                        go.Scatter(
                            x=tasep_couple[x_var],
                            y=tasep_couple[y_var],
                            mode='markers',
                            marker=dict(color='black', size=tasep_size, symbol='star', line=dict(width=0.5, color='white')),
                            hovertext=hover_text_tasep,
                            hoverinfo='text',
                            showlegend=(i == 0),  # Only show legend once
                            name='TASEP (Ω=0)'
                        ),
                        row=1, col=i+1
                    )

        # Update axes
        fig.update_yaxes(title_text=labels.get(y_var, y_var), row=1, col=1)
        for i in range(len(couples)):
            fig.update_xaxes(title_text=labels.get(x_var, x_var), row=1, col=i+1)
            if log_x:
                fig.update_xaxes(type="log", row=1, col=i+1)
            if log_y:
                fig.update_yaxes(type="log", row=1, col=i+1)

        fig.update_layout(
            height=500,
            template="plotly_white",
            font=dict(size=12)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistics per couple
        stat_cols = st.columns(len(couples))
        for i, couple in enumerate(couples):
            couple_df = plot_df[plot_df['couple'] == couple]
            corr = couple_df[x_var].corr(couple_df[y_var])
            stat_cols[i].caption(f"**{couple}**: R={corr:.3f}, R²={corr**2:.3f}, N={len(couple_df)}")

# Data table (collapsible)
with st.expander("View Data Table"):
    st.dataframe(df, height=300)

    # Download button
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="Download data (CSV)",
        data=csv_data,
        file_name="langmuir_data.csv",
        mime="text/csv"
    )
