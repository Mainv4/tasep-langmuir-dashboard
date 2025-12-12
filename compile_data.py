#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compile Langmuir simulation data for dashboard scatter plots.

Creates a CSV with one row per (alpha_in, alpha_out, Omega, segment) containing:
- Parameters: alpha_in, alpha_out, Omega, segment
- Local observables: rho_local, D_local
- Global observables: rho_mean, D_eff

Usage:
    python3 compile_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import re
from scipy.optimize import curve_fit

BASE_DIR = Path(__file__).parent.parent


def parse_params_from_filename(filename):
    """Extract parameters from filename pattern."""
    pattern = r'alpha_in_(\d+\.\d+)_alpha_out_(\d+\.\d+)_Omega_A_(\d+\.\d+)'
    match = re.search(pattern, str(filename))
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    return None, None, None


def fit_diffusion(time, msd, fit_range=(0.1, 0.4)):
    """
    Fit MSD to extract diffusion coefficient D.

    Fits MSD = 6*D*t in the specified time range (as fraction of max time).

    Args:
        time: time array
        msd: MSD array
        fit_range: (start_frac, end_frac) of max time for fitting

    Returns:
        D: diffusion coefficient
    """
    if len(time) < 10 or np.all(msd == 0):
        return np.nan

    # Select fit range
    t_max = time.max()
    t_min_fit = t_max * fit_range[0]
    t_max_fit = t_max * fit_range[1]

    mask = (time >= t_min_fit) & (time <= t_max_fit) & (time > 0) & (msd > 0)
    if mask.sum() < 5:
        return np.nan

    t_fit = time[mask]
    msd_fit = msd[mask]

    try:
        # Linear fit: MSD = 6*D*t => slope = 6*D
        coeffs = np.polyfit(t_fit, msd_fit, 1)
        D = coeffs[0] / 6.0
        return D if D > 0 else np.nan
    except Exception:
        return np.nan


def load_segment_density():
    """Load segment density profiles from DATA_DENSITY_LANGMUIR."""
    data = []
    density_dir = BASE_DIR / "DATA_DENSITY_LANGMUIR"

    if not density_dir.exists():
        print(f"Warning: {density_dir} not found")
        return pd.DataFrame()

    for f in density_dir.glob("*_density_segments_n_10.dat"):
        alpha_in, alpha_out, Omega = parse_params_from_filename(f.name)
        if alpha_in is None:
            continue

        try:
            arr = np.loadtxt(f)
            for row in arr:
                segment_center, rho, std, sem = row
                data.append({
                    'alpha_in': alpha_in,
                    'alpha_out': alpha_out,
                    'Omega': Omega,
                    'segment': int(segment_center),
                    'rho_local': rho,
                    'rho_local_std': std
                })
        except Exception as e:
            print(f"Error loading {f}: {e}")

    print(f"Loaded density for {len(data)} segment-parameter combinations")
    return pd.DataFrame(data)


def load_segment_msd():
    """Load segment MSD data and extract D_local and MSD at specific timelags."""
    data = []
    msd_dir = BASE_DIR / "DATA_MSD" / "segments"

    # Timelags at which to extract MSD values
    timelags = [20, 100, 200, 1000, 2000, 6000]

    if not msd_dir.exists():
        print(f"Warning: {msd_dir} not found")
        return pd.DataFrame()

    # Segment centers (matching density segments)
    n_segments = 30
    segment_size = 10  # beads per segment
    segment_centers = np.arange(segment_size/2, 300, segment_size).astype(int)

    for f in msd_dir.glob("*_msd_segments.dat"):
        alpha_in, alpha_out, Omega = parse_params_from_filename(f.name)
        if alpha_in is None:
            continue

        try:
            arr = np.loadtxt(f)
            time = arr[:, 0]  # First column is time

            # Find indices for specific timelags
            timelag_indices = {}
            for tlag in timelags:
                idx = np.where(np.isclose(time, tlag))[0]
                if len(idx) > 0:
                    timelag_indices[tlag] = idx[0]

            # Columns 1-30 are MSD for each segment
            for seg_idx in range(n_segments):
                msd = arr[:, seg_idx + 1]
                D_local = fit_diffusion(time, msd)

                row = {
                    'alpha_in': alpha_in,
                    'alpha_out': alpha_out,
                    'Omega': Omega,
                    'segment': segment_centers[seg_idx],
                    'D_local': D_local
                }

                # Add MSD at each timelag
                for tlag in timelags:
                    if tlag in timelag_indices:
                        row[f'MSD_t{tlag}'] = msd[timelag_indices[tlag]]
                    else:
                        row[f'MSD_t{tlag}'] = np.nan

                data.append(row)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    print(f"Loaded MSD for {len(data)} segment-parameter combinations")
    return pd.DataFrame(data)


def load_segment_rg():
    """Load segment Rg data from DATA_RG_SEGMENTS_LANGMUIR."""
    data = []
    rg_dir = BASE_DIR / "DATA_RG_SEGMENTS_LANGMUIR" / "disjoint_n_10"

    if not rg_dir.exists():
        print(f"Warning: {rg_dir} not found")
        return pd.DataFrame()

    for f in rg_dir.glob("*_rg_segments.dat"):
        alpha_in, alpha_out, Omega = parse_params_from_filename(f.name)
        if alpha_in is None:
            continue

        try:
            arr = np.loadtxt(f)
            for row in arr:
                segment_center, rg, std, sem = row
                data.append({
                    'alpha_in': alpha_in,
                    'alpha_out': alpha_out,
                    'Omega': Omega,
                    'segment': int(segment_center),
                    'Rg_local': rg,
                    'Rg_local_std': std
                })
        except Exception as e:
            print(f"Error loading {f}: {e}")

    print(f"Loaded Rg for {len(data)} segment-parameter combinations")
    return pd.DataFrame(data)


def load_spatial_density():
    """Load spatial density data (n_neighbors) from DATA_SPATIAL_DENSITY_LANGMUIR/disjoint_n_10."""
    data = []
    density_dir = BASE_DIR / "DATA_SPATIAL_DENSITY_LANGMUIR" / "disjoint_n_10"

    if not density_dir.exists():
        print(f"Warning: {density_dir} not found")
        return pd.DataFrame()

    for f in density_dir.glob("*_spatial_density.dat"):
        alpha_in, alpha_out, Omega = parse_params_from_filename(f.name)
        if alpha_in is None:
            continue

        try:
            arr = np.loadtxt(f)
            for row in arr:
                segment_center, n_neighbors, std, sem = row
                data.append({
                    'alpha_in': alpha_in,
                    'alpha_out': alpha_out,
                    'Omega': Omega,
                    'segment': int(segment_center),
                    'n_neighbors_3s': n_neighbors,
                    'n_neighbors_3s_std': std
                })
        except Exception as e:
            print(f"Error loading {f}: {e}")

    print(f"Loaded spatial density for {len(data)} segment-parameter combinations")
    return pd.DataFrame(data)


def load_global_rg():
    """Load global Rg (chain-level) from DATA_RG."""
    rg_dir = BASE_DIR / "DATA_RG"

    if not rg_dir.exists():
        print(f"Warning: {rg_dir} not found")
        return pd.DataFrame()

    # Group files by parameter set
    param_rg = {}
    for f in rg_dir.glob("*_rg.dat"):
        alpha_in, alpha_out, Omega = parse_params_from_filename(f.name)
        if alpha_in is None:
            continue

        key = (alpha_in, alpha_out, Omega)
        if key not in param_rg:
            param_rg[key] = []

        try:
            arr = np.loadtxt(f)
            # Format: time, Rg - take mean Rg
            rg_values = arr[:, 1]
            param_rg[key].append(np.mean(rg_values))
        except Exception as e:
            print(f"Error loading {f}: {e}")

    # Average across replicates
    data = []
    for (alpha_in, alpha_out, Omega), rg_list in param_rg.items():
        data.append({
            'alpha_in': alpha_in,
            'alpha_out': alpha_out,
            'Omega': Omega,
            'Rg_chain': np.mean(rg_list)
        })

    print(f"Loaded global Rg for {len(data)} parameter combinations")
    return pd.DataFrame(data)


def load_activity_fluctuations():
    """Load theoretical activity fluctuation profiles from TASEP-Langmuir theory."""
    fluct_dir = BASE_DIR.parent / "src" / "test_results_langmuir" / "chain_300" / "data"

    if not fluct_dir.exists():
        print(f"Warning: {fluct_dir} not found")
        return pd.DataFrame()

    data = []
    n_segments = 30
    segment_size = 10
    segment_centers = np.arange(segment_size / 2, 300, segment_size).astype(int)

    # Pattern: std_N300_K1.0_alpha_0.1_beta_0.1_Omega_D_0.001.txt
    pattern = r'std_N300_K1\.0_alpha_(\d+\.\d+)_beta_(\d+\.\d+)_Omega_D_(\d+\.\d+)\.txt'

    for f in fluct_dir.glob("std_N300_K1.0_*.txt"):
        match = re.search(pattern, f.name)
        if not match:
            continue

        alpha_in = float(match.group(1))
        alpha_out = float(match.group(2))
        Omega = float(match.group(3))

        try:
            arr = np.loadtxt(f, comments='#')
            # Format: position, std_density (300 values)
            std_profile = arr[:, 1]

            # Average into segments
            for seg_idx in range(n_segments):
                start = seg_idx * segment_size
                end = (seg_idx + 1) * segment_size
                sigma_seg = np.mean(std_profile[start:end])

                data.append({
                    'alpha_in': alpha_in,
                    'alpha_out': alpha_out,
                    'Omega': Omega,
                    'segment': segment_centers[seg_idx],
                    'sigma_rho_theory': sigma_seg
                })
        except Exception as e:
            print(f"Error loading {f}: {e}")

    print(f"Loaded theoretical fluctuations for {len(data)} segment-parameter combinations")
    return pd.DataFrame(data)


def load_global_observables():
    """Load global observables: mean density and D_eff from COM MSD."""
    data = []

    # Load D_eff from COM MSD
    msd_dir = BASE_DIR / "DATA_MSD" / "COM"
    if msd_dir.exists():
        for f in msd_dir.glob("*_msd.dat"):
            alpha_in, alpha_out, Omega = parse_params_from_filename(f.name)
            if alpha_in is None:
                continue

            try:
                arr = np.loadtxt(f)
                time = arr[:, 0]
                msd = arr[:, 1]
                D_eff = fit_diffusion(time, msd)

                data.append({
                    'alpha_in': alpha_in,
                    'alpha_out': alpha_out,
                    'Omega': Omega,
                    'D_eff': D_eff
                })
            except Exception as e:
                print(f"Error loading {f}: {e}")

    df_D = pd.DataFrame(data)

    # Load mean density from full profile
    density_data = []
    density_dir = BASE_DIR / "DATA_DENSITY_LANGMUIR"
    if density_dir.exists():
        for f in density_dir.glob("*_density_profile.dat"):
            alpha_in, alpha_out, Omega = parse_params_from_filename(f.name)
            if alpha_in is None:
                continue

            try:
                arr = np.loadtxt(f)
                rho_mean = np.mean(arr[:, 1])  # Mean of all site densities

                density_data.append({
                    'alpha_in': alpha_in,
                    'alpha_out': alpha_out,
                    'Omega': Omega,
                    'rho_mean': rho_mean
                })
            except Exception as e:
                print(f"Error loading {f}: {e}")

    df_rho = pd.DataFrame(density_data)

    # Merge D_eff and rho_mean
    if len(df_D) > 0 and len(df_rho) > 0:
        df_global = df_D.merge(df_rho, on=['alpha_in', 'alpha_out', 'Omega'], how='outer')
    elif len(df_D) > 0:
        df_global = df_D
    elif len(df_rho) > 0:
        df_global = df_rho
    else:
        df_global = pd.DataFrame()

    print(f"Loaded {len(df_global)} global parameter combinations")
    return df_global


def main():
    print("Compiling Langmuir simulation data...")
    print(f"Base directory: {BASE_DIR}")
    print()

    # Load all data
    df_density = load_segment_density()
    df_msd = load_segment_msd()
    df_rg = load_segment_rg()
    df_spatial = load_spatial_density()
    df_fluct = load_activity_fluctuations()
    df_global = load_global_observables()
    df_global_rg = load_global_rg()

    if len(df_density) == 0:
        print("Error: No density data found")
        return

    # Start with density data
    df = df_density.copy()

    # Merge with MSD data
    if len(df_msd) > 0:
        df = df.merge(df_msd, on=['alpha_in', 'alpha_out', 'Omega', 'segment'], how='left')

    # Merge with Rg data
    if len(df_rg) > 0:
        df = df.merge(df_rg, on=['alpha_in', 'alpha_out', 'Omega', 'segment'], how='left')

    # Merge with spatial density data
    if len(df_spatial) > 0:
        df = df.merge(df_spatial, on=['alpha_in', 'alpha_out', 'Omega', 'segment'], how='left')

    # Merge with theoretical fluctuations
    if len(df_fluct) > 0:
        df = df.merge(df_fluct, on=['alpha_in', 'alpha_out', 'Omega', 'segment'], how='left')

    # Merge with global observables
    if len(df_global) > 0:
        df = df.merge(df_global, on=['alpha_in', 'alpha_out', 'Omega'], how='left')

    # Merge with global Rg
    if len(df_global_rg) > 0:
        df = df.merge(df_global_rg, on=['alpha_in', 'alpha_out', 'Omega'], how='left')

    # Add derived columns
    df['alpha_ratio'] = df['alpha_in'] / df['alpha_out']
    df['log_Omega'] = np.log10(df['Omega'])

    # Sort and save
    df = df.sort_values(['alpha_in', 'alpha_out', 'Omega', 'segment'])

    output = BASE_DIR / "dashboard" / "data_compiled.csv"
    output.parent.mkdir(exist_ok=True)
    df.to_csv(output, index=False)

    print()
    print(f"Saved {len(df)} rows to {output}")
    print(f"Columns: {list(df.columns)}")
    print()
    print("Summary statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()
