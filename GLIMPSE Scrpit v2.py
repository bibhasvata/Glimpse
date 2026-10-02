#!/usr/bin/env python3
"""
GLIMPSE Scrpit v2.py

Single-file pipeline: trains spatial isoscape (XGBoost), processes temporal data, performs M-sweep
and correlation analysis for six sources, computes SHAP, cross-validation, and writes
publication-ready figures (PNG + PDF) and CSV outputs into ./glimpse_outputs by default.

Usage examples:
  python GLIMPSE_Scrpit_v2.py --data_dir "/path/to" --f_name "GLIMPSE Dataset" --use_xgb --m_start -1 --m_end 1 --m_steps 201 --make_figures

Author: Bibhasvata Dasgupta
Date: 2025-08-11
"""

import os
import sys
import argparse
import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.stats import pearsonr

# optional packages
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# tqdm
try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('GLIMPSE')

# -------------------------------
# Helper I/O
# -------------------------------

def read_data(data_dir: str, f_name: str, sheet_name: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f_name + '.xlsx')
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_excel(path, sheet_name=sheet_name)
    logger.info(f"Read sheet '{sheet_name}' from {path} with shape {df.shape}")
    return df

# -------------------------------
# Spatial model training
# -------------------------------

def process_train_test_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler, List[str]]:
    # Construct features/target consistent with original script
    X = df[['Latitude', 'Longitude', 'Elevation', 'Temperature', 'δ18O']].copy()
    y = df[['Class', 'Season']].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    y_train_encoded = pd.get_dummies(y_train).multiply(X_train['Temperature'], axis=0)
    y_test_encoded = pd.get_dummies(y_test).multiply(X_test['Temperature'], axis=0)

    # drop Temperature from predictors
    X_train_drop = X_train.drop(columns=['Temperature']).copy()
    X_test_drop = X_test.drop(columns=['Temperature']).copy()

    y_train_new = X_train_drop['δ18O'].copy()
    y_test_new = X_test_drop['δ18O'].copy()

    X_train_drop = X_train_drop.drop(columns=['δ18O']).copy()
    X_test_drop = X_test_drop.drop(columns=['δ18O']).copy()

    column_names = list(X_train_drop.columns)

    X_train_new = np.hstack((X_train_drop.values, y_train_encoded.values))
    X_test_new = np.hstack((X_test_drop.values, y_test_encoded.values))

    column_names.extend(list(y_train_encoded.columns))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_new)
    X_test_scaled = scaler.transform(X_test_new)

    return X_train_scaled, X_test_scaled, y_train_new.reset_index(drop=True), y_test_new.reset_index(drop=True), scaler, column_names


def train_and_evaluate_model(X_train_scaled: np.ndarray, y_train: pd.Series, X_test_scaled: np.ndarray, y_test: pd.Series, xgb_params: dict) -> Tuple[object,float,float,float,float,np.ndarray,np.ndarray]:
    if XGB_AVAILABLE:
        model = xgb.XGBRegressor(**xgb_params)
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=xgb_params.get('n_estimators', 100), max_depth=xgb_params.get('max_depth',3), random_state=42)

    model.fit(X_train_scaled, y_train)

    y_pred_test = model.predict(X_test_scaled)
    y_pred_train = model.predict(X_train_scaled)

    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    logger.info(f"Training complete. Test R2 = {r2_test:.3f}, MSE = {mse:.3f}")

    return model, mse, mae, r2_test, r2_train, y_pred_train, y_pred_test

# -------------------------------
# Temporal processing (ported from original code)
# -------------------------------

def interchange_columns(df_temporal_encoded):
    # Interchange column values
    df_temporal_encoded[['Class_Rain', 'Class_Cryo']] = df_temporal_encoded[['Class_Cryo', 'Class_Rain']]
    return df_temporal_encoded

def process_temporal_data(df_temporal, model, scaler):
    df_temporal_encoded = pd.get_dummies(df_temporal[['Latitude', 'Longitude', 'Elevation', 'Temperature', 'Season', 'Class_Rain', 'Class_GW', 'Class_Cryo']])

    # Multiply columns with 'Temperature'
    columns_to_multiply = ['Class_Rain', 'Class_GW', 'Class_Cryo', 'Season_Monsoon', 'Season_Post Monsoon', 'Season_Spring', 'Season_Winter']
    df_temporal_encoded[columns_to_multiply] = df_temporal_encoded[columns_to_multiply].multiply(df_temporal_encoded['Temperature'], axis=0)

    # Drop 'Temperature' column
    df_temporal_encoded = df_temporal_encoded.drop('Temperature', axis=1)

    df_temporal_encoded_shifted = interchange_columns(df_temporal_encoded)
    
    # Prediction
    df_temporal = predict_temporal_data(model, df_temporal, df_temporal_encoded_shifted, scaler)

    return df_temporal

def predict_temporal_data(model, df_temporal, df_temporal_encoded, scaler):
    # Predict on the df_temporal set for rain
    df_temporal_encoded_scaled = scaler.transform(df_temporal_encoded)
    df_temporal['y_pred_temporal_rain'] = model.predict(df_temporal_encoded_scaled)

    interchange_columns(df_temporal_encoded)

    # Predict on the df_temporal set for cryo
    df_temporal_encoded_scaled = scaler.transform(df_temporal_encoded)
    df_temporal['y_pred_temporal_snow'] = model.predict(df_temporal_encoded_scaled)

    # Predicting temporal δ18O of Precipitation (Rain + Snow) from spatial model in diurnal proportion of rain and snow
    df_temporal['Sink Precipitation δ18O'] = ((df_temporal['Snowfall'] / (df_temporal['Snowfall'] + df_temporal['Rainfall'])) * df_temporal['y_pred_temporal_snow']) + ((df_temporal['Rainfall'] / (df_temporal['Snowfall'] + df_temporal['Rainfall'])) * df_temporal['y_pred_temporal_rain'])

    # Fraction of atmospheric vapour precipitated (F) for Rayleigh Distillation
    df_temporal['F'] = df_temporal['GPM'] / df_temporal['PWV']
    df_temporal.loc[df_temporal['F'] > 1, 'F'] = 1

    # Define constants
    t1 = 6.7123 * (10**3)
    t2 = 1.6664 * (10**6)
    t3 = 0.35041 * (10**9)

    # Fractionation factor (alpha) at the sink atmosphere calculated from Cloud Top Temperature (CTT) at Sink and 6xSources
    T1_sink = t1 / df_temporal['Temp_Sink']
    T2_sink = t2 / (df_temporal['Temp_Sink']**2)
    T3_sink = t3 / (df_temporal['Temp_Sink']**3)

    df_temporal['Alpha_Sink'] = np.exp((-7.685 + T1_sink - T2_sink + T3_sink) / 1000)

    for i in range(1, 7):
        T1_source = t1 / df_temporal[f'Temp_Source{i}']
        T2_source = t2 / (df_temporal[f'Temp_Source{i}']**2)
        T3_source = t3 / (df_temporal[f'Temp_Source{i}']**3)
        df_temporal[f'Alpha_Source{i}'] = np.exp((-7.685 + T1_source - T2_source + T3_source) / 1000)

    # Predicting temporal δ18O of Sink Vapour
    mask = df_temporal['F'] < 1
    df_temporal.loc[mask, 'Sink Vapour δ18O'] = df_temporal.loc[mask, 'Sink Precipitation δ18O'] - (np.log(1 - df_temporal.loc[mask, 'F']) / df_temporal.loc[mask, 'Alpha_Sink'])
    df_temporal.loc[~mask, 'Sink Vapour δ18O'] = df_temporal.loc[~mask, 'Sink Precipitation δ18O']

    for i in range(1, 7):
        # Constant K
        df_temporal[f'K_Source{i}'] = (df_temporal['Alpha_Sink'] - df_temporal[f'Alpha_Source{i}']) / df_temporal['Sink Vapour δ18O']

        # K term
        df_temporal[f'Kterm_Source{i}'] = (df_temporal['AV_Sink']**df_temporal['Alpha_Sink']) - (df_temporal[f'AV_Source{i}']**df_temporal[f'Alpha_Source{i}']) * df_temporal['Sink Vapour δ18O'] * df_temporal[f'K_Source{i}'] / (df_temporal['Alpha_Sink'] - df_temporal[f'Alpha_Source{i}'])

        # M term
        df_temporal[f'Mterm_Source{i}'] = ((df_temporal['AV_Sink']**(2 * df_temporal['Alpha_Sink'] - 2)) - (df_temporal[f'AV_Source{i}']**(2 * df_temporal[f'Alpha_Source{i}'] - 2))) * M / (df_temporal['Alpha_Sink'] - df_temporal[f'Alpha_Source{i}'])

        # δ18O of source vapour
        df_temporal[f'δ18O_Vapour_Source{i}'] = df_temporal[f'Kterm_Source{i}'] - df_temporal[f'Mterm_Source{i}']

    # Combining δ18O_Vapour of all 6 sources as a function of their seasonality
    df_temporal['Source_Combined_Seasonally'] = (df_temporal['δ18O_Vapour_Source1']*df_temporal['Traj_Source1'] +
                                                 df_temporal['δ18O_Vapour_Source2']*df_temporal['Traj_Source2'] +
                                                 df_temporal['δ18O_Vapour_Source3']*df_temporal['Traj_Source3'] +
                                                 df_temporal['δ18O_Vapour_Source4']*df_temporal['Traj_Source4'] +
                                                 df_temporal['δ18O_Vapour_Source5']*df_temporal['Traj_Source5'] +
                                                 df_temporal['δ18O_Vapour_Source6']*df_temporal['Traj_Source6']) / 6

    return df_temporal

# -------------------------------
# Correlation and plotting utilities
# -------------------------------

def scatter_plot_with_correlation(df_temporal: pd.DataFrame, z_threshold: float=3.0, out_dir: str='./glimpse_outputs'):
    # ensure δ18O sources exist by running prediction with M currently set (global M)
    # Note: This function is not used in the main pipeline, but kept for compatibility
    # The main pipeline uses process_temporal_data directly

    plot_temporal_evolution(df_temporal, z_threshold)

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    r_values = []
    p_values = []
    std_errs = []

    for i in range(1,7):
        source_col = f'δ18O_Vapour_Source{i}'
        data = df_temporal[['Ghalegaon GNIP δ18O', source_col]].dropna().copy()
        z_scores = np.abs((data[source_col] - data[source_col].mean()) / (data[source_col].std() + 1e-12))
        outliers = data[z_scores > z_threshold]
        data = data[z_scores <= z_threshold]
        if len(data) < 3:
            r_values.append(np.nan); p_values.append(np.nan); std_errs.append(np.nan)
            continue
        sns.scatterplot(x=data['Ghalegaon GNIP δ18O'], y=data[source_col], label=f'{source_col}', ax=axs[0])
        r,p = pearsonr(data['Ghalegaon GNIP δ18O'], data[source_col])
        n = len(data)
        std_err = np.sqrt((1 - r**2) / (n - 2)) if n>2 else np.nan
        r_values.append(r); p_values.append(p); std_errs.append(std_err)

    legend_labels = [f'{source_col} (r={r:.2f}, SE={se:.2f}, p={p:.2e})' for source_col, r, se, p in zip([f'Source{i}' for i in range(1,7)], r_values, std_errs, p_values)]
    axs[0].legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    axs[0].set_xlabel('Ghalegaon GNIP δ$^{18}$O (‰)')
    axs[0].set_ylabel('Vapour Source δ$^{18}$O (‰)')
    axs[0].set_ylim(-70, 30)
    axs[0].set_title(f'Scatter Plots for all Sources (Outliers > {z_threshold} S.D removed) vs. Ghalegaon GNIP')

    df = df_temporal.rename(columns=lambda x: x.replace('δ18O_Vapour_', ''))
    df[['Source1','Source2','Source3','Source4','Source5','Source6','Source_Combined_Seasonally','Ghalegaon GNIP δ18O']].boxplot(ax=axs[1])
    axs[1].set_ylabel('δ$^{18}$O (‰)')
    axs[1].set_ylim(-100, 30)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha='right')
    axs[1].set_title('Box Plot')

    plt.tight_layout()
    for ext in ['png','pdf']:
        plt.savefig(os.path.join(out_dir, 'Filtered sources.'+ext), dpi=300)
    plt.close()
    return r_values, p_values, std_errs

# We'll reuse your original plot_temporal_evolution function but call it after predict

def plot_temporal_evolution(df_temporal: pd.DataFrame, z_threshold: float = 3.0, out_dir: str = './glimpse_outputs'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    for i in range(1, 7):
        source_col = f'δ18O_Vapour_Source{i}'
        data = df_temporal[['Day', source_col]].dropna()
        if len(data)==0:
            continue
        z_scores = np.abs((data[source_col] - data[source_col].mean()) / (data[source_col].std() + 1e-12))
        data = data[z_scores <= z_threshold]
        ax1.scatter(data['Day'], data[source_col], color=f'C{i-1}', label='Source {}'.format(i))

    ax1.scatter(df_temporal['Day'], df_temporal['Ghalegaon GNIP δ18O'], color='black', label='Ghalegaon GNIP (Terai)')
    if 'Source_Combined_Seasonally' in df_temporal.columns:
        ax1.scatter(df_temporal['Day'], df_temporal['Source_Combined_Seasonally'], color='gold', marker='*', label='Source Vapour Combined', alpha=0.5)
    ax1.set_title(f"Temporal evolution of Source Vapour (Outliers > {z_threshold} S.D removed)")
    ax1.set_ylabel(r'δ$^{18}$O (‰)')
    ax1.set_ylim(-70, 30)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax2.scatter(df_temporal['Day'], df_temporal['Ghalegaon GNIP δ18O'], color='black', label='Ghalegaon GNIP (Terai)')
    if 'Source_Combined_Seasonally' in df_temporal.columns:
        ax2.scatter(df_temporal['Day'], df_temporal['Source_Combined_Seasonally'], color='gold', marker='*', label='Source Vapour Combined')
    if 'Sink Vapour δ18O' in df_temporal.columns:
        ax2.scatter(df_temporal['Day'], df_temporal['Sink Vapour δ18O'], color='violet', marker='+', label='Sink Vapour')
    if 'Sink Precipitation δ18O' in df_temporal.columns:
        ax2.scatter(df_temporal['Day'], df_temporal['Sink Precipitation δ18O'], color='black', marker='+', label='Sink Precipitation', alpha=0.5)
    if 'y_pred_temporal_rain' in df_temporal.columns:
        ax2.scatter(df_temporal['Day'], df_temporal['y_pred_temporal_rain'], color='red', marker='.', label='Sink Rain', alpha=0.1)
    if 'y_pred_temporal_snow' in df_temporal.columns:
        ax2.scatter(df_temporal['Day'], df_temporal['y_pred_temporal_snow'], color='blue', marker='.', label='Sink Snow', alpha=0.1)

    ax2.set_xlabel(r'Day')
    ax2.set_ylabel(r'δ$^{18}$O (‰)')
    ax2.set_title('Temporal evolution of Source and Sink Vapour in the Kosi region')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    for ext in ['png','pdf']:
        plt.savefig(os.path.join(out_dir, 'source_vs_day.'+ext), dpi=300)
    plt.close()

# -------------------------------
# M sweep (record r/p/se for all sources)
# -------------------------------

def run_m_sweep(model, scaler, df_temporal_base, m_start, m_end, m_steps, z_threshold, placeholder_means):
    global M  # Use global M variable
    M_values = np.linspace(m_start, m_end, m_steps)
    records = []
    iterator = M_values
    if TQDM:
        iterator = tqdm(M_values, desc='M sweep')

    for M_val in iterator:
        M = M_val  # Set global M value
        # Process temporal data with current M value
        df_pred = process_temporal_data(df_temporal_base, model, scaler)
        
        r_vals = []
        p_vals = []
        se_vals = []
        for i in range(1,7):
            col = f'δ18O_Vapour_Source{i}'
            if col not in df_pred.columns:
                r_vals.append(np.nan); p_vals.append(np.nan); se_vals.append(np.nan); continue
            data = df_pred[['Ghalegaon GNIP δ18O', col]].dropna().copy()
            if len(data) < 3:
                r_vals.append(np.nan); p_vals.append(np.nan); se_vals.append(np.nan); continue
            z = np.abs((data[col] - data[col].mean()) / (data[col].std() + 1e-12))
            data_clean = data[z <= z_threshold]
            if len(data_clean) < 3:
                r_vals.append(np.nan); p_vals.append(np.nan); se_vals.append(np.nan); continue
            r,p = pearsonr(data_clean['Ghalegaon GNIP δ18O'], data_clean[col])
            se = np.sqrt((1 - r*r) / (len(data_clean) - 2)) if len(data_clean) > 2 else np.nan
            r_vals.append(r); p_vals.append(p); se_vals.append(se)
        rec = {'M': float(M_val)}
        for i in range(1,7):
            rec[f'r{i}'] = r_vals[i-1]
            rec[f'p{i}'] = p_vals[i-1]
            rec[f'se{i}'] = se_vals[i-1]
        records.append(rec)
    results_df = pd.DataFrame.from_records(records)
    return results_df

# -------------------------------
# Main
# -------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description='GLIMPSE Scrpit v2: full pipeline with plotting')
    parser.add_argument('--data_dir', default='.', help='Directory containing GLIMPSE Dataset.xlsx')
    parser.add_argument('--f_name', default='GLIMPSE Dataset', help='Excel filename (without .xlsx)')
    parser.add_argument('--use_xgb', action='store_true', help='Use XGBoost if available')
    parser.add_argument('--m_start', type=float, default=-0.1)
    parser.add_argument('--m_end', type=float, default=0.1)
    parser.add_argument('--m_steps', type=int, default=201)
    parser.add_argument('--z_threshold', type=float, default=3.0)
    parser.add_argument('--output_dir', default='./glimpse_outputs')
    parser.add_argument('--make_figures', action='store_true', help='Generate publication figures')
    args = parser.parse_args(argv)

    # Set global variables
    global M, z_threshold
    M = 0.0
    z_threshold = args.z_threshold

    out_dir = args.output_dir
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = os.path.join(os.getcwd(), 'glimpse_outputs'); os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Using output dir: {out_dir}")

    # Read
    df_spatial = read_data(args.data_dir, args.f_name, 'Spatial')
    df_temporal_raw = read_data(args.data_dir, args.f_name, 'Temporal')

    # Prepare temporal by ensuring Source columns are shifted according to shift_amount logic used earlier
    source_columns = [col for col in df_temporal_raw.columns if 'Source' in col]
    df_temporal_source = df_temporal_raw[source_columns]
    df_temporal_v2 = df_temporal_raw.drop(columns=source_columns)
    shift_amount = -2
    new_indices = (df_temporal_source.index + shift_amount) % df_temporal_source.shape[0]
    df_temporal_source_shifted = df_temporal_source.iloc[new_indices]
    df_temporal_source_shifted.index = df_temporal_v2.index
    df_temporal = pd.concat([df_temporal_v2, df_temporal_source_shifted], axis=1)

    # Process spatial data and train model
    X_train_scaled, X_test_scaled, y_train_new, y_test_new, scaler, feature_names = process_train_test_data(df_spatial)

    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }

    model, mse, mae, r2_test, r2_train, y_train_pred, y_test_pred = train_and_evaluate_model(X_train_scaled, y_train_new, X_test_scaled, y_test_new, xgb_params)

    # Save training metrics
    pd.DataFrame([{'mse': mse, 'mae': mae, 'r2_test': r2_test, 'r2_train': r2_train}]).to_csv(os.path.join(out_dir, 'training_metrics.csv'), index=False)

    # CV
    try:
        X_full = df_spatial.drop(columns=['δ18O'])
        # transform full X with same encoding as training
        # for simplicity reuse process_train_test_data preprocess approach (encode class-season * temperature like before)
        # we'll compute CV on scaled features created from df_spatial
        Xp = None
        # quick recreation of scaled full X
        # WARNING: this follows the same sequence as process_train_test_data but on full dataset
        y_dummy = df_spatial[['Class','Season']]
        y_dum_enc = pd.get_dummies(y_dummy).multiply(df_spatial['Temperature'], axis=0)
        Xnum = df_spatial[['Latitude','Longitude','Elevation']].values
        X_full_mat = np.hstack([Xnum, y_dum_enc.values])
        Xs_full = scaler.transform(X_full_mat)
        cv_scores = cross_val_score(model, Xs_full, df_spatial['δ18O'].values, cv=5, scoring='r2')
        pd.DataFrame({'cv_fold': np.arange(1, len(cv_scores)+1), 'r2': cv_scores}).to_csv(os.path.join(out_dir, 'training_cv_scores.csv'), index=False)
        logger.info(f"Saved CV scores to {os.path.join(out_dir, 'training_cv_scores.csv')} (mean R2 = {cv_scores.mean():.3f})")
    except Exception:
        logger.exception('CV failed')

    # SHAP
    if SHAP_AVAILABLE:
        try:
            explainer = shap.Explainer(model)
            shap_vals = explainer(X_train_scaled)
            shap_png = os.path.join(out_dir, 'shap_summary.png')
            plt.figure(figsize=(6,6))
            shap.summary_plot(shap_vals, feature_names=feature_names, show=False)
            plt.tight_layout(); plt.savefig(shap_png, dpi=200); plt.close()
            logger.info(f"Saved SHAP summary to {shap_png}")
        except Exception:
            logger.exception('SHAP failed')

    # Run M sweep and save
    results_df = run_m_sweep(
    model, scaler, df_temporal,
    args.m_start, args.m_end, args.m_steps,
    args.z_threshold, placeholder_means=None)

    results_df.to_csv(os.path.join(out_dir, 'M_sweep_results.csv'), index=False)
    logger.info(f"Saved M sweep results to {os.path.join(out_dir, 'M_sweep_results.csv')}")

    # Best per source
    best_rows = []
    for i in range(1,7):
        col = f'r{i}'
        if col in results_df.columns and results_df[col].notna().any():
            idx = results_df[col].idxmax()
            best_rows.append({'source': f'source{i}', 'M': float(results_df.loc[idx,'M']), 'r': float(results_df.loc[idx,col]), 'p': float(results_df.loc[idx,f'p{i}']), 'se': float(results_df.loc[idx,f'se{i}'])})
    pd.DataFrame(best_rows).to_csv(os.path.join(out_dir, 'M_sweep_best_per_source.csv'), index=False)
    logger.info(f"Saved best-per-source summary to {os.path.join(out_dir, 'M_sweep_best_per_source.csv')}")

    # Create temporal predictions for best M (use best M for source5 as example)
    try:
        best_df = pd.read_csv(os.path.join(out_dir, 'M_sweep_best_per_source.csv'))
        if not best_df.empty:
            # construct a mapping from source->M
            best_map = {row['source']: row['M'] for _, row in best_df.iterrows()}
        else:
            best_map = {}
    except Exception:
        best_map = {}

    # Produce temporal df for M=0 and for best per-source if available
    M = 0.0
    df_pred_M0 = process_temporal_data(df_temporal, model, scaler)
    df_pred_M0.head(10).to_csv(os.path.join(out_dir, 'temporal_predictions_M0_head10.csv'), index=False)
    # Save full temporal for best source5 if present
    if 'source5' in best_map:
        M = float(best_map['source5'])
        df_pred_best5 = process_temporal_data(df_temporal, model, scaler)
        df_pred_best5.to_csv(os.path.join(out_dir, f"temporal_predictions_best_source5_M_{best_map['source5']:.6f}.csv"), index=False)

    # Make figures if requested
    scatter_plot_with_correlation(df_temporal)
    if args.make_figures:
        # Ensure df_temporal has δ18O_Vapour_SourceX computed for a chosen M (use M from best_map source5 or 0)
        M_for_plots = float(best_map.get('source5', 0.0))
        M = M_for_plots
        df_temporal_pred = process_temporal_data(df_temporal, model, scaler)

        # Figure: scatter train/test observed vs predicted
        plt.figure(figsize=(6,6))
        plt.scatter(y_test_new, y_test_pred, color='red', marker='+', label='Test data')
        plt.scatter(y_train_new, y_train_pred, color='blue', marker='x', label='Training data')
        plt.text(0.95, 0.2, f'R-squared = {r2_test:.2f}', transform=plt.gca().transAxes, color='red', horizontalalignment='right')
        plt.text(0.95, 0.15, f'R-squared = {r2_train:.2f}', transform=plt.gca().transAxes, color='blue', horizontalalignment='right')
        plt.text(0.95, 0.1, f'Mean Square Error = {mse:.2f}', transform=plt.gca().transAxes, color='red', horizontalalignment='right')
        plt.text(0.95, 0.05, f'Mean Absolute Error = {mae:.2f}', transform=plt.gca().transAxes, color='red', horizontalalignment='right')
        plt.xlabel(r'Observed δ$^{18}$O (‰)')
        plt.ylabel(r'Predicted δ$^{18}$O (‰)')
        plt.title('XGB Model Performance')
        plt.legend()
        for ext in ['png','pdf']:
            plt.savefig(os.path.join(out_dir, f'scatter_train_test.{ext}'), dpi=300)
        plt.close()

        # SHAP bar (if available)
        if SHAP_AVAILABLE:
            try:
                shap_vals = shap_vals if 'shap_vals' in locals() else None
                if shap_vals is None:
                    expl = shap.Explainer(model)
                    shap_vals = expl(X_train_scaled)
                shap_vals_arr = np.abs(shap_vals.values).mean(axis=0)
                shap_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': shap_vals_arr})
                shap_df = shap_df.sort_values('mean_abs_shap', ascending=True)
                shap_df.plot.barh(x='feature', y='mean_abs_shap', legend=False, figsize=(6,6))
                plt.xlabel('Mean |SHAP value|')
                for ext in ['png','pdf']:
                    plt.savefig(os.path.join(out_dir, f'shap_bar.{ext}'), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception:
                logger.exception('SHAP plotting failed')

        # r vs M per source
        results_df = pd.read_csv(os.path.join(out_dir, 'M_sweep_results.csv'))
        best_df = pd.read_csv(os.path.join(out_dir, 'M_sweep_best_per_source.csv'))
        # convert best_df to lookup if needed
        best_lookup = {row['source']: row['M'] for _, row in best_df.iterrows()} if not best_df.empty else {}

        fig, axes = plt.subplots(3,2, figsize=(10,10))
        for i, ax in enumerate(axes.flatten(), start=1):
            ax.plot(results_df['M'], results_df[f'r{i}'])
            m_best = best_lookup.get(f'source{i}', None)
            if m_best is not None:
                ax.axvline(float(m_best), color='red', linestyle='--')
            ax.set_title(f'Source {i}')
            ax.set_xlabel('M'); ax.set_ylabel('r')
        plt.tight_layout()
        for ext in ['png','pdf']:
            plt.savefig(os.path.join(out_dir, f'r_vs_M_per_source.{ext}'), dpi=300)
        plt.close()

        # Temporal evolution plots (as in original code)
        plot_temporal_evolution(df_temporal_pred, args.z_threshold)
        # Save boxplot and other multi-panels
        # Box plot
        source_cols = [f'δ18O_Vapour_Source{i}' for i in range(1,7)]
        if all(c in df_temporal_pred.columns for c in source_cols):
            plt.figure(figsize=(10,5))
            df_temp_box = df_temporal_pred.rename(columns=lambda x: x.replace('δ18O_Vapour_', ''))
            cols = ['Source1','Source2','Source3','Source4','Source5','Source6',
                    'Source_Combined_Seasonally','Ghalegaon GNIP δ18O']

            # Prepare the data for plt.boxplot
            data = [df_temp_box[col].dropna() for col in cols]

            # Create wrapped labels
            def wrap_label(text):
                return text.replace('_', '\n').replace(' ', '\n')

            wrapped_labels = [wrap_label(c) for c in cols]

            # Plot directly with matplotlib
            plt.boxplot(data, tick_labels=wrapped_labels)
            plt.ylabel('δ$^{18}$O (‰)')
            plt.title('Boxplot of source vapour δ18O and Ghalegaon')

            for ext in ['png', 'pdf']:
                plt.savefig(os.path.join(out_dir, f'boxplot_sources.{ext}'), dpi=300)

            plt.close()

        # Save temporal scatter/ET layout from original plot_temporal_evolution (create separate PNG)
        # Create scatter vs ET panel (two columns layout)
        try:
            fig, axes = plt.subplots(ncols=2, figsize=(12,8), gridspec_kw={'width_ratios':[8.5,1.5]})
            markers = ['o','^','s','D','P']
            et_sources = ['Source Open water evaporation','Source Canopy water evaporation','Source Direct Evaporation from Bare Soil','Source Snow Evaporation','Source Transpiration']
            for i, source in enumerate(et_sources):
                if source in df_temporal_pred.columns:
                    axes[0].scatter(df_temporal_pred['Source_Combined_Seasonally'], df_temporal_pred[source], c=df_temporal_pred['Day'], cmap='twilight', marker=markers[i], s=50, label=source)
            norm = Normalize(vmin=df_temporal_pred['Day'].min(), vmax=df_temporal_pred['Day'].max())
            scalar_mappable = cm.ScalarMappable(norm=norm, cmap='twilight')
            scalar_mappable.set_array([])
            cbar = fig.colorbar(scalar_mappable, ax=axes[0], ticks=range(0,366,50), pad=0.02)
            cbar.set_label('Day', labelpad=-45, y=1.03)
            axes[0].set_xlabel(r'δ$^{18}$O (‰)'); axes[0].set_ylabel(r'ET (kg/m$^2$/day)'); axes[0].set_title('Seasonally combined Source Vapour vs. Source resolved ET'); axes[0].set_xlim(-12,4)
            legend = axes[0].legend(loc='lower left', bbox_to_anchor=(0.05,0.05))
            for handle in legend.legend_handles:
                handle.set_color('black')
            axes[1].barh(df_temporal_pred['Day'], df_temporal_pred['ET Source (kg/m2/day)'], color='black', height=0.5)
            axes[1].set_yticks([]); axes[1].set_ylim(-1,366); axes[1].set_xlabel(r'ET (kg/m$^2$/day)'); axes[1].set_title('Source combined ET')
            plt.tight_layout()
            for ext in ['png','pdf']:
                plt.savefig(os.path.join(out_dir, 'source_vs_ET_panel.'+ext), dpi=300)
            plt.close()
        except Exception:
            logger.exception('Source vs ET plot failed')

    logger.info('Pipeline finished successfully.')

if __name__ == '__main__':
    main()
