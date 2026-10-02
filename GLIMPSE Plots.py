# GLIMPSE full pipeline with integrated figure generation
import os
import argparse
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from tqdm import tqdm

# === Helper Functions ===
def read_data(data_dir, f_name, sheet_name):
    path = os.path.join(data_dir, f_name + '.xlsx')
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_excel(path, sheet_name=sheet_name)
    return df

def process_train_test_data(df):
    X = df[['Latitude', 'Longitude', 'Elevation', 'Temperature', 'δ18O']]
    y = df[['Class', 'Season']]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    y_train_encoded = pd.get_dummies(y_train).multiply(X_train['Temperature'], axis=0)
    y_test_encoded = pd.get_dummies(y_test).multiply(X_test['Temperature'], axis=0)
    X_train.drop('Temperature', axis=1, inplace=True)
    X_test.drop('Temperature', axis=1, inplace=True)
    y_train_new = X_train['δ18O']
    y_test_new = X_test['δ18O']
    X_train = X_train.drop('δ18O', axis=1)
    X_test = X_test.drop('δ18O', axis=1)
    column_names = list(X_train.columns)
    X_train_new = np.hstack((X_train, y_train_encoded))
    X_test_new = np.hstack((X_test, y_test_encoded))
    column_names.extend(list(y_train_encoded.columns))
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_new)
    X_test_scaled = scaler.transform(X_test_new)
    return X_train_scaled, X_test_scaled, y_train_new, y_test_new, scaler, column_names

def train_and_evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test, feature_names):
    model = xgb.XGBRegressor(objective='reg:squarederror', max_depth=3, learning_rate=0.1,
                             n_estimators=100, subsample=0.8, colsample_bytree=0.8)
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    return model, y_train_pred, y_test_pred, mse, mae, r2_test, r2_train

def make_figures(out_dir, y_train, y_test, y_train_pred, y_test_pred, feature_names, model, X_train_scaled, m_sweep_df, m_best_df, df_temporal):
    os.makedirs(out_dir, exist_ok=True)
    # Scatter train/test
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_test_pred, c='red', label=f"Test (R²={r2_score(y_test, y_test_pred):.2f})")
    plt.scatter(y_train, y_train_pred, c='blue', alpha=0.6, label=f"Train (R²={r2_score(y_train, y_train_pred):.2f})")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
    plt.xlabel("Observed δ$^{18}$O (‰)")
    plt.ylabel("Predicted δ$^{18}$O (‰)")
    plt.legend()
    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{out_dir}/scatter_train_test.{ext}", dpi=300)
    plt.close()
    # SHAP summary + bar
    explainer = shap.Explainer(model, X_train_scaled)
    shap_values = explainer(X_train_scaled)
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    for ext in ["png","pdf"]:
        plt.savefig(f"{out_dir}/shap_summary.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    shap_means = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": shap_means})
    shap_df.sort_values("mean_abs_shap", ascending=True).plot.barh(x="feature", y="mean_abs_shap", legend=False)
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{out_dir}/shap_bar.{ext}", dpi=300)
    plt.close()
    # r vs M per source
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten(), start=1):
        ax.plot(m_sweep_df["M"], m_sweep_df[f"r{i}"])
        ax.axvline(m_best_df.loc[f"source{i}", "M"], color="red", linestyle="--")
        ax.set_title(f"Source {i}")
    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{out_dir}/r_vs_M_per_source.{ext}", dpi=300)
    plt.close()
    # Temporal for best M per source
    for i in range(1, 7):
        plt.figure(figsize=(10,4))
        plt.plot(df_temporal["Day"], df_temporal["Ghalegaon GNIP δ18O"], label="Ghalegaon GNIP", color="black")
        plt.plot(df_temporal["Day"], df_temporal[f"δ18O_Vapour_Source{i}"], label=f"Source {i}")
        plt.xlabel("Day"); plt.ylabel("δ$^{18}$O (‰)"); plt.legend()
        for ext in ["png","pdf"]:
            plt.savefig(f"{out_dir}/temporal_best_source{i}.{ext}", dpi=300)
        plt.close()
    # Boxplot sources
    source_cols = [f"δ18O_Vapour_Source{i}" for i in range(1,7)]
    df_temporal[source_cols].boxplot()
    plt.ylabel("δ$^{18}$O (‰)")
    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{out_dir}/boxplot_sources.{ext}", dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--f_name", type=str, default="GLIMPSE Dataset")
    args = parser.parse_args()
    OUT_DIR = "./glimpse_outputs"
    os.makedirs(OUT_DIR, exist_ok=True)
    df_spatial = read_data(args.data_dir, args.f_name, 'Spatial')
    df_temporal = read_data(args.data_dir, args.f_name, 'Temporal')
    X_train_scaled, X_test_scaled, y_train_new, y_test_new, scaler, feature_names = process_train_test_data(df_spatial)
    model, y_train_pred, y_test_pred, mse, mae, r2_test, r2_train = train_and_evaluate_model(
        X_train_scaled, y_train_new, X_test_scaled, y_test_new, feature_names
    )
    # Dummy sweep outputs for plotting example
    m_sweep_df = pd.DataFrame({"M": np.linspace(-0.1,0.1,21), **{f"r{i}": np.random.rand(21) for i in range(1,7)}})
    m_best_df = pd.DataFrame({"M": np.random.choice(m_sweep_df["M"],6), "r": np.random.rand(6)},
                             index=[f"source{i}" for i in range(1,7)])
    make_figures(OUT_DIR, y_train_new, y_test_new, y_train_pred, y_test_pred, feature_names,
                 model, X_train_scaled, m_sweep_df, m_best_df, df_temporal)
