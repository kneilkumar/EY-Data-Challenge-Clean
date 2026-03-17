"""
MAXIMUM AGGRESSION R² SCRIPT
No constraints. No mercy. No chill.

Strategies:
- KNN at 8 k values + mean/median/std/q25/q75/logspace/uniform variants
- Polynomial interactions on top features
- RFE to 130 features
- Optuna 60 trials XGB + 60 LGBM + 40 CatBoost per target
- Extra Trees + Random Forest + MLP in ensemble
- GroupKFold OOF stacking with Ridge meta-learner
- Pseudo-labeling on confident test samples
- Two-pass target chaining (TA<-EC, EC<-TA, DRP<-TA)
- 12-seed blend on full data
- Aggressive clipping both ends
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

X = pd.read_csv("train_features.csv").reset_index(drop=True)
V = pd.read_csv("test_features.csv").reset_index(drop=True)

TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
DROP_COLS   = TARGET_COLS + ['Latitude', 'Longitude', 'Sample Date', 'station_id', 'xx', 'Litho']

TA  = X['Total Alkalinity'].reset_index(drop=True)
EC  = X['Electrical Conductance'].reset_index(drop=True)
DRP = X['Dissolved Reactive Phosphorus'].reset_index(drop=True)
groups = X['station_id']

BOUNDS = {
    'Total Alkalinity':              (float(TA.min()  * 0.5), float(TA.max()  * 1.5)),
    'Electrical Conductance':        (float(EC.min()  * 0.5), float(EC.max()  * 1.5)),
    'Dissolved Reactive Phosphorus': (float(DRP.min() * 0.5), float(DRP.max() * 1.5)),
}
print("Bounds:", BOUNDS)

# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, val_idx = next(gss.split(X, TA, groups=groups))
train_idx = np.array(train_idx)
val_idx   = np.array(val_idx)

print(f"Train: {X['station_id'].iloc[train_idx].nunique()} stations | "
      f"Val: {X['station_id'].iloc[val_idx].nunique()} stations")

# ---------------------------------------------------------------------------
# KNN features
# ---------------------------------------------------------------------------

print("\nComputing KNN features...")

train_df    = X.iloc[train_idx]
station_df  = train_df.groupby(['Latitude', 'Longitude'])[TARGET_COLS].mean().reset_index()
station_med = train_df.groupby(['Latitude', 'Longitude'])[TARGET_COLS].median().reset_index()
station_std = train_df.groupby(['Latitude', 'Longitude'])[TARGET_COLS].std().fillna(0).reset_index()
station_q25 = train_df.groupby(['Latitude', 'Longitude'])[TARGET_COLS].quantile(0.25).reset_index()
station_q75 = train_df.groupby(['Latitude', 'Longitude'])[TARGET_COLS].quantile(0.75).reset_index()

station_coords = station_df[['Latitude', 'Longitude']].values
train_coords   = X[['Latitude', 'Longitude']].values
test_coords    = V[['Latitude', 'Longitude']].values

knn_map = {
    'Total Alkalinity':              'knn_TA_mean',
    'Electrical Conductance':        'knn_EC_mean',
    'Dissolved Reactive Phosphorus': 'knn_DRP_mean',
}

FORCED_COLS = []
n_stations = len(station_coords)

for k in [2, 3, 5, 8, 10, 15, 20, 30]:
    k_use = min(k, n_stations - 1)
    knn = KNeighborsRegressor(n_neighbors=k_use, weights='distance')
    for target_col, feat_base in knn_map.items():
        feat_name = f"{feat_base}_k{k}"
        knn.fit(station_coords, station_df[target_col].values)
        X[feat_name] = knn.predict(train_coords)
        V[feat_name] = knn.predict(test_coords)
        FORCED_COLS.append(feat_name)

for stat_name, stat_df in [('median', station_med), ('std', station_std),
                             ('q25', station_q25), ('q75', station_q75)]:
    for target_col, feat_base in knn_map.items():
        feat_name = f"{feat_base}_{stat_name}"
        knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
        knn.fit(station_coords, stat_df[target_col].values)
        X[feat_name] = knn.predict(train_coords)
        V[feat_name] = knn.predict(test_coords)
        FORCED_COLS.append(feat_name)

for k in [5, 10]:
    k_use = min(k, n_stations - 1)
    knn_u = KNeighborsRegressor(n_neighbors=k_use, weights='uniform')
    for target_col, feat_base in knn_map.items():
        feat_name = f"{feat_base}_uniform_k{k}"
        knn_u.fit(station_coords, station_df[target_col].values)
        X[feat_name] = knn_u.predict(train_coords)
        V[feat_name] = knn_u.predict(test_coords)
        FORCED_COLS.append(feat_name)

for target_col, feat_base in knn_map.items():
    feat_name = f"{feat_base}_logspace"
    knn_log = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn_log.fit(station_coords, np.log1p(station_df[target_col].values))
    X[feat_name] = np.expm1(knn_log.predict(train_coords))
    V[feat_name] = np.expm1(knn_log.predict(test_coords))
    FORCED_COLS.append(feat_name)

for target_col, feat_name in knn_map.items():
    X[feat_name] = X[f"{feat_name}_k5"]
    V[feat_name] = V[f"{feat_name}_k5"]
    if feat_name not in FORCED_COLS:
        FORCED_COLS.append(feat_name)

print(f"KNN cols: {len(FORCED_COLS)}")

# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

print("Engineering interactions...")

for df in [X, V]:
    c = df.columns.tolist()
    if '3m_mean_q' in c and 'sg_clay_t8k_mean' in c:
        df['runoff_x_clay_8k']        = df['3m_mean_q'] * df['sg_clay_t8k_mean']
    if '3m_mean_ppt' in c and 'lith_sc' in c:
        df['ppt_x_carbonate']         = df['3m_mean_ppt'] * df['lith_sc']
    if 'z_score_q' in c and 'sg_ph_t8k_mean' in c:
        df['runoff_anomaly_x_ph']     = df['z_score_q'] * df['sg_ph_t8k_mean']
    if '3m_mean_soil' in c and 'cropland_upstream_frac' in c:
        df['soil_x_crop']             = df['3m_mean_soil'] * df['cropland_upstream_frac']
    if 'sg_cec_t8k_mean' in c and 'sg_ph_t8k_mean' in c:
        df['cec_x_ph']                = df['sg_cec_t8k_mean'] * df['sg_ph_t8k_mean']
    if '3m_mean_def' in c and 'urban_upstream_frac' in c:
        df['deficit_x_urban']         = df['3m_mean_def'] * df['urban_upstream_frac']
    if 'sg_ph_updown_ratio' in c and 'lith_sc' in c:
        df['ph_ratio_x_carbonate']    = df['sg_ph_updown_ratio'] * df['lith_sc']
    if 'knn_EC_mean' in c and 'knn_TA_mean' in c:
        df['knn_EC_over_TA']          = df['knn_EC_mean'] / (df['knn_TA_mean'] + 1e-6)
    if 'knn_TA_mean_k3' in c and 'knn_TA_mean_k20' in c:
        df['knn_TA_local_vs_regional']= df['knn_TA_mean_k3'] - df['knn_TA_mean_k20']
    if 'knn_EC_mean_k3' in c and 'knn_EC_mean_k20' in c:
        df['knn_EC_local_vs_regional']= df['knn_EC_mean_k3'] - df['knn_EC_mean_k20']

# Polynomial interactions on top KNN + geology
poly_cols = [c for c in ['knn_TA_mean', 'knn_EC_mean', 'knn_DRP_mean',
                          'sg_ph_t8k_mean', 'sg_clay_t8k_mean', 'lith_sc',
                          'sg_cec_t8k_mean', '3m_mean_q', '3m_mean_soil']
             if c in X.columns]
if len(poly_cols) >= 2:
    poly    = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_tr = poly.fit_transform(X[poly_cols].fillna(0))
    poly_te = poly.transform(V[poly_cols].fillna(0))
    poly_names = poly.get_feature_names_out(poly_cols)
    n_added = 0
    for i, name in enumerate(poly_names):
        if ' ' in name:  # interaction terms have a space between feature names
            col_name = f"poly_{name.replace(' ', '_')}"
            X[col_name] = poly_tr[:, i]
            V[col_name] = poly_te[:, i]
            n_added += 1
    print(f"  Added {n_added} polynomial interaction features")

# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

feature_cols = [c for c in X.columns if c not in DROP_COLS]
X_feat = X[feature_cols].astype(float).reset_index(drop=True)
V_feat = V[feature_cols].astype(float).reset_index(drop=True)

print(f"Starting features: {X_feat.shape}")

# ---------------------------------------------------------------------------
# RFE to 130
# ---------------------------------------------------------------------------

print("\nRunning RFE to 130...")

N_FEATURES = 130
rfe_est = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                        random_state=42, verbosity=0)

selected_cols = set(FORCED_COLS)
for target_name, y_raw in [('TA', TA), ('EC', EC), ('DRP', DRP)]:
    print(f"  RFE for {target_name}...")
    n_sel = min(100, X_feat.shape[1] - 1)
    rfe = RFE(estimator=rfe_est, n_features_to_select=n_sel, step=10)
    rfe.fit(X_feat.iloc[train_idx].fillna(0), np.log1p(y_raw.iloc[train_idx]))
    cols = X_feat.columns[rfe.support_].tolist()
    selected_cols.update(cols)
    print(f"    {len(cols)} selected — union: {len(selected_cols)}")

if len(selected_cols) > N_FEATURES:
    non_forced = [c for c in selected_cols if c not in FORCED_COLS]
    imp_m = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    imp_m.fit(X_feat[list(selected_cols)].iloc[train_idx].fillna(0),
              np.log1p(TA.iloc[train_idx]))
    imp    = pd.Series(imp_m.feature_importances_, index=list(selected_cols))
    top_nf = imp[non_forced].nlargest(N_FEATURES - len(FORCED_COLS)).index.tolist()
    selected_cols = FORCED_COLS + top_nf

selected_cols = list(dict.fromkeys(selected_cols))
print(f"Final features: {len(selected_cols)}")

X_feat = X_feat[selected_cols].reset_index(drop=True)
V_feat = V_feat[selected_cols].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Optuna helpers
# ---------------------------------------------------------------------------

def tune_xgb(X_tr, y_tr, X_val, y_val, n_trials=60):
    def objective(trial):
        m = XGBRegressor(
            n_estimators     = trial.suggest_int("n", 300, 3000),
            learning_rate    = trial.suggest_float("lr", 0.003, 0.05, log=True),
            max_depth        = trial.suggest_int("d", 3, 6),
            subsample        = trial.suggest_float("sub", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("col", 0.4, 1.0),
            min_child_weight = trial.suggest_int("mcw", 3, 40),
            reg_alpha        = trial.suggest_float("a", 0.01, 10.0, log=True),
            reg_lambda       = trial.suggest_float("l", 0.1, 20.0, log=True),
            gamma            = trial.suggest_float("g", 0.0, 5.0),
            random_state=42, verbosity=0,
        )
        m.fit(X_tr, y_tr)
        return r2_score(y_val, m.predict(X_val))
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_lgbm(X_tr, y_tr, X_val, y_val, n_trials=60):
    def objective(trial):
        m = LGBMRegressor(
            n_estimators      = trial.suggest_int("n", 300, 3000),
            learning_rate     = trial.suggest_float("lr", 0.003, 0.05, log=True),
            max_depth         = trial.suggest_int("d", 3, 6),
            num_leaves        = trial.suggest_int("nl", 15, 63),
            subsample         = trial.suggest_float("sub", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("col", 0.4, 1.0),
            min_child_samples = trial.suggest_int("mcs", 5, 60),
            reg_alpha         = trial.suggest_float("a", 0.01, 10.0, log=True),
            reg_lambda        = trial.suggest_float("l", 0.1, 20.0, log=True),
            random_state=42, verbose=-1,
        )
        m.fit(X_tr, y_tr)
        return r2_score(y_val, m.predict(X_val))
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_cat(X_tr, y_tr, X_val, y_val, n_trials=40):
    def objective(trial):
        m = CatBoostRegressor(
            iterations    = trial.suggest_int("n", 300, 2000),
            learning_rate = trial.suggest_float("lr", 0.005, 0.1, log=True),
            depth         = trial.suggest_int("d", 4, 8),
            l2_leaf_reg   = trial.suggest_float("l2", 1.0, 20.0, log=True),
            random_seed=42, verbose=0,
        )
        m.fit(X_tr.fillna(-999), y_tr)
        return r2_score(y_val, m.predict(X_val.fillna(-999)))
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# ---------------------------------------------------------------------------
# Core train function
# ---------------------------------------------------------------------------

SEEDS = [42, 7, 13, 21, 99, 123, 256, 512, 1337, 2024, 31415, 999]
N_MODELS = 6  # xgb, lgbm, cat, et, rf, mlp

def train_target(y_raw, target_name, log_transform=True,
                 extra_cols_train=None, extra_cols_test=None):
    print(f"\n{'='*60}")
    print(f"TARGET: {target_name} | log={log_transform}")
    print(f"{'='*60}")

    lo, hi = BOUNDS[target_name]

    Xf = X_feat.copy()
    Vf = V_feat.copy()
    if extra_cols_train:
        for col, vals in extra_cols_train.items():
            Xf[col] = np.array(vals)
            Vf[col] = np.array(extra_cols_test[col])

    X_tr  = Xf.iloc[train_idx].reset_index(drop=True)
    X_val = Xf.iloc[val_idx].reset_index(drop=True)

    y_tr_raw  = y_raw.iloc[train_idx].reset_index(drop=True)
    y_val_raw = y_raw.iloc[val_idx].reset_index(drop=True)
    y_tr   = np.log1p(y_tr_raw).values  if log_transform else y_tr_raw.values
    y_val  = np.log1p(y_val_raw).values if log_transform else y_val_raw.values
    y_full = np.log1p(y_raw).values     if log_transform else y_raw.values

    def inv(p):
        p = np.expm1(np.array(p)) if log_transform else np.array(p)
        return np.clip(p, lo, hi)

    knn_col = knn_map[target_name]
    print(f"  KNN R²: {r2_score(y_val_raw, np.clip(X_val[knn_col], lo, hi)):.4f}")

    # Tune all three
    print("  Tuning XGBoost (60 trials)...")
    xgb_best = tune_xgb(X_tr, y_tr, X_val, y_val, n_trials=60)
    print("  Tuning LightGBM (60 trials)...")
    lgbm_best = tune_lgbm(X_tr, y_tr, X_val, y_val, n_trials=60)
    print("  Tuning CatBoost (40 trials)...")
    cat_best = tune_cat(X_tr, y_tr, X_val, y_val, n_trials=40)

    # Fit base models
    xgb_m  = XGBRegressor(**xgb_best, random_state=42, verbosity=0)
    xgb_m.fit(X_tr, y_tr)

    lgbm_m = LGBMRegressor(**lgbm_best, random_state=42, verbose=-1)
    lgbm_m.fit(X_tr, y_tr)

    cat_m  = CatBoostRegressor(**cat_best, random_seed=42, verbose=0)
    cat_m.fit(X_tr.fillna(-999), y_tr)

    et_m = ExtraTreesRegressor(n_estimators=500, max_features=0.5,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
    et_m.fit(X_tr.fillna(0), y_tr)

    rf_m = RandomForestRegressor(n_estimators=500, max_features=0.5,
                                  min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_m.fit(X_tr.fillna(0), y_tr)

    scaler_mlp = StandardScaler()
    X_tr_s = scaler_mlp.fit_transform(X_tr.fillna(0))
    X_val_s = scaler_mlp.transform(X_val.fillna(0))
    Xf_s    = scaler_mlp.transform(Xf.fillna(0))
    Vf_s    = scaler_mlp.transform(Vf.fillna(0))

    mlp_m = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation='relu',
                          learning_rate_init=0.001, max_iter=500, random_state=42,
                          early_stopping=True, validation_fraction=0.1,
                          n_iter_no_change=20)
    mlp_m.fit(X_tr_s, y_tr)

    # Val scores
    p_xgb  = xgb_m.predict(X_val)
    p_lgbm = lgbm_m.predict(X_val)
    p_cat  = cat_m.predict(X_val.fillna(-999))
    p_et   = et_m.predict(X_val.fillna(0))
    p_rf   = rf_m.predict(X_val.fillna(0))
    p_mlp  = mlp_m.predict(X_val_s)

    for name, pred in [('XGB', p_xgb), ('LGBM', p_lgbm), ('CAT', p_cat),
                        ('ET', p_et), ('RF', p_rf), ('MLP', p_mlp)]:
        print(f"  {name:4s} R²: {r2_score(y_val_raw, inv(pred)):.4f}")

    meta_val = np.column_stack([p_xgb, p_lgbm, p_cat, p_et, p_rf, p_mlp])

    # GroupKFold OOF stacking
    gkf = GroupKFold(n_splits=5)
    fold_groups = X['station_id'].iloc[train_idx].values
    oof = np.zeros((len(train_idx), N_MODELS))

    for fold_tr, fold_val in gkf.split(X_tr, y_tr, groups=fold_groups):
        Xf_tr, Xf_val = X_tr.iloc[fold_tr], X_tr.iloc[fold_val]
        yf = y_tr[fold_tr]

        m = XGBRegressor(**xgb_best, random_state=42, verbosity=0)
        m.fit(Xf_tr, yf); oof[fold_val, 0] = m.predict(Xf_val)

        m = LGBMRegressor(**lgbm_best, random_state=42, verbose=-1)
        m.fit(Xf_tr, yf); oof[fold_val, 1] = m.predict(Xf_val)

        m = CatBoostRegressor(**cat_best, random_seed=42, verbose=0)
        m.fit(Xf_tr.fillna(-999), yf); oof[fold_val, 2] = m.predict(Xf_val.fillna(-999))

        m = ExtraTreesRegressor(n_estimators=200, max_features=0.5,
                                 min_samples_leaf=5, random_state=42, n_jobs=-1)
        m.fit(Xf_tr.fillna(0), yf); oof[fold_val, 3] = m.predict(Xf_val.fillna(0))

        m = RandomForestRegressor(n_estimators=200, max_features=0.5,
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
        m.fit(Xf_tr.fillna(0), yf); oof[fold_val, 4] = m.predict(Xf_val.fillna(0))

        s = StandardScaler()
        m = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300,
                          random_state=42, early_stopping=True)
        m.fit(s.fit_transform(Xf_tr.fillna(0)), yf)
        oof[fold_val, 5] = m.predict(s.transform(Xf_val.fillna(0)))

    scaler_meta = StandardScaler()
    ridge = Ridge(alpha=1.0)
    ridge.fit(scaler_meta.fit_transform(oof), y_tr)
    stacked_val = ridge.predict(scaler_meta.transform(meta_val))
    print(f"  STACKED R²: {r2_score(y_val_raw, inv(stacked_val)):.4f}")

    # OOF for chaining
    oof_full = np.zeros(len(Xf))
    oof_full[train_idx] = inv(ridge.predict(scaler_meta.transform(oof)))
    oof_full[val_idx]   = inv(stacked_val)

    # Pseudo-labeling on confident test samples
    pseudo_preds = np.column_stack([
        xgb_m.predict(Vf),
        lgbm_m.predict(Vf),
        cat_m.predict(Vf.fillna(-999)),
        et_m.predict(Vf.fillna(0)),
        rf_m.predict(Vf.fillna(0)),
        mlp_m.predict(Vf_s),
    ])
    pseudo_mean = pseudo_preds.mean(axis=1)
    pseudo_std  = pseudo_preds.std(axis=1)
    conf_thresh = np.percentile(pseudo_std, 30)
    confident   = pseudo_std < conf_thresh
    print(f"  Pseudo-label: {confident.sum()} confident test samples")

    if confident.sum() > 0:
        Xf_aug = pd.concat([Xf, Vf[confident].reset_index(drop=True)], ignore_index=True)
        y_aug  = np.concatenate([y_full, pseudo_mean[confident]])
    else:
        Xf_aug = Xf
        y_aug  = y_full

    Xf_aug_s = scaler_mlp.transform(Xf_aug.fillna(0))

    # Seed blend on full + pseudo data
    n_xgb  = max(int(xgb_best.get('n_estimators', 500)), 200)
    n_lgbm = max(int(lgbm_best.get('n_estimators', 500)), 200)
    n_cat  = max(int(cat_best.get('iterations', 500)), 200)

    test_preds = []
    for i, seed in enumerate(SEEDS):
        print(f"  Seed {seed} ({i+1}/{len(SEEDS)})...", end=" ", flush=True)

        xm = XGBRegressor(**{**xgb_best, 'n_estimators': n_xgb},
                           random_state=seed, verbosity=0)
        xm.fit(Xf_aug, y_aug)

        lm = LGBMRegressor(**{**lgbm_best, 'n_estimators': n_lgbm},
                            random_state=seed, verbose=-1)
        lm.fit(Xf_aug, y_aug)

        cm = CatBoostRegressor(**{**cat_best, 'iterations': n_cat},
                                random_seed=seed, verbose=0)
        cm.fit(Xf_aug.fillna(-999), y_aug)

        em = ExtraTreesRegressor(n_estimators=300, max_features=0.5,
                                  min_samples_leaf=5, random_state=seed, n_jobs=-1)
        em.fit(Xf_aug.fillna(0), y_aug)

        seed_stack = np.column_stack([
            xm.predict(Vf),
            lm.predict(Vf),
            cm.predict(Vf.fillna(-999)),
            em.predict(Vf.fillna(0)),
            rf_m.predict(Vf.fillna(0)),   # RF fixed — reusing original
            mlp_m.predict(Vf_s),           # MLP fixed — reusing original
        ])
        test_preds.append(ridge.predict(scaler_meta.transform(seed_stack)))
        print("done")

    blended = inv(np.mean(test_preds, axis=0))
    print(f"  Final — min:{blended.min():.3f} max:{blended.max():.3f} mean:{blended.mean():.3f}")

    return blended, oof_full

# ---------------------------------------------------------------------------
# PASS 1 — bootstrap
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("PASS 1 — Bootstrap")
print("="*60)

ta_pred_p1, ta_oof_p1 = train_target(TA, "Total Alkalinity", log_transform=False)
ec_pred_p1, ec_oof_p1 = train_target(EC, "Electrical Conductance")

# ---------------------------------------------------------------------------
# PASS 2 — chained: TA<-EC, EC<-TA, DRP<-TA only
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("PASS 2 — Chained")
print("="*60)

ta_pred, ta_oof = train_target(
    TA, "Total Alkalinity",
    log_transform=False,
    extra_cols_train={"pred_EC": ec_oof_p1},
    extra_cols_test ={"pred_EC": ec_pred_p1},
)

ec_pred, ec_oof = train_target(
    EC, "Electrical Conductance",
    extra_cols_train={"pred_TA": ta_oof_p1},
    extra_cols_test ={"pred_TA": ta_pred_p1},
)

drp_pred, _ = train_target(
    DRP, "Dissolved Reactive Phosphorus",
    extra_cols_train={"pred_TA": ta_oof},
    extra_cols_test ={"pred_TA": ta_pred},
)

# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

submission = pd.read_csv("submission_template.csv")
submission["Total Alkalinity"]              = ta_pred
submission["Electrical Conductance"]        = ec_pred
submission["Dissolved Reactive Phosphorus"] = drp_pred

submission.to_csv("submission_unhinged.csv", index=False)
print(f"\n{'='*60}")
print("DONE. submission_unhinged.csv")
print(submission[TARGET_COLS].describe())