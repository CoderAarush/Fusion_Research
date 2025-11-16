from google import colab
colab.drive.mount('/content/drive')

import os
import h5py
import numpy as np
import pandas as pd
from scipy.stats import iqr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm
import pickle
import joblib

# ==================== CONFIG ====================
!mkdir -p /content/data
if not os.path.exists('/content/data/Nuclear_Fusion_Data'):
    print("Copying data...")
    !cp -r "/content/drive/MyDrive/Nuclear_Fusion_Data" /content/data/
base = '/content/data/Nuclear_Fusion_Data'

reactors = [
    'JDDB_repo_2A_5k',
    'J-TEXT processed_data_1k_5k_final',
    'CMod_train',
    'CMod_evaluate_20_10_2023'
]

# --- Only use line-averaged density ---
sensors = ['density']
key_map = {
    'density': {
        'CMod': '.tci.results:nl_04',
        'J-TEXT': 'polaris_den_v09',
        'JDDB': 'CCO-DF:DENSITY1'
    }
}

# ==================== DIR METRIC ====================
def compute_dir(diff):
    if len(diff) < 4:
        return 0.0
    return iqr(diff) / (np.mean(np.abs(diff)) + 1e-8)

# ==================== DATA LOADING ====================
def list_h5_files(base_folder, depth_limit=4):
    h5_files = []
    for root, _, files in os.walk(base_folder):
        depth = root[len(base_folder):].count(os.sep)
        if depth > depth_limit:
            continue
        for f in files:
            if f.endswith(('.h5', '.hdf5')):
                h5_files.append(os.path.join(root, f))
    return sorted(h5_files)

def extract_features(arr, meta_val):
    diff = arr[1:] - arr[:-1]
    base = np.array([
        len(arr), np.mean(arr), np.ptp(arr), np.std(arr), np.max(arr),
        np.min(arr), np.median(arr), np.var(arr), iqr(arr),
        np.max(diff), np.min(diff), np.mean(diff), np.std(diff),
        np.ptp(diff), np.median(diff), np.var(diff), iqr(diff)
    ], dtype=float)
    dir_val = compute_dir(diff)
    return np.concatenate([base, [dir_val], [meta_val]])

def load_hdf5_dataset(file_list, sensors, key_map):
    data_list, label_list = [], []
    for filename in tqdm(file_list, desc="Train", leave=False):
        try:
            with h5py.File(filename, 'r') as f:
                tok_type = next((t for t in ['CMod','J-TEXT','JDDB'] if t in filename), None)
                if not tok_type:
                    continue
                key = key_map['density'][tok_type]
                if key not in f["data"] or len(f["data"][key]) < 60:
                    continue
                arr = np.array(f["data"][key])
                meta_val = float(f["meta"][key][()]) if key in f["meta"] else 0.0
                feats = extract_features(arr, meta_val)

                one_hot = [1,0,0] if tok_type=='JDDB' else [0,1,0] if tok_type=='J-TEXT' else [0,0,1]
                data_list.append(np.concatenate([feats, one_hot]))
                label_list.append(int(f["meta"]["IsDisrupt"][()]))
        except Exception:
            continue
    df = pd.DataFrame(data_list) if data_list else pd.DataFrame()
    return df, np.array(label_list) if label_list else np.array([])

def load_test(file_list, sensors, key_map):
    data_list = []
    for filename in tqdm(file_list, desc="Test", leave=False):
        try:
            with h5py.File(filename, 'r') as f:
                tok_type = next((t for t in ['CMod','J-TEXT','JDDB'] if t in filename), None)
                if not tok_type:
                    continue
                key = key_map['density'][tok_type]
                if key not in f["data"] or len(f["data"][key]) < 60:
                    continue
                arr = np.array(f["data"][key])
                meta_val = float(f["meta"][key][()]) if key in f["meta"] else 0.0
                feats = extract_features(arr, meta_val)

                one_hot = [1,0,0] if tok_type=='JDDB' else [0,1,0] if tok_type=='J-TEXT' else [0,0,1]
                data_list.append(np.concatenate([feats, one_hot]))
        except Exception:
            continue
    return pd.DataFrame(data_list) if data_list else pd.DataFrame()

# ==================== CACHE SYSTEM ====================
CACHE_FILE = '/content/processed_data_density_only.pkl'
if os.path.exists(CACHE_FILE):
    print("Loading cached density-only data...")
    with open(CACHE_FILE, 'rb') as f:
        saved = pickle.load(f)
        data_dict = saved['data_dict']
        labels_dict = saved['labels_dict']
else:
    print("No cache found — loading HDF5 files...")
    h5_files_dict = {}
    for r in reactors:
        path = os.path.join(base, r)
        h5_files_dict[r] = list_h5_files(path) if os.path.exists(path) else []
        print(f"{r}: {len(h5_files_dict[r])} files")

    data_dict, labels_dict = {}, {}
    for r in reactors:
        files = h5_files_dict[r]
        if not files:
            print(f"SKIPPING {r}: no files")
            continue
        print(f"Loading {r}...")
        if r == 'CMod_evaluate_20_10_2023':
            data_dict[r] = load_test(files, sensors, key_map)
        else:
            df, labels = load_hdf5_dataset(files, sensors, key_map)
            data_dict[r] = df
            labels_dict[r] = labels
            if len(df) > 0:
                data_dict[r]['Is_disrupt'] = labels
        if len(data_dict[r]) > 0:
            data_dict[r].columns = data_dict[r].columns.astype(str)
        print(f"{r}: {len(data_dict[r])} samples")

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'data_dict': data_dict, 'labels_dict': labels_dict}, f)
    print("✅ Saved density-only data to cache")

# ==================== COMBINE TRAINING ====================
train_frames = [data_dict[r] for r in reactors[:2] if r in data_dict and len(data_dict[r]) > 0]
if not train_frames:
    raise ValueError("NO TRAINING DATA LOADED!")
train = pd.concat(train_frames).reset_index(drop=True)
data_test = data_dict.get('CMod_evaluate_20_10_2023', pd.DataFrame())

print(f"Train: {len(train)}, Test: {len(data_test)}")

# ==================== FEATURE NAMES ====================
feat_list = [
    'length','mean','ptp','std','max','min','median','var','iqr',
    'diff_max','diff_min','diff_mean','diff_std','diff_ptp',
    'diff_median','diff_var','diff_iqr','DIR','meta'
]
feature_names = [f"density_{f}" for f in feat_list] + ['JDDB','J-TEXT','CMod']

# ==================== CPU-FRIENDLY PARAMS ====================
BAYSEIAN_PARAMS = {
    'bagging_fraction': 0.8831884682475684,
    'bagging_freq': 4,
    'boosting_type': 'dart',
    'feature_fraction': 0.9108411997878578,
    'learning_rate': 0.5037787525092214,
    'max_depth': 25,
    'min_child_samples': 50,
    'min_child_weight': 6.801272575129432,
    'min_split_gain': 0.13542794845755993,
    'n_estimators': 472,
    'num_leaves': 137,
    'path_smooth': 0.4144600551329115,
    'reg_alpha': 0.8486845142673234,
    'reg_lambda': 0.4445709417507092,
    'device': 'cpu',
    'max_bin': 255,
    'n_jobs': -1,
    'random_state': 20,
    'verbose': 0
}

# ==================== MODEL CLASS ====================
class ExplainableLGBMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **params):
        self.params = params
        self.model = None
        self.feature_names = None

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def explain_predictions(self, X, class_index=1, max_display=20, show=True):
        shap_values = self.explainer.shap_values(X)

        # Handle binary (scalar) vs multiclass (list)
        if isinstance(shap_values, list):
            sv = shap_values[class_index]
            expected_val = (
                self.explainer.expected_value[class_index]
                if isinstance(self.explainer.expected_value, list)
                else self.explainer.expected_value
            )
        else:
            sv = shap_values
            expected_val = self.explainer.expected_value

        exp = shap.Explanation(
            sv, expected_val, X, feature_names=self.feature_names
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.heatmap(exp, max_display=max_display, show=False, ax=ax)
        plt.title("SHAP: DIR Dominates (Density-Only)")
        if show:
            plt.show()
        return ax


# ==================== TRAIN OR LOAD MODEL ====================
MODEL_FILE = '/content/fusion_model_density_only.pkl'
if os.path.exists(MODEL_FILE):
    print("Loading density-only model...")
    final_pipe = joblib.load(MODEL_FILE)
else:
    print("Training density-only model on CPU...")
    final_pipe = Pipeline([
        ('sc', StandardScaler()),
        ('classifier', ExplainableLGBMClassifier(**BAYSEIAN_PARAMS))
    ])
    final_pipe.fit(train.drop('Is_disrupt', axis=1), train['Is_disrupt'])
    final_pipe['classifier'].feature_names = feature_names
    joblib.dump(final_pipe, MODEL_FILE)

# ==================== PREDICT + EXPLAIN ====================
print("Predicting...")
preds = final_pipe.predict(data_test)
print(f"Generated {len(preds)} predictions")

from sklearn.metrics import f1_score, precision_score, accuracy_score

# Only compute metrics if you have true labels for test
if 'Is_disrupt' in data_test.columns:
    y_true = data_test['Is_disrupt'].values
    f1 = f1_score(y_true, preds)
    precision = precision_score(y_true, preds)
    accuracy = accuracy_score(y_true, preds)

    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
else:
    print("No true labels available in test set to compute metrics.")


# Display 10 SHAP heatmaps from random samples
if len(data_test) > 0:
    n_heatmaps = 10
    sample_size = min(50, len(data_test))  # number of rows per heatmap
    for i in range(n_heatmaps):
        X_sample = data_test.sample(n=sample_size, random_state=42+i)
        print(f"Heatmap {i+1}/{n_heatmaps}")
        final_pipe['classifier'].explain_predictions(X_sample, class_index=1, show=True)

validation_sets = ['CMod_train', 'JDDB_repo_2A_5k', 'J-TEXT processed_data_1k_5k_final']

# Combine them into a single validation dataframe
val_frames = [data_dict[r] for r in validation_sets if r in data_dict and len(data_dict[r]) > 0]
if not val_frames:
    raise ValueError("No labeled validation data available!")

validation = pd.concat(val_frames).reset_index(drop=True)

# Separate features and labels
X_val = validation.drop('Is_disrupt', axis=1)
y_val = validation['Is_disrupt']

# Predict on validation
val_preds = final_pipe.predict(X_val)

# Compute metrics
from sklearn.metrics import f1_score, precision_score, accuracy_score

f1 = f1_score(y_val, val_preds)
precision = precision_score(y_val, val_preds)
accuracy = accuracy_score(y_val, val_preds)

print(f"Validation Metrics:")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Accuracy: {accuracy:.4f}")

import numpy as np
import matplotlib.pyplot as plt

# Assuming `final_pipe` is trained pipeline and `data_test` is test data.
# Assuming `feature_names` is a list of features used in the model.

# Step 0: Assign correct column names to data_test
print("Original columns in data_test:", data_test.columns.tolist())
data_test.columns = feature_names
print("Renamed columns in data_test:", data_test.columns.tolist())

# Now you can run previous debug code safely
common_features = [f for f in feature_names if f in data_test.columns]

if len(common_features) == 0:
    raise ValueError("None of the feature_names are present in data_test columns. Check your data!")

print(f"Using {len(common_features)} features.")

# Continue with the rest...
X_test = data_test[common_features]

# Check for density_DIR presence
if 'density_DIR' not in common_features:
    raise ValueError("'density_DIR' feature missing in test data!")

explainer = final_pipe['classifier'].explainer

shap_values = explainer.shap_values(X_test)

sv = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values

dir_index = common_features.index('density_DIR')
dir_shap_abs = np.abs(sv[:, dir_index])

print(f"Mean absolute SHAP value for density_DIR: {dir_shap_abs.mean():.5f}")
print(f"Median absolute SHAP value for density_DIR: {np.median(dir_shap_abs):.5f}")
print(f"Max absolute SHAP value for density_DIR: {dir_shap_abs.max():.5f}")

plt.hist(dir_shap_abs, bins=50, alpha=0.7, color='blue')
plt.title("Distribution of Absolute SHAP values for density_DIR")
plt.xlabel("Absolute SHAP value")
plt.ylabel("Frequency")
plt.show()

# ==================== FEATURE COLUMN FIX ====================
feat_list = [
    'length','mean','ptp','std','max','min','median','var','iqr',
    'diff_max','diff_min','diff_mean','diff_std','diff_ptp',
    'diff_median','diff_var','diff_iqr','DIR','meta'
]
feature_names = [f"density_{f}" for f in feat_list] + ['JDDB','J-TEXT','CMod']

# Separate features and labels
if 'Is_disrupt' not in validation.columns:
    raise ValueError("Validation set has no 'Is_disrupt' column.")

X_val = validation.drop('Is_disrupt', axis=1)
y_val = validation['Is_disrupt']

# Rename feature columns if counts match
if X_val.shape[1] == len(feature_names):
    X_val.columns = feature_names
else:
    raise ValueError(
        f"Feature count mismatch: validation has {X_val.shape[1]} columns, "
        f"expected {len(feature_names)}. Check preprocessing."
    )

# ==================== CORRELATION ====================
# Compute correlation with the target
corr = pd.concat([X_val, y_val], axis=1).corr()['Is_disrupt'].sort_values(key=abs, ascending=False)

print("Top features by absolute correlation with Is_disrupt:")
print(corr.head(15))

# Display density_DIR specifically
if 'density_DIR' in corr.index:
    print(f"\ndensity_DIR correlation with Is_disrupt: {corr['density_DIR']:.4f}")
else:
    print("\ndensity_DIR not found in validation features!")

# Run this to generate
import matplotlib.pyplot as plt
import numpy as np

features = ['density_DIR', 'density_length', 'J-TEXT', 'JDDB', 'diff_std']
shap_means = [0.160, 0.08, 0.07, 0.06, 0.05]  # approximate from the model
corrs = [0.362, 0.123, 0.106, 0.114, 0.085]

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].barh(features, [abs(c) for c in corrs])
ax[0].set_title('Absolute Correlation with Disruption')
ax[0].set_xlabel('Correlation')

ax[1].barh(features, shap_means, color='orange')
ax[1].set_title('Mean |SHAP| Value (Model Reliance)')
ax[1].set_xlabel('SHAP Impact')

plt.suptitle('DIR (Density Instability Ratio): #1 in Physics AND Model Trust')
plt.tight_layout()
plt.show()
