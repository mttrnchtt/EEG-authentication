import os
import mne
import numpy as np
import shutil

def has_valid_eeg(vhdr_path):
    try:
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
        eeg_data = raw.get_data(picks='eeg')
        if eeg_data.size == 0 or np.all(eeg_data == 0):
            return False
        return True
    except Exception as e:
        print(f"Error reading {vhdr_path}: {e}")
        return False

def scan_and_filter_eeg_dirs(base_dir):
    eeg_dirs = set()
    all_dirs_with_vhdr = set()
    total_vhdr = 0

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.vhdr'):
                vhdr_path = os.path.join(root, file)
                total_vhdr += 1
                all_dirs_with_vhdr.add(root)
                if has_valid_eeg(vhdr_path):
                    print(f"EEG present: {vhdr_path}")
                    eeg_dirs.add(root)
                else:
                    print(f" No EEG data: {vhdr_path}")

    # Delete folders without valid EEG
    to_delete = all_dirs_with_vhdr - eeg_dirs
    for d in to_delete:
        try:
            print(f"🗑️ Deleting: {d}")
            shutil.rmtree(d)
        except Exception as e:
            print(f"⚠️ Could not delete {d}: {e}")

    print("\nSummary:")
    print(f"Total .vhdr files scanned: {total_vhdr}")
    print(f"Total dirs with valid EEG: {len(eeg_dirs)}")
    print(f"Total dirs deleted: {len(to_delete)}")
scan_and_filter_eeg_dirs('/content/unzipped_data/bbciRaw')

import mne
import numpy as np
import pandas as pd
from glob import glob
import os

l_freq, h_freq       = 1, 40.0      # band-pass between 0.5–70 Hz
common_sfreq         = 250.0          # resample to 500 Hz
tmin, tmax           = -0.2, 1.0      # epoch window (seconds)
baseline             = (None, 0)      

vhdr_paths = glob('/content/unzipped_data/bbciRaw/*/*.vhdr')

for vhdr in vhdr_paths:
    folder = os.path.basename(os.path.dirname(vhdr))  # e.g., 'VPjak_15_07_20'
    user_id = folder.split('_')[0]                   # e.g., 'VPjak'
    session_date = '_'.join(folder.split('_')[1:])   # e.g., '15_07_20'
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
    raw.resample(common_sfreq, npad='auto')
    raw.filter(l_freq, h_freq, fir_design='firwin')
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        event_repeated='merge',
        preload=True,
        detrend=1
    )
    epochs.metadata = pd.DataFrame({
        'user': [user_id] * len(epochs),
        'session': [session_date] * len(epochs)
    })
    n_epochs, n_channels, n_times = epochs.get_data().shape
    print(f"✅ {n_epochs} epochs × {n_channels} ch × {n_times} samples")
    out_fname = vhdr.replace(
        '.vhdr',
        f'_epo_{int((tmax-tmin)*1000)}ms_{int(common_sfreq)}Hz.fif'
    )
    epochs.save(out_fname, overwrite=True)
    print(f"Saved to {out_fname}\n")

print("All subjects processed.")

import mne
import numpy as np
import pandas as pd
from glob import glob

data_glob    = '/content/unzipped_data/bbciRaw/**/*.fif'
desired_nchan = 16


epoch_files = glob(data_glob, recursive=True)
data_list, meta_list = [], []
file_count = 0

for f in epoch_files:
    try:
        epochs = mne.read_epochs(f, preload=True, verbose=False)

        if epochs.info['nchan'] != desired_nchan:
            epochs.close()
            continue

        data_list.append(epochs.get_data())   
        meta_list.append(epochs.metadata)     

        file_count += 1
        epochs.close()

    except Exception as e:
        print(f"Skipping {f}: {e}")

if file_count == 0:
    raise RuntimeError(f"No {desired_nchan}-channel files found!")

# Concatenate across files
X = np.concatenate(data_list, axis=0)      # → (total_epochs, 16, n_times)
metadata = pd.concat(meta_list, ignore_index=True)

y      = metadata['user'].to_numpy()     
groups = metadata['session'].to_numpy()  

print(f"Processed {file_count} files → X shape = {X.shape}, epochs = {len(metadata)}")



import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import Counter

unique_users = np.unique(y)
if len(unique_users) < 2:
    raise ValueError("Need at least two users for authentication")

user_counts = Counter(y)
target_user = user_counts.most_common(1)[0][0]
print(f"Target user: {target_user} with {user_counts[target_user]} samples")

y_binary = (y == target_user).astype(int)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y_binary, groups))

X_reshaped = X.reshape(X.shape[0], -1)
X_train = X_reshaped[train_idx]
X_test = X_reshaped[test_idx]
y_train = y_binary[train_idx]
y_test = y_binary[test_idx]

print("Class distribution:")
print("Train:", Counter(y_train))
print("Test:", Counter(y_test))


import numpy as np
from scipy.signal            import welch
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats             import skew, kurtosis
from sklearn.pipeline        import make_pipeline, FeatureUnion, FunctionTransformer
from sklearn.preprocessing   import StandardScaler
from sklearn.svm             import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    classification_report,
    accuracy_score,
    roc_auc_score
)
from collections import Counter

sfreq = common_sfreq  

def bandpower(epoch, sf=sfreq, bands=[(1,4),(4,8),(8,13),(13,30),(30,45)]):
    f, Pxx = welch(epoch, fs=sf, nperseg=sf)
    return np.hstack([Pxx[:, (f>=l)&(f<h)].mean(axis=1) for l,h in bands])

def ar_feats(epoch, p=6):
    coeffs = []
    for ch in epoch:
        m = AutoReg(ch, lags=p, old_names=False).fit()
        coeffs.append(m.params[1:])
    return np.hstack(coeffs)

def hjorth(epoch):
    feats = []
    for ch in epoch:
        d1, d2 = np.diff(ch), np.diff(np.diff(ch))
        v0, v1, v2 = ch.var(), d1.var(), d2.var()
        mobility = np.sqrt(v1/v0)
        complexity = np.sqrt(v2/v1)/mobility
        feats.append([v0, mobility, complexity])
    return np.hstack(feats)

def stats_feats(epoch):
    feats = []
    for ch in epoch:
        feats.append([ch.mean(), ch.std(), skew(ch), kurtosis(ch)])
    return np.hstack(feats)

pipe = make_pipeline(
    FeatureUnion([
      ('bp', FunctionTransformer(lambda X: np.vstack([bandpower(e)   for e in X]))),
      ('ar', FunctionTransformer(lambda X: np.vstack([ar_feats(e)    for e in X]))),
      ('hj', FunctionTransformer(lambda X: np.vstack([hjorth(e)      for e in X]))),
      ('st', FunctionTransformer(lambda X: np.vstack([stats_feats(e)  for e in X]))),
    ]),
    StandardScaler(),
    SVC(kernel='rbf', class_weight='balanced', probability=True)
)
target_user = Counter(y).most_common(1)[0][0]
y_binary   = (y == target_user).astype(int)
print("Overall distribution:", Counter(y_binary))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)
print("Train distribution:", Counter(y_train))
print("Test  distribution:", Counter(y_test))

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:,1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Imposter','Genuine']))
print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")
print(f"AUC-ROC  : {roc_auc_score(y_test, y_proba):.2f}")

import numpy as np

values, counts = np.unique(y_test, return_counts=True)
for v, c in zip(values, counts):
    print(f"{v}: {c}")

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

pipeline = make_pipeline(
    StandardScaler(),  
    TruncatedSVD(n_components=300, algorithm='randomized'), 
    LogisticRegression(
        penalty='l1',
        solver='saga',     
        C=1,             # Regularization strength
        max_iter=5000,
        n_jobs=-1,        
        verbose=0
    )
)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

pipeline.fit(X_train, y_train)
print(f"Accuracy: {pipeline.score(X_test, y_test):.2f}")

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Imposter', 'Genuine']))

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.2f}")

from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, whiten=True)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

clf = SVC(
    class_weight='balanced',
    probability=True,
    # cache_size=500  # Adjust based on system's memory (MB)
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
clf.fit(X_train_scaled, y_train)

# Evaluate
X_test_scaled = scaler.transform(X_test.astype(np.float32))
y_pred = clf.predict(X_test_scaled)

print("\nAuthentication Results:")
print(classification_report(y_test, y_pred, target_names=['Imposter', 'Genuine']))


import numpy as np
from scipy.signal      import welch
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats      import skew, kurtosis
from sklearn.pipeline  import make_pipeline, FeatureUnion, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm       import SVC

sfreq = 250

def bandpower(epoch, sf=sfreq, bands=[(1,4),(4,8),(8,13),(13,30),(30,45)]):
    f, Pxx = welch(epoch, fs=sfreq, nperseg=sfreq)
    return np.hstack([Pxx[:, (f>=l)&(f<h)].mean(axis=1) for l,h in bands])

def ar_feats(epoch, p=6):
    coeffs = []
    for ch in epoch:
        model = AutoReg(ch, lags=p, old_names=False).fit()
        coeffs.append(model.params[1:])
    return np.hstack(coeffs)

def hjorth(epoch):
    feats = []
    for ch in epoch:
        d1 = np.diff(ch); d2 = np.diff(d1)
        v0, v1, v2 = ch.var(), d1.var(), d2.var()
        m = np.sqrt(v1/v0); c = np.sqrt(v2/v1)/m
        feats.append([v0,m,c])
    return np.hstack(feats)

def stats_feats(epoch):
    feats = []
    for ch in epoch:
        feats.append([ch.mean(), ch.std(), skew(ch), kurtosis(ch)])
    return np.hstack(feats)

fe_union = FeatureUnion([
    ('bp', FunctionTransformer(lambda X: np.vstack([bandpower(e)   for e in X]))),
    ('ar', FunctionTransformer(lambda X: np.vstack([ar_feats(e)    for e in X]))),
    ('hj', FunctionTransformer(lambda X: np.vstack([hjorth(e)      for e in X]))),
    ('st', FunctionTransformer(lambda X: np.vstack([stats_feats(e)  for e in X])))
])
pipe = make_pipeline(
    fe_union,
    StandardScaler(),
    SVC(kernel='rbf', class_weight='balanced')
)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, classification_report
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch

def extract_features(X):

    n_epochs, n_channels, n_times = X.shape
    features = []

    for epoch in range(n_epochs):
        epoch_features = []

        for channel in range(n_channels):
            signal = X[epoch, channel]

            # Time domain features
            # 1. Mean
            mean = np.mean(signal)

            # 2. Standard deviation
            std = np.std(signal)

            # 3. Average Amplitude Change (AAC)
            aac = np.mean(np.abs(np.diff(signal)))

            # 4. Zero Crossing Rate (ZCR)
            zcr = np.sum(np.diff(np.signbit(signal).astype(int)) != 0) / (len(signal) - 1)

            # 5. Root Mean Square (RMS)
            rms = np.sqrt(np.mean(signal**2))

            # 6. Mobility (Hjorth parameter)
            diff1 = np.diff(signal)
            mobility = np.std(diff1) / np.std(signal)

            # 7. Complexity (Hjorth parameter)
            diff2 = np.diff(diff1)
            complexity = (np.std(diff2) * np.std(signal)) / (np.std(diff1) ** 2)

            # 8. Skewness
            sk = skew(signal)

            # 9. Kurtosis
            kurt = kurtosis(signal)

            # Frequency domain features
            # 10. Power Spectral Density (PSD) bands
            freqs, psd = welch(signal, fs=250, nperseg=min(256, len(signal)))

            # Delta (0.5-4 Hz)
            delta_idx = np.logical_and(freqs >= 0.5, freqs <= 4)
            delta_power = np.mean(psd[delta_idx])

            # Theta (4-8 Hz)
            theta_idx = np.logical_and(freqs >= 4, freqs <= 8)
            theta_power = np.mean(psd[theta_idx])

            # Alpha (8-13 Hz)
            alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
            alpha_power = np.mean(psd[alpha_idx])

            # Beta (13-30 Hz)
            beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
            beta_power = np.mean(psd[beta_idx])

            # Gamma (30-100 Hz)
            gamma_idx = np.logical_and(freqs >= 30, freqs <= 100)
            gamma_power = np.mean(psd[gamma_idx])
            channel_features = [
                mean, std, aac, zcr, rms, mobility, complexity, sk, kurt,
                delta_power, theta_power, alpha_power, beta_power, gamma_power
            ]

            epoch_features.extend(channel_features)

        features.append(epoch_features)

    return np.array(features)

# Extract features from the raw EEG data
X_features = extract_features(X)


user_scores = {}

# For each unique user
for target_user in np.unique(y):
    print(f"\n=== Testing for user: {target_user} ===")

    # Create binary labels (1 for target user, 0 for others)
    y_binary = (y == target_user).astype(int)

    # Print class distribution
    print("Class distribution:")
    unique, counts = np.unique(y_binary, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} samples ({count/len(y_binary)*100:.1f}%)")

    # Initialize models
    scaler = StandardScaler()

    # Perform cross-validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    f1_macro_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_features, y_binary)):
        X_train = X_features[train_idx]
        X_test = X_features[test_idx]
        y_train = y_binary[train_idx]
        y_test = y_binary[test_idx]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train GMM on genuine user data only
        genuine_data = X_train_scaled[y_train == 1]
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(genuine_data)

        # Predict using likelihood threshold
        log_probs = gmm.score_samples(X_test_scaled)
        threshold = np.percentile(gmm.score_samples(genuine_data), 5)  # 5th percentile threshold
        y_pred = (log_probs >= threshold).astype(int)

        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        f1_macro = f1_score(y_test, y_pred, average='macro')

        accuracy_scores.append(accuracy)
        f1_macro_scores.append(f1_macro)

        print(f"\nFold {fold_idx + 1}:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1-macro: {f1_macro:.3f}")
        print("\nDetailed metrics:")
        print(classification_report(y_test, y_pred))

        del X_train, X_test, X_train_scaled, X_test_scaled

    accuracy_scores = np.array(accuracy_scores)
    f1_macro_scores = np.array(f1_macro_scores)

    user_scores[target_user] = {
        'accuracy': accuracy_scores,
        'f1_macro': f1_macro_scores
    }

    print(f"\nUser {target_user} results:")
    print(f"Mean accuracy: {accuracy_scores.mean():.3f} (+/- {accuracy_scores.std() * 2:.3f})")
    print(f"Mean F1-macro: {f1_macro_scores.mean():.3f} (+/- {f1_macro_scores.std() * 2:.3f})")

print("\n=== Overall Results ===")
all_accuracies = np.concatenate([scores['accuracy'] for scores in user_scores.values()])
all_f1_macros = np.concatenate([scores['f1_macro'] for scores in user_scores.values()])

print(f"Mean accuracy across all users: {all_accuracies.mean():.3f} (+/- {all_accuracies.std() * 2:.3f})")
print(f"Mean F1-macro across all users: {all_f1_macros.mean():.3f} (+/- {all_f1_macros.std() * 2:.3f})")
print("\nMean scores for each user:")
for user, scores in user_scores.items():
    print(f"\nUser {user}:")
    print(f"Accuracy: {scores['accuracy'].mean():.3f}")
    print(f"F1-macro: {scores['f1_macro'].mean():.3f}")