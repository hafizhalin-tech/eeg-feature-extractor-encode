import os
import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch, spectrogram
from scipy.stats import skew, kurtosis
from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif

# --- Helper functions (same semantics as your original program, extended) ---

def bandpower(freqs, psd, band):
    band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    if np.any(band_idx):
        return np.trapz(psd[band_idx], freqs[band_idx])
    return 0.0


def hjorth_parameters(signal):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)
    var_zero = np.var(signal)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)
    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero) if var_zero != 0 else 0
    complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 != 0 and mobility != 0 else 0
    return activity, mobility, complexity


def petrosian_fd(signal):
    diff = np.diff(signal)
    N_delta = np.sum(diff[1:] * diff[:-1] < 0)
    n = len(signal)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta) + 1e-12))


def approximate_entropy(U, m=2, r=0.2):
    U = np.array(U)
    N = len(U)
    def _phi(m):
        x = np.array([U[i:i+m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r * np.std(U), axis=0) / (N - m + 1)
        return np.sum(np.log(C + 1e-12)) / (N - m + 1)
    return abs(_phi(m) - _phi(m+1))


def spectral_centroid(freqs, psd):
    norm = np.sum(psd) + 1e-12
    return np.sum(freqs * psd) / norm


def spectral_edge_frequency(freqs, psd, edge=0.9):
    cumsum = np.cumsum(psd)
    total = cumsum[-1]
    idx = np.searchsorted(cumsum, edge * total)
    return freqs[min(idx, len(freqs)-1)]


# --- Feature extraction for a single segment ---

def extract_features(segment, sfreq, channels):
    features = {}
    for i, ch_data in enumerate(segment.T):
        ch_name = channels[i] if i < len(channels) else f"Ch{i}"

        # PSD via Welch
        freqs, psd = welch(ch_data, sfreq, nperseg=sfreq*2)
        psd_norm = psd / (np.sum(psd) + 1e-12)

        # Basic spectral
        features[f"mean_psd_{ch_name}"] = np.mean(psd)
        features[f"var_psd_{ch_name}"] = np.var(psd)
        features[f"spectral_entropy_{ch_name}"] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        features[f"psd_centroid_{ch_name}"] = spectral_centroid(freqs, psd)
        features[f"psd_sedge_{ch_name}"] = spectral_edge_frequency(freqs, psd, edge=0.9)

        # Band powers and relative powers
        bands = {"delta":(0.5,4), "theta":(4,8), "alpha":(8,12), "beta":(13,30), "gamma":(30,45)}
        total_power = 0
        for bname, brange in bands.items():
            p = bandpower(freqs, psd, brange)
            features[f"{bname}_{ch_name}"] = p
            total_power += p
        for bname in bands.keys():
            features[f"rel_{bname}_{ch_name}"] = features[f"{bname}_{ch_name}"] / (total_power + 1e-12)

        # Ratios
        features[f"alpha_beta_ratio_{ch_name}"] = features[f"alpha_{ch_name}"] / (features[f"beta_{ch_name}"] + 1e-12)
        features[f"delta_theta_ratio_{ch_name}"] = features[f"delta_{ch_name}"] / (features[f"theta_{ch_name}"] + 1e-12)

        # Time-domain
        features[f"mean_time_{ch_name}"] = np.mean(ch_data)
        features[f"std_time_{ch_name}"] = np.std(ch_data)
        features[f"rms_{ch_name}"] = np.sqrt(np.mean(ch_data**2))
        features[f"skew_{ch_name}"] = skew(ch_data)
        features[f"kurtosis_{ch_name}"] = kurtosis(ch_data)
        features[f"zero_crossings_{ch_name}"] = ((ch_data[:-1] * ch_data[1:]) < 0).sum()

        # Hjorth
        act, mob, comp = hjorth_parameters(ch_data)
        features[f"hjorth_activity_{ch_name}"] = act
        features[f"hjorth_mobility_{ch_name}"] = mob
        features[f"hjorth_complexity_{ch_name}"] = comp

        # Nonlinear
        features[f"approx_entropy_{ch_name}"] = approximate_entropy(ch_data)
        features[f"petrosian_fd_{ch_name}"] = petrosian_fd(ch_data)

        # Time-frequency via spectrogram summary
        f_spec, t_spec, Sxx = spectrogram(ch_data, sfreq, nperseg=int(sfreq*1))
        Sxx_mean = np.mean(Sxx)
        Sxx_var = np.var(Sxx)
        features[f"spec_mean_{ch_name}"] = Sxx_mean
        features[f"spec_var_{ch_name}"] = Sxx_var
        features[f"spec_maxfreq_{ch_name}"] = f_spec[np.argmax(np.mean(Sxx, axis=1))]

    return features


# --- Streamlit UI ---

st.set_page_config(page_title="EEG Feature Extractor", layout="wide")
st.title("EEG Feature Extractor — Streamlit")

with st.sidebar:
    st.header("Settings")
    sampling_rate = st.number_input("Sampling rate (Hz)", value=128, min_value=1)
    data_duration_sec = st.number_input("Data duration to use (sec)", value=50, min_value=1)
    num_segments = st.number_input("Number of segments per file", value=6, min_value=1)
    overlap_percent = st.slider("Overlap percent", 0, 90, 50)
    channels_text = st.text_area("Channel names (comma separated)", value=",".join(['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']))
    channels = [c.strip() for c in channels_text.split(",") if c.strip()]

st.markdown("Drop multiple CSV/XLSX EEG files below. Filenames should contain label after the first underscore: e.g. participant1_label.xlsx")
uploaded_files = st.file_uploader("Upload EEG files (CSV or XLSX)", type=["csv","xlsx"], accept_multiple_files=True)

if uploaded_files:
    all_features = []
    labels = []

    progress_bar = st.progress(0)
    total_files = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        # read into dataframe
        fname = uploaded_file.name
        if fname.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        total_samples = int(sampling_rate * data_duration_sec)
        df = df.iloc[:total_samples, :len(channels)]
        data = df.to_numpy()

        # label parsing
        if "_" in fname:
            label = fname.split("_",1)[1].rsplit(".",1)[0]
        else:
            label = st.text_input(f"Label for {fname}", value="unknown")

        segment_length = int(len(data) / num_segments)
        step_size = int(segment_length * (1 - overlap_percent / 100))
        if step_size < 1:
            step_size = 1

        for start in range(0, len(data) - segment_length + 1, step_size):
            end = start + segment_length
            segment = data[start:end, :]
            feats = extract_features(segment, sampling_rate, channels)
            all_features.append(feats)
            labels.append(label)

        progress_bar.progress((idx + 1) / total_files)

    features_df = pd.DataFrame(all_features)
    features_df["Label"] = labels
    
    # --- One-Hot Encode Labels ---
    label_dummies = pd.get_dummies(features_df["Label"], prefix="Label")
    features_df = pd.concat([features_df.drop(columns=["Label"]), label_dummies], axis=1)
    
    st.subheader("Extracted features preview (with one-hot encoded labels)")
    st.dataframe(features_df.head())
    
    # For MI, treat each label column separately (sum or max)
    label_cols = [col for col in features_df.columns if col.startswith("Label_")]
    X = features_df.drop(columns=label_cols)
    y = features_df[label_cols].idxmax(axis=1) if len(label_cols) > 1 else features_df[label_cols[0]]

  
    with st.spinner("Calculating mutual information scores..."):
        mi_scores = mutual_info_classif(X.fillna(0), y, discrete_features=False)
    mi_df = pd.DataFrame({"feature": X.columns, "MI": mi_scores}).sort_values("MI", ascending=False)

    st.subheader("Mutual Information (MI) scores")
    st.table(mi_df.head(30))

    mi_min = float(mi_df["MI"].min())
    mi_max = float(mi_df["MI"].max())

    # Initialize session state
    if "mi_threshold" not in st.session_state:
        st.session_state.mi_threshold = 0.0
    
    # Use the stored threshold as default value
    threshold = st.sidebar.slider(
        "Minimum MI to keep feature", 
        min_value=0.0, 
        max_value=float(mi_max),
        value=float(st.session_state.mi_threshold),
        step=max(mi_max / 100.0, 1e-6),
        key="mi_threshold"
    )


    selected_features = mi_df[mi_df["MI"] >= threshold]["feature"].tolist()
    if not selected_features:
        st.warning("No features pass the MI threshold. Lower the threshold.")
    else:
        st.success(f"{len(selected_features)} features selected (MI >= {threshold:.6f})")
        final_df = features_df[selected_features + ["Label"]]
        st.subheader("Final feature table (selected)")
        st.dataframe(final_df.head())

        # Normalize
        if st.checkbox("Normalize selected features (MinMax) ", value=True):
            scaler = MinMaxScaler()
            final_df[selected_features] = scaler.fit_transform(final_df[selected_features].fillna(0))

        # Save options
        save_to_csv = st.button("Save selected features to CSV")
        if save_to_csv:
            csv_buffer = io.StringIO()
            final_df.to_csv(csv_buffer, index=False)
            st.download_button(label="Download features CSV", data=csv_buffer.getvalue(), file_name="eeg_selected_features.csv", mime="text/csv")

        # Also allow previewing MI-ranked features for manual deselection
        st.subheader("MI-ranked features (top 50)")
        st.dataframe(mi_df.head(50))

else:
    st.info("Upload EEG CSV/XLSX files to start feature extraction.")

st.markdown("---")
st.markdown("**Notes:** This app runs entirely in the browser/server where Streamlit is hosted. It does not mount Google Drive. To process files in Drive, download them and upload here (or connect your GitHub/Cloud storage to Streamlit deployment).")

