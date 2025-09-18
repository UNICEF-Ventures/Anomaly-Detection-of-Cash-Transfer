

import os
import streamlit as st
import pandas as pd
from io import StringIO
import re
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.ensemble import IsolationForest

from modules.llm_explainer import explain_record 
import torch

from huggingface_hub import InferenceClient
from streamlit_chat import message as chat_message

from modules.ad_combined import run_ad_combined
#from modules.amount_spike import run_amount_spike
#from modules.freq_surge import run_freq_surge
#from modules.district_surge import run_district_surge
#from modules.id_integrity import run_id_integrity
#from modules.cycle_irregularity import run_cycle_irregularity
#from modules.desc_inconsistency import run_desc_inconsistency


# ----------------------------------
# LOADING THE generate.py + DATA
# ---------------------------------

import sys
from pathlib import Path
import importlib.util

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[1] 
DATA_CSV = REPO_ROOT / "synthetic_cash_transfer_data.csv"
GEN_PY = REPO_ROOT / "generate.py"


def _import_generate_module():
    """Dynamically import generate.py if present."""
    if not GEN_PY.exists():
        return None
    spec = importlib.util.spec_from_file_location("generate_module", GEN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

@st.cache_data
def load_bz_data():
    """
    Load beneficiary history data for the dashboard.
    - It will load if CSV already exists
    - If CSV is missing, it will autogenerate using generate.py
    """
    if not DATA_CSV.exists():
        gen = _import_generate_module()
        if gen is None:
            st.error("generate.py not found in repo root; cannot generate dataset.")
            return pd.DataFrame()
        # Generate synthetic dataset
        bz_df = gen.generate(
            beneficiaries=10_000, seed=42, locale="ar_SA",
            min_cycles=2, max_cycles=5, base_amount_noise=0.20
        )
        bz_df.to_csv(DATA_CSV, index=False, encoding="utf-8-sig")
        
    bz_df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
    return bz_df

# ----------------------------------
# COMPUTE ANOMALIES
# ---------------------------------
@st.cache_data
def compute_all(tx_df: pd.DataFrame, bz_df: pd.DataFrame) -> dict:
# def compute_all(tx_df, bz_df):
    return {
      "AD Combined 1.0":    run_ad_combined(bz_df),
      #"Amount Spike":       run_amount_spike(bz_df),
      #"District Surge":     run_district_surge(tx_df),
      #"ID Integrity":       run_id_integrity(bz_df),
      #"Cycle Irregularity": run_cycle_irregularity(bz_df),
    }


#___________________________________________________________________________________________________



# ----------------------------------
# WEB UI APP
# ----------------------------------

st.markdown("<h1 style='text-align: center;'>UNICEF YSC AI - Anomaly Detection Center</h1>", unsafe_allow_html=True)

hist_tx = load_tx_data()
hist_bz = load_bz_data()


# --------------------------------------------------
# WEB UI APP – UPLOAD FEATURE FOR NEW PAYMENT CYCLE
# --------------------------------------------------
uploaded_bz = st.sidebar.file_uploader(
    "Upload Current Cycle Beneficiary List (CSV)", type="csv"
)
if uploaded_bz:
    uploaded_df = pd.read_csv(uploaded_bz, dtype=str)

    # 1) Accept either `cycle_number` or `payment_cycle`
    if 'cycle_number' in uploaded_df.columns:
        uploaded_df.rename(columns={'cycle_number': 'payment_cycle'}, inplace=True)

    # 2) Ensure we now have `payment_cycle`
    if 'payment_cycle' not in uploaded_df.columns:
        st.error("Uploaded file must include a `cycle_number` or `payment_cycle` column.")
    else:
        curr_cycle = uploaded_df['payment_cycle'].iloc[0]

        # ——— Dynamic renaming ———
        # a) beneficiary names to 'beneficiary_names'
        for col in ('beneficiary_name_v2', 'beneficiary_name'):
            if col in uploaded_df.columns:
                uploaded_df.rename(columns={col: 'beneficiary_names'}, inplace=True)
                break

        # b) payment amounts to 'payment_amount'
        for col in ('payment_amount_yer_final', 'amount_yer'):
            if col in uploaded_df.columns:
                uploaded_df.rename(columns={col: 'payment_amount'}, inplace=True)
                break

        # c) description to 'activity_desc'
        for col in ('activity_description', 'activit_description'):
            if col in uploaded_df.columns:
                uploaded_df.rename(columns={col: 'activity_desc'}, inplace=True)
                break

        # 3) Select only the unified columns + payment_cycle
        keep_cols = [
            'verification_code',
            'beneficiary_names',
            'payment_amount',
            'activity_desc',
            'activty_duration',
            'id_number',
            'phone_number',
            'payment_cycle'
        ]
        missing = set(keep_cols) - set(uploaded_df.columns)
        if missing:
            st.error(f"Uploaded file is missing columns: {missing}")
        else:
            uploaded_df = uploaded_df[keep_cols].copy()

            # 4) Replace history for this cycle, then append
            hist_bz = pd.concat([
                hist_bz[hist_bz['payment_cycle'] != int(curr_cycle)],
                uploaded_df
            ], ignore_index=True)

            st.sidebar.success(f"Using uploaded beneficiary list for cycle {curr_cycle}")


# --------------------------------
# WEB UI APP - COMPUTE ANOMALIES
# --------------------------------
results = compute_all(hist_tx, hist_bz)

# Select anomaly type
choice = st.sidebar.selectbox("Anomaly Type", list(results.keys()))
df = results[choice]

if df.empty:
    st.info("No anomalies detected.")
    st.stop()

# -----------------------------
# WEB UI APP - FILTER BY CYCLE
# -----------------------------
if 'payment_cycle' in df.columns:
    df['payment_cycle'] = pd.to_numeric(df['payment_cycle'], errors='coerce')
    cycles = sorted(df['payment_cycle'].dropna().unique().astype(int))
    latest_cycle = cycles[-1]

    # checkbox to grab all cycles at once
    select_all = st.sidebar.checkbox("Select all cycles", value=False)

    if select_all:
        selected_cycles = cycles
    else:
        # latest cycle is selected initially
        selected_cycles = st.sidebar.multiselect(
            "Filter by Payment Cycle",
            options=cycles,
            default=[latest_cycle]
        )

    df = df[df['payment_cycle'].isin(selected_cycles)]


# -----------------------------------------
# WEB UI APP - DISPLAY AND DOWNLOAD RESULTS
# -----------------------------------------
st.subheader(f"{choice} — {len(df)} records flagged")
st.dataframe(df)

# Print results to csv
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, file_name=f"{choice}.csv")

# Print results to excel xlsx
#csv = df.to_excel(index=False).encode("utf-8")
#st.download_button("Download XLSX", data=csv, file_name=f"{choice}.xlsx")


#Explanation MODEL
#______________________________________________________________________________

anoms = df[df['combined_anomaly']].copy()

st.markdown("## Explainability")

# This will allow user pick a single flagged row to explain
# assuming `anoms` is our dataframe of flagged anomalies
if not anoms.empty:
    idx = st.selectbox("Pick anomaly to explain", anoms.index.tolist())
    record = anoms.loc[idx].to_dict()
    explanation_obj = explain_record(record)

    st.subheader("Flagged for Review: Here’s Why")
    st.write(explanation_obj["explanation"])

    st.subheader("Suggested Next Steps")
    for step in explanation_obj["next_steps"]:
        st.write(f"- {step}")

