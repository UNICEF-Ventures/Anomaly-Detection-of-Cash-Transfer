
# ____________________________________APP_____________________________________________

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.preprocessing import normalize
from sklearn.ensemble import IsolationForest



def run_ad_combined(bz_df):
    # 1. Base DataFrame and type casting
    df = bz_df.copy().reset_index(drop=False).rename(columns={'index': 'id'})
    
    df['payment_amount'] = pd.to_numeric(df['payment_amount'], errors='coerce')
    df['payment_cycle'] = pd.to_numeric(df['payment_cycle'], errors='coerce')
    
    # 2. Historical amount metrics per beneficiary
    df.sort_values(by=['verification_code', 'payment_cycle'], inplace=True)
    df['avg_past'] = df.groupby('verification_code')['payment_amount'] \
                     .transform(lambda x: x.expanding().mean().shift(1))
    df['std_past'] = df.groupby('verification_code')['payment_amount'] \
                     .transform(lambda x: x.expanding().std().shift(1))
    
    # 3. Frequency features
    df['count_pp'] = df.groupby(['verification_code', 'payment_cycle'])['id'] \
                     .transform('count')
    df['cycles_since_last'] = df.groupby('verification_code')['payment_cycle'] \
                             .diff()
    
    # 4. Normalize Arabic descriptions
    #def normalize_arabic(text):
    def normalize_arabic(text: str) -> str:
        """
        Normalize Arabic text by removing diacritics and standardizing letters.
        """
        if not isinstance(text, str):
            return ""
        
        # Remove Arabic diacritics (Tashkeel)
        diacritics_pattern = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
        text = re.sub(diacritics_pattern, '', text)
        
        # Standardize forms of Alif and other letters
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ى', 'ي')
        
        return text
    
    df['normalized_desc'] = df['activity_desc'].apply(normalize_arabic)
    
    
    # Calculate z-score
    df['z_score_amount'] = np.where(
        (df['std_past'].isna()) | (df['std_past'] == 0),
        0,
        (df['payment_amount'] - df['avg_past']) / df['std_past']
    )
    
    
    # 5. Rule-based anomaly flags
    df['flag_amount_gt3x'] = df['avg_past'].notna() & (df['payment_amount'] > 3 * df['avg_past'])
    df['flag_z_score_amount'] = df['z_score_amount'].abs() > 3
    df['flag_freq'] = df['count_pp'] > 3
    
    # 6. Semantic embedding and clustering
    
    # We'll compute embeddings for unique normalized descriptions.
    unique_desc = df['normalized_desc'].unique()
    desc_df = pd.DataFrame({'normalized_desc': unique_desc})
    
    # Using a multilingual sentence transformer model (which supports Arabic).
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    desc_df['embedding'] = desc_df['normalized_desc'].apply(lambda s: model.encode(s))
    
    # Embeddings are stacked into a matrix.
    embeddings = np.vstack(desc_df['embedding'].values)
    
    # Scale embeddings.
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    
    # Embeddings are clustered using hdbscan with cosine similarity.
    embeddings_norm = normalize(embeddings_scaled, norm='l2')
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    desc_df['cluster'] = clusterer.fit_predict(embeddings_norm)
    
    
    # 7. Merge cluster labels and compute description z-score
    df = df.merge(desc_df[['normalized_desc','cluster']], on='normalized_desc', how='left')
    
    # For each cluster, historical metrics are calculated for the amount.
    cluster_stats = df.groupby('cluster')['payment_amount'] \
                      .agg(desc_avg='mean', desc_std='std').reset_index()
    
    # Merging aggregated metrics back into df on cluster.
    df = df.merge(cluster_stats, on='cluster', suffixes=('', '_cluster'))
    
    # We'll compute a z-score for each record relative to its cluster's historical metrics.
    df['z_score_desc'] = (df['payment_amount'] - df['desc_avg']) / df['desc_std']
    
    # Flag as anomaly if absolute z-score > threshold
    df['flag_z_desc'] = df['z_score_desc'].abs() > 3
    
    
    # 8. Combine rule-based flags
    df['rule_flag'] = (
        df['flag_amount_gt3x'] |
        df['flag_z_score_amount']   |
        df['flag_freq']       |
        df['flag_z_desc']
    )
    
    # 9. Isolation Forest on numeric features only
    num_feats = ['payment_amount','z_score_amount','count_pp']
    df_num = df.dropna(subset=num_feats).copy()
    
    iso_num = IsolationForest(contamination=0.05, random_state=42)
    df_num['iforest_num'] = iso_num.fit_predict(df_num[num_feats]) == -1
    
    # Merge back into main df
    df = df.merge(df_num[['id','iforest_num']], on='id', how='left')
    df['iforest_num'] = df['iforest_num'].fillna(False)
    
    # 10. Isolation Forest on embeddings only (straightforward merge)
    iso_emb = IsolationForest(contamination=0.05, random_state=42)
    desc_df['iforest_emb'] = iso_emb.fit_predict(embeddings_scaled) == -1
    
    # Merge back into main df
    df = df.merge(desc_df[['normalized_desc','iforest_emb']], on='normalized_desc', how='left')
    df['iforest_emb'] = df['iforest_emb'].fillna(False)
    
    # 11. Final combined anomaly flag
    #df['combined_anomaly'] = df['rule_flag'] & df['iforest_num'] | df['iforest_emb']
    df['combined_anomaly'] = ((df['rule_flag'] & df['iforest_num']) | df['iforest_emb'])
    
    # 12. Build human-readable explanations
    flag_meanings = {
      'flag_amount_gt3x': "This beneficiary will be receiving an amount that is more than three times higher than their usual (historical) payment",
      'flag_freq':         "Will receive more than three payments in this single cycle",
      'flag_z_desc':       "The payment amount doesn’t match what’s normally paid for this service",
      'iforest_emb':       "This activity description uses phrasing that’s very different from the usual patterns we see"
    }
    
    
    
    def make_explanation(row):
        reasons = []
        for flag, text in flag_meanings.items():
            if bool(row.get(flag, False)):
                reasons.append(text)
        return "; ".join(reasons) or "—"
    
    # Apply it to every row
    df['Explanation'] = df.apply(make_explanation, axis=1)
    
    # 13. Filter only the anomalies and return
    anomalies = df[df['combined_anomaly']].copy()
    return anomalies
    
    
