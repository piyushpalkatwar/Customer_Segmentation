# ================================================================
# PROJECT 2 : AI Customer Segmentation  (RFM + K-Means Clustering)
# Domain    : Retail / D2C E-Commerce
# AI Angle  : Combines SQL-style RFM scoring with K-Means clustering
#             to auto-discover customer segments + profile each one.
# CSVs Used : csv/customers.csv | csv/transactions.csv | csv/behavior.csv
#
# RUN: python customer_segmentation.py
# ================================================================

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.cluster          import KMeans
from sklearn.preprocessing    import StandardScaler
from sklearn.decomposition    import PCA
from sklearn.metrics          import silhouette_score
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(BASE, '..', 'csv')
OUT  = BASE

print("=" * 58)
print("  PROJECT 2 — AI Customer Segmentation (RFM + K-Means)")
print("=" * 58)

# ── 1. LOAD DATA ───────────────────────────────────────────
customers = pd.read_csv(os.path.join(CSV, 'customers.csv'), parse_dates=['signup_date'])
txns      = pd.read_csv(os.path.join(CSV, 'transactions.csv'), parse_dates=['txn_date'])
behavior  = pd.read_csv(os.path.join(CSV, 'behavior.csv'))

print(f"\n✅  Loaded  →  customers: {len(customers):,}  |  "
      f"transactions: {len(txns):,}  |  behavior: {len(behavior):,}")

# ── 2. RFM FEATURE ENGINEERING ─────────────────────────────
snapshot_date = pd.Timestamp('2025-01-01')

rfm = (txns.groupby('customer_id')
           .agg(
               last_purchase =('txn_date','max'),
               frequency     =('txn_id','count'),
               monetary      =('amount','sum')
           ).reset_index())

rfm['recency'] = (snapshot_date - rfm['last_purchase']).dt.days
rfm.drop(columns='last_purchase', inplace=True)

# Merge behavioral features
rfm = rfm.merge(behavior, on='customer_id', how='left').fillna(0)
rfm = rfm.merge(customers[['customer_id','age','channel']], on='customer_id', how='left')

# Encode channel
rfm['channel_num'] = LabelEncoder_simple = {'App':0,'Web':1,'In-Store':2}
rfm['channel_enc'] = rfm['channel'].map(LabelEncoder_simple).fillna(0).astype(int)

print(f"\n    RFM stats:")
print(f"    Recency  : mean={rfm['recency'].mean():.0f} days,  median={rfm['recency'].median():.0f}")
print(f"    Frequency: mean={rfm['frequency'].mean():.1f},  median={rfm['frequency'].median():.0f}")
print(f"    Monetary : mean=₹{rfm['monetary'].mean():,.0f},  median=₹{rfm['monetary'].median():,.0f}")

# ── 3. RFM SCORING (Quintile-based, like SQL NTILE) ────────
for col, asc in [('recency',True), ('frequency',False), ('monetary',False)]:
    label = col[0].upper()
    # Use rank-based scoring to avoid duplicate bin edge issues
    if asc:
        rfm[f'{label}_score'] = pd.qcut(rfm[col].rank(method='first'), q=5,
                                          labels=[5,4,3,2,1]).astype(int)
    else:
        rfm[f'{label}_score'] = pd.qcut(rfm[col].rank(method='first'), q=5,
                                          labels=[1,2,3,4,5]).astype(int)

rfm['RFM_total'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
rfm['rfm_segment'] = pd.cut(rfm['RFM_total'], bins=[2,5,7,10,12,16],
                             labels=['Lost','At Risk','Potential','Loyal','Champion'],
                             include_lowest=True)

print("\n--- RFM Segment Distribution ---")
seg_counts = rfm['rfm_segment'].value_counts()
for seg, cnt in seg_counts.items():
    pct = cnt / len(rfm) * 100
    bar = '█' * int(pct / 2)
    print(f"  {seg:<12} {bar:<25} {cnt:>4} ({pct:.1f}%)")

# ── 4. K-MEANS CLUSTERING (AI Layer) ───────────────────────
print("\n--- K-Means Clustering ---")

cluster_features = ['recency','frequency','monetary',
                    'app_sessions_30d','pages_viewed_30d',
                    'cart_abandons_30d','wishlist_items',
                    'email_opens_30d','referrals_made']

X_cluster = rfm[cluster_features].fillna(0)
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_cluster)

# Find optimal K using silhouette score
sil_scores = {}
for k in range(2, 8):
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, lbl)

best_k = max(sil_scores, key=sil_scores.get)
print(f"   Silhouette scores: { {k:f'{v:.3f}' for k,v in sil_scores.items()} }")
print(f"   Optimal K = {best_k}  (silhouette = {sil_scores[best_k]:.3f})")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

# Profile each cluster
cluster_profile = rfm.groupby('kmeans_cluster')[cluster_features + ['RFM_total']].mean().round(1)
print("\n--- Cluster Profiles ---")
print(cluster_profile.to_string())

# Auto-label clusters based on RFM total + monetary
cluster_labels = {}
for cid, row in cluster_profile.iterrows():
    if row['monetary'] > cluster_profile['monetary'].quantile(0.75):
        cluster_labels[cid] = 'High-Value'
    elif row['recency'] < cluster_profile['recency'].quantile(0.33):
        cluster_labels[cid] = 'Active-Occasional'
    elif row['recency'] > cluster_profile['recency'].quantile(0.67):
        cluster_labels[cid] = 'Dormant'
    else:
        cluster_labels[cid] = 'Regular'

rfm['cluster_label'] = rfm['kmeans_cluster'].map(cluster_labels)
print("\n--- Auto-Labelled Clusters ---")
for cid, label in cluster_labels.items():
    n = (rfm['kmeans_cluster'] == cid).sum()
    avg_rev = rfm[rfm['kmeans_cluster']==cid]['monetary'].mean()
    print(f"  Cluster {cid} → {label:<20} | {n:>4} customers | Avg spend ₹{avg_rev:,.0f}")

# ── 5. PCA for 2-D VISUALISATION ──────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
rfm['pca1'] = X_pca[:, 0]
rfm['pca2'] = X_pca[:, 1]

# ── 6. CATEGORY PREFERENCE ─────────────────────────────────
cat_seg = (txns.merge(rfm[['customer_id','rfm_segment']], on='customer_id')
               .groupby(['rfm_segment','category'])['amount'].sum()
               .unstack(fill_value=0))

# ── 7. CHANNEL ANALYSIS ────────────────────────────────────
channel_ltv = (txns.merge(customers[['customer_id','channel']], on='customer_id')
                   .groupby('channel')['amount'].agg(['sum','mean','count']))
channel_ltv.columns = ['total_revenue','avg_order','transactions']
channel_ltv['ltv_per_customer'] = channel_ltv['total_revenue'] / customers['channel'].value_counts()
print("\n--- Channel LTV ---")
print(channel_ltv.round(0).to_string())

# ── 8. VISUALISATIONS ──────────────────────────────────────
fig = plt.figure(figsize=(20, 13))
fig.suptitle("PROJECT 2 — AI Customer Segmentation Dashboard  (D2C Retail)", fontsize=15, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

palette = {'Champion':'#27AE60','Loyal':'#2980B9','Potential':'#F39C12',
           'At Risk':'#E67E22','Lost':'#E74C3C'}

# Plot 1: RFM segment revenue treemap-style bar
ax1 = fig.add_subplot(gs[0, 0])
seg_rev = rfm.groupby('rfm_segment')['monetary'].sum().sort_values(ascending=True)
colors1 = [palette.get(s,'grey') for s in seg_rev.index]
bars1 = ax1.barh(seg_rev.index.astype(str), seg_rev.values/1e6, color=colors1)
ax1.bar_label(bars1, fmt='₹%.1fM', padding=3, fontsize=9)
ax1.set_title('Total Revenue by RFM Segment', fontweight='bold')
ax1.set_xlabel('Total Revenue (₹M)')

# Plot 2: K-Means PCA scatter
ax2 = fig.add_subplot(gs[0, 1])
cluster_colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for cid in range(best_k):
    mask = rfm['kmeans_cluster'] == cid
    ax2.scatter(rfm.loc[mask,'pca1'], rfm.loc[mask,'pca2'],
                c=[cluster_colors[cid]], s=20, alpha=0.6,
                label=f"C{cid}: {cluster_labels[cid]}")
ax2.set_title(f'K-Means Clusters (K={best_k}) — PCA View', fontweight='bold')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)')
ax2.legend(fontsize=8)

# Plot 3: Silhouette score curve (elbow / optimal K)
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(list(sil_scores.keys()), list(sil_scores.values()),
         'o-', color='#8E44AD', linewidth=2.5, markersize=8)
ax3.axvline(best_k, color='#E74C3C', linestyle='--', label=f'Best K={best_k}')
ax3.set_title('Silhouette Score vs K\n(Higher = Better Separation)', fontweight='bold')
ax3.set_xlabel('Number of Clusters (K)'); ax3.set_ylabel('Silhouette Score')
ax3.legend(); ax3.grid(True, alpha=0.3)

# Plot 4: RFM scatter (Recency vs Monetary, coloured by segment)
ax4 = fig.add_subplot(gs[1, 0])
for seg, color in palette.items():
    mask = rfm['rfm_segment'] == seg
    ax4.scatter(rfm.loc[mask,'recency'], rfm.loc[mask,'monetary']/1e3,
                c=color, s=18, alpha=0.55, label=seg)
ax4.set_title('Recency vs Monetary\n(Coloured by RFM Segment)', fontweight='bold')
ax4.set_xlabel('Recency (days since purchase)'); ax4.set_ylabel('Monetary Spend (₹K)')
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.2)

# Plot 5: Category preference heatmap
ax5 = fig.add_subplot(gs[1, 1])
cat_norm = cat_seg.div(cat_seg.sum(axis=1), axis=0) * 100
sns.heatmap(cat_norm, annot=True, fmt='.0f', cmap='Blues', ax=ax5,
            linewidths=0.5, cbar_kws={'label':'% of Segment Spend'})
ax5.set_title('Category Preference by Segment\n(% of spend)', fontweight='bold')
ax5.tick_params(axis='x', rotation=30, labelsize=8)
ax5.tick_params(axis='y', rotation=0, labelsize=8)

# Plot 6: Behavioral heatmap by cluster
ax6 = fig.add_subplot(gs[1, 2])
beh_cols = ['app_sessions_30d','cart_abandons_30d','wishlist_items','referrals_made']
beh_profile = rfm.groupby('cluster_label')[beh_cols].mean()
beh_norm    = (beh_profile - beh_profile.min()) / (beh_profile.max() - beh_profile.min())
sns.heatmap(beh_norm, annot=beh_profile.round(1), fmt='.1f', cmap='RdYlGn',
            ax=ax6, linewidths=0.5)
ax6.set_title('Behaviour Signals by Cluster\n(Normalised)', fontweight='bold')
ax6.tick_params(axis='x', rotation=30, labelsize=8)
ax6.tick_params(axis='y', rotation=0, labelsize=8)

plt.savefig(os.path.join(OUT, 'p2_segmentation_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n✅  Dashboard saved → p2_segmentation_dashboard.png")

# ── 9. EXPORT OUTPUTS ──────────────────────────────────────
rfm[['customer_id','recency','frequency','monetary',
     'R_score','F_score','M_score','RFM_total',
     'rfm_segment','kmeans_cluster','cluster_label']].to_csv(
    os.path.join(CSV, 'customer_segments_output.csv'), index=False)

cluster_profile.reset_index().to_csv(
    os.path.join(CSV, 'cluster_profiles_output.csv'))

print("✅  2 output CSVs saved to /csv/")

print("\n--- INTERVIEW TALKING POINTS ---")
champ_rev = rfm[rfm['rfm_segment']=='Champion']['monetary'].sum()
tot_rev   = rfm['monetary'].sum()
print(f"• RFM scoring: Champions = {(rfm['rfm_segment']=='Champion').mean()*100:.0f}% of customers but {champ_rev/tot_rev*100:.0f}% of revenue")
print(f"• Optimal K = {best_k} clusters (silhouette = {sil_scores[best_k]:.3f})")
print(f"• App channel LTV > Web by {(channel_ltv.loc['App','avg_order']/channel_ltv.loc['Web','avg_order']-1)*100:.0f}%")
print(f"• Merged behavioral data with RFM — 9 features total for clustering")
print(f"• Architecture: CSV → RFM quintiles → StandardScaler → KMeans → PCA viz")
