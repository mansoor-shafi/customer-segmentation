import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Main background */
.stApp {
    background-color: #0f0f13;
    color: #e8e4dc;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #16161d;
    border-right: 1px solid #2a2a35;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a24, #1f1f2e);
    border: 1px solid #2a2a40;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-card .label {
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 32px;
    font-family: 'DM Serif Display', serif;
    color: #f0c060;
}

/* Cluster badges */
.cluster-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin: 4px;
}

/* Section divider */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #f0c060;
    border-bottom: 1px solid #2a2a40;
    padding-bottom: 8px;
    margin-top: 32px;
    margin-bottom: 16px;
}

/* Streamlit elements */
.stButton > button {
    background: linear-gradient(135deg, #f0c060, #e09030);
    color: #0f0f13;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.stFileUploader {
    border: 2px dashed #2a2a40;
    border-radius: 12px;
    padding: 10px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #16161d;
    border-radius: 10px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #888;
    font-family: 'DM Sans', sans-serif;
}
.stTabs [aria-selected="true"] {
    background-color: #f0c060 !important;
    color: #0f0f13 !important;
    border-radius: 8px;
    font-weight: 600;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #2a2a40;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Cluster metadata ──────────────────────────────────────────────────────────
CLUSTER_COLORS = {0: '#4e9af1', 1: '#f0a050', 2: '#50c878', 3: '#e05060'}
CLUSTER_NAMES  = {
    0: "At-Risk Customers",
    1: "Occasional Buyers",
    2: "High-Value Loyals",
    3: "New / Inactive",
}
CLUSTER_DESC = {
    0: "Previously active customers who haven't purchased recently. Worth re-engagement campaigns.",
    1: "Moderate spenders who buy occasionally. Respond well to promotions and reminders.",
    2: "Most valuable segment — frequent, recent, and high-spending. Prioritize retention.",
    3: "Low recency, low frequency, low spend. May be one-time or lapsed customers.",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛍️ Customer Segmentation")
    st.markdown("*RFM Analysis + K-Means Clustering*")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=4)
    remove_outliers = st.checkbox("Remove Outliers (IQR method)", value=True)
    st.markdown("---")
    st.markdown("### 📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload Online Retail II (.xlsx)",
        type=["xlsx", "csv"],
        help="Upload the Online Retail II dataset"
    )
    st.markdown("---")
    st.markdown("<small style='color:#555'>Capstone Project · Customer Analytics</small>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 32px 0 16px 0'>
    <h1 style='font-size:42px; margin:0; color:#f0c060;'>Customer Segmentation</h1>
    <p style='color:#888; font-size:16px; margin-top:6px;'>
        RFM Analysis &amp; K-Means Clustering · Online Retail II Dataset
    </p>
</div>
""", unsafe_allow_html=True)

# ── No file state ─────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a24,#1f1f2e); border:1px solid #2a2a40;
                border-radius:16px; padding:48px; text-align:center; margin-top:32px;'>
        <div style='font-size:56px; margin-bottom:16px;'>📂</div>
        <h2 style='color:#f0c060; font-family:"DM Serif Display",serif;'>Upload Your Dataset</h2>
        <p style='color:#888; max-width:480px; margin:0 auto; line-height:1.7;'>
            Upload the <strong style='color:#e8e4dc;'>Online Retail II</strong> Excel file using the
            sidebar to begin the full segmentation pipeline.
        </p>
        <div style='margin-top:28px; display:flex; justify-content:center; gap:24px; flex-wrap:wrap;'>
            <div style='color:#aaa; font-size:14px;'>✅ Data Exploration</div>
            <div style='color:#aaa; font-size:14px;'>✅ Data Cleaning</div>
            <div style='color:#aaa; font-size:14px;'>✅ RFM Features</div>
            <div style='color:#aaa; font-size:14px;'>✅ Outlier Removal</div>
            <div style='color:#aaa; font-size:14px;'>✅ K-Means Clustering</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file, sheet_name=0)

with st.spinner("Loading data..."):
    df = load_data(uploaded_file)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview",
    "🧹 Cleaning",
    "📐 RFM Features",
    "📈 Clustering",
    "🎯 Segments"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DATA OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Raw Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ["Total Rows", "Total Columns", "Unique Customers", "Unique Invoices"],
        [f"{len(df):,}", df.shape[1],
         f"{df['Customer ID'].nunique():,}" if 'Customer ID' in df.columns else "N/A",
         f"{df['Invoice'].nunique():,}" if 'Invoice' in df.columns else "N/A"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Sample Records</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown('<div class="section-title">Missing Values</div>', unsafe_allow_html=True)
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
    missing = missing[missing["Missing Count"] > 0]
    if missing.empty:
        st.success("No missing values found!")
    else:
        st.dataframe(missing, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CLEANING
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Data Cleaning Pipeline</div>', unsafe_allow_html=True)

    cleaned_df = df.copy()
    rows_start = len(cleaned_df)

    # Step 1 — Valid invoices (6-digit)
    cleaned_df["Invoice"] = cleaned_df["Invoice"].astype("str")
    cleaned_df = cleaned_df[cleaned_df["Invoice"].str.match(r"^\d{6}$")]
    r1 = rows_start - len(cleaned_df)

    # Step 2 — Valid stock codes
    cleaned_df["StockCode"] = cleaned_df["StockCode"].astype("str")
    cleaned_df = cleaned_df[
        cleaned_df["StockCode"].str.match(r"^\d{5}$") |
        cleaned_df["StockCode"].str.match(r"^\d{5}[a-zA-Z]+$")
    ]
    r2 = (rows_start - r1) - len(cleaned_df)

    # Step 3 — Drop missing Customer IDs
    before = len(cleaned_df)
    cleaned_df.dropna(subset=["Customer ID"], inplace=True)
    r3 = before - len(cleaned_df)

    # Step 4 — Remove zero/negative prices
    before = len(cleaned_df)
    cleaned_df = cleaned_df[cleaned_df["Price"] > 0]
    r4 = before - len(cleaned_df)

    rows_end = len(cleaned_df)
    pct_kept = rows_end / rows_start * 100

    steps = [
        ("1", "Valid Invoice Format", f"Kept only 6-digit numeric invoices", f"−{r1:,} rows"),
        ("2", "Valid Stock Code Format", f"Kept standard 5-digit stock codes", f"−{r2:,} rows"),
        ("3", "Remove Missing Customer IDs", f"Dropped rows without Customer ID", f"−{r3:,} rows"),
        ("4", "Remove Zero Price Items", f"Dropped rows where Price = 0", f"−{r4:,} rows"),
    ]
    for num, title, desc, impact in steps:
        st.markdown(f"""
        <div style='background:#1a1a24; border:1px solid #2a2a40; border-left:4px solid #f0c060;
                    border-radius:8px; padding:14px 20px; margin-bottom:12px; display:flex;
                    justify-content:space-between; align-items:center;'>
            <div>
                <span style='color:#f0c060; font-weight:700; margin-right:10px;'>Step {num}</span>
                <strong style='color:#e8e4dc;'>{title}</strong>
                <div style='color:#888; font-size:13px; margin-top:4px;'>{desc}</div>
            </div>
            <div style='color:#e05060; font-weight:600; font-size:15px; white-space:nowrap;'>{impact}</div>
        </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, label, val in zip(
        [c1, c2, c3],
        ["Original Rows", "Rows After Cleaning", "Data Retained"],
        [f"{rows_start:,}", f"{rows_end:,}", f"{pct_kept:.1f}%"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — RFM FEATURES
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Feature Engineering — RFM</div>', unsafe_allow_html=True)

    cleaned_df["InvoiceDate"] = pd.to_datetime(cleaned_df["InvoiceDate"])
    cleaned_df["SalesLineTotal"] = cleaned_df["Quantity"] * cleaned_df["Price"]

    aggregated_df = cleaned_df.groupby("Customer ID", as_index=False).agg(
        MonetaryValue=("SalesLineTotal", "sum"),
        Frequency=("Invoice", "nunique"),
        LastInvoiceDate=("InvoiceDate", "max")
    )
    max_date = aggregated_df["LastInvoiceDate"].max()
    aggregated_df["Recency"] = (max_date - aggregated_df["LastInvoiceDate"]).dt.days

    col1, col2, col3 = st.columns(3)
    for col, name, color, desc in zip(
        [col1, col2, col3],
        ["Recency", "Frequency", "Monetary"],
        ["#e05060", "#4e9af1", "#50c878"],
        ["Days since last purchase", "Number of unique invoices", "Total spend (£)"]
    ):
        col.markdown(f"""
        <div style='background:#1a1a24; border:1px solid #2a2a40; border-top:4px solid {color};
                    border-radius:10px; padding:20px; text-align:center;'>
            <div style='font-size:28px; font-family:"DM Serif Display",serif; color:{color};'>R·F·M</div>
            <div style='font-size:18px; font-weight:600; color:#e8e4dc; margin-top:6px;'>{name}</div>
            <div style='font-size:13px; color:#888; margin-top:4px;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">RFM Table (Sample)</div>', unsafe_allow_html=True)
    st.dataframe(aggregated_df[["Customer ID","Recency","Frequency","MonetaryValue"]].head(10),
                 use_container_width=True)

    st.markdown('<div class="section-title">Distributions</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='#0f0f13')
    for ax, col, color, title in zip(
        axes,
        ["MonetaryValue", "Frequency", "Recency"],
        ["#4e9af1", "#50c878", "#e05060"],
        ["Monetary Value", "Frequency", "Recency (days)"]
    ):
        ax.set_facecolor('#16161d')
        ax.hist(aggregated_df[col], bins=30, color=color, edgecolor='none', alpha=0.85)
        ax.set_title(title, color='#e8e4dc', fontsize=13)
        ax.tick_params(colors='#666')
        for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Outlier removal
    if remove_outliers:
        def iqr_filter(df, col):
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            return df[(df[col] <= Q3 + 1.5*IQR) & (df[col] >= Q1 - 1.5*IQR)]
        non_outliers_df = iqr_filter(aggregated_df, "MonetaryValue")
        non_outliers_df = iqr_filter(non_outliers_df, "Frequency")
        removed = len(aggregated_df) - len(non_outliers_df)
        st.info(f"🔍 Outlier removal (IQR): **{removed}** outlier customers removed — **{len(non_outliers_df):,}** customers remain.")
    else:
        non_outliers_df = aggregated_df.copy()
        st.info("ℹ️ Outlier removal is disabled.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">K-Means Clustering</div>', unsafe_allow_html=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(non_outliers_df[["MonetaryValue", "Frequency", "Recency"]])
    scaled_df = pd.DataFrame(scaled, index=non_outliers_df.index,
                             columns=["MonetaryValue", "Frequency", "Recency"])

    with st.spinner("Running K-Means for k = 2 to 12..."):
        inertias, sil_scores, k_vals = [], [], range(2, 13)
        for k in k_vals:
            km = KMeans(n_clusters=k, random_state=42, max_iter=1000, n_init=10)
            labels = km.fit_predict(scaled_df)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(scaled_df, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0f0f13')
    for ax, values, ylabel, title, color in zip(
        [ax1, ax2],
        [inertias, sil_scores],
        ["Inertia", "Silhouette Score"],
        ["Elbow Curve — Inertia", "Silhouette Scores"],
        ["#f0c060", "#50c878"]
    ):
        ax.set_facecolor('#16161d')
        ax.plot(list(k_vals), values, marker='o', color=color, linewidth=2.5, markersize=7)
        ax.axvline(x=n_clusters, color='#e05060', linestyle='--', alpha=0.7, label=f'Selected k={n_clusters}')
        ax.set_xlabel('Number of Clusters (k)', color='#888')
        ax.set_ylabel(ylabel, color='#888')
        ax.set_title(title, color='#e8e4dc', fontsize=13)
        ax.tick_params(colors='#666')
        ax.legend(facecolor='#1a1a24', labelcolor='#e8e4dc', edgecolor='#2a2a40')
        for spine in ax.spines.values(): spine.set_color('#2a2a40')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Final clustering
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, max_iter=1000, n_init=10)
    cluster_labels = kmeans_final.fit_predict(scaled_df)
    non_outliers_df = non_outliers_df.copy()
    non_outliers_df["Cluster"] = cluster_labels

    final_sil = silhouette_score(scaled_df, cluster_labels)
    c1, c2, c3 = st.columns(3)
    for col, label, val in zip(
        [c1, c2, c3],
        ["Clusters", "Customers Segmented", "Silhouette Score"],
        [n_clusters, f"{len(non_outliers_df):,}", f"{final_sil:.3f}"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">3D Cluster Visualization</div>', unsafe_allow_html=True)
    fig = plt.figure(figsize=(10, 8), facecolor='#0f0f13')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#16161d')

    colors_list = ['#4e9af1','#f0a050','#50c878','#e05060','#c084fc','#fb923c','#34d399','#f472b6']
    for cid in sorted(non_outliers_df["Cluster"].unique()):
        mask = non_outliers_df["Cluster"] == cid
        name = CLUSTER_NAMES.get(cid, f"Cluster {cid}") if n_clusters == 4 else f"Cluster {cid}"
        ax.scatter(
            non_outliers_df.loc[mask, "MonetaryValue"],
            non_outliers_df.loc[mask, "Frequency"],
            non_outliers_df.loc[mask, "Recency"],
            c=colors_list[cid % len(colors_list)], alpha=0.7, s=18, label=name
        )

    ax.set_xlabel('Monetary Value', color='#888', labelpad=10)
    ax.set_ylabel('Frequency', color='#888', labelpad=10)
    ax.set_zlabel('Recency', color='#888', labelpad=10)
    ax.tick_params(colors='#555')
    ax.set_title('Customer Clusters in RFM Space', color='#e8e4dc', fontsize=14, pad=20)
    ax.legend(facecolor='#1a1a24', labelcolor='#e8e4dc', edgecolor='#2a2a40', fontsize=9)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">Customer Segment Profiles</div>', unsafe_allow_html=True)

    cluster_summary = non_outliers_df.groupby("Cluster")[["MonetaryValue","Frequency","Recency"]].mean().round(2)
    cluster_summary["Count"] = non_outliers_df.groupby("Cluster")["Customer ID"].count().values

    if n_clusters == 4:
        cols = st.columns(2)
        for i, (cid, row) in enumerate(cluster_summary.iterrows()):
            name = CLUSTER_NAMES.get(cid, f"Cluster {cid}")
            desc = CLUSTER_DESC.get(cid, "")
            color = colors_list[cid % len(colors_list)]
            with cols[i % 2]:
                st.markdown(f"""
                <div style='background:#1a1a24; border:1px solid #2a2a40; border-left:5px solid {color};
                            border-radius:10px; padding:20px; margin-bottom:16px;'>
                    <div style='font-size:18px; font-weight:700; color:{color};'>{name}</div>
                    <div style='color:#888; font-size:13px; margin:6px 0 14px 0;'>{desc}</div>
                    <div style='display:flex; gap:20px; flex-wrap:wrap;'>
                        <div><div style='color:#555; font-size:11px; text-transform:uppercase; letter-spacing:1px;'>Avg Spend</div>
                             <div style='color:#e8e4dc; font-weight:600;'>£{row["MonetaryValue"]:,.0f}</div></div>
                        <div><div style='color:#555; font-size:11px; text-transform:uppercase; letter-spacing:1px;'>Avg Orders</div>
                             <div style='color:#e8e4dc; font-weight:600;'>{row["Frequency"]:.1f}</div></div>
                        <div><div style='color:#555; font-size:11px; text-transform:uppercase; letter-spacing:1px;'>Avg Recency</div>
                             <div style='color:#e8e4dc; font-weight:600;'>{row["Recency"]:.0f} days</div></div>
                        <div><div style='color:#555; font-size:11px; text-transform:uppercase; letter-spacing:1px;'>Customers</div>
                             <div style='color:#e8e4dc; font-weight:600;'>{int(row["Count"]):,}</div></div>
                    </div>
                </div>""", unsafe_allow_html=True)
    else:
        st.dataframe(cluster_summary, use_container_width=True)

    st.markdown('<div class="section-title">RFM Distribution by Cluster</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0f0f13')
    for ax, metric in zip(axes, ["MonetaryValue", "Frequency", "Recency"]):
        ax.set_facecolor('#16161d')
        palette = {cid: colors_list[cid % len(colors_list)]
                   for cid in non_outliers_df["Cluster"].unique()}
        sns.violinplot(data=non_outliers_df, x="Cluster", y=metric,
                       palette=palette, ax=ax, hue="Cluster", legend=False)
        ax.set_title(metric, color='#e8e4dc', fontsize=13)
        ax.set_xlabel("Cluster", color='#888')
        ax.set_ylabel(metric, color='#888')
        ax.tick_params(colors='#666')
        for spine in ax.spines.values(): spine.set_color('#2a2a40')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-title">Download Segmented Customers</div>', unsafe_allow_html=True)
    output = non_outliers_df[["Customer ID", "MonetaryValue", "Frequency", "Recency", "Cluster"]].copy()
    if n_clusters == 4:
        output["Segment Name"] = output["Cluster"].map(CLUSTER_NAMES)
    csv = output.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Segmented Customer CSV",
        data=csv,
        file_name="customer_segments.csv",
        mime="text/csv"
    )
