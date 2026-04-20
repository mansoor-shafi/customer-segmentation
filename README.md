# Customer Segmentation — RFM Analysis & K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-green?style=flat-square)

A full-stack customer analytics web application built with Streamlit that segments customers using RFM Analysis (Recency, Frequency, Monetary) and K-Means Clustering. Upload any retail transaction dataset and instantly get actionable customer segments, marketing strategies, and exportable reports.

---

## Features

- Smart Column Detection — Automatically maps dataset columns using fuzzy matching
- Automated Data Cleaning — 4-step pipeline: invoice validation, stock codes, missing IDs, zero prices
- RFM Feature Engineering — Computes Recency, Frequency, and Monetary Value per customer
- Auto K-Detection — Combines Silhouette Score and Elbow Method to find the optimal number of clusters
- 3D Cluster Visualization — Scatter plot of customers in RFM space
- Marketing Strategy Engine — Tailored re-engagement strategies per customer segment
- Business Analytics Dashboard — Revenue share, customer share, RFM correlation heatmap
- Customer Search and Filter — Look up any customer by ID and filter by RFM range
- Export — Download segmented data and cluster summary as CSV

---

## App Tabs

| Tab | Description |
|-----|-------------|
| Data Overview | Raw data summary, missing value detection |
| Cleaning | Step-by-step data cleaning pipeline with row impact |
| RFM Features | Feature engineering, distribution plots, outlier removal |
| Clustering | K-Means with Elbow and Silhouette charts, 3D visualization |
| Segments | Segment profiles with average RFM values |
| Strategies | Actionable marketing tactics per segment |
| Analytics | Revenue and customer pie charts, RFM heatmap |
| Search and Filter | Customer lookup and dynamic RFM range filtering |
| Export | Download segmented data and cluster summary |

---

## How It Works

```
Raw Transactions
      |
      v
Smart Column Detection (fuzzy mapping)
      |
      v
Data Cleaning (invoice format, stock codes, missing IDs, zero prices)
      |
      v
RFM Feature Engineering
  - Recency   = Days since last purchase
  - Frequency = Number of unique invoices
  - Monetary  = Total spend per customer
      |
      v
StandardScaler Normalization
      |
      v
K-Means Clustering (Auto K via Silhouette + Elbow, or manual override)
      |
      v
Segment Profiles + Marketing Strategies + Export
```

---

## Customer Segments (4-cluster mode)

| Segment | Description | Strategy |
|---------|-------------|----------|
| High-Value Loyals | Frequent, recent, high spend | VIP program, referrals |
| Occasional Buyers | Moderate spend, buys sometimes | Loyalty points, seasonal offers |
| At-Risk Customers | Previously active, now drifting | Win-back campaigns, discounts |
| New / Inactive | Low recency, low frequency | Welcome series, 2nd purchase push |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Streamlit | Web app framework |
| Pandas and NumPy | Data manipulation |
| Scikit-learn | K-Means, StandardScaler, Silhouette Score |
| Matplotlib and Seaborn | Data visualization |

---

## Installation and Usage

```bash
# Clone the repository
git clone https://github.com/mansoor-shafi/customer-segmentation.git
cd customer-segmentation

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser and upload your dataset.

---

## Dataset Format

The app accepts .csv or .xlsx files with retail transaction data. Required fields (auto-detected):

| Field | Description |
|-------|-------------|
| Invoice | Transaction or order ID |
| Customer ID | Unique customer identifier |
| InvoiceDate | Date of transaction |
| Quantity | Units purchased |
| Price | Unit price |

The app uses fuzzy matching to auto-detect column names even if they are named differently (e.g. order_id, custid, unit_price).

---

## Author

**Mansoor Shafi**  
B.Tech IT Student (2022-2026)  
[LinkedIn](https://www.linkedin.com/in/mansoor-shafi-431ab224b) | [GitHub](https://github.com/mansoor-shafi)

---

## License

This project is open source and available under the MIT License.
