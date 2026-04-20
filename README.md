# customer-segmentation
🛍️ Customer Segmentation — RFM Analysis & K-Means Clustering
�
�
�
�
A full-stack customer analytics web application built with Streamlit that segments customers using RFM Analysis (Recency, Frequency, Monetary) and K-Means Clustering. Upload any retail transaction dataset and instantly get actionable customer segments, marketing strategies, and exportable reports.
🚀 Live Demo
Upload your retail .csv or .xlsx dataset and the app automatically detects columns, cleans data, engineers RFM features, finds the optimal number of clusters, and delivers a full analytics dashboard.
✨ Features
Smart Column Detection — Automatically maps your dataset's columns using fuzzy matching, no manual setup needed
Automated Data Cleaning — 4-step pipeline: validates invoices, stock codes, removes missing customer IDs and zero-price rows
RFM Feature Engineering — Computes Recency, Frequency, and Monetary Value per customer from raw transactions
Auto K-Detection — Combines Silhouette Score and Elbow Method to find the optimal number of clusters automatically
3D Cluster Visualization — Interactive 3D scatter plot of customers in RFM space
Marketing Strategy Engine — Generates tailored re-engagement strategies for each customer segment
Business Analytics Dashboard — Revenue share, customer share, RFM correlation heatmap, and more
Customer Search & Filter — Look up any customer by ID and filter by segment or RFM range
Export — Download segmented customer data and cluster summary reports as CSV
📊 App Tabs
Tab
Description
📋 Data Overview
Raw data summary, missing value detection
🧹 Cleaning
Step-by-step data cleaning pipeline with row impact
📐 RFM Features
Feature engineering, distribution plots, outlier removal
🔵 Clustering
K-Means with Elbow + Silhouette charts, 3D visualization
👥 Segments
Segment profiles with average RFM values
🎯 Strategies
Actionable marketing tactics per segment
📈 Analytics
Revenue/customer pie charts, RFM heatmap
🔍 Search & Filter
Customer lookup and dynamic RFM range filtering
⬇️ Export
Download segmented data and cluster summary
🧠 How It Works
Raw Transactions
      │
      ▼
Smart Column Detection (fuzzy mapping)
      │
      ▼
Data Cleaning (invoice format, stock codes, missing IDs, zero prices)
      │
      ▼
RFM Feature Engineering
  • Recency   = Days since last purchase
  • Frequency = Number of unique invoices
  • Monetary  = Total spend per customer
      │
      ▼
StandardScaler Normalization
      │
      ▼
K-Means Clustering
  • Auto K via Silhouette + Elbow (or manual override)
      │
      ▼
Segment Profiles + Marketing Strategies + Export
🗂️ Customer Segments (4-cluster mode)
Segment
Description
Strategy
👑 High-Value Loyals
Frequent, recent, high spend
VIP program, referrals
🛒 Occasional Buyers
Moderate spend, buys sometimes
Loyalty points, seasonal offers
⚠️ At-Risk Customers
Previously active, now drifting
Win-back campaigns, discounts
🌱 New / Inactive
Low recency, low frequency
Welcome series, 2nd purchase push
🛠️ Tech Stack
Tool
Purpose
Python
Core language
Streamlit
Web app framework
Pandas & NumPy
Data manipulation
Scikit-learn
K-Means clustering, StandardScaler, Silhouette Score
Matplotlib & Seaborn
Data visualization
⚙️ Installation & Usage
# Clone the repository
git clone https://github.com/mansoor-shafi/customer-segmentation.git
cd customer-segmentation

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
Then open http://localhost:8501 in your browser and upload your dataset.
📁 Dataset Format
The app accepts .csv or .xlsx files with retail transaction data. Required fields (auto-detected):
Field
Description
Invoice
Transaction/order ID
Customer ID
Unique customer identifier
InvoiceDate
Date of transaction
Quantity
Units purchased
Price
Unit price
The app uses fuzzy matching to auto-detect your column names — even if they're named differently (e.g. order_id, custid, unit_price).
👨‍💻 Author
Mansoor Shafi
B.Tech IT Student (2022–2026)
LinkedIn · GitHub
📄 License
This project is open source and available under the MIT License.