
# 🧠 AI Customer Segmentation using RFM Analysis + K-Means Clustering

## 📌 Project Overview

This project builds an **AI-powered Customer Segmentation System** for a retail / D2C e-commerce business using **RFM Analysis (Recency, Frequency, Monetary)** and **Machine Learning clustering techniques**.

The goal is to automatically identify valuable customer groups, analyze customer behavior, and help businesses make **data-driven marketing and retention decisions**.

---

## 🚀 Business Problem

E-commerce companies generate huge volumes of customer transaction data, but identifying customer buying patterns manually is difficult.

This project solves that problem by:

* Segmenting customers based on purchasing behavior
* Identifying high-value and at-risk customers
* Discovering hidden customer groups using machine learning
* Improving customer retention and targeted marketing strategies

---

## 🛠 Tech Stack

* **Python**
* **Pandas** → Data Cleaning & Analysis
* **NumPy** → Numerical Operations
* **Matplotlib** → Data Visualization
* **Seaborn** → Statistical Visualization
* **Scikit-learn** → Machine Learning Models
* **K-Means Clustering** → Customer Segmentation
* **PCA (Principal Component Analysis)** → Dimensionality Reduction
* **RFM Analysis** → Customer Scoring Framework

---

## 📊 Project Workflow

### 1️⃣ Data Collection

Project uses three datasets:

* Customer Dataset
* Transaction Dataset
* Customer Behavior Dataset

### 2️⃣ Feature Engineering

Created important business metrics:

* Recency → Days since last purchase
* Frequency → Number of purchases
* Monetary → Total customer spend

### 3️⃣ RFM Scoring

Applied quintile-based scoring system to classify customers into:

* Champion Customers
* Loyal Customers
* Potential Customers
* At Risk Customers
* Lost Customers

### 4️⃣ Machine Learning Clustering

Applied **K-Means Clustering** to automatically discover hidden customer segments based on behavioral patterns.

### 5️⃣ Cluster Optimization

Used **Silhouette Score** to determine the optimal number of clusters.

### 6️⃣ PCA Visualization

Reduced dimensions using PCA to visualize customer clusters in 2D space.

### 7️⃣ Customer Behavior Analysis

Analyzed behavioral metrics like:

* App Sessions
* Wishlist Activity
* Cart Abandonment
* Email Engagement
* Referral Activity

---

## 📈 Key Insights Generated

✔ High-value customers contribute major revenue share

✔ Customer clusters automatically detected using AI

✔ Loyal customers identified for retention campaigns

✔ Dormant customers identified for re-engagement strategies

✔ Channel-wise customer lifetime value analyzed

✔ Category preference analyzed across customer segments

---

## 📉 Visualizations Created

The project generates a dashboard containing:

* Revenue by RFM Segment
* K-Means Cluster Scatter Plot
* Silhouette Score Optimization Curve
* Recency vs Monetary Distribution
* Category Preference Heatmap
* Behavioral Cluster Analysis

---

## 🎯 Machine Learning Concepts Used

* Unsupervised Learning
* Customer Segmentation
* Feature Scaling
* Cluster Analysis
* PCA Dimensionality Reduction
* Behavioral Analytics
* Data Visualization

---

## 📂 Output Files

Generated output files:

* `customer_segments_output.csv` → Final segmented customer data
* `cluster_profiles_output.csv` → Cluster summary statistics
* `p2_segmentation_dashboard.png` → Dashboard visualization

---

## 💡 Business Impact

This solution helps businesses:

* Improve customer retention
* Personalize marketing campaigns
* Increase customer lifetime value (LTV)
* Reduce customer churn
* Optimize product recommendations

---

## 📌 Future Improvements

* Deploy model using API
* Build interactive dashboard using Power BI / Streamlit
* Real-time customer segmentation pipeline
* Recommendation engine integration
* Customer churn prediction model

---

## 👨‍💻 Author

**Piyush Palkatwar**

Aspiring **AI/ML Engineer | Data Scientist | Generative AI Enthusiast**

Building real-world machine learning projects focused on solving business problems using data and artificial intelligence.
