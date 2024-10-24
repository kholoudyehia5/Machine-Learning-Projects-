# Telecom Customer Churn

### Introduction
This report outlines the final project after a long and insightful journey, focusing on Data Exploration, Preprocessing, Insights, and where I applied various data science techniques to understand and predict customer behavior.

### Objective 
This project focuses on analyzing customer behavior in the telecom industry to predict churn and identify meaningful customer segments. This analysis will empower telecom businesses to improve customer retention and tailor services to different customer segments.

### Key Features
**1. Churn Prediction**: 
- Supervised Learning: Developing models to classify customers as likely to churn or remain, using features like service usage, contract type, and payment methods. 
- Model Comparison: Applying multiple algorithms (Logistic Regression, Random Forest, XGBoost) and select the best model based on evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

**2. Customer Segmentation**: 
- Unsupervised Learning: Implementing clustering algorithms (K-Means, DBSCAN Clustering) to segment customers based on their behavior and demographics. 
- Optimal Clustering: Using techniques like the elbow method or silhouette score to determine the best number of clusters, ensuring meaningful segmentation.

**3. Visualization & Reporting**: 
- Visualize the churn prediction results, customer clusters, and their characteristics using Python libraries such as Matplotlib and Seaborn. 
- Present insights and recommendations in a structured format, emphasizing actionable business strategies.

**4. Deployment**:
- Save a model as .pkl file for using in depolyment.
- Deploy model and run with Streamlit app.

### Results
- **Best Model in Supervised Learning**: XGBoost Classifier with an accuracy of 95% and ROC AUC of 99%.
- **Best Model in Unsupervised Learning**: DBSCAN Clustering Algorithm with Silhouette Score 0.74%.

