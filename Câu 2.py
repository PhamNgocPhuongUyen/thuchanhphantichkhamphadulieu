#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[2]:


df = pd.read_csv("online_retail_cleaned.csv")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df.head()


# In[3]:


print("Kích thước dữ liệu:", df.shape)
print("Các cột dữ liệu:")
print(df.columns)


# In[4]:


analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
print("Ngày phân tích:", analysis_date)


# In[5]:


monetary_col = 'Revenue' if 'Revenue' in df.columns else 'TotalPrice'

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (analysis_date - x.max()).days,
    'InvoiceNo': 'nunique',
    monetary_col: 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm.head()


# In[6]:


rfm.describe()


# In[7]:


rfm['Monetary'] = np.log1p(rfm['Monetary'])


# In[8]:


scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(
    rfm[['Recency', 'Frequency', 'Monetary']]
)


# In[9]:


print("Silhouette score theo số cụm k:")
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    score = silhouette_score(rfm_scaled, labels)
    print(f"k = {k}, silhouette = {score:.4f}")


# In[10]:


kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

rfm.head()


# In[11]:


cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Num_Customers'})

cluster_summary


# In[12]:


plt.figure(figsize=(8, 6))
for c in sorted(rfm['Cluster'].unique()):
    temp = rfm[rfm['Cluster'] == c]
    plt.scatter(
        temp['Frequency'],
        temp['Monetary'],
        label=f'Cluster {c}',
        alpha=0.6
    )
plt.xlabel("Frequency")
plt.ylabel("Monetary (log)")
plt.title("Phân nhóm khách hàng theo RFM + K-Means")
plt.legend()
plt.show()


# In[15]:


cluster_summary = rfm.groupby('Cluster').agg(
    Num_Customers=('CustomerID', 'count'),
    Total_Revenue=('Monetary', 'sum'),
    Avg_Revenue=('Monetary', 'mean')
).reset_index()
cluster_summary['Customer_Ratio (%)'] = (
    cluster_summary['Num_Customers'] / cluster_summary['Num_Customers'].sum() * 100
)
cluster_summary['Revenue_Ratio (%)'] = (
    cluster_summary['Total_Revenue'] / cluster_summary['Total_Revenue'].sum() * 100
)
cluster_summary['Customer_Ratio (%)'] = cluster_summary['Customer_Ratio (%)'].round(2)
cluster_summary['Revenue_Ratio (%)'] = cluster_summary['Revenue_Ratio (%)'].round(2)
print(cluster_summary)


# In[17]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.bar(
    [f'Cụm {c}' for c in cluster_summary['Cluster']],
    cluster_summary['Num_Customers']
)
plt.xlabel("Cụm khách hàng")
plt.ylabel("Số lượng khách hàng")
plt.title("Quy mô các cụm khách hàng")
plt.show()


# In[19]:


plt.figure(figsize=(8, 5))
plt.bar(
    [f'Cụm {c}' for c in cluster_summary['Cluster']],
    cluster_summary['Revenue_Ratio (%)']
)
plt.xlabel("Cụm khách hàng")
plt.ylabel("Tỷ trọng doanh thu (%)")
plt.title("Tỷ trọng doanh thu theo cụm khách hàng")
plt.show()


# In[20]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Biểu đồ 1: Quy mô khách hàng
ax1.bar([f'Cụm {c}' for c in cluster_summary['Cluster']], 
        cluster_summary['Num_Customers'],
        color=['blue', 'red', 'green', 'orange'])
ax1.set_xlabel("Cụm khách hàng")
ax1.set_ylabel("Số lượng khách hàng")
ax1.set_title("Quy mô các cụm khách hàng")

# Biểu đồ 2: Tỷ trọng doanh thu
ax2.bar([f'Cụm {c}' for c in cluster_summary['Cluster']], 
        cluster_summary['Revenue_Ratio (%)'],
        color=['blue', 'red', 'green', 'orange'])
ax2.set_xlabel("Cụm khách hàng")
ax2.set_ylabel("Tỷ trọng doanh thu (%)")
ax2.set_title("Tỷ trọng doanh thu theo cụm")

plt.tight_layout()
plt.show()


# In[ ]:




