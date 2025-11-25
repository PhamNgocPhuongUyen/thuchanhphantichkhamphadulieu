# ===================================================================
# BƯỚC 3: PHÂN TÍCH ĐƠN BIẾN (UNIVARIATE ANALYSIS)
# ===================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dữ liệu
train = pd.read_csv('train.csv')

print("=" * 70)
print("PHÂN TÍCH ĐƠN BIẾN - TARGET VARIABLE")
print("=" * 70)

# ========== PHÂN TÍCH BIẾN MỤC TIÊU (SURVIVED) ==========

# Đếm số lượng
survived_counts = train['Survived'].value_counts()
print("\nPhân bố Survived:")
print(survived_counts)
print(f"\nTỷ lệ sống sót: {(survived_counts[1]/len(train)*100):.2f}%")
print(f"Tỷ lệ tử vong: {(survived_counts[0]/len(train)*100):.2f}%")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
survived_counts.plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'])
axes[0].set_title('Survival Count', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Survived (0=No, 1=Yes)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xticklabels(['Died (0)', 'Survived (1)'], rotation=0)
axes[0].grid(axis='y', alpha=0.3)

# Thêm giá trị lên cột
for i, v in enumerate(survived_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Pie chart
colors = ['#e74c3c', '#2ecc71']
explode = (0.05, 0.05)
axes[1].pie(survived_counts.values, labels=['Died', 'Survived'], 
           autopct='%1.1f%%', startangle=90, colors=colors,
           explode=explode, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Survival Rate', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('target_variable_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("PHÂN TÍCH CÁC BIẾN PHÂN LOẠI (CATEGORICAL)")
print("=" * 70)

# ========== PHÂN TÍCH CÁC BIẾN PHÂN LOẠI ==========

categorical_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, col in enumerate(categorical_cols):
    # Đếm giá trị
    value_counts = train[col].value_counts().sort_index()
    
    print(f"\n{col}:")
    print(value_counts)
    
    # Vẽ biểu đồ
    value_counts.plot(kind='bar', ax=axes[idx], color='steelblue')
    axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col, fontsize=11)
    axes[idx].set_ylabel('Count', fontsize=11)
    axes[idx].grid(axis='y', alpha=0.3)
    
    # Thêm giá trị lên cột
    for i, v in enumerate(value_counts.values):
        axes[idx].text(i, v + 5, str(v), ha='center')

# Xóa subplot thừa
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('categorical_variables_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("PHÂN TÍCH CÁC BIẾN SỐ (NUMERICAL)")
print("=" * 70)

# ========== PHÂN TÍCH CÁC BIẾN SỐ ==========

numerical_cols = ['Age', 'Fare']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for idx, col in enumerate(numerical_cols):
    # Loại bỏ giá trị null
    data = train[col].dropna()
    
    # Thống kê
    print(f"\n{col}:")
    print(f"  Mean   : {data.mean():.2f}")
    print(f"  Median : {data.median():.2f}")
    print(f"  Std    : {data.std():.2f}")
    print(f"  Min    : {data.min():.2f}")
    print(f"  Max    : {data.max():.2f}")
    print(f"  Q1     : {data.quantile(0.25):.2f}")
    print(f"  Q3     : {data.quantile(0.75):.2f}")
    
    # Histogram
    axes[idx, 0].hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[idx, 0].set_title(f'Histogram of {col}', fontsize=12, fontweight='bold')
    axes[idx, 0].set_xlabel(col, fontsize=11)
    axes[idx, 0].set_ylabel('Frequency', fontsize=11)
    axes[idx, 0].axvline(data.mean(), color='red', linestyle='--', 
                         linewidth=2, label=f'Mean: {data.mean():.2f}')
    axes[idx, 0].axvline(data.median(), color='green', linestyle='--', 
                         linewidth=2, label=f'Median: {data.median():.2f}')
    axes[idx, 0].legend()
    axes[idx, 0].grid(alpha=0.3)
    
    # Boxplot
    axes[idx, 1].boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
    axes[idx, 1].set_title(f'Boxplot of {col}', fontsize=12, fontweight='bold')
    axes[idx, 1].set_ylabel(col, fontsize=11)
    axes[idx, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('numerical_variables_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== KIỂM TRA OUTLIERS ==========

print("\n" + "=" * 70)
print("PHÁT HIỆN OUTLIERS (SỬ DỤNG IQR METHOD)")
print("=" * 70)

for col in numerical_cols:
    data = train[col].dropna()
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    print(f"\n{col}:")
    print(f"  Lower bound: {lower_bound:.2f}")
    print(f"  Upper bound: {upper_bound:.2f}")
    print(f"  Number of outliers: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)")

print("\n✓ Hoàn thành bước 3: Phân tích đơn biến")
print("✓ Đã lưu các biểu đồ:")
print("  - target_variable_analysis.png")
print("  - categorical_variables_analysis.png")
print("  - numerical_variables_analysis.png")