# ===================================================================
# BƯỚC 4: PHÂN TÍCH HAI BIẾN (BIVARIATE ANALYSIS)
# ===================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dữ liệu
train = pd.read_csv('train.csv')

print("=" * 70)
print("PHÂN TÍCH MỐI QUAN HỆ GIỮA CÁC BIẾN VỚI TARGET")
print("=" * 70)

# ========== SURVIVAL RATE THEO CÁC BIẾN PHÂN LOẠI ==========

categorical_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

# Tính survival rate
print("\nTỷ lệ sống sót theo từng biến:")
print("-" * 70)

for col in categorical_cols:
    survival_rate = train.groupby(col)['Survived'].agg(['sum', 'count', 'mean'])
    survival_rate.columns = ['Survived', 'Total', 'Survival_Rate']
    survival_rate['Survival_Rate'] = (survival_rate['Survival_Rate'] * 100).round(2)
    
    print(f"\n{col}:")
    print(survival_rate)

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, col in enumerate(categorical_cols):
    # Tạo crosstab
    ct = pd.crosstab(train[col], train['Survived'], normalize='index') * 100
    
    # Vẽ stacked bar chart
    ct.plot(kind='bar', stacked=True, ax=axes[idx], 
           color=['#e74c3c', '#2ecc71'], alpha=0.8)
    axes[idx].set_title(f'Survival Rate by {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col, fontsize=11)
    axes[idx].set_ylabel('Percentage (%)', fontsize=11)
    axes[idx].legend(['Died', 'Survived'], loc='upper right')
    axes[idx].grid(axis='y', alpha=0.3)
    
    # Thêm phần trăm lên cột
    for container in axes[idx].containers:
        axes[idx].bar_label(container, fmt='%.1f%%', label_type='center')

# Xóa subplot thừa
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('survival_by_categorical.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== SURVIVAL RATE THEO BIẾN SỐ ==========

print("\n" + "=" * 70)
print("PHÂN TÍCH BIẾN SỐ VỚI TARGET")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Age vs Survived
axes[0, 0].hist([train[train['Survived']==0]['Age'].dropna(),
                train[train['Survived']==1]['Age'].dropna()],
               bins=20, label=['Died', 'Survived'], 
               color=['#e74c3c', '#2ecc71'], alpha=0.7)
axes[0, 0].set_title('Age Distribution by Survival', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Age', fontsize=11)
axes[0, 0].set_ylabel('Count', fontsize=11)
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Boxplot Age vs Survived
train.boxplot(column='Age', by='Survived', ax=axes[0, 1],
             patch_artist=True, grid=False)
axes[0, 1].set_title('Age by Survival Status', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Survived (0=No, 1=Yes)', fontsize=11)
axes[0, 1].set_ylabel('Age', fontsize=11)
plt.sca(axes[0, 1])
plt.xticks([1, 2], ['Died', 'Survived'])

# Fare vs Survived
axes[1, 0].hist([train[train['Survived']==0]['Fare'].dropna(),
                train[train['Survived']==1]['Fare'].dropna()],
               bins=30, label=['Died', 'Survived'], 
               color=['#e74c3c', '#2ecc71'], alpha=0.7)
axes[1, 0].set_title('Fare Distribution by Survival', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Fare', fontsize=11)
axes[1, 0].set_ylabel('Count', fontsize=11)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_xlim(0, 300)  # Giới hạn để dễ nhìn

# Boxplot Fare vs Survived
train.boxplot(column='Fare', by='Survived', ax=axes[1, 1],
             patch_artist=True, grid=False)
axes[1, 1].set_title('Fare by Survival Status', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Survived (0=No, 1=Yes)', fontsize=11)
axes[1, 1].set_ylabel('Fare', fontsize=11)
axes[1, 1].set_ylim(0, 300)  # Giới hạn để dễ nhìn
plt.sca(axes[1, 1])
plt.xticks([1, 2], ['Died', 'Survived'])

plt.tight_layout()
plt.savefig('survival_by_numerical.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== PHÂN TÍCH TƯƠNG TÁC GIỮA CÁC BIẾN ==========

print("\n" + "=" * 70)
print("PHÂN TÍCH TƯƠNG TÁC: PCLASS & SEX VS SURVIVED")
print("=" * 70)

# Tạo crosstab
pclass_sex_survival = train.groupby(['Pclass', 'Sex'])['Survived'].agg(['sum', 'count', 'mean'])
pclass_sex_survival.columns = ['Survived', 'Total', 'Survival_Rate']
pclass_sex_survival['Survival_Rate'] = (pclass_sex_survival['Survival_Rate'] * 100).round(2)
print(pclass_sex_survival)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Grouped bar chart
survival_pivot = train.pivot_table(values='Survived', index='Pclass', 
                                   columns='Sex', aggfunc='mean') * 100
survival_pivot.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'], alpha=0.8)
axes[0].set_title('Survival Rate by Class and Gender', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Passenger Class', fontsize=12)
axes[0].set_ylabel('Survival Rate (%)', fontsize=12)
axes[0].set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=0)
axes[0].legend(['Female', 'Male'], title='Gender')
axes[0].grid(axis='y', alpha=0.3)

# Heatmap
survival_heatmap = train.pivot_table(values='Survived', index='Sex', 
                                     columns='Pclass', aggfunc='mean')
sns.heatmap(survival_heatmap, annot=True, fmt='.2%', cmap='RdYlGn', 
           ax=axes[1], cbar_kws={'label': 'Survival Rate'})
axes[1].set_title('Survival Rate Heatmap (Sex vs Class)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Passenger Class', fontsize=12)
axes[1].set_ylabel('Gender', fontsize=12)

plt.tight_layout()
plt.savefig('interaction_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== CORRELATION MATRIX ==========

print("\n" + "=" * 70)
print("MA TRẬN TƯƠNG QUAN (CORRELATION MATRIX)")
print("=" * 70)

# Chọn các biến số để tính correlation
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
correlation_matrix = train[numeric_cols].corr()

print("\nCorrelation với Survived:")
print(correlation_matrix['Survived'].sort_values(ascending=False))

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
           center=0, square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== INSIGHTS ==========

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("""
1. PCLASS (Hạng ghế):
   ✓ Hạng 1 có tỷ lệ sống sót cao nhất (~63%)
   ✓ Hạng 3 có tỷ lệ sống sót thấp nhất (~24%)
   
2. SEX (Giới tính):
   ✓ Nữ có tỷ lệ sống sót rất cao (~74%)
   ✓ Nam có tỷ lệ sống sót thấp (~19%)
   ✓ "Women and children first" policy!
   
3. PCLASS & SEX (Tương tác):
   ✓ Nữ hạng 1 & 2: >90% sống sót
   ✓ Nam hạng 3: chỉ ~14% sống sót
   
4. AGE (Tuổi):
   ✓ Trẻ em có tỷ lệ sống sót cao hơn
   ✓ Người cao tuổi có xu hướng sống sót thấp hơn
   
5. FARE (Giá vé):
   ✓ Giá vé cao tương quan với sống sót cao
   ✓ Phản ánh vị trí cabin và hạng ghế
   
6. SIBSP & PARCH (Gia đình):
   ✓ Có gia đình nhỏ (1-2 người) tốt hơn đi một mình
   ✓ Gia đình quá lớn làm giảm tỷ lệ sống sót
""")

print("\n✓ Hoàn thành bước 4: Phân tích hai biến")
print("✓ Đã lưu các biểu đồ:")
print("  - survival_by_categorical.png")
print("  - survival_by_numerical.png")
print("  - interaction_analysis.png")
print("  - correlation_matrix.png")