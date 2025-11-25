# ===================================================================
# BƯỚC 6: XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH
# ===================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("BƯỚC 6: XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH")
print("=" * 70)

# ========== LOAD DỮ LIỆU ĐÃ XỬ LÝ ==========

print("\nLoad dữ liệu đã xử lý...")
train_processed = pd.read_csv('train_processed.csv')

# Tách features và target
X_train = train_processed.drop('Survived', axis=1)
y_train = train_processed['Survived']

print(f"✓ X_train shape: {X_train.shape}")
print(f"✓ y_train shape: {y_train.shape}")

# ========== KHỞI TẠO CÁC MÔ HÌNH ==========

print("\n" + "=" * 70)
print("KHỞI TẠO CÁC MÔ HÌNH")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'XGBoost': XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss'),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

print(f"✓ Đã khởi tạo {len(models)} mô hình")
for name in models.keys():
    print(f"  - {name}")

# ========== CROSS-VALIDATION ==========

print("\n" + "=" * 70)
print("ĐÁNH GIÁ MÔ HÌNH BẰNG CROSS-VALIDATION (5-FOLD)")
print("=" * 70)

# Cấu hình cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary để lưu kết quả
cv_results = {}

print("\nĐang thực hiện cross-validation...")
print("-" * 70)

for name, model in models.items():
    # Tính cross-validation scores
    scores = cross_val_score(model, X_train, y_train, cv=cv, 
                            scoring='accuracy', n_jobs=-1)
    
    cv_results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores
    }
    
    print(f"{name:25} | Mean: {scores.mean():.4f} | Std: {scores.std():.4f}")

# ========== VISUALIZE CV RESULTS ==========

print("\n" + "=" * 70)
print("TRỰC QUAN HÓA KẾT QUẢ CROSS-VALIDATION")
print("=" * 70)

# Tạo dataframe từ kết quả
cv_df = pd.DataFrame({
    'Model': list(cv_results.keys()),
    'Mean_Accuracy': [v['mean'] for v in cv_results.values()],
    'Std': [v['std'] for v in cv_results.values()]
})
cv_df = cv_df.sort_values('Mean_Accuracy', ascending=False)

# Plot 1: Bar chart với error bars
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].barh(cv_df['Model'], cv_df['Mean_Accuracy'], 
            xerr=cv_df['Std'], color='steelblue', alpha=0.7)
axes[0].set_xlabel('Accuracy', fontsize=12)
axes[0].set_title('Model Comparison - Cross-Validation Accuracy', 
                 fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Thêm giá trị
for i, (model, acc) in enumerate(zip(cv_df['Model'], cv_df['Mean_Accuracy'])):
    axes[0].text(acc + 0.01, i, f'{acc:.4f}', va='center', fontsize=10)

# Plot 2: Boxplot cho tất cả fold scores
all_scores = []
all_labels = []
for name in cv_df['Model']:
    all_scores.extend(cv_results[name]['scores'])
    all_labels.extend([name] * len(cv_results[name]['scores']))

bp_df = pd.DataFrame({'Model': all_labels, 'Score': all_scores})
bp_df['Model'] = pd.Categorical(bp_df['Model'], categories=cv_df['Model'], ordered=True)

bp_df.boxplot(column='Score', by='Model', ax=axes[1], patch_artist=True,
             boxprops=dict(facecolor='lightblue', alpha=0.7))
axes[1].set_title('Cross-Validation Score Distribution', 
                 fontsize=14, fontweight='bold')
axes[1].set_xlabel('Model', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
plt.sca(axes[1])
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('model_comparison_cv.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Đã lưu biểu đồ: model_comparison_cv.png")

# ========== TRAIN MÔ HÌNH TỐT NHẤT TRÊN TOÀN BỘ TRAIN SET ==========

print("\n" + "=" * 70)
print("TRAIN MÔ HÌNH TỐT NHẤT TRÊN TOÀN BỘ TRAIN SET")
print("=" * 70)

# Chọn mô hình tốt nhất
best_model_name = cv_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"Mô hình tốt nhất: {best_model_name}")
print(f"Cross-validation accuracy: {cv_df.iloc[0]['Mean_Accuracy']:.4f}")

# Train trên toàn bộ train set
print(f"\nĐang train {best_model_name}...")
best_model.fit(X_train, y_train)
print("✓ Hoàn thành training")

# Đánh giá trên training set
y_train_pred = best_model.predict(X_train)
y_train_proba = best_model.predict_proba(X_train)[:, 1]

train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)

print("\n" + "-" * 70)
print("TRAINING SET METRICS")
print("-" * 70)
print(f"Accuracy  : {train_accuracy:.4f}")
print(f"Precision : {train_precision:.4f}")
print(f"Recall    : {train_recall:.4f}")
print(f"F1-Score  : {train_f1:.4f}")
print(f"ROC-AUC   : {train_auc:.4f}")

# ========== FEATURE IMPORTANCE (NẾU MODEL HỖ TRỢ) ==========

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

if hasattr(best_model, 'feature_importances_'):
    # Lấy feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Features quan trọng nhất:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Visualize
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top 15 Feature Importance - {best_model_name}', 
             fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Đã lưu biểu đồ: feature_importance.png")
    
elif hasattr(best_model, 'coef_'):
    # Đối với Logistic Regression
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': best_model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("\nTop 10 Features quan trọng nhất (theo coefficient):")
    print(feature_importance.head(10).to_string(index=False))
    
    # Visualize
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    colors = ['red' if x < 0 else 'green' for x in top_features['Coefficient']]
    plt.barh(top_features['Feature'], top_features['Coefficient'], color=colors, alpha=0.7)
    plt.xlabel('Coefficient', fontsize=12)
    plt.title(f'Top 15 Feature Coefficients - {best_model_name}', 
             fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_coefficients.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Đã lưu biểu đồ: feature_coefficients.png")
else:
    print(f"\n{best_model_name} không hỗ trợ feature importance")

# ========== LƯU MÔ HÌNH ==========

print("\n" + "=" * 70)
print("LƯU MÔ HÌNH")
print("=" * 70)

import pickle

# Lưu mô hình tốt nhất
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Đã lưu mô hình: best_model.pkl")

# Lưu tên mô hình
with open('best_model_name.txt', 'w') as f:
    f.write(best_model_name)
print("✓ Đã lưu tên mô hình: best_model_name.txt")

# Lưu kết quả cross-validation
cv_df.to_csv('cv_results.csv', index=False)
print("✓ Đã lưu kết quả CV: cv_results.csv")

print("\n" + "=" * 70)
print("✓ HOÀN THÀNH BƯỚC 6: XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH")
print("=" * 70)
print(f"\nMô hình tốt nhất: {best_model_name}")
print(f"CV Accuracy: {cv_df.iloc[0]['Mean_Accuracy']:.4f} ± {cv_df.iloc[0]['Std']:.4f}")
print(f"Training Accuracy: {train_accuracy:.4f}")