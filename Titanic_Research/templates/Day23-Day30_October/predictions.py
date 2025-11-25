# ===================================================================
# BƯỚC 8: DỰ ĐOÁN TRÊN TEST SET VÀ TẠO SUBMISSION FILE
# ===================================================================

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("BƯỚC 8: DỰ ĐOÁN TRÊN TEST SET")
print("=" * 70)

# ========== LOAD MÔ HÌNH VÀ DỮ LIỆU ==========

print("\nLoad mô hình và dữ liệu...")

# Kiểm tra xem có mô hình đã tune không
if os.path.exists('tuned_model.pkl'):
    with open('tuned_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Đã load: tuned_model.pkl (mô hình đã tune)")
    model_type = "Tuned Model"
else:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Đã load: best_model.pkl (mô hình baseline)")
    model_type = "Baseline Model"

# Load tên mô hình
with open('best_model_name.txt', 'r') as f:
    model_name = f.read().strip()
print(f"✓ Model: {model_name}")

# Load dữ liệu
train_processed = pd.read_csv('train_processed.csv')
test_processed = pd.read_csv('test_processed.csv')

print(f"✓ Train shape: {train_processed.shape}")
print(f"✓ Test shape: {test_processed.shape}")

# ========== CHUẨN BỊ DỮ LIỆU ==========

print("\n" + "=" * 70)
print("CHUẨN BỊ DỮ LIỆU")
print("=" * 70)

# Train data
X_train = train_processed.drop('Survived', axis=1)
y_train = train_processed['Survived']

# Test data
test_ids = test_processed['PassengerId'].copy()
X_test = test_processed.drop('PassengerId', axis=1)

print(f"✓ X_train shape: {X_train.shape}")
print(f"✓ y_train shape: {y_train.shape}")
print(f"✓ X_test shape: {X_test.shape}")

# Đảm bảo cùng thứ tự columns
X_test = X_test[X_train.columns]
print("✓ Đã đồng bộ columns giữa train và test")

# ========== ĐÁNH GIÁ LẠI TRÊN TRAINING SET ==========

print("\n" + "=" * 70)
print("ĐÁNH GIÁ TRÊN TRAINING SET")
print("=" * 70)

# Predict trên train set
y_train_pred = model.predict(X_train)

# Tính metrics
train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")

# Classification report
print("\nClassification Report (Training Set):")
print(classification_report(y_train, y_train_pred, 
                          target_names=['Died', 'Survived']))

# Confusion Matrix
cm_train = confusion_matrix(y_train, y_train_pred)

# Visualize confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training confusion matrix
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
           xticklabels=['Died', 'Survived'],
           yticklabels=['Died', 'Survived'])
axes[0].set_title(f'Confusion Matrix - Training Set\n{model_type}', 
                 fontsize=12, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=11)
axes[0].set_xlabel('Predicted', fontsize=11)

# Thêm tỷ lệ
for i in range(2):
    for j in range(2):
        text = axes[0].text(j + 0.5, i + 0.7, 
                          f'({cm_train[i, j]/cm_train[i].sum()*100:.1f}%)',
                          ha='center', va='center', fontsize=9, color='gray')

# ========== DỰ ĐOÁN TRÊN TEST SET ==========

print("\n" + "=" * 70)
print("DỰ ĐOÁN TRÊN TEST SET")
print("=" * 70)

# Predict
print("Đang dự đoán...")
y_test_pred = model.predict(X_test)

# Predict probabilities (nếu model hỗ trợ)
if hasattr(model, 'predict_proba'):
    y_test_proba = model.predict_proba(X_test)[:, 1]
    print("✓ Đã tính probability scores")
else:
    y_test_proba = None
    print("⚠ Model không hỗ trợ predict_proba")

# Thống kê dự đoán
n_survived = y_test_pred.sum()
n_died = len(y_test_pred) - n_survived

print(f"\nKết quả dự đoán:")
print(f"  Survived: {n_survived} ({n_survived/len(y_test_pred)*100:.1f}%)")
print(f"  Died    : {n_died} ({n_died/len(y_test_pred)*100:.1f}%)")

# Visualize prediction distribution
prediction_counts = pd.Series(y_test_pred).value_counts()

axes[1].bar(['Died', 'Survived'], 
           [prediction_counts.get(0, 0), prediction_counts.get(1, 0)],
           color=['#e74c3c', '#2ecc71'], alpha=0.7)
axes[1].set_title('Test Set Predictions\nDistribution', 
                 fontsize=12, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

# Thêm số lượng lên cột
for i, v in enumerate([prediction_counts.get(0, 0), prediction_counts.get(1, 0)]):
    axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Đã lưu biểu đồ: prediction_results.png")

# ========== TẠO SUBMISSION FILE ==========

print("\n" + "=" * 70)
print("TẠO SUBMISSION FILE")
print("=" * 70)

# Tạo submission dataframe
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': y_test_pred
})

# Lưu file
submission.to_csv('submission.csv', index=False)
print("✓ Đã lưu: submission.csv")

# Hiển thị 10 dòng đầu
print("\n10 dòng đầu của submission file:")
print(submission.head(10))

# ========== TẠO SUBMISSION FILE VỚI PROBABILITIES ==========

if y_test_proba is not None:
    submission_with_proba = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': y_test_pred,
        'Survival_Probability': y_test_proba
    })
    
    submission_with_proba.to_csv('submission_with_probabilities.csv', index=False)
    print("\n✓ Đã lưu: submission_with_probabilities.csv (bao gồm probabilities)")
    
    # Phân tích confidence
    print("\n" + "-" * 70)
    print("PHÂN TÍCH CONFIDENCE")
    print("-" * 70)
    
    # Các dự đoán với confidence cao (>0.9)
    high_conf_died = ((y_test_pred == 0) & (y_test_proba < 0.1)).sum()
    high_conf_survived = ((y_test_pred == 1) & (y_test_proba > 0.9)).sum()
    
    # Các dự đoán không chắc chắn (0.4-0.6)
    uncertain = ((y_test_proba >= 0.4) & (y_test_proba <= 0.6)).sum()
    
    print(f"High confidence 'Died' (prob < 0.1)      : {high_conf_died}")
    print(f"High confidence 'Survived' (prob > 0.9)  : {high_conf_survived}")
    print(f"Uncertain predictions (0.4 < prob < 0.6) : {uncertain}")
    
    # Visualize confidence distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(y_test_proba, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    plt.xlabel('Survival Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Survival Probabilities', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    survived_probs = y_test_proba[y_test_pred == 1]
    died_probs = y_test_proba[y_test_pred == 0]
    
    plt.hist([died_probs, survived_probs], bins=30, 
            label=['Predicted Died', 'Predicted Survived'],
            color=['#e74c3c', '#2ecc71'], alpha=0.7)
    plt.xlabel('Survival Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Probability Distribution by Prediction', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_confidence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Đã lưu biểu đồ: prediction_confidence.png")

# ========== SUMMARY ==========

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

summary = f"""
Model Information:
  - Model Type: {model_name}
  - Version: {model_type}
  - Training Accuracy: {train_accuracy:.4f}

Test Set Predictions:
  - Total Predictions: {len(y_test_pred)}
  - Predicted Survived: {n_survived} ({n_survived/len(y_test_pred)*100:.1f}%)
  - Predicted Died: {n_died} ({n_died/len(y_test_pred)*100:.1f}%)

Output Files:
  ✓ submission.csv - Main submission file
  {'✓ submission_with_probabilities.csv - With confidence scores' if y_test_proba is not None else ''}
  ✓ prediction_results.png - Visualization
  {'✓ prediction_confidence.png - Confidence analysis' if y_test_proba is not None else ''}

Next Steps:
  1. Submit submission.csv to Kaggle
  2. Analyze results and feature importance
  3. Consider ensemble methods for better performance
"""

print(summary)

print("\n" + "=" * 70)
print("✓ HOÀN THÀNH BƯỚC 8: DỰ ĐOÁN TRÊN TEST SET")
print("=" * 70)