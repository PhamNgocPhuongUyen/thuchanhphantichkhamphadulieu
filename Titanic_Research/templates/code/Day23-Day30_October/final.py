# ===================================================================
# BƯỚC 9: TẠO BÁO CÁO TỔNG KẾT
# ===================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
from datetime import datetime

print("=" * 70)
print("BƯỚC 9: TẠO BÁO CÁO TỔNG KẾT")
print("=" * 70)

# ========== LOAD TẤT CẢ THÔNG TIN CẦN THIẾT ==========

print("\nĐang thu thập thông tin...")

# Load model info
with open('best_model_name.txt', 'r') as f:
    model_name = f.read().strip()

# Load CV results
cv_results = pd.read_csv('cv_results.csv')

# Load best params nếu có
if os.path.exists('best_params.json'):
    with open('best_params.json', 'r') as f:
        best_params = json.load(f)
    has_tuning = True
else:
    best_params = None
    has_tuning = False

# Load submission
submission = pd.read_csv('submission.csv')

# Load processed data info
train_processed = pd.read_csv('train_processed.csv')
test_processed = pd.read_csv('test_processed.csv')

print("✓ Đã thu thập đầy đủ thông tin")

# ========== TẠO VISUALIZATION TỔNG QUAN ==========

print("\n" + "=" * 70)
print("TẠO VISUALIZATION TỔNG QUAN")
print("=" * 70)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. So sánh hiệu suất mô hình
ax1 = fig.add_subplot(gs[0, :2])
cv_results_sorted = cv_results.sort_values('Mean_Accuracy', ascending=True)
colors = ['red' if x == model_name else 'steelblue' for x in cv_results_sorted['Model']]
ax1.barh(cv_results_sorted['Model'], cv_results_sorted['Mean_Accuracy'],
        xerr=cv_results_sorted['Std'], color=colors, alpha=0.7)
ax1.set_xlabel('Cross-Validation Accuracy', fontsize=11)
ax1.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
for i, (model, acc) in enumerate(zip(cv_results_sorted['Model'], 
                                     cv_results_sorted['Mean_Accuracy'])):
    ax1.text(acc + 0.005, i, f'{acc:.4f}', va='center', fontsize=9)

# 2. Thông tin mô hình tốt nhất
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
best_score = cv_results.iloc[0]['Mean_Accuracy']
best_std = cv_results.iloc[0]['Std']
info_text = f"""
BEST MODEL

Model: {model_name}

CV Score:
{best_score:.4f} ± {best_std:.4f}

Rank: #1/{len(cv_results)}
"""
ax2.text(0.5, 0.5, info_text, transform=ax2.transAxes,
        fontsize=11, verticalalignment='center', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# 3. Phân phối dự đoán
ax3 = fig.add_subplot(gs[1, 0])
pred_counts = submission['Survived'].value_counts()
colors_pred = ['#e74c3c', '#2ecc71']
ax3.pie([pred_counts.get(0, 0), pred_counts.get(1, 0)], 
       labels=['Died', 'Survived'],
       autopct='%1.1f%%', startangle=90, colors=colors_pred,
       textprops={'fontsize': 10, 'fontweight': 'bold'})
ax3.set_title('Test Set Predictions', fontsize=12, fontweight='bold')

# 4. Phân phối dữ liệu huấn luyện
ax4 = fig.add_subplot(gs[1, 1])
train_survived = (train_processed['Survived'] == 1).sum()
train_died = (train_processed['Survived'] == 0).sum()
ax4.bar(['Died', 'Survived'], [train_died, train_survived],
       color=['#e74c3c', '#2ecc71'], alpha=0.7)
ax4.set_title('Training Data Distribution', fontsize=12, fontweight='bold')
ax4.set_ylabel('Count', fontsize=10)
ax4.grid(axis='y', alpha=0.3)
for i, v in enumerate([train_died, train_survived]):
    ax4.text(i, v + 10, str(v), ha='center', fontweight='bold')

# 5. Thông tin về số lượng của tính năng
ax5 = fig.add_subplot(gs[1, 2])
n_features = len(train_processed.columns) - 1  # Exclude target
feature_info = f"""
FEATURES

Original: 11 columns
Engineered: {n_features} features

Key Features:
• Pclass
• Sex
• Age
• Fare
• FamilySize
• Title
• Has_Cabin
"""
ax5.axis('off')
ax5.text(0.1, 0.5, feature_info, transform=ax5.transAxes,
        fontsize=10, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# 6. Pipeline tổng quan
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')

pipeline_steps = [
    "1. EDA",
    "2. Missing Data",
    "3. Feature Engineering",
    "4. Encoding",
    "5. Model Training",
    "6. Hyperparameter Tuning" if has_tuning else "6. (No Tuning)",
    "7. Prediction",
    "8. Submission"
]

# Vẽ pipeline
for i, step in enumerate(pipeline_steps):
    x = (i + 0.5) / len(pipeline_steps)
    color = 'lightgreen' if i < len(pipeline_steps) - 1 else 'lightcoral'
    ax6.add_patch(plt.Rectangle((x - 0.05, 0.3), 0.1, 0.4, 
                               facecolor=color, edgecolor='black', linewidth=2))
    ax6.text(x, 0.5, step, ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    if i < len(pipeline_steps) - 1:
        ax6.arrow(x + 0.05, 0.5, 0.03, 0, head_width=0.1, head_length=0.01,
                 fc='black', ec='black')

ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)
ax6.set_title('ML Pipeline', fontsize=13, fontweight='bold', pad=20)

plt.suptitle('Titanic Survival Prediction - Project Summary', 
            fontsize=16, fontweight='bold', y=0.98)

plt.savefig('project_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Đã lưu biểu đồ: project_summary.png")

# ========== TẠO DETAILED REPORT ==========

print("\n" + "=" * 70)
print("TẠO DETAILED REPORT")
print("=" * 70)

report_content = f"""
{'=' * 80}
                    TITANIC SURVIVAL PREDICTION
                         FINAL REPORT
{'=' * 80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'=' * 80}
1. TỔNG QUAN DỰ ÁN
{'=' * 80}

Mục tiêu:
  Dự đoán sự sống sót của hành khách trên tàu Titanic bằng máy học

Dataset:
  • Training Set: {len(train_processed)} mẫu
  • Test Set: {len(test_processed)} mẫu
  • Features: {len(train_processed.columns) - 1} (không bao gồm mục tiêu)

{'=' * 80}
2. TIỀN XỬ LÝ DỮ LIỆU & KỸ THUẬT TÍNH NĂNG
{'=' * 80}

Xử lý dữ liệu bị thiếu:
  • Age: Điền giá trị median (trung vị) bởi biến Pclass và biến Sex
  • Cabin: Tạo biến Has_Cabin (1 nếu có Cabin, 0 nếu không)
  • Embarked: Điền giá trị phổ biến nhất (S) bằng mode
  • Fare: Điền giá trị trung vị từ biến Pclass

Kỹ thuật tính năng:
  • Title: Trích danh xưng từ tên (Mr, Mrs, Miss, Master, Rare)
  • FamilySize: SibSp + Parch + 1
  • IsAlone: 1 nếu FamilySize = 1, ngược lại 0
  • AgeGroup: Phân loại độ tuổi thành các nhóm
  • FareGroup: Phân loại giá vé thành các nhóm theo phân vị

Mã hóa biến:
  • Label Encoding: Sex, Title, AgeGroup, FareGroup
  • One-Hot Encoding: Embarked
  • Standardization: Age, Fare, SibSp, Parch, FamilySize

{'=' * 80}
3. MÔ HÌNH MÁY HỌC & ĐÁNH GIÁ
{'=' * 80}

Models Tested: {len(cv_results)}

Top 3 Mô hình (by CV Accuracy):
"""

for idx, row in cv_results.head(3).iterrows():
    report_content += f"\n  {idx+1}. {row['Model']:25} : {row['Mean_Accuracy']:.4f} ± {row['Std']:.4f}"

report_content += f"""

Selected Model: {model_name}
  • Cross-Validation Accuracy: {cv_results.iloc[0]['Mean_Accuracy']:.4f} ± {cv_results.iloc[0]['Std']:.4f}
  • Ranking: #1 out of {len(cv_results)} models

{'=' * 80}
4. ĐIỀU CHỈNH SIÊU THAM SỐ
{'=' * 80}
"""

if has_tuning:
    report_content += f"""
Method: Randomized/Grid Search with 5-Fold Cross-Validation

Best Parameters:
"""
    for param, value in best_params.items():
        report_content += f"  • {param}: {value}\n"
else:
    report_content += """
Status: Not performed (using default parameters)
"""

report_content += f"""
{'=' * 80}
5. DỰ ĐOÁN BỘ KIỂM TRA
{'=' * 80}

Total Predictions: {len(submission)}

Prediction Distribution:
  • Survived: {(submission['Survived'] == 1).sum()} ({(submission['Survived'] == 1).sum() / len(submission) * 100:.1f}%)
  • Died: {(submission['Survived'] == 0).sum()} ({(submission['Survived'] == 0).sum() / len(submission) * 100:.1f}%)

Comparison with Training Set:
  • Train Survival Rate: {(train_processed['Survived'] == 1).sum() / len(train_processed) * 100:.1f}%
  • Test Survival Rate: {(submission['Survived'] == 1).sum() / len(submission) * 100:.1f}%
  • Difference: {abs((submission['Survived'] == 1).sum() / len(submission) * 100 - (train_processed['Survived'] == 1).sum() / len(train_processed) * 100):.1f}%

{'=' * 80}
6. NHỮNG THÔNG TIN CHÍNH TỪ EDA
{'=' * 80}

Factors Influencing Survival:

1. Gender (Sex):
   • Female survival rate: ~74%
   • Male survival rate: ~19%
   → "Women and children first" policy clearly visible

2. Passenger Class (Pclass):
   • 1st Class survival: ~63%
   • 2nd Class survival: ~47%
   • 3rd Class survival: ~24%
   → Higher class had better survival chances

3. Age:
   • Children (<12) had higher survival rates
   • Elderly passengers had lower survival rates

4. Family Size:
   • Small families (2-4 members) had better survival
   • Solo travelers and large families struggled

5. Fare:
   • Higher fares correlated with survival
   • Reflected cabin location and class

{'=' * 80}
7. OUTPUT FILES
{'=' * 80}

Generated Files:
  ✓ submission.csv - Main Kaggle submission file
  ✓ train_processed.csv - Preprocessed training data
  ✓ test_processed.csv - Preprocessed test data
  ✓ best_model.pkl - Trained model (baseline)
  {'✓ tuned_model.pkl - Hyperparameter tuned model' if has_tuning else ''}
  {'✓ best_params.json - Best hyperparameters' if has_tuning else ''}
  ✓ cv_results.csv - Cross-validation results
  ✓ project_summary.png - Visual summary
  
Visualizations:
  ✓ missing_data_analysis.png
  ✓ target_variable_analysis.png
  ✓ categorical_variables_analysis.png
  ✓ numerical_variables_analysis.png
  ✓ survival_by_categorical.png
  ✓ survival_by_numerical.png
  ✓ interaction_analysis.png
  ✓ correlation_matrix.png
  ✓ model_comparison_cv.png
  ✓ feature_importance.png (if available)
  ✓ prediction_results.png

{'=' * 80}
8. ĐỀ XUẤT CẢI TIẾN
{'=' * 80}

Potential Enhancements:

1. Feature Engineering:
   • Extract more information from Name (nationality, etc.)
   • Create interaction features (e.g., Sex × Pclass)
   • Use Cabin deck information more effectively

2. Model Ensemble:
   • Voting Classifier (combine multiple models)
   • Stacking (use predictions as features)
   • Blending (weighted average of predictions)

3. Advanced Techniques:
   • Feature selection (remove low-importance features)
   • Advanced imputation (KNN, MICE)
   • Cross-validation with different strategies

4. External Data:
   • Historical information about Titanic
   • Crew member information
   • Cabin layout data

{'=' * 80}
9. KẾT LUẬN
{'=' * 80}

Summary:
  • Successfully built and evaluated {len(cv_results)} machine learning models
  • Selected {model_name} as the best performer
  • Achieved CV accuracy of {cv_results.iloc[0]['Mean_Accuracy']:.4f}
  • Generated predictions for {len(submission)} test samples
  
The analysis revealed that gender, passenger class, and age were the most
important factors in survival. The "women and children first" evacuation
policy is clearly evident in the data.

Next Steps:
  1. Submit predictions to Kaggle for evaluation
  2. Analyze Kaggle leaderboard score
  3. Iterate on feature engineering and model selection
  4. Consider ensemble methods for improved performance

{'=' * 80}
                         END OF REPORT
{'=' * 80}
"""

# Lưu report
with open('FINAL_REPORT.txt', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("✓ Đã lưu báo cáo chi tiết: FINAL_REPORT.txt")

# In ra màn hình
print("\n" + "=" * 70)
print("PREVIEW FINAL REPORT")
print("=" * 70)
print(report_content)

# ========== TẠO README FILE ==========

readme_content = f"""# Titanic Survival Prediction - Machine Learning Project

## Tổng quan dự án

Dự án này triển khai một quy trình học máy hoàn chỉnh để dự đoán sự sống sót của hành khách trên tàu Titanic.

**Mô hình tốt nhất:** {model_name}  
**Độ chính xác của xác thực chéo:** {cv_results.iloc[0]['Mean_Accuracy']:.4f} ± {cv_results.iloc[0]['Std']:.4f}

## Cấu trúc dự án

```
titanic-prediction/
├── train.csv                          # Dữ liệu huấn luyện ban đầu
├── test.csv                           # Dữ liệu thử nghiệm gốc
├── train_processed.csv                # Tiền xử lý dữ liệu huấn luyện
├── test_processed.csv                 # Tiền xử lý dữ liệu thử nghiệm
├── submission.csv                     # Dự đoán cuối cùng
├── best_model.pkl                     # Thuật toán mô hình tốt nhất
{'├── tuned_model.pkl                     # Mô hình điều chỉnh' if has_tuning else ''}
{'├── best_params.json                   # Siêu tham số tốt nhất' if has_tuning else ''}
├── cv_results.csv                     # Kết quả xác thực chéo
├── FINAL_REPORT.txt                   # Báo cáo chi tiết dự án
├── README.md                          # Hướng dẫn và tổng quan dự án
└── visualizations/                    # Thư mục chứa các biểu đồ
```

## Hướng dẫn thực thi

### Yêu cầu cài đặt
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Các bước thực thi

1. **Tải và Khám phá Dữ liệu**
```python
python load_data.py
```

2. **Phân tích Dữ liệu Thiếu**
```python
python missing_analysis.py
```

3. **Phân tích Đơn biến**
```python
python univariate_analysis.py
```

4. **Phân tích Đa biến**
```python
python bivariate_analysis.py
```

5. **Tiền xử lý Dữ liệu**
```python
python preprocessing.py
```

6. **Xây dựng và Huấn luyện Mô hình**
```python
python model.py
```

7. **Tối ưu hóa hyperparameters**
```python
python hyperparameter_tuning.py
```

8. **Dự đoán và Tạo file Kết quả**
```python
python prediction.py
```

9. **Báo cáo Tổng kết**
```python
python final.py
```

## Tóm tắt Kết quả

- **Các mô hình đã được Thử nghiệm:** {len(cv_results)}
- **Mô hình Tốt nhất:** {model_name}
- **Độ chính xác của CV:** {cv_results.iloc[0]['Mean_Accuracy']:.4f}
- **Kiểm tra dự đoán:** {len(submission)} samples

### Phân phối Dự đoán trên Test Set
- **Survived:** {(submission['Survived'] == 1).sum()} ({(submission['Survived'] == 1).sum() / len(submission) * 100:.1f}%)
- **Died:** {(submission['Survived'] == 0).sum()} ({(submission['Survived'] == 0).sum() / len(submission) * 100:.1f}%)

## Các Kỹ thuật Chính Sử Dụng

1. **Kỹ thuật tính năng:**
   - Trích xuất tiêu đề từ Tên
   - Tính toán quy mô gia đình
   - Phân nhóm tuổi
   - Phân loại giá vé
   - Chỉ báo khoang hành khách

2. **Tiền xử lý dữ liệu:**
   - Suy đoán giá trị bị thiếu
   - Mã hóa danh mục
   - Tỷ lệ tính năng

3. **Đánh giá mô hình:**
   - Kiểm định chéo phân tầng 5 lần
   - So sánh nhiều mô hình
   - Điều chỉnh siêu tham số

## Các Biểu đồ và Trực quan hóa

Tất cả hình ảnh trực quan được lưu dưới dạng tệp PNG:
- missing_data_analysis.png
- categorical_variables_analysis.png
- numerical_variables_analysis.png
- correlation_matrix.png
- model_comparison_cv.png
- prediction_results.png
- project_summary.png

## Khuyến nghị Cải tiến

1. Phương pháp tổng hợp (Voting, Stacking)
2. Thêm tính năng kỹ thuật
3. Kỹ thuật quy imputation nâng cao
4. Tích hợp dữ liệu bên ngoài

---
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("\n✓ Đã tạo README.md")

# ========== FINAL SUMMARY ==========

print("\n" + "=" * 70)
print("✓ HOÀN THÀNH BƯỚC 9: TẠO BÁO CÁO TỔNG KẾT")
print("=" * 70)

print(f"""
 DỰ ÁN ĐÃ HOÀN THÀNH THÀNH CÔNG!

Generated Files:
  • FINAL_REPORT.txt - Chi tiết đầy đủ về dự án
  • README.md - Hướng dẫn và tổng quan
  • project_summary.png - Visualization tổng hợp

Model Performance:
  • Best Model: {model_name}
  • CV Accuracy: {cv_results.iloc[0]['Mean_Accuracy']:.4f} ± {cv_results.iloc[0]['Std']:.4f}

Submission:
  • File: submission.csv
""")

print("=" * 70)