# ===================================================================
# BƯỚC 7: HYPERPARAMETER TUNING
# ===================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("BƯỚC 7: HYPERPARAMETER TUNING")
print("=" * 70)

# ========== LOAD DỮ LIỆU VÀ MÔ HÌNH ==========

print("\nLoad dữ liệu và thông tin mô hình tốt nhất...")
train_processed = pd.read_csv('train_processed.csv')

X_train = train_processed.drop('Survived', axis=1)
y_train = train_processed['Survived']

# Đọc tên mô hình tốt nhất
with open('best_model_name.txt', 'r') as f:
    best_model_name = f.read().strip()

print(f"✓ Mô hình tốt nhất từ bước trước: {best_model_name}")

# ========== ĐỊNH NGHĨA PARAMETER GRIDS ==========

print("\n" + "=" * 70)
print("ĐỊNH NGHĨA PARAMETER GRIDS")
print("=" * 70)

# Parameter grids cho từng mô hình
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },
    
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 0.9, 1.0]
    },
    
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
}

# ========== CHỌN PHƯƠNG PHÁP TUNING ==========

# Sử dụng RandomizedSearchCV cho các mô hình phức tạp
# GridSearchCV cho các mô hình đơn giản

use_randomized = best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']

if use_randomized:
    print(f"\nSử dụng RandomizedSearchCV cho {best_model_name}")
    print("(Tìm kiếm ngẫu nhiên để tiết kiệm thời gian)")
    n_iter = 50  # Số lần thử ngẫu nhiên
else:
    print(f"\nSử dụng GridSearchCV cho {best_model_name}")
    print("(Tìm kiếm toàn bộ grid)")

# ========== KHỞI TẠO MÔ HÌNH VÀ SEARCH ==========

print("\n" + "=" * 70)
print("BẮT ĐẦU HYPERPARAMETER TUNING")
print("=" * 70)

# Khởi tạo mô hình cơ bản
if best_model_name == 'Random Forest':
    base_model = RandomForestClassifier(random_state=42)
elif best_model_name == 'Gradient Boosting':
    base_model = GradientBoostingClassifier(random_state=42)
elif best_model_name == 'XGBoost':
    base_model = XGBClassifier(random_state=42, eval_metric='logloss')
elif best_model_name == 'Logistic Regression':
    base_model = LogisticRegression(random_state=42, max_iter=1000)
elif best_model_name == 'SVM':
    base_model = SVC(random_state=42, probability=True)
else:
    print(f"⚠ Không có parameter grid cho {best_model_name}")
    print("Sử dụng mô hình với hyperparameters mặc định")
    base_model = None

# Thực hiện tuning nếu có parameter grid
if best_model_name in param_grids and base_model is not None:
    param_grid = param_grids[best_model_name]
    
    print(f"\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Cấu hình cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Thực hiện search
    print(f"\nĐang tìm kiếm hyperparameters tốt nhất...")
    print("(Quá trình này có thể mất vài phút...)")
    
    if use_randomized:
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    else:
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
    
    # Fit search
    search.fit(X_train, y_train)
    
    # ========== KẾT QUẢ TUNING ==========
    
    print("\n" + "=" * 70)
    print("KẾT QUẢ HYPERPARAMETER TUNING")
    print("=" * 70)
    
    print(f"\nBest Score (CV Accuracy): {search.best_score_:.4f}")
    print(f"\nBest Parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    
    # So sánh với mô hình baseline
    with open('best_model.pkl', 'rb') as f:
        baseline_model = pickle.load(f)
    
    from sklearn.model_selection import cross_val_score
    baseline_scores = cross_val_score(baseline_model, X_train, y_train, cv=cv, scoring='accuracy')
    baseline_mean = baseline_scores.mean()
    
    print(f"\n" + "-" * 70)
    print("SO SÁNH VỚI BASELINE")
    print("-" * 70)
    print(f"Baseline (default params)  : {baseline_mean:.4f}")
    print(f"Tuned (best params)        : {search.best_score_:.4f}")
    print(f"Improvement                : {(search.best_score_ - baseline_mean):.4f} ({((search.best_score_ - baseline_mean)/baseline_mean * 100):.2f}%)")
    
    # ========== VISUALIZE TOP RESULTS ==========
    
    print("\n" + "=" * 70)
    print("TRỰC QUAN HÓA KẾT QUẢ")
    print("=" * 70)
    
    # Lấy kết quả từ cv_results_
    results_df = pd.DataFrame(search.cv_results_)
    
    # Top 10 configurations
    top_results = results_df.nlargest(10, 'mean_test_score')[
        ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    ]
    
    print("\nTop 10 Configurations:")
    for idx, row in top_results.iterrows():
        print(f"\nRank {row['rank_test_score']}:")
        print(f"  Score: {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
        print(f"  Params: {row['params']}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Top 20 configurations
    top20 = results_df.nlargest(20, 'mean_test_score')
    x_pos = np.arange(len(top20))
    
    axes[0].barh(x_pos, top20['mean_test_score'], 
                xerr=top20['std_test_score'], color='steelblue', alpha=0.7)
    axes[0].set_yticks(x_pos)
    axes[0].set_yticklabels([f"Config {i+1}" for i in range(len(top20))])
    axes[0].set_xlabel('CV Accuracy', fontsize=12)
    axes[0].set_title('Top 20 Hyperparameter Configurations', 
                     fontsize=14, fontweight='bold')
    axes[0].axvline(x=baseline_mean, color='red', linestyle='--', 
                   linewidth=2, label='Baseline')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].invert_yaxis()
    
    # Plot 2: Score distribution
    axes[1].hist(results_df['mean_test_score'], bins=30, 
                color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(search.best_score_, color='green', linestyle='--', 
                   linewidth=2, label=f'Best: {search.best_score_:.4f}')
    axes[1].axvline(baseline_mean, color='red', linestyle='--', 
                   linewidth=2, label=f'Baseline: {baseline_mean:.4f}')
    axes[1].set_xlabel('CV Accuracy', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of CV Scores', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Đã lưu biểu đồ: hyperparameter_tuning_results.png")
    
    # ========== LƯU MÔ HÌNH ĐÃ TUNE ==========
    
    print("\n" + "=" * 70)
    print("LƯU MÔ HÌNH ĐÃ TUNE")
    print("=" * 70)
    
    # Lưu mô hình tốt nhất
    tuned_model = search.best_estimator_
    
    with open('tuned_model.pkl', 'wb') as f:
        pickle.dump(tuned_model, f)
    print("✓ Đã lưu mô hình: tuned_model.pkl")
    
    # Lưu best parameters
    import json
    with open('best_params.json', 'w') as f:
        # Convert numpy types to Python types
        params_json = {}
        for k, v in search.best_params_.items():
            if isinstance(v, np.integer):
                params_json[k] = int(v)
            elif isinstance(v, np.floating):
                params_json[k] = float(v)
            else:
                params_json[k] = v
        json.dump(params_json, f, indent=4)
    print("✓ Đã lưu best parameters: best_params.json")
    
    # Lưu tuning results
    top_results.to_csv('tuning_results.csv', index=False)
    print("✓ Đã lưu tuning results: tuning_results.csv")
    
    print("\n" + "=" * 70)
    print("✓ HOÀN THÀNH BƯỚC 7: HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"\nMô hình: {best_model_name}")
    print(f"Best CV Score: {search.best_score_:.4f}")
    print(f"Improvement: +{((search.best_score_ - baseline_mean)/baseline_mean * 100):.2f}%")
    
else:
    print("\n⚠ Không thực hiện tuning cho mô hình này")
    print("Sử dụng mô hình với hyperparameters mặc định từ bước 6")