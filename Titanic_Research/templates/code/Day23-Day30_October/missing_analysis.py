# ===================================================================
# BƯỚC 2: PHÂN TÍCH DỮ LIỆU THIẾU (MISSING DATA)
# ===================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dữ liệu (nếu chưa load)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("=" * 70)
print("PHÂN TÍCH DỮ LIỆU THIẾU")
print("=" * 70)

# Hàm kiểm tra missing data
def missing_data_analysis(df, dataset_name='Dataset'):
    """
    Phân tích dữ liệu thiếu trong dataframe
    """
    # Tính số lượng và tỷ lệ missing
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    # Tạo dataframe kết quả
    missing_df = pd.DataFrame({
        'Column': missing_count.index,
        'Missing_Count': missing_count.values,
        'Missing_Percent': missing_percent.values
    })
    
    # Chỉ lấy các cột có missing data
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Percent', ascending=False
    ).reset_index(drop=True)
    
    print(f"\n{dataset_name}:")
    print("-" * 70)
    if len(missing_df) == 0:
        print("✓ Không có dữ liệu thiếu!")
    else:
        print(missing_df.to_string(index=False))
    
    return missing_df

# Phân tích train set
train_missing = missing_data_analysis(train, 'TRAIN SET')

# Phân tích test set
test_missing = missing_data_analysis(test, 'TEST SET')

# Visualize missing data
print("\n" + "=" * 70)
print("TRỰC QUAN HÓA DỮ LIỆU THIẾU")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Train set
if len(train_missing) > 0:
    axes[0].barh(train_missing['Column'], train_missing['Missing_Percent'], color='coral')
    axes[0].set_xlabel('Tỷ lệ thiếu (%)', fontsize=12)
    axes[0].set_title('Missing Data - Train Set', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Thêm giá trị lên cột
    for i, v in enumerate(train_missing['Missing_Percent']):
        axes[0].text(v + 1, i, f'{v:.1f}%', va='center')
else:
    axes[0].text(0.5, 0.5, 'Không có dữ liệu thiếu', 
                ha='center', va='center', fontsize=14)
    axes[0].set_xlim(0, 1)

# Test set
if len(test_missing) > 0:
    axes[1].barh(test_missing['Column'], test_missing['Missing_Percent'], color='skyblue')
    axes[1].set_xlabel('Tỷ lệ thiếu (%)', fontsize=12)
    axes[1].set_title('Missing Data - Test Set', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Thêm giá trị lên cột
    for i, v in enumerate(test_missing['Missing_Percent']):
        axes[1].text(v + 1, i, f'{v:.1f}%', va='center')
else:
    axes[1].text(0.5, 0.5, 'Không có dữ liệu thiếu', 
                ha='center', va='center', fontsize=14)
    axes[1].set_xlim(0, 1)

plt.tight_layout()
plt.savefig('missing_data_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Đã lưu biểu đồ: missing_data_analysis.png")

# Nhận xét và đề xuất
print("\n" + "=" * 70)
print("NHẬN XÉT VÀ ĐỀ XUẤT XỬ LÝ")
print("=" * 70)
print("""
1. AGE (Tuổi):
   - Train: ~20% thiếu, Test: ~21% thiếu
   - Đề xuất: Impute bằng median theo Pclass và Sex
   
2. CABIN (Cabin):
   - Train: ~77% thiếu, Test: ~78% thiếu
   - Đề xuất: Tạo biến has_cabin (0/1) hoặc extract deck letter
   
3. EMBARKED (Cảng đi):
   - Train: 2 giá trị thiếu
   - Đề xuất: Impute bằng mode (giá trị phổ biến nhất)
   
4. FARE (Giá vé):
   - Test: 1 giá trị thiếu
   - Đề xuất: Impute bằng median theo Pclass
""")

print("\n✓ Hoàn thành bước 2: Phân tích dữ liệu thiếu")