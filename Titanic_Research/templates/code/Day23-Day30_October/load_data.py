# ===================================================================
# BƯỚC 1: LOAD DỮ LIỆU VÀ KIỂM TRA THÔNG TIN CƠ BẢN
# ===================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Cấu hình hiển thị
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load dữ liệu
print("=" * 70)
print("LOAD DỮ LIỆU")
print("=" * 70)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"\n✓ Train set shape: {train.shape}")
print(f"✓ Test set shape: {test.shape}")

# Hiển thị 5 dòng đầu
print("\n" + "=" * 70)
print("5 DÒNG ĐẦU TIÊN CỦA TRAIN SET")
print("=" * 70)
print(train.head())

# Thông tin cơ bản về dữ liệu
print("\n" + "=" * 70)
print("THÔNG TIN CƠ BẢN VỀ TRAIN SET")
print("=" * 70)
print(train.info())

# Thống kê mô tả cho các biến số
print("\n" + "=" * 70)
print("THỐNG KÊ MÔ TẢ CÁC BIẾN SỐ")
print("=" * 70)
print(train.describe())

# Thống kê mô tả cho các biến phân loại
print("\n" + "=" * 70)
print("THỐNG KÊ MÔ TẢ CÁC BIẾN PHÂN LOẠI")
print("=" * 70)
print(train.describe(include=['object']))

# Kiểm tra giá trị duy nhất của mỗi cột
print("\n" + "=" * 70)
print("SỐ LƯỢNG GIÁ TRỊ DUY NHẤT CỦA MỖI CỘT")
print("=" * 70)
for col in train.columns:
    print(f"{col:15} : {train[col].nunique():5} giá trị duy nhất")

print("\n✓ Hoàn thành bước 1: Load và kiểm tra dữ liệu cơ bản")