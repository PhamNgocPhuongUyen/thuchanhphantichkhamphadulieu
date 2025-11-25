# ===================================================================
# BƯỚC 5: TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)
# ===================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dữ liệu
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Lưu PassengerId của test set để submit sau
test_ids = test['PassengerId'].copy()

print("=" * 70)
print("BƯỚC 5: TIỀN XỬ LÝ DỮ LIỆU")
print("=" * 70)

# ========== KẾT HỢP TRAIN VÀ TEST ĐỂ XỬ LÝ ĐỒNG NHẤT ==========

print("\nKết hợp train và test set...")
# Lưu target variable
y_train = train['Survived'].copy()

# Xóa cột Survived khỏi train để kết hợp với test
train_temp = train.drop('Survived', axis=1)

# Kết hợp
all_data = pd.concat([train_temp, test], axis=0, sort=False).reset_index(drop=True)
print(f"✓ Shape sau khi kết hợp: {all_data.shape}")

# ========== XỬ LÝ MISSING VALUES ==========

print("\n" + "=" * 70)
print("XỬ LÝ MISSING VALUES")
print("=" * 70)

# 1. AGE: Impute bằng median theo Pclass và Sex
print("\n1. Xử lý Age...")
all_data['Age'] = all_data.groupby(['Pclass', 'Sex'])['Age'].transform(
    lambda x: x.fillna(x.median())
)
print(f"   ✓ Missing Age sau xử lý: {all_data['Age'].isnull().sum()}")

# 2. EMBARKED: Impute bằng mode
print("\n2. Xử lý Embarked...")
all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].mode()[0])
print(f"   ✓ Missing Embarked sau xử lý: {all_data['Embarked'].isnull().sum()}")

# 3. FARE: Impute bằng median theo Pclass
print("\n3. Xử lý Fare...")
all_data['Fare'] = all_data.groupby('Pclass')['Fare'].transform(
    lambda x: x.fillna(x.median())
)
print(f"   ✓ Missing Fare sau xử lý: {all_data['Fare'].isnull().sum()}")

# 4. CABIN: Tạo biến has_cabin
print("\n4. Xử lý Cabin...")
all_data['Has_Cabin'] = all_data['Cabin'].notna().astype(int)
print(f"   ✓ Đã tạo biến Has_Cabin")

# ========== FEATURE ENGINEERING ==========

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

# 1. TITLE: Extract từ Name
print("\n1. Tạo biến Title từ Name...")
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Nhóm các title hiếm
title_mapping = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs',
    'Master': 'Master',
    'Dr': 'Rare',
    'Rev': 'Rare',
    'Col': 'Rare',
    'Major': 'Rare',
    'Mlle': 'Miss',
    'Countess': 'Rare',
    'Ms': 'Miss',
    'Lady': 'Rare',
    'Jonkheer': 'Rare',
    'Don': 'Rare',
    'Dona': 'Rare',
    'Mme': 'Mrs',
    'Capt': 'Rare',
    'Sir': 'Rare'
}
all_data['Title'] = all_data['Title'].map(title_mapping)
print(f"   ✓ Title distribution:")
print(all_data['Title'].value_counts())

# 2. FAMILY SIZE: SibSp + Parch + 1
print("\n2. Tạo biến FamilySize...")
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
print(f"   ✓ FamilySize range: {all_data['FamilySize'].min()} - {all_data['FamilySize'].max()}")

# 3. IS ALONE: Người đi một mình
print("\n3. Tạo biến IsAlone...")
all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)
print(f"   ✓ Số người đi một mình: {all_data['IsAlone'].sum()}")

# 4. AGE GROUP: Phân nhóm tuổi
print("\n4. Tạo biến AgeGroup...")
all_data['AgeGroup'] = pd.cut(all_data['Age'], 
                              bins=[0, 12, 18, 35, 60, 100],
                              labels=['Child', 'Teenager', 'Adult', 'Middle_Age', 'Senior'])
print(f"   ✓ AgeGroup distribution:")
print(all_data['AgeGroup'].value_counts())

# 5. FARE GROUP: Phân nhóm giá vé
print("\n5. Tạo biến FareGroup...")
all_data['FareGroup'] = pd.qcut(all_data['Fare'], q=4, 
                                labels=['Low', 'Medium', 'High', 'Very_High'],
                                duplicates='drop')
print(f"   ✓ FareGroup distribution:")
print(all_data['FareGroup'].value_counts())

# ========== ENCODING CATEGORICAL VARIABLES ==========

print("\n" + "=" * 70)
print("ENCODING CATEGORICAL VARIABLES")
print("=" * 70)

# 1. Label Encoding cho biến ordinal và binary
print("\n1. Label Encoding...")

# Sex: female=1, male=0
all_data['Sex'] = all_data['Sex'].map({'female': 1, 'male': 0})
print(f"   ✓ Sex encoded")

# Title
title_encode = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
all_data['Title'] = all_data['Title'].map(title_encode)
print(f"   ✓ Title encoded")

# AgeGroup
age_encode = {'Child': 0, 'Teenager': 1, 'Adult': 2, 'Middle_Age': 3, 'Senior': 4}
all_data['AgeGroup'] = all_data['AgeGroup'].map(age_encode)
print(f"   ✓ AgeGroup encoded")

# FareGroup
fare_encode = {'Low': 0, 'Medium': 1, 'High': 2, 'Very_High': 3}
all_data['FareGroup'] = all_data['FareGroup'].map(fare_encode)
print(f"   ✓ FareGroup encoded")

# 2. One-Hot Encoding cho Embarked
print("\n2. One-Hot Encoding cho Embarked...")
embarked_dummies = pd.get_dummies(all_data['Embarked'], prefix='Embarked', drop_first=True)
all_data = pd.concat([all_data, embarked_dummies], axis=1)
print(f"   ✓ Embarked one-hot encoded: {embarked_dummies.columns.tolist()}")

# ========== XÓA CÁC CỘT KHÔNG CẦN THIẾT ==========

print("\n" + "=" * 70)
print("XÓA CÁC CỘT KHÔNG CẦN THIẾT")
print("=" * 70)

columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']
all_data = all_data.drop(columns_to_drop, axis=1)
print(f"✓ Đã xóa các cột: {columns_to_drop}")
print(f"✓ Shape sau khi xóa: {all_data.shape}")

# ========== SCALING NUMERICAL FEATURES ==========

print("\n" + "=" * 70)
print("SCALING NUMERICAL FEATURES")
print("=" * 70)

# Các cột cần scale
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']

scaler = StandardScaler()
all_data[numerical_features] = scaler.fit_transform(all_data[numerical_features])
print(f"✓ Đã scale các features: {numerical_features}")

# ========== TÁCH LẠI TRAIN VÀ TEST ==========

print("\n" + "=" * 70)
print("TÁCH LẠI TRAIN VÀ TEST SET")
print("=" * 70)

# Tách dựa vào số lượng ban đầu
train_len = len(train_temp)

X_train = all_data[:train_len].copy()
X_test = all_data[train_len:].copy()

print(f"✓ X_train shape: {X_train.shape}")
print(f"✓ X_test shape: {X_test.shape}")
print(f"✓ y_train shape: {y_train.shape}")

# ========== LƯU DỮ LIỆU ĐÃ XỬ LÝ ==========

print("\n" + "=" * 70)
print("LƯU DỮ LIỆU ĐÃ XỬ LÝ")
print("=" * 70)

# Lưu train set
train_processed = X_train.copy()
train_processed['Survived'] = y_train
train_processed.to_csv('train_processed.csv', index=False)
print("✓ Đã lưu: train_processed.csv")

# Lưu test set
test_processed = X_test.copy()
test_processed['PassengerId'] = test_ids.values
test_processed.to_csv('test_processed.csv', index=False)
print("✓ Đã lưu: test_processed.csv")

# Hiển thị các features cuối cùng
print("\n" + "=" * 70)
print("CÁC FEATURES SAU TIỀN XỬ LÝ")
print("=" * 70)
print(f"Số lượng features: {X_train.shape[1]}")
print(f"\nDanh sách features:")
for i, col in enumerate(X_train.columns, 1):
    print(f"{i:2}. {col}")

# Kiểm tra missing values cuối cùng
print("\n" + "=" * 70)
print("KIỂM TRA MISSING VALUES CUỐI CÙNG")
print("=" * 70)
print(f"Train set: {X_train.isnull().sum().sum()} missing values")
print(f"Test set: {X_test.isnull().sum().sum()} missing values")

print("\n" + "=" * 70)
print("✓ HOÀN THÀNH BƯỚC 5: TIỀN XỬ LÝ DỮ LIỆU")
print("=" * 70)