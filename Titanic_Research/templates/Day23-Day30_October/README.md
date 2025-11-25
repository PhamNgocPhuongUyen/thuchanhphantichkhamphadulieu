# Titanic Survival Prediction - Machine Learning Project

## Tổng quan dự án

Dự án này triển khai một quy trình học máy hoàn chỉnh để dự đoán sự sống sót của hành khách trên tàu Titanic.

**Mô hình tốt nhất:** Gradient Boosting  
**Độ chính xác của xác thực chéo:** 0.8361 ± 0.0182

## Cấu trúc dự án

```
titanic-prediction/
├── train.csv                          # Dữ liệu huấn luyện ban đầu
├── test.csv                           # Dữ liệu thử nghiệm gốc
├── train_processed.csv                # Tiền xử lý dữ liệu huấn luyện
├── test_processed.csv                 # Tiền xử lý dữ liệu thử nghiệm
├── submission.csv                     # Dự đoán cuối cùng
├── best_model.pkl                     # Thuật toán mô hình tốt nhất
├── tuned_model.pkl                     # Mô hình điều chỉnh
├── best_params.json                   # Siêu tham số tốt nhất
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

- **Các mô hình đã được Thử nghiệm:** 8
- **Mô hình Tốt nhất:** Gradient Boosting
- **Độ chính xác của CV:** 0.8361
- **Kiểm tra dự đoán:** 418 samples

### Phân phối Dự đoán trên Test Set
- **Survived:** 150 (35.9%)
- **Died:** 268 (64.1%)

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
**Generated:** 2025-11-02 09:22:09
