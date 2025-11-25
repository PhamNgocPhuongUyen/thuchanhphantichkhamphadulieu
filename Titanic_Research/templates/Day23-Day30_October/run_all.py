import subprocess
import sys

# Danh sách các file cần chạy theo thứ tự
files = [
    'load_data.py',
    'missing_analysis.py',
    'univariate_analysis.py',
    'bivariate_analysis.py',
    'data_preprocessing.py',
    'model.py',
    'hyperparameter_tuning.py',
    'predictions.py',
    'final.py'
]

print("=" * 70)
print("BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE")
print("=" * 70)

for i, file in enumerate(files, 1):
    print(f"\n[{i}/{len(files)}] Đang chạy: {file}")
    print("-" * 70)
    
    result = subprocess.run([sys.executable, file], 
                          capture_output=False, 
                          text=True)
    
    if result.returncode != 0:
        print(f"❌ LỖI khi chạy {file}")
        break
    else:
        print(f"✅ Hoàn thành {file}")

print("\n" + "=" * 70)
print("✅ HOÀN THÀNH TOÀN BỘ PIPELINE!")
print("=" * 70)