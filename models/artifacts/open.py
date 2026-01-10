import joblib
import pandas as pd
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 1. Đường dẫn file
model_path = "model.joblib"
extractor_path = "extractor.joblib"
prep_path = "preprocessor.joblib"

# 2. Load các object lên bộ nhớ
print("Đang load models...")
loaded_model = joblib.load(model_path)          # Đây là XGBoost model
loaded_extractor = joblib.load(extractor_path)  # Đây là Feature Extractor
loaded_prep = joblib.load(prep_path)            # Đây là Preprocessor

print("Đã load thành công!")
print("Loại model:", type(loaded_model))

# 3. Ví dụ cách dùng (Inference)
# Giả sử bạn có dữ liệu mới (ví dụ load từ file test)
# df_new = pd.read_csv("data/processed/test.csv").head(1)
# X_new = df_new.drop(columns=['label']) 

# X_feat = loaded_extractor.transform(X_new)
# X_proc = loaded_prep.transform(X_feat)
# prediction = loaded_model.predict(X_proc)

# print("Dự đoán:", prediction)