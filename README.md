# Hệ thống Phát hiện Xâm nhập (IDS) AI Thời gian thực

Framework IDS AI chuẩn Production bằng Python, tích hợp Huấn luyện Thông minh (Ensemble, Optuna, SHAP) và Inference Thời gian thực.

## Cấu trúc Dự án
```text
ai_ids_project/
├── configs/                # Cấu hình (YAML)
├── data_ingestion/         # Stream & Batch Loaders
├── feature_extraction/     # Stateful Feature Engineering (Đảm bảo nhất quán Train/Serve)
├── preprocessing/          # Làm sạch & Chuẩn hóa dữ liệu
├── models/                 # Wrappers cho XGBoost/LightGBM & Artifacts
├── inference/              # FastAPI Inference Engine
├── dashboard/              # Streamlit Monitoring UI
├── utils/                  # Loggers & Serialization
└── scripts/                # Các pipeline huấn luyện
    ├── train_smart.py      # Huấn luyện Nâng cao (Ensemble + XAI)
    ├── train.py            # Huấn luyện Cơ bản
    └── process_pcap.py     # Chuyển đổi PCAP sang CSV
```

## Cài đặt

1.  **Cài đặt Thư viện**:
    ```bash
    pip install -r requirements.txt
    pip install optuna shap xgboost lightgbm
    ```

2.  **Chuẩn bị Dữ liệu**:
    Chuyển đổi file `.pcap` thô sang CSV để train:
    ```bash
    python main.py process --input data/raw/traffic.pcap --output data/processed/traffic.csv --label 0
    ```
    *Lưu ý: File CSV cần có các cột như `ts`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `proto`, `service`, `conn_state`, `duration`, `orig_pkts`, v.v.*

## Huấn luyện Nâng cao (`train_smart.py`)

Chuẩn mực mới cho việc huấn luyện model hiệu năng cao.

### Tính năng
*   **Dual Ensemble**: Train song song LightGBM và XGBoost, sau đó gộp kết quả dự đoán (Average Voting).
*   **Feature Selection**: Tự động chọn ra Top-K features quan trọng nhất.
*   **Optuna Tuning**: Tự động tối ưu hóa siêu tham số (Hyperparameter Optimization).
*   **Explainable AI (SHAP)**: Giải thích *lý do* tại sao model đưa ra quyết định (Global Importance & Local Explanation).

### Ví dụ Sử dụng

#### 1. Train Nhanh (Binary Classification)
Train nhanh model với tính năng tự động chọn lọc features (Top 20 features).
```bash
python scripts/train_smart.py --train_csv data/processed/train.csv --test_csv data/processed/test.csv --task binary --top_k 20 --outdir experiments/binary_fast
```

#### 2. Train chuẩn Production (Full Pipeline)
Sử dụng cho model tốt nhất để deploy.
- **Tune**: Có (Optuna)
- **Trials**: 30
- **Ensemble**: Có
- **XAI**: Có
```bash
python scripts/train_smart.py --train_csv data/processed/train.csv --test_csv data/processed/test.csv --task binary --top_k 20 --tune 1 --n_trials 30 --outdir experiments/production_v1
```

#### 3. Phân loại Đa lớp (Multiclass Classification)
Phát hiện cụ thể từng loại tấn công (DoS, PortScan, v.v.).
```bash
python scripts/train_smart.py --train_csv data/processed/train.csv --test_csv data/processed/test.csv --task multiclass --top_k 30 --tune 1 --outdir experiments/multiclass_v1
```

### Kết quả đầu ra (Artifacts)
Kiểm tra thư mục `--outdir` (ví dụ `experiments/production_v1/`) để xem:
*   `models/ensemble_bundle.joblib`: Model cuối cùng đã sẵn sàng deploy.
*   `reports/model_comparison.csv`: Bảng so sánh hiệu năng (LGBM vs XGB vs Ensemble).
*   `reports/shap_importance.csv`: Danh sách độ quan trọng của các features.
*   `reports/shap_sample_explanations.json`: Giải thích chi tiết cho các mẫu test.

## Inference & Giám sát

1.  **Khởi chạy API Inference**:
    ```bash
    python main.py serve
    ```
    Swagger UI: `http://localhost:8000/docs`

2.  **Khởi chạy Dashboard**:
    ```bash
    python main.py dashboard
    ```
    Truy cập tại: `http://localhost:8501`

## Tính năng Chính
-   **Nhất quán**: Logic Feature Extraction giống hệt nhau giữa Training và Inference.
-   **Hiệu năng**: Ensemble LightGBM/XGBoost + FastAPI cho tốc độ phản hồi cực nhanh.
-   **Minh bạch**: Tích hợp phân tích SHAP trực tiếp vào pipeline.
