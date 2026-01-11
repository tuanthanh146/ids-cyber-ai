# Hướng Dẫn Sử Dụng Hệ Thống IDS AI (SET.MD)

Tài liệu này mô tả chi tiết cách hoạt động và hướng dẫn sử dụng hệ thống Phát hiện Xâm nhập (IDS) sử dụng Trí tuệ Nhân tạo (AI).

## 1. Giới Thiệu
Hệ thống này sử dụng các thuật toán máy học tiên tiến (LightGBM, XGBoost) để phân tích lưu lượng mạng và phát hiện các cuộc tấn công mạng trong thời gian thực. Hệ thống bao gồm các thành phần:
- **Training**: Huấn luyện mô hình từ dữ liệu lịch sử.
- **Inference API**: API dự đoán thời gian thực.
- **Dashboard**: Giao diện trực quan hóa cảnh báo và giải thích AI (XAI).

## 2. Luồng Hoạt Động Của Dự Án (Project Workflow)

Hệ thống hoạt động theo quy trình khép kín từ thu thập dữ liệu đến cảnh báo:

1.  **Dữ liệu đầu vào (Input)**:
    - Hệ thống tiếp nhận các gói tin mạng (Packet) từ tệp PCAP hoặc luồng mạng trực tiếp.
    - Script `process_pcap.py` chịu trách nhiệm trích xuất thông tin thô (IP nguồn/đích, Port, Protocol, Flags...) từ các gói tin này.

2.  **Tiền xử lý (Preprocessing)**:
    - Dữ liệu thô được làm sạch, xử lý giá trị thiếu (NaN) và chuẩn hóa.
    - Các đặc trưng mới được tạo ra (Feature Engineering) để tăng độ chính xác cho mô hình.

3.  **Phân tích AI (Inference Engine)**:
    - **API Server** (`inference/api.py`) nhận dữ liệu gói tin qua endpoint `/predict`.
    - Dữ liệu được đưa qua mô hình Ensemble (LightGBM + XGBoost) đã huấn luyện.
    - Mô hình tính toán:
        - **Dự đoán nhãn**: Tấn công hay Bình thường.
        - **Điểm rủi ro (Risk Score)**: Mức độ nguy hiểm của kết nối.
        - **SHAP Values**: Giải thích lý do tại sao kết nối bị đánh dấu là tấn công.

4.  **Lưu trữ & Cảnh báo (Storage & Alerting)**:
    - Nếu phát hiện tấn công, hệ thống ghi log vào `logs/alerts.jsonl`.
    - Dashboard đọc log này để hiển thị cảnh báo thời gian thực cho quản trị viên.

## 3. Cách Hoạt Động Của Mô Hình

Mô hình hoạt động theo quy trình 6 bước (như trong `scripts/train_smart.py`):

1.  **Thu thập & Làm sạch dữ liệu**:
    - Dữ liệu đầu vào là file CSV (được chuyển đổi từ PCAP mạng).
    - Tự động xử lý giá trị thiếu, vô cực và chuẩn hóa dữ liệu số.

2.  **Lựa chọn đặc trưng thông minh (Feature Selection)**:
    - Sử dụng LightGBM để đánh giá độ quan trọng của từng đặc trưng (feature importance).
    - Chỉ giữ lại **Top-N** (mặc định 30) đặc trưng quan trọng nhất để tăng tốc độ và độ chính xác.

3.  **Tối ưu tham số (Hyperparameter Tuning)**:
    - Sử dụng **Optuna** để tự động tìm bộ tham số tốt nhất cho LightGBM và XGBoost.

4.  **Huấn luyện Ensemble (Kết hợp mô hình)**:
    - Huấn luyện song song **LightGBM** và **XGBoost**.
    - Kết hợp kết quả từ cả hai mô hình (Ensemble) để đưa ra quyết định cuối cùng, giúp giảm thiểu báo động giả (False Positive).

5.  **Giải thích AI (Explainable AI - SHAP)**:
    - Sử dụng thư viện **SHAP** để giải thích *tại sao* mô hình lại coi một kết nối là tấn công.
    - Cung cấp thông tin chi tiết về các đặc trưng đóng góp nhiều nhất vào quyết định (ví dụ: `dst_port`, `flow_duration`).

## 3. Hướng Dẫn Sử Dụng

Bạn có thể tương tác với hệ thống thông qua `main.py`.

### A. Xử lý dữ liệu (Process PCAP)
Chuyển đổi file PCAP thô thành CSV để huấn luyện hoặc kiểm tra.
```bash
python main.py process --input data/traffic.pcap --output data/traffic.csv --label 1
```
*   `--label 1`: Gán nhãn là tấn công (dùng 0 cho dữ liệu sạch).

### B. Huấn luyện mô hình (Train)
Huấn luyện mô hình mới từ dữ liệu CSV.
```bash
python main.py train --data data/traffic.csv --output runs/my_model --demo
```
*   Hệ thống sẽ tự động chạy Feature Selection, Tuning và lưu mô hình vào `runs/my_model`.

### C. Chạy API Dự đoán (Serve)
Khởi chạy API để nhận dữ liệu và dự đoán thời gian thực.
```bash
python main.py serve --host 0.0.0.0 --port 8000
```

### D. Khởi chạy Dashboard
Xem các cảnh báo và biểu đồ trực quan.
```bash
python main.py dashboard
```
*   Dashboard sẽ chạy tại `http://localhost:8501`.

## 4. Triển Khai Lên Streamlit Cloud

Để đưa Dashboard lên Internet:
1.  Đẩy code lên GitHub (bao gồm file `streamlit_app.py` mới tạo).
2.  Truy cập [share.streamlit.io](https://share.streamlit.io/).
3.  Chọn repo và deploy file `streamlit_app.py`.
4.  **Lưu ý**: Trên Cloud, hệ thống sẽ sử dụng dữ liệu mẫu (`data/sample_alerts.jsonl`) vì không thể truy cập log trên máy tính cá nhân của bạn.
