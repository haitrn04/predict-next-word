# Hướng dẫn Nhanh - Next Word Prediction

## Bắt đầu nhanh trong 3 bước

### Bước 1: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Hoặc cài thủ công:

```bash
pip install datasets underthesea numpy jupyter
```

### Bước 2: Chạy các notebooks theo thứ tự

#### 2.1. Tiền xử lý dữ liệu

```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

Chạy tất cả các cell (Cell → Run All). Notebook này sẽ:
- Tải 10,000 dòng từ VTSNLP dataset
- Tiền xử lý và tokenize
- Lưu vào `data/processed/`

**Thời gian:** ~5-10 phút

#### 2.2. Huấn luyện mô hình

```bash
jupyter notebook notebooks/02_model_training.ipynb
```

Chạy tất cả các cell. Notebook này sẽ:
- Train mô hình N-gram
- Train mô hình LSTM (5 epochs)
- Lưu vào `models/`

**Thời gian:** ~10-30 phút (tùy cấu hình máy)

#### 2.3. Test mô hình

```bash
jupyter notebook notebooks/03_model_testing.ipynb
```

Chạy tất cả các cell để test mô hình.

### Bước 3: Sử dụng

Trong notebook 03, sử dụng hàm `predict_next_word()`:

```python
# Ví dụ đơn giản
input_text = "tôi đi học bằng "
predictions = predict_next_word(input_text, 3)
print(predictions)
```

## Ví dụ sử dụng chi tiết

### Ví dụ 1: Dự đoán với LSTM

```python
input_text = "tôi đi học bằng "
result = predict_next_word(input_text, top_k=3, model_type='lstm')
print(result)
# Output: ['xe', 'đường', 'cách']
```

### Ví dụ 2: Dự đoán với N-gram

```python
input_text = "hôm nay trời đẹp "
result = predict_next_word(input_text, top_k=5, model_type='ngram')
print(result)
# Output: ['quá', 'lắm', 'thật', 'như', 'vậy']
```

### Ví dụ 3: So sánh cả 2 mô hình

```python
input_text = "việt nam là "
result = predict_next_word(input_text, top_k=3, model_type='both')

print("N-gram:", result['ngram'])
print("LSTM:", result['lstm'])
```

### Ví dụ 4: Nhiều test cases

```python
test_cases = [
    "tôi thích ăn ",
    "chúng tôi đang ",
    "học sinh đi ",
]

for test in test_cases:
    predictions = predict_next_word(test, 3, model_type='lstm')
    print(f"{test} → {predictions}")
```

## Format theo yêu cầu

```python
input = "tôi đi học bằng "
print(predict_next_word(input, 3))
# Output: ['xe đạp', 'xe buýt', 'đi bộ']  # Ví dụ
```

## Xử lý lỗi thường gặp

### Lỗi 1: Module not found

```bash
pip install datasets underthesea numpy
```

### Lỗi 2: File not found (vocabulary.pkl, models, etc.)

→ Chạy lại notebook 01 và 02 theo thứ tự

### Lỗi 3: CUDA/GPU errors

→ Mô hình chỉ dùng NumPy, không cần GPU

### Lỗi 4: Memory error khi load dataset

→ Giảm NUM_SAMPLES trong notebook 01 (ví dụ: 5000 thay vì 10000)

## Tùy chỉnh

### Thay đổi số lượng dữ liệu

Trong `01_data_preprocessing.ipynb`:

```python
NUM_SAMPLES = 5000  # Thay đổi từ 10000
```

### Thay đổi tham số mô hình

Trong `02_model_training.ipynb`:

```python
# N-gram
ngram_model = NgramModel(n=4, smoothing=0.01)  # Thay n=3 thành n=4

# LSTM
lstm_model = SimpleLSTM(
    vocab_size=vocab_size,
    embedding_dim=100,  # Tăng từ 50
    hidden_dim=256,     # Tăng từ 128
    max_seq_len=max_seq_len
)
```

### Thay đổi training parameters

```python
epochs = 10          # Tăng từ 5
batch_size = 64      # Giảm từ 128
learning_rate = 0.001  # Giảm từ 0.01
```

## Cấu trúc thư mục sau khi hoàn thành

```
nwp/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_testing.ipynb
├── data/
│   ├── raw/
│   └── processed/
│       ├── vocabulary.pkl          ← Từ điển
│       ├── training_data.pkl       ← Dữ liệu train
│       ├── tokenized_texts.pkl     ← Văn bản đã token
│       └── config.pkl              ← Cấu hình
├── models/
│   ├── ngram_model.pkl            ← Mô hình N-gram
│   └── lstm_model.pkl             ← Mô hình LSTM
├── README.md
├── QUICKSTART.md
└── requirements.txt
```

## Tips

1. **Chạy notebook theo thứ tự**: 01 → 02 → 03
2. **Lưu output**: Sau mỗi notebook, kiểm tra file đã được tạo
3. **RAM**: Cần ít nhất 4GB RAM để chạy thoải mái
4. **Thời gian**: Tổng thời gian ~30-60 phút cho toàn bộ quá trình

## Câu hỏi thường gặp

**Q: Tôi có thể dùng dataset khác không?**
A: Có, chỉnh sửa phần load dataset trong notebook 01.

**Q: Làm sao để cải thiện độ chính xác?**
A:
- Tăng số lượng dữ liệu
- Tăng số epochs
- Tăng kích thước mô hình (embedding_dim, hidden_dim)
- Điều chỉnh learning rate

**Q: Mô hình nào tốt hơn?**
A:
- N-gram: Nhanh, tốt cho ngữ cảnh ngắn
- LSTM: Chậm hơn, tốt cho ngữ cảnh dài và phức tạp

**Q: Có thể deploy lên production không?**
A: Code hiện tại chỉ để học tập. Để production:
- Dùng framework chuyên nghiệp (PyTorch/TensorFlow)
- Tối ưu hóa inference
- Thêm error handling
- Xây dựng API

## Liên hệ & Đóng góp

Nếu gặp vấn đề, vui lòng tạo issue hoặc liên hệ.

---

**Chúc bạn thành công!** 🎉
