# Hệ thống Dự đoán Từ Tiếp theo (Next Word Prediction)

Dự án xây dựng mô hình dự đoán từ tiếp theo cho tiếng Việt, sử dụng N-gram và LSTM được tự xây dựng từ đầu.

## Mô tả

Hệ thống này giúp dự đoán từ tiếp theo dựa trên văn bản đầu vào, tương tự như các công cụ gợi ý từ trên bàn phím di động.

**Đặc điểm:**
- Sử dụng dataset VTSNLP (10,000 dòng văn bản tiếng Việt)
- Tiền xử lý văn bản với thư viện `underthesea`
- 2 mô hình được xây dựng hoàn toàn từ đầu (không dùng sklearn, PyTorch, TensorFlow):
  - **N-gram Model**: Dựa trên thống kê n-gram với Laplace smoothing
  - **LSTM Model**: Mạng neural LSTM được xây dựng bằng NumPy thuần túy

## Cấu trúc dự án

```
nwp/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb    # Tiền xử lý dữ liệu
│   ├── 02_model_training.ipynb        # Huấn luyện mô hình
│   └── 03_model_testing.ipynb         # Test và sử dụng mô hình
├── data/
│   ├── raw/                           # Dữ liệu gốc
│   └── processed/                     # Dữ liệu đã xử lý
│       ├── vocabulary.pkl
│       ├── training_data.pkl
│       ├── tokenized_texts.pkl
│       └── config.pkl
├── models/
│   ├── ngram_model.pkl               # Mô hình N-gram
│   └── lstm_model.pkl                # Mô hình LSTM
└── README.md
```

## Cài đặt

### Yêu cầu
- Python 3.7+
- Jupyter Notebook

### Cài đặt thư viện

```bash
pip install datasets underthesea numpy pickle-mixin
```

## Hướng dẫn sử dụng

### Bước 1: Tiền xử lý dữ liệu

Mở và chạy notebook `01_data_preprocessing.ipynb`:

```python
# Notebook này sẽ:
# 1. Tải 10,000 dòng từ VTSNLP dataset
# 2. Làm sạch và tokenize văn bản
# 3. Xây dựng vocabulary
# 4. Tạo training sequences
# 5. Lưu dữ liệu đã tiền xử lý
```

**Output:** Các file trong `data/processed/`

### Bước 2: Huấn luyện mô hình

Mở và chạy notebook `02_model_training.ipynb`:

```python
# Notebook này sẽ:
# 1. Load dữ liệu đã tiền xử lý
# 2. Xây dựng và train mô hình N-gram
# 3. Xây dựng và train mô hình LSTM
# 4. Đánh giá mô hình
# 5. Lưu mô hình
```

**Output:** Các file trong `models/`

### Bước 3: Sử dụng mô hình

Mở và chạy notebook `03_model_testing.ipynb`:

```python
# Load mô hình và test
from notebooks import *

# Sử dụng hàm predict_next_word()
input_text = "tôi đi học bằng "
predictions = predict_next_word(input_text, 3)
print(predictions)
# Output: ['xe đạp', 'xe buýt', 'đi bộ']  # (ví dụ)
```

## API Sử dụng

### Hàm chính: `predict_next_word()`

```python
def predict_next_word(input_text, top_k=3, model_type='both'):
    """
    Dự đoán top k từ tiếp theo

    Args:
        input_text (str): Văn bản đầu vào
        top_k (int): Số lượng từ dự đoán muốn trả về (mặc định: 3)
        model_type (str): 'ngram', 'lstm', hoặc 'both' (mặc định: 'both')

    Returns:
        list hoặc dict: Danh sách các từ được dự đoán
    """
```

### Ví dụ sử dụng

```python
# Sử dụng cả 2 mô hình
predictions = predict_next_word("tôi đi học bằng ", 3, model_type='both')
print("N-gram:", predictions['ngram'])
print("LSTM:", predictions['lstm'])

# Chỉ dùng LSTM
lstm_predictions = predict_next_word("tôi đi học bằng ", 3, model_type='lstm')
print(lstm_predictions)
# Output: ['xe', 'đường', 'taxi']

# Chỉ dùng N-gram
ngram_predictions = predict_next_word("tôi đi học bằng ", 5, model_type='ngram')
print(ngram_predictions)
# Output: ['xe', 'đường', 'taxi', 'cách', 'phương']
```

## Chi tiết kỹ thuật

### 1. Tiền xử lý dữ liệu

- **Làm sạch văn bản**: Loại bỏ URL, email, ký tự đặc biệt
- **Tokenization**: Sử dụng `underthesea.word_tokenize()`
- **Vocabulary**: Chỉ giữ các từ xuất hiện >= 2 lần
- **Encoding**: Chuyển đổi từ thành indices
- **Sequences**: Tạo cặp (input, target) cho training

### 2. Mô hình N-gram

**Đặc điểm:**
- Trigram (n=3) model
- Sử dụng Laplace smoothing để xử lý unseen n-grams
- Tính xác suất dựa trên tần suất xuất hiện

**Công thức:**
```
P(word | context) = (count(context, word) + α) / (count(context) + α * |V|)
```

**Ưu điểm:**
- Nhanh, hiệu quả
- Không cần training phức tạp
- Hoạt động tốt với ngữ cảnh ngắn

### 3. Mô hình LSTM

**Kiến trúc:**
- Embedding layer: 50 dimensions
- LSTM layer: 128 hidden units
- Output layer: Softmax over vocabulary

**Các thành phần LSTM:**
- Forget gate: Quyết định thông tin nào bị loại bỏ
- Input gate: Quyết định thông tin nào được thêm vào
- Cell state: Lưu trữ thông tin dài hạn
- Output gate: Quyết định output

**Training:**
- Loss function: Cross-entropy
- Optimizer: Gradient descent (simplified)
- Epochs: 5
- Batch size: 128

**Ưu điểm:**
- Có thể học được dependencies dài hạn
- Xử lý được ngữ cảnh phức tạp
- Linh hoạt hơn N-gram

## Đánh giá

### Metrics sử dụng

1. **Top-k Accuracy**: Từ đúng có nằm trong k dự đoán hàng đầu không?
2. **Perplexity**: Độ không chắc chắn của mô hình
3. **Inference Speed**: Thời gian dự đoán

### Kết quả mong đợi

- **N-gram**: Nhanh (~1-2ms), tốt cho ngữ cảnh ngắn
- **LSTM**: Chậm hơn (~10-20ms), tốt cho ngữ cảnh dài

## Mở rộng

Các hướng phát triển:

1. **Cải thiện mô hình:**
   - Thêm attention mechanism
   - Sử dụng BiLSTM
   - Tăng kích thước dataset

2. **Tối ưu hóa:**
   - Quantization
   - Model pruning
   - Caching

3. **Ứng dụng:**
   - Xây dựng API REST
   - Tích hợp vào bàn phím
   - Mobile app

## Lưu ý

- **Mô hình được xây dựng hoàn toàn từ đầu** không sử dụng sklearn, PyTorch, TensorFlow
- LSTM implementation là simplified version, không có full BPTT (Backpropagation Through Time)
- Để sử dụng trong production, nên cân nhắc sử dụng framework chuyên nghiệp

## Tác giả

Dự án được xây dựng theo yêu cầu bài tập NLP.

## License

MIT License
