## 🙋🏼 Dự án: Phân lọai phân khúc khách hàng (Customer Segmentation Classification)

Dự án này nhằm xây dựng một hệ thống Machine Learning để dự đoán phân khúc khách hàng dựa trên một số thông tin cá nhân như độ tuổi, kinh nghiệm làm việc, tình trạng hôn nhân, v.v. Dữ liệu được lấy từ nền tảng Kaggle.

Quy trình thực hiện bao gồm:

- Tiền xử lý dữ liệu,

- Xử lý đặc trưng,

- Huấn luyện và đánh giá mô hình phân loại từ đơn giản như Logistic Regression đến nâng cao như XGBoost và LightGBM.

Cuối cùng, mô hình được triển khai dưới dạng ứng dụng web để người dùng có thể dễ dàng nhập thông tin và nhận kết quả dự đoán trực tiếp.

## 📄 Mô tả dữ liệu

Bộ dữ liệu được lấy trên Kaggle, đây là một bộ dữ liệu bình thường không trong cuộc thi nào cả.

Link: https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation

### 📌 Các đặc trưng trong dữ liệu:

Khách hàng có 11 loại thông tin bao gồm:
`ID`: Số thứ tự của khách hàng - số nguyên.

`Gender`: Giới tính của khách hàng - chuỗi.

`Ever_Married`: Tình trạng hôn nhân (đã kết hôn hay chưa?) - chuỗi

`Age`: Tuổi của khách hàng - số nguyên.

`Graduated`: Khách hàng đã tốt nghiệp hay chưa? - chuỗi.

`Profession`: Lĩnh vực chuyên môn của khách hàng - chuỗi.

`Work_Experience`: Kinh nghiệm làm việc của khách hàng - số thực.

`Spending_Score`: Điều kiện mua sắm của khách hàng - chuỗi.

`Family_Size`: Số lượng thành viên trong gia đình của khách hàng - số nguyên.

`Var_1`: Loại thông tin ẩn danh của khách hàng - chuỗi.

`Segmentation` (Target): Hạng phân khúc của khách hàng - chuỗi.

### 📌 Nhận xét

Nhiều giá trị NULL ở một số cột, cần có phương pháp phù hợp để xử lý.

Các đặc trưng đa dạng từ dạng số, phân loại đến nhị phân.

## 🎯Mô tả bài toán

### 🎯 Mục tiêu

Một công ty ô tô đang có kế hoạch mở rộng sang các thị trường mới với các sản phẩm hiện có (P1, P2, P3, P4 và P5). Qua nghiên cứu thị trường, họ nhận thấy hành vi người tiêu dùng ở thị trường mới khá tương đồng với thị trường hiện tại.

Tại thị trường hiện tại, khách hàng đã được đội ngũ bán hàng phân thành 4 nhóm (A, B, C, D) và áp dụng các chiến lược tiếp cận riêng cho từng nhóm, mang lại hiệu quả đáng kể.

Hiện nay, công ty muốn áp dụng lại chiến lược này cho 2.627 khách hàng tiềm năng mới tại các thị trường mới.

### 🎯 Nhiệm vụ

Xây dựng một mô hình phân loại để dự đoán phân khúc khách hàng (A, B, C, D) dựa trên thông tin cá nhân như: độ tuổi, kinh nghiệm làm việc, tình trạng hôn nhân, thu nhập, v.v.

- **Input**: Một hàng dữ liệu chứa thông tin của một khách hàng.

- **Output**: Dự đoán khách hàng đó thuộc phân khúc nào (A/B/C/D).

Vì phân khúc là một biến phân loại, đây là một bài toán phân loại (Classification). Mục tiêu là đạt được độ chính xác cao với mức độ nhầm lẫn thấp nhất trong dự đoán.

### 🎯 Lợi ích và ứng dụng

Bài toán có ứng dụng thực tiễn trong kinh doanh, hỗ trợ tiềm kiếm và xác định khách hàng tiềm năng, từ đó lập kế hoạch tăng doanh thu của cửa hàng.

Dataset có đa dạng các loại biến từ phân loại đến số học, giúp khai thác được hết các yếu tố ảnh hưởng đến phân khúc khách hàng.

Kích thước bộ dữ liệu khá lớn, tạo điều kiện thử nghiệm nhiều mô hình, tinh chỉnh và chọn được mô hình tối ưu.

## 📁 Cấu trúc thư mục dự án

```

Customer_Segmentation_Classification/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── Train.csv
│   └── dataPreprocessing.csv
│
├── model/
│   ├── lgbm_best.pkl
│   ├── lgr_best.pkl
│   ├── rdf_best.pkl
│   ├── xgb_best.pkl
│   └── stacking_model.pkl
│
├── notebooks/
│   ├── Data_Visualize.ipynb
│   ├── Modeling.ipynb
│   └── ensemble_model.ipynb
│
├── pages/
│   ├── 1_Dataset.py
│   ├── 2_Visualization.py
│   └── 3_Model.py
│
└── src/
    ├── data_loader.py
    ├── data_preprocessing.py
    └── feature.py
```

## 🚀Cách chạy dự án

### Cài đặt môi trường

- Cài đặt các thư viện cần thiết
<pre> <code> pip install -r requirements.txt</code> </pre>

### Cài biến môi trường `MODEL_DIR` (link dẫn đến folder project)

- Windows
<pre><code>set MODEL_DIR=.\Customer_Segmentation_Classification</code></pre>

- macOS/Linux
<pre><code>export MODEL_DIR=.\Customer_Segmentation_Classification</code> </pre>

### Chạy chương trình chính

- Chạy ứng dụng Streamlit bằng lệnh sau:
<pre><code> streamlit run app.py </code></pre>

## 🌐 Giao diện web

Sau khi chạy, ứng dụng web sẽ tự động mở trên trình duyệt mặc định.
Tại đây, bạn có thể:

- Xem dữ liệu

- Trực quan hóa đặc trưng

- Dự đoán phân khúc khách hàng mới dựa vào thông tin cá nhân
