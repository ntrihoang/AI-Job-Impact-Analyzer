
# AI JOB IMPACT & CAREER TRANSITION ANALYSIS

## Trạng thái dự án

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| Data Preparation & Database | O*NET + OECD, SQLite, validation | Hoàn thành |
| EDA | Phân tích phân phối, AI Exposure, quan hệ giữa các feature | Hoàn thành |
| Feature Representation | 41 Work Activities LV (chính) và PCA 14 components (so sánh) | Hoàn thành |
| Unsupervised Learning | K-Means, ứng viên K = 4, diễn giải cụm | Hoàn thành |
| AI Exposure Analysis | Phân tích theo cụm, Kruskal–Wallis, Mann–Whitney U (Holm) | Hoàn thành |
| Skill Similarity & Career Transition | Cosine similarity, Top-K occupation tương đồng | Tiếp theo |
| Skill Gap Analysis | So sánh skill vector giữa nghề hiện tại và nghề mục tiêu | Chưa thực hiện |
| Evaluation với O*NET Related Occupations | Top-K overlap, similarity distribution | Chưa thực hiện |
| Visualization & Streamlit | Occupation Explorer, Clusters, Career Transition | Chưa thực hiện |

---



### 1.1. Ý tưởng

Dự án xây dựng một hệ thống Data Science nhằm phân tích đặc điểm của các nghề nghiệp, khám phá các nhóm nghề có đặc điểm tương đồng bằng **Unsupervised Learning** và tìm kiếm các hướng chuyển đổi nghề nghiệp dựa trên mức độ tương đồng về kỹ năng.

Thay vì xây dựng một mô hình dự đoán nghề nào sẽ bị AI thay thế, dự án sử dụng **OECD AI Exposure** như một chỉ số đã được OECD xây dựng để phân tích dữ liệu nghề nghiệp. Chỉ số này là mức phơi nhiễm với AI, không phải xác suất mất việc hay xác suất một nghề bị tự động hóa hoàn toàn.

Trọng tâm của dự án là:

```text
Occupation Data
       ↓
Data Cleaning & Feature Engineering
       ↓
Exploratory Data Analysis
       ↓
Unsupervised Learning
       ↓
Occupation Clustering
       ↓
AI Exposure Analysis
       ↓
Skill Similarity
       ↓
Career Transition
       ↓
Evaluation & Interpretation
```

Dashboard chỉ đóng vai trò là lớp trình bày kết quả, không phải trọng tâm của dự án.

---

# 2. Mục tiêu dự án

## 2.1. Mục tiêu chính

Dự án hướng tới 3 mục tiêu chính:

### Mục tiêu 1 — Khám phá cấu trúc nghề nghiệp

Sử dụng dữ liệu O*NET để tìm ra các nhóm occupation có đặc điểm tương đồng dựa trên **41 Work Activities Level (LV)**, áp dụng **K-Means Clustering**.

Sau khi có cụm, OECD AI Exposure được ghép vào theo `occupation_code`, rồi trả lời câu hỏi:

> Các cluster khác nhau có mức AI Exposure khác nhau như thế nào?

Kết quả thực tế với K = 4 (911 occupation có Work Activities LV):

| Cluster | Tên mô tả | Số occupation | Mean AI Exposure |
|---|---|---:|---:|
| 0 | General Physical & Material-Handling | 204 | 0.922 |
| 1 | Interpersonal, Administrative & Digital-Service | 221 | 0.822 |
| 2 | Analytical, Advisory & Organizational-Leadership | 258 | 0.768 |
| 3 | Mechanical, Equipment & Process-Operations | 228 | 0.870 |

AI Exposure **không** được dùng làm input của K-Means; nó chỉ là chỉ số bên ngoài để phân tích sau phân cụm.

### Mục tiêu 2 — Phân tích Career Transition

Xây dựng hệ thống tìm kiếm các occupation có skill profile tương đồng với occupation hiện tại.

Quy trình:

```text
Current Occupation
       ↓
Skill Vector
       ↓
Cosine Similarity
       ↓
Similar Occupations
       ↓
Skill Gap Analysis
       ↓
Potential Career Transition
```

Hệ thống không quyết định nghề nghiệp cho người dùng mà cung cấp thông tin về:

- Skill similarity
- Skill gap
- Education requirement
- AI Exposure
- Occupation cluster

để người dùng tự đánh giá các hướng chuyển đổi.

---

# 3. Phạm vi MVP

## 3.1. Bao gồm

MVP tập trung vào:

- O*NET occupation data
- O*NET skills
- O*NET tasks
- O*NET work activities (41 activities, dùng Level làm feature cho clustering)
- O*NET education/job zone
- O*NET Related Occupations
- OECD AI Exposure
- Data cleaning
- Data integration
- Feature engineering
- EDA
- K-Means clustering
- AI Exposure analysis theo cụm
- Skill similarity
- Skill gap analysis
- Evaluation
- Visualization
- Streamlit ở mức cơ bản để trình bày kết quả

---

# 4. Data Preparation & Database

## 4.1. Tổng quan

Phần Data của project tập trung vào việc xây dựng một cơ sở dữ liệu nghề nghiệp thống nhất, phục vụ cho các bước phân tích và Machine Learning ở các giai đoạn sau.

Pipeline dữ liệu:

```text
Raw Data
   │
   ├── O*NET Occupation Data
   ├── O*NET Skills
   ├── O*NET Education
   ├── O*NET Work Activities
   ├── O*NET Related Occupations
   │
   └── OECD AI Exposure
          │
          ↓
     Data Cleaning
          │
          ↓
   Data Integration
          │
          ↓
  Feature Engineering
          │
          ↓
      SQLite Database
          │
          ↓
       Data Validation
          │
          ↓
   EDA / Machine Learning
```

Database sử dụng **SQLite**, với `occupation_code` là khóa định danh trung tâm để liên kết dữ liệu giữa các bảng.

---

## 4.2. Nguồn dữ liệu

### O*NET

O*NET được sử dụng làm nguồn dữ liệu chính về đặc điểm nghề nghiệp. Các nhóm dữ liệu được sử dụng gồm:

- Occupations
- Essential Skills
- Education
- Work Activities
- Related Occupations

`occupation_code` được sử dụng làm identifier chính thay vì `occupation_title`, để tránh merge dữ liệu dựa trên tên nghề vì tên nghề có thể được viết khác nhau giữa các nguồn.

### OECD AI Exposure

OECD AI Exposure được sử dụng như một chỉ số bên ngoài để phân tích mức độ tiếp xúc của occupation với AI.

Dữ liệu OECD sử dụng:

```text
OCC_Code
AI Capability Gap Index_Rev. norm.
```

Trong đó `AI Capability Gap Index_Rev. norm.` được dùng làm `ai_exposure_score`, vì chỉ số này nằm trong khoảng `0–1`, với giá trị cao hơn thể hiện mức exposure cao hơn theo định nghĩa của nguồn dữ liệu.

Không sử dụng `AI Capability Gap Index_Total` vì hướng diễn giải của chỉ số này khác với chỉ số normalized được chọn.

---

## 4.3. Occupation Master Data

Bảng `occupations` là **master table** của toàn bộ database.

```sql
CREATE TABLE occupations (
    occupation_code TEXT PRIMARY KEY,
    occupation_title TEXT NOT NULL,
    description TEXT,
    job_zone INTEGER,
    ai_exposure_score REAL
);
```

| Column              | Description               |
| ------------------- | ------------------------- |
| `occupation_code`   | Mã occupation, khóa chính |
| `occupation_title`  | Tên occupation            |
| `description`       | Mô tả occupation          |
| `job_zone`          | Job Zone của occupation   |
| `ai_exposure_score` | AI exposure từ OECD       |

`ai_exposure_score` được cập nhật từ OECD dựa trên `occupation_code`.

---

## 4.4. Occupation Education

Education được lưu trong một bảng riêng thay vì chỉ giữ một `education_level` duy nhất trong bảng `occupations`.

```sql
CREATE TABLE occupation_education (
    occupation_code TEXT NOT NULL,
    education_category INTEGER NOT NULL,
    percentage REAL,
    PRIMARY KEY (occupation_code, education_category),
    FOREIGN KEY (occupation_code)
        REFERENCES occupations(occupation_code)
);

CREATE INDEX idx_education_occupation
ON occupation_education(occupation_code);
```

Kết quả kiểm tra dữ liệu:

```text
Rows:                  10,812
Unique occupations:       901
Duplicate pairs:            0
Missing values:              0
Matched occupation codes:  901 / 901
```

Education được giữ dưới dạng phân phối theo category thay vì làm mất thông tin bằng cách chọn một education level duy nhất. Các occupation không có education data không bị ép giá trị giả vào database.

---

## 4.5. Occupation Skills

Essential Skills được sử dụng để xây dựng skill representation cho occupation.

Raw data có 18,200 rows, tổ chức theo occupation-skill với hai loại measurement:

```text
IM = Importance
LV = Level
```

Sau khi pivot, bảng `occupation_skills` gồm:

```text
occupation_code
skill_id
skill_name
importance_score
level_score
normalized_weight
```

Kiểm tra sau xử lý:

```text
Occupation-skill combinations: 9,100
Importance values:             9,100
Level values:                 9,100
Missing importance:                0
Missing level:                    0
```

```text
Importance: mean = 3.103655, std = 0.771478, min = 1, max = 5
Level:      mean = 3.099026, std = 1.066853, min = 0, max = 6
```

Có 165 records `Level = 0` (khoảng 1.813%). Các giá trị này được giữ lại vì có thể là giá trị hợp lệ thể hiện level requirement rất thấp hoặc không đáng kể.

Skill vector của occupation sẽ được dùng ở giai đoạn sau cho:

- Skill similarity
- Cosine similarity
- Skill gap analysis
- Career transition analysis

---

## 4.6. Occupation Work Activities

Work Activities là nguồn dữ liệu chính cho bước phân cụm.

Raw data chứa hai loại measurement `IM` (Importance) và `LV` (Level), được pivot từ dạng:

```text
occupation + activity + scale + score
```

thành:

```text
occupation + activity + importance + level
```

Hai measurement được giữ riêng vì mô tả hai khía cạnh khác nhau:

- **Importance (IM):** activity quan trọng như thế nào đối với occupation.
- **Level (LV):** occupation yêu cầu activity đó ở mức độ nào.

Project không trực tiếp gộp `Importance × Level`. Thay vào đó:

```text
LV → feature cho clustering
IM → EDA / phân tích riêng
```

### Coverage của Work Activities

Master occupation dataset có khoảng **1,016 occupations**, trong khi Work Activities có đầy đủ dữ liệu cho khoảng **911 occupations**. Project **không drop** occupation khỏi master table:

```text
occupations
    └── giữ toàn bộ occupation

occupation_features / work activities
    └── có thể thiếu đối với occupation
        không đủ Work Activities
```

Các occupation thiếu feature được xử lý ở bước Machine Learning, khi tạo dataset cho K-Means (dataset phân cụm gồm 911 occupation).

---

## 4.7. OECD AI Exposure Integration

OECD data được đọc từ Excel workbook, trong đó sheet `Data` có nhiều dòng metadata/header trước phần dữ liệu occupation. Dữ liệu thực tế bắt đầu từ row 4.

Kết quả mapping OECD – O*NET:

```text
O*NET occupations:       1,016
OECD occupations:          879
Matched occupations:       879
Unmatched OECD:               0
O*NET without OECD:         137
```

Tức 879 / 1,016 ≈ 86.5% occupation trong master table có AI exposure. 137 occupation còn lại giữ `NULL` thay vì `0`, vì `NULL` biểu diễn **không có dữ liệu**, trong khi `0` mang nghĩa exposure bằng 0.

---

# 5. Database Schema

Database được thiết kế xoay quanh `occupation_code`:

```text
                    occupations
                         │
   ┌─────────────┬───────┼──────────────┬──────────────────┐
   ▼             ▼       ▼              ▼                  ▼
occupation_  occupation_ occupation_  occupation_work_   related_
skills       education   tasks        activities         occupations
   │
   ▼
occupation_features
```

Các bảng chính:

```text
occupations
occupation_skills
occupation_education
occupation_tasks
occupation_work_activities
occupation_features
related_occupations
```

## 5.1. Nguyên tắc xử lý dữ liệu

1. **Không merge bằng occupation title.** Dùng `occupation_code` làm identifier chính.
2. **Không biến missing thành zero.** Không có OECD exposure ≠ AI exposure = 0, giữ `NULL`.
3. **Không tự động xóa các giá trị Level = 0.** `LV = 0` có thể là giá trị hợp lệ trong O*NET.
4. **Phân biệt raw data và engineered features.** `importance_score`, `level_score` là dữ liệu nguồn; `normalized_weight` và các feature tổng hợp do project xây dựng là engineered features.
5. **AI Exposure không được dùng để tạo occupational clusters.** `ai_exposure_score` được lưu trong `occupations` nhưng không là input của K-Means:

```text
O*NET Work Activities LV
          ↓
       K-Means
          ↓
 Occupational Clusters
          ↓
Compare / Analyze
          ↑
 OECD AI Exposure
```

AI Exposure là **external indicator** dùng để phân tích cụm sau khi clustering, không quyết định việc hình thành cụm.

## 5.2. Data Validation

Sau khi insert dữ liệu vào SQLite, database được kiểm tra trước khi chuyển sang EDA và Machine Learning:

- **Primary key:** duplicate hoặc NULL `occupation_code`.
- **Foreign key:** các bảng con có occupation tồn tại trong `occupations`.
- **Duplicate records:** occupation + skill, occupation + education_category, occupation + task, occupation + related occupation.
- **Numerical range:** `importance_score`, `level_score`, `normalized_weight`, `ai_exposure_score`, occupational features.
- **Missing values:** phân biệt missing do source coverage với missing do lỗi xử lý dữ liệu.

---

# 6. EDA

EDA tập trung vào việc hiểu dữ liệu trước khi machine learning.

- **Occupation distribution:** số lượng occupation, education, job zone, skill distribution.
- **AI Exposure:** distribution, occupation có exposure cao/thấp, quan hệ với đặc điểm nghề nghiệp.
- **Feature relationships:** quan hệ giữa các feature tổng hợp ban đầu (Cognitive, Social, Physical, Digital, Routine) và AI Exposure. Lưu ý: correlation không đồng nghĩa với causation.
- **Skill analysis:** skill phổ biến, skill importance, skill distribution giữa các occupation.

Phát hiện quan trọng: một số feature tổng hợp có tương quan cao, đặc biệt giữa Cognitive, Information/Digital và Administrative/Routine. Điều này đặt ra câu hỏi về ảnh hưởng của cách nhóm hoạt động thủ công đối với cấu trúc phân cụm, và dẫn đến thay đổi biểu diễn ở phần 7.

---

# 7. Feature Engineering — Work Activities Representation

## 7.1. Mục tiêu

Chuyển dữ liệu Work Activities từ O\*NET thành ma trận đặc trưng phục vụ bài toán phân cụm nghề nghiệp.

Dự án sử dụng **41 Work Activities Level (LV)** làm biểu diễn chính. Cách tiếp cận này giữ lại thông tin ở cấp từng hoạt động công việc, giúp mô tả và so sánh đặc trưng giữa các nhóm nghề sau khi phân cụm.

## 7.2. Xây dựng ma trận đặc trưng

Mỗi occupation được biểu diễn bằng một vector gồm 41 giá trị Level, tương ứng với 41 hoạt động công việc trong O\*NET.

Một số hoạt động tiêu biểu:

- `Handling and Moving Objects`
- `Making Decisions and Solving Problems`
- `Analyzing Data or Information`
- `Working with Computers`
- `Controlling Machines and Processes`

Ma trận được xây dựng bằng cách pivot dữ liệu từ bảng `occupation_work_activities`, sử dụng `occupation_code` làm định danh nghề và `activity` làm đặc trưng.

Trước khi phân cụm, dữ liệu được chuẩn hóa để hạn chế ảnh hưởng của khác biệt về thang đo giữa các đặc trưng.

### Kiểm tra chất lượng dữ liệu

Dữ liệu Work Activities được sử dụng có các đặc điểm:

| Chỉ tiêu | Kết quả |
|---|---:|
| Số occupation | 911 |
| Số hoạt động trên mỗi occupation | 41 |
| Tổng số bản ghi nguồn | 37,351 |
| Giá trị thiếu ở `importance_score` và `level_score` | 0 |
| Cặp occupation–activity trùng lặp | 0 |

Sau khi pivot, ma trận có 911 hàng và 41 đặc trưng, không có giá trị LV bị thiếu.

## 7.3. Lý do lựa chọn 41 Work Activities LV

Trong giai đoạn đầu, dự án xây dựng năm đặc trưng tổng hợp từ các nhóm hoạt động công việc:

- `cognitive_intensity`
- `social_intensity`
- `physical_operational_intensity`
- `information_digital_intensity`
- `administrative_routine_intensity`

Các đặc trưng này được định nghĩa trong phạm vi dự án, không phải các nhóm phân loại chính thức của O\*NET. EDA cho thấy một số đặc trưng có tương quan cao và việc tổng hợp phụ thuộc vào cách nhóm hoạt động do dự án lựa chọn.

Vì vậy, dự án chuyển sang sử dụng trực tiếp 41 Work Activities LV làm biểu diễn chính nhằm:

- Giữ lại ý nghĩa cụ thể của từng hoạt động công việc.
- Hạn chế sự phụ thuộc vào cách tổng hợp đặc trưng thủ công.
- Hỗ trợ diễn giải các cluster bằng những hoạt động có mức độ nổi bật tương đối.
- Tạo cơ sở nhất quán cho việc phân tích mối liên hệ giữa đặc điểm hoạt động công việc và AI Exposure.

Năm đặc trưng tổng hợp ban đầu không được sử dụng trong pipeline clustering chính thức.

---

# 8. Unsupervised Learning — Occupational Clustering

## 8.1. Bài toán

Dự án đặt ra câu hỏi nghiên cứu:

> Các nghề nghiệp có thể được phân thành những nhóm nào dựa trên đặc điểm hoạt động công việc, và các nhóm đó khác nhau như thế nào về AI Exposure?

Đây là bài toán phân cụm không giám sát. Mô hình không sử dụng nhãn mục tiêu `y`.

- **Input:** Ma trận 41 Work Activities LV đã chuẩn hóa.
- **Model:** K-Means.
- **Output:** Nhãn cluster của từng occupation và hồ sơ đặc trưng của các cluster.
- **Phân tích tiếp theo:** Ghép nhãn cluster với OECD AI Exposure để so sánh phân phối điểm giữa các nhóm.

AI Exposure không được sử dụng làm đầu vào cho K-Means, nhằm tránh việc phân cụm trực tiếp theo biến sẽ được phân tích ở bước sau.

## 8.2. Phương pháp K-Means

K-Means phân chia dữ liệu thành K cụm bằng cách tối thiểu hóa tổng bình phương khoảng cách từ mỗi điểm dữ liệu đến tâm cụm tương ứng:


$J=\sum_{i=1}^{n}\left\|x_i-\mu_{c_i}\right\|^2$

Trong đó:

- $x_i$ là vector đặc trưng của occupation thứ \(i\).
- $\mu_{c_i}$ là tâm của cluster được gán cho occupation đó.
- K là số cluster được xác định trước.

### Quy trình thực nghiệm

1. Xây dựng ma trận 41 Work Activities LV.
2. Chuẩn hóa các đặc trưng.
3. Chạy K-Means với K từ 2 đến 8.
4. Tính các clustering metrics để so sánh kết quả.
5. Xem xét kích thước cụm và khả năng diễn giải theo hoạt động công việc.
6. Sử dụng (K=4) làm phương án phân tích chính hiện tại.
7. Lưu nhãn cluster để phục vụ phân tích AI Exposure và các bước tiếp theo.

## 8.3. So sánh các giá trị K

Dự án đánh giá các cấu hình K-Means với K=2 đến 8 bằng bốn chỉ số:

| Metric | Ý nghĩa | Hướng đánh giá |
|---|---|---|
| Inertia | Tổng bình phương khoảng cách đến tâm cụm | Thường giảm khi tăng K |
| Silhouette Score | Mức độ gắn kết trong cụm và tách biệt giữa các cụm | Cao hơn thường tốt hơn |
| Davies–Bouldin Index | Mức độ tương đồng giữa các cụm | Thấp hơn thường tốt hơn |
| Calinski–Harabasz Index | Tương quan giữa độ phân tán giữa cụm và trong cụm | Cao hơn thường tốt hơn |

### Kết quả thực nghiệm trên 41 Work Activities LV

| K | Inertia | Silhouette | Davies–Bouldin | Calinski–Harabasz |
|---:|---:|---:|---:|---:|
| 2 | 24718.852 | 0.2799 | 1.3453 | 464.529 |
| 3 | 21640.614 | 0.1786 | 1.7253 | 329.589 |
| 4 | 19349.927 | 0.1747 | 1.7008 | 281.258 |
| 5 | 18071.636 | 0.1535 | 1.7691 | 241.637 |
| 6 | 16975.542 | 0.1395 | 1.8182 | 217.252 |
| 7 | 16259.275 | 0.1407 | 1.7965 | 195.447 |
| 8 | 15719.113 | 0.1301 | 1.8651 | 177.524 |

### Lựa chọn K = 4

Kết quả cho thấy Silhouette Score cao nhất trong các cấu hình được thử nghiệm tại K=2. Tuy nhiên, lựa chọn K=2 tạo ra cách phân chia thô hơn và có thể không cung cấp đủ mức độ chi tiết cho mục tiêu mô tả các nhóm nghề nghiệp.

Dự án lựa chọn K=4 làm phương án phân tích chính dựa trên sự cân nhắc giữa:

- Khả năng mô tả các nhóm nghề với đặc điểm hoạt động khác nhau.
- Kích thước của các cluster.
- Khả năng diễn giải kết quả bằng các hoạt động công việc cụ thể.
- Các clustering metrics thu được từ thực nghiệm.

Đây là lựa chọn phục vụ mục tiêu phân tích của dự án, không phải khẳng định K=4 là số cụm tối ưu tuyệt đối.

Silhouette Score tại K=4 khoảng 0.1747, cho thấy các cụm chưa tách biệt mạnh. Do đó, kết quả được xem là một cách phân nhóm hữu ích để khám phá và mô tả dữ liệu, không phải bằng chứng về các ranh giới tự nhiên hoàn toàn rõ ràng giữa các nghề.

## 8.4. Kết quả phân cụm với K = 4

Mô hình phân chia 911 occupation thành bốn cluster:

| Cluster | Số occupation | Đặc trưng chính | Tên mô tả đề xuất |
|---|---:|---|---|
| 0 | 204 | Xử lý vật thể và hoạt động thể chất tổng quát | General Physical & Material-Handling Occupations |
| 1 | 221 | Quan hệ liên cá nhân, giao tiếp, hành chính và dịch vụ số | Interpersonal, Administrative & Digital-Service Occupations |
| 2 | 258 | Phân tích, tư vấn, chiến lược và điều phối tổ chức | Analytical, Advisory & Organizational-Leadership Occupations |
| 3 | 228 | Máy móc, thiết bị, sửa chữa và vận hành quy trình | Mechanical, Equipment & Process-Operations Occupations |
| **Tổng** | **911** | | |

Các tên cluster là nhãn mô tả do dự án xây dựng, không phải phân loại nghề nghiệp chính thức của O\*NET. Việc diễn giải dựa trên các hoạt động có mức Level trung bình nổi bật tương đối so với những cluster còn lại, kết hợp với việc xem xét các occupation tiêu biểu.

### Đặc điểm của từng cluster

**Cluster 0 — General Physical & Material-Handling Occupations**

Nhóm này nổi bật ở `Handling and Moving Objects` và `Performing General Physical Activities`. Các hoạt động phân tích thông tin và tư vấn thấp hơn đáng kể so với một số cluster khác.

**Cluster 1 — Interpersonal, Administrative & Digital-Service Occupations**

Nhóm này nổi bật ở giao tiếp với người ngoài tổ chức, làm việc với máy tính, thu thập thông tin và duy trì quan hệ liên cá nhân. Các hoạt động xử lý vật thể và kiểm soát máy móc tương đối thấp.

**Cluster 2 — Analytical, Advisory & Organizational-Leadership Occupations**

Nhóm này có mức độ nổi bật tương đối ở tư vấn, phân tích dữ liệu, diễn giải thông tin, bố trí nhân sự và định hướng hoạt động tổ chức.

**Cluster 3 — Mechanical, Equipment & Process-Operations Occupations**

Nhóm này nổi bật ở sửa chữa thiết bị cơ khí, kiểm soát máy móc, xử lý vật thể, vận hành thiết bị và kiểm tra thiết bị hoặc vật liệu.

Các cluster được mô tả theo hồ sơ hoạt động công việc tổng thể. Không nên diễn giải chúng thành các nhóm nghề hoàn toàn tách biệt hoặc cho rằng mọi occupation trong một cluster đều có đặc điểm giống hệt nhau.

---

# 9. AI Exposure Analysis by Cluster

## 9.1. Mục tiêu

Phân tích liệu các nhóm occupation được hình thành từ Work Activities LV có phân phối AI Exposure khác nhau hay không.

AI Exposure được sử dụng như một biến phân tích sau phân cụm, không phải đầu vào của mô hình K-Means.

## 9.2. Ghép dữ liệu và xử lý giá trị thiếu

Nhãn cluster được ghép với `ai_exposure_score` trong bảng `occupations` bằng khóa `occupation_code`.

Các occupation không có điểm AI Exposure từ OECD vẫn được giữ trong kết quả phân cụm. Giá trị thiếu không bị thay bằng 0.

Khi tính thống kê AI Exposure theo cluster, chỉ các occupation có điểm hợp lệ được sử dụng. Số lượng occupation có dữ liệu và tỷ lệ coverage được báo cáo để thể hiện mức độ đầy đủ của dữ liệu.

## 9.3. Thống kê mô tả theo cluster

| Cluster | Occupations có dữ liệu | Coverage | Mean | Median |
|---|---:|---:|---:|---:|
| 0 | 198/204 | 97.06% | 0.9221 | 0.9311 |
| 1 | 210/221 | 95.02% | 0.8221 | 0.8219 |
| 2 | 247/258 | 95.74% | 0.7680 | 0.7635 |
| 3 | 224/228 | 98.25% | 0.8698 | 0.8742 |

Trong 879 occupation có dữ liệu AI Exposure:

- Mean toàn bộ tập dữ liệu: **0.8416**.
- Median toàn bộ tập dữ liệu: **0.8557**.

### Phát hiện chính

- **Cluster 0** có mean và median AI Exposure cao nhất.
- **Cluster 3** đứng thứ hai.
- **Cluster 1** có điểm thấp hơn mặt bằng chung một chút.
- **Cluster 2** có mean và median thấp nhất.

Chênh lệch mean giữa Cluster 0 và Cluster 2 khoảng 0.1541 điểm; chênh lệch median khoảng 0.1676 điểm.

Kết quả cho thấy các nhóm nghề có hồ sơ hoạt động công việc khác nhau cũng có phân phối AI Exposure khác nhau trong dữ liệu đang phân tích. Tuy nhiên, không thể từ kết quả này suy luận rằng hoạt động thể chất tự thân có AI Exposure cao hơn hoạt động trí óc, bởi mỗi cluster được hình thành từ nhiều đặc trưng cùng lúc.

Điểm AI Exposure cũng không phải xác suất mất việc hoặc xác suất một occupation bị AI thay thế.

## 9.4. Trực quan hóa và kiểm định thống kê

Boxplot được sử dụng để so sánh trung vị, độ phân tán và các giá trị ngoại lệ của AI Exposure giữa bốn cluster.

### Kiểm định Kruskal–Wallis

Kết quả kiểm định:

- H statistic: **329.5844**
- p-value: **3.9255 × 10⁻⁷¹**

Ở mức ý nghĩa 0.05, có bằng chứng thống kê cho thấy phân phối AI Exposure không giống nhau giữa tất cả các cluster.

Các kiểm định Mann–Whitney U theo từng cặp, với hiệu chỉnh Holm, cho thấy cả sáu cặp cluster đều có khác biệt có ý nghĩa thống kê ở mức 0.05.

Các kiểm định hậu nghiệm được lưu làm kết quả bổ sung. Phần trình bày chính tập trung vào thống kê mô tả, biểu đồ và phát hiện tổng quát.

Ý nghĩa thống kê không tự động chứng minh rằng mọi khác biệt đều lớn về mặt thực tiễn. Những kết quả này cũng không chứng minh quan hệ nhân quả giữa đặc điểm công việc và AI Exposure.

## 9.5. Kết luận giai đoạn Unsupervised Learning

Giai đoạn Unsupervised Learning đã thực hiện các công việc chính:

1. Xây dựng biểu diễn 911 occupation bằng 41 Work Activities LV.
2. Thử nghiệm K-Means với K=2 đến K=8.
3. Chọn K=4 làm phương án phân tích chính dựa trên mục tiêu mô tả và khả năng diễn giải.
4. Xác định và mô tả bốn cluster dựa trên đặc trưng hoạt động công việc.
5. Ghép OECD AI Exposure với nhãn cluster bằng `occupation_code`.
6. So sánh AI Exposure giữa các cluster bằng thống kê mô tả, trực quan hóa và kiểm định phi tham số.

Kết quả cho thấy AI Exposure có phân phối khác nhau giữa các nhóm nghề được hình thành từ Work Activities LV. Đây là cơ sở để tiếp tục nghiên cứu cấu trúc nghề nghiệp và kết hợp với giai đoạn Skill Similarity và Career Transition Analysis nhằm khám phá các lựa chọn chuyển đổi nghề nghiệp dựa trên dữ liệu.


# 10. Career Transition

## 10.1. Skill Similarity

Mỗi occupation được biểu diễn thành một skill vector, ví dụ:

```text
Accountant
[
  Mathematics = 0.82,
  Critical Thinking = 0.91,
  Reading = 0.88,
  ...
]
```

Tính **Cosine Similarity** giữa occupation hiện tại và toàn bộ occupation khác.

## 10.2. Top-K Similar Occupations

```text
Accountant

1. Financial Analyst       0.91
2. Financial Manager       0.86
3. Data Analyst            0.79
4. Business Analyst        0.77
5. ...
```

(Ví dụ minh họa, chưa phải kết quả thực tế.) Không gọi đây là "nghề tốt hơn"; đây là **occupations có skill profile tương đồng**.

---

# 11. Skill Gap Analysis

```text
Current Occupation
        ↓
Candidate Occupation
        ↓
Compare Skill Vectors
        ↓
Skill Gap
```

Ví dụ minh họa:

```text
Accountant → Data Analyst

Existing strengths:
✓ Mathematics
✓ Critical Thinking

Skill gaps:
→ SQL
→ Statistics
→ Data Analysis
```

Mục tiêu là giải thích tại sao hai occupation tương đồng hoặc khác nhau, không chỉ đưa ra một con số similarity.

---

# 12. Evaluation

## 12.1. Clustering Evaluation (đã thực hiện)

- Inertia, Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Index với K = 2–8
- So sánh 41 Work Activities LV và PCA 14 components tại K = 4
- Kích thước cụm và khả năng diễn giải

Việc đánh giá độ ổn định cụm khi thay đổi random seed vẫn có thể bổ sung nếu cần.

## 12.2. Career Recommendation Evaluation (chưa thực hiện)

Dùng O*NET Related Occupations làm baseline:

```text
O*NET Related Occupations
        VS
Our Skill Similarity
```

Đánh giá: Top-K overlap, số occupation chung, similarity distribution, các trường hợp recommendation khác biệt.

Mục tiêu không phải chứng minh hệ thống "đúng tuyệt đối", mà kiểm tra xem skill-based similarity có tạo ra kết quả hợp lý và có mức tương đồng nhất định với dữ liệu tham chiếu hay không.

---

# 13. Visualization & Streamlit

Dashboard chỉ là lớp trình bày kết quả, gồm:

### Occupation Explorer
Người dùng chọn occupation; hiển thị occupation information, skills, education, job zone, AI Exposure, cluster.

### Occupation Clusters
Cluster distribution, PCA visualization, cluster characteristics, AI Exposure distribution.

### Career Transition
Người dùng chọn occupation; hiển thị candidate occupation, skill similarity, AI Exposure, skill gap, education.

Không cần xây dựng dashboard quá phức tạp.

---

# 14. Toàn bộ pipeline

```text
                    DATA SOURCES
                         │
             ┌───────────┴───────────┐
             │                       │
           O*NET                   OECD
             │                       │
             │                 AI Exposure
             │                       │
             ↓                       ↓
       DATA COLLECTION ───────→ DATA INTEGRATION
             │                       │
             └───────────┬───────────┘
                         ↓
                  DATA CLEANING
                         ↓
                FEATURE ENGINEERING
                         │
             ┌───────────┴───────────┐
             ↓                       ↓
   41 Work Activities LV         Skill Matrix
   (PCA 14 để so sánh)               │
             │                       │
             ↓                       ↓
          K-MEANS             COSINE SIMILARITY
             │                       │
             ↓                       ↓
     OCCUPATION CLUSTERS       CAREER TRANSITION
             │                       │
             ↓                       │
   AI EXPOSURE ANALYSIS              │
             │                       │
             └───────────┬───────────┘
                         ↓
                    EVALUATION
                         ↓
                  VISUALIZATION
                         ↓
                    STREAMLIT
```

---

# 15. Câu hỏi nghiên cứu chính

- **RQ1.** Các occupation có thể được phân nhóm như thế nào dựa trên đặc điểm hoạt động công việc? *(đã trả lời ở mức K = 4)*
- **RQ2.** Các occupation cluster khác nhau có mức AI Exposure khác nhau như thế nào? *(đã trả lời, xem mục 9)*
- **RQ3.** Có thể dùng skill similarity để xác định occupation có tiềm năng chuyển đổi nghề nghiệp không?
- **RQ4.** Recommendation dựa trên skill similarity tương đồng với O*NET Related Occupations ở mức nào?
- **RQ5.** Skill gap giữa occupation hiện tại và occupation mục tiêu có thể giải thích career transition như thế nào?

---

# 16. Kết quả đầu ra của MVP

### Dataset

```text
occupations
occupation_skills
occupation_education
occupation_tasks
occupation_work_activities
occupation_features
related_occupations
```

### Data Science results

```text
EDA
+
Feature Engineering (41 Work Activities LV, PCA 14)
+
K-Means Clustering (K = 4)
+
Cluster Analysis
+
AI Exposure Analysis
+
Skill Similarity
+
Skill Gap
+
Evaluation
```

### Final system

Người dùng chọn một occupation (ví dụ `Accountant`), hệ thống trả về:

```text
Occupation Profile
        ↓
Cluster
        ↓
AI Exposure
        ↓
Similar Occupations
        ↓
Skill Similarity
        ↓
Skill Gap
        ↓
O*NET Baseline Comparison
```

---

# 17. Hướng mở rộng sau MVP

Chỉ sau khi MVP hoàn thành mới cân nhắc mở rộng:

```text
MVP
│
├── Vietnam labor market
├── ILO
├── Job postings
├── Real-world demand
├── Salary
├── Employment trends
└── More advanced recommendation
```

Nếu thêm Việt Nam, có thể nghiên cứu:

```text
Global occupation analysis
        +
Vietnam labor market data
        ↓
Vietnam-specific career transition analysis
```


