# AI JOB IMPACT & CAREER TRANSITION ANALYSIS

## 1. Tổng quan dự án

### 1.1. Ý tưởng

Dự án xây dựng một hệ thống Data Science nhằm phân tích đặc điểm của các nghề nghiệp, khám phá các nhóm nghề có đặc điểm tương đồng bằng **Unsupervised Learning** và tìm kiếm các hướng chuyển đổi nghề nghiệp dựa trên mức độ tương đồng về kỹ năng.

Thay vì xây dựng một mô hình dự đoán nghề nào sẽ bị AI thay thế, dự án sử dụng **OECD AI Exposure** như một chỉ số đã được OECD xây dựng để phân tích dữ liệu nghề nghiệp.

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
Skill Similarity
       ↓
Career Transition
       ↓
AI Exposure Analysis
       ↓
Evaluation & Interpretation
```

Dashboard chỉ đóng vai trò là lớp trình bày kết quả, không phải trọng tâm của dự án.

---

# 2. Mục tiêu dự án

## 2.1. Mục tiêu chính

Dự án hướng tới 3 mục tiêu chính:

### Mục tiêu 1 — Khám phá cấu trúc nghề nghiệp

Sử dụng dữ liệu O*NET để tìm ra các nhóm occupation có đặc điểm tương đồng dựa trên:

- Skills
- Knowledge
- Abilities
- Work Activities
- Occupational characteristics
- Các feature được nhóm tự xây dựng

Áp dụng Unsupervised Learning, dự kiến sử dụng K-Means Clustering.
Ví dụ sau khi K-Means tạo ra:
- Cluster 0  
- Cluster 1  
- Cluster 2
- Cluster 3
- Cluster 4

chúng ta lấy OECD Exposure và gắn vào:
Occupation     | Cluster     | AI Exposure
-------------- | ----------- | -----------
Accountant     | Cluster 0   | 0.XX
Data Scientist | Cluster 0   | 0.XX
Nurse          | Cluster 1   | 0.XX
Teacher        | Cluster 1   | 0.XX
Electrician    | Cluster 3   | 0.XX

...

Sau đó hỏi:
Các cluster khác nhau có mức AI Exposure khác nhau như thế nào?

Ví dụ minh họa:
Cluster 0 → Average AI Exposure = 0.76  
Cluster 1 → Average AI Exposure = 0.61  
Cluster 2 → Average AI Exposure = 0.43  
Cluster 3 → Average AI Exposure = 0.32  


---

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
- O*NET knowledge/abilities/work activities khi cần cho feature engineering
- O*NET education/job zone
- O*NET Related Occupations
- OECD AI Exposure
- Data cleaning
- Data integration
- Feature engineering
- EDA
- K-Means clustering
- Skill similarity
- Skill gap analysis
- Evaluation
- Visualization
- Streamlit ở mức cơ bản để trình bày kết quả

---


# Data Preparation & Database

## 1. Tổng quan

Phần Data của project tập trung vào việc xây dựng một cơ sở dữ liệu nghề nghiệp thống nhất, phục vụ cho các bước phân tích và Machine Learning ở các giai đoạn sau.

Pipeline dữ liệu hiện tại:

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

# 2. Nguồn dữ liệu

## 2.1. O\*NET

O\*NET được sử dụng làm nguồn dữ liệu chính về đặc điểm nghề nghiệp.

Các nhóm dữ liệu được sử dụng gồm:

- Occupations
- Essential Skills
- Education
- Work Activities
- Related Occupations

`occupation_code` được sử dụng làm identifier chính thay vì `occupation_title`.

Điều này giúp tránh việc merge dữ liệu dựa trên tên nghề, vì tên nghề có thể khác nhau về cách viết giữa các nguồn dữ liệu.

---

## 2.2. OECD AI Exposure

OECD AI Exposure được sử dụng như một chỉ số bên ngoài để phân tích mức độ tiếp xúc của occupation với AI.

Dữ liệu OECD sử dụng:

```text
OCC_Code
AI Capability Gap Index_Rev. norm.
```

Trong đó:

```text
AI Capability Gap Index_Rev. norm.
```

được sử dụng làm:

```text
ai_exposure_score
```

vì chỉ số này được biểu diễn trong khoảng `0–1`, với giá trị cao hơn thể hiện mức exposure cao hơn theo định nghĩa của nguồn dữ liệu.

Không sử dụng:

```text
AI Capability Gap Index_Total
```

vì hướng diễn giải của chỉ số này khác với chỉ số normalized được lựa chọn.

---
# 3. Data thu thập
## 3.1 Occupation Master Data

Bảng `occupations` đóng vai trò là **master table** của toàn bộ database.

Schema:

```sql
CREATE TABLE occupations (
    occupation_code TEXT PRIMARY KEY,
    occupation_title TEXT NOT NULL,
    description TEXT,
    job_zone INTEGER,
    ai_exposure_score REAL
);
```

Trong đó:

| Column              | Description               |
| ------------------- | ------------------------- |
| `occupation_code`   | Mã occupation, khóa chính |
| `occupation_title`  | Tên occupation            |
| `description`       | Mô tả occupation          |
| `job_zone`          | Job Zone của occupation   |
| `ai_exposure_score` | AI exposure từ OECD       |

`ai_exposure_score` được cập nhật từ OECD dựa trên `occupation_code`.

---

## 3.2 Occupation Education

Education được lưu trong một bảng riêng thay vì chỉ giữ một `education_level` duy nhất trong bảng `occupations`.

Schema:

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

### Validation

Kết quả kiểm tra dữ liệu:

```text
Rows:                  10,812
Unique occupations:       901
Duplicate pairs:            0
Missing values:              0
Matched occupation codes:  901 / 901
```

Do đó, education data được giữ dưới dạng phân phối theo category thay vì làm mất thông tin bằng cách chọn một education level duy nhất.

Các occupation không có education data không bị ép giá trị giả vào database.

---

## 3.3. Occupation Skills

Essential Skills được sử dụng để xây dựng skill representation cho occupation.

Raw data có:

```text
18,200 rows
```

và được tổ chức theo occupation-skill với hai loại measurement:

```text
IM = Importance
LV = Level
```

Sau khi pivot dữ liệu, mỗi occupation-skill combination có:

```text
skill_id
skill_name
importance_score
level_score
```

Bảng `occupation_skills` sử dụng các thuộc tính:

```text
occupation_code
skill_id
skill_name
importance_score
level_score
normalized_weight
```

---

## 3.4. Kiểm tra Skill Data

Sau khi xử lý:

```text
Occupation-skill combinations: 9,100
Importance values:             9,100
Level values:                 9,100
Missing importance:                0
Missing level:                    0
```

Thống kê:

```text
Importance:
mean = 3.103655
std  = 0.771478
min  = 1
max  = 5

Level:
mean = 3.099026
std  = 1.066853
min  = 0
max  = 6
```

Có:

```text
Level = 0: 165 records
```

tương đương khoảng:

```text
1.813%
```

Các giá trị `Level = 0` được giữ lại vì đây có thể là giá trị hợp lệ thể hiện level requirement rất thấp hoặc không đáng kể. Không tự động coi chúng là missing hoặc lỗi.

---


Vector này sẽ được sử dụng ở giai đoạn sau cho:

- Skill similarity
- Cosine similarity
- Skill gap analysis
- Career transition analysis

---

## 3.6. Occupation Work Activities

Work Activities được sử dụng để xây dựng các đặc trưng cấp occupation.

Raw data chứa hai loại measurement:

```text
IM = Importance
LV = Level
```

Dữ liệu được pivot từ dạng:

```text
occupation + activity + scale + score
```

thành:

```text
occupation + activity + importance + level
```



---

### Work Activities: Importance và Level

Hai measurement được giữ riêng vì chúng mô tả hai khía cạnh khác nhau.

### Importance (IM)

Cho biết activity quan trọng như thế nào đối với occupation.

### Level (LV)

Cho biết occupation yêu cầu activity đó ở mức độ/level như thế nào.

Do đó, project không trực tiếp gộp:

```text
Importance × Level
```

để tạo occupational activity features.

Thay vào đó:

```text
LV
↓
Occupational features

IM
↓
Separate EDA / analysis
```

Cách này giúp giữ lại ý nghĩa riêng của hai measurement.

---

### Work Activity Feature Engineering

Từ O\*NET Work Activities, project xây dựng 5 occupational dimensions.

Các dimensions này là **feature do project định nghĩa**, không phải năm category chính thức của O\*NET.

Feature được tính bằng cách lấy mean `LV` của các activities thuộc cùng một nhóm.

---

### Kiểm tra Coverage của Work Activities

Work Activities có coverage nhỏ hơn master occupation table.

Có khoảng:

```text
1,016 occupations
```

trong master occupation dataset.

Trong khi Work Activities có khoảng:

```text
911 occupations
```

có đầy đủ activity data cần thiết.

Do đó, project **không drop các occupation khỏi master table**.

Thay vào đó:

```text
occupations
    └── giữ toàn bộ occupation

occupation_features
    └── có thể có NULL đối với occupation
        không đủ Work Activities
```

Điều này giữ cho `occupations` là master dataset nhất quán.

Các occupation thiếu feature sẽ được xử lý ở bước Machine Learning sau, khi tạo dataset phù hợp cho K-Means.


---

## 3.7 OECD AI Exposure Integration

OECD data được đọc từ Excel workbook, trong đó sheet `Data` có nhiều dòng metadata/header trước phần dữ liệu occupation.

Dữ liệu thực tế bắt đầu từ row 4.


---

**OECD - O\*NET Mapping Validation**

Kết quả mapping:

```text
O*NET occupations:       1,016
OECD occupations:          879
Matched occupations:       879
Unmatched OECD:               0
O*NET without OECD:         137
```

Như vậy:

```text
879 / 1,016 ≈ 86.5%
```

occupation trong master table có AI exposure từ OECD.

137 occupation còn lại không có OECD exposure.

Các giá trị này được giữ là:

```text
NULL
```

thay vì gán:

```text
0
```

vì `NULL` biểu diễn **không có dữ liệu**, trong khi `0` sẽ mang nghĩa là exposure bằng 0.

---

# 4. Database Schema

Database hiện tại được thiết kế xoay quanh `occupation_code`.

```text
                    occupations
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
occupation_skills  occupation_education  occupation_tasks
        │
        │
        ▼
occupation_features

        occupations
             │
             ▼
   related_occupations
```

Các bảng chính:

```text
occupations
occupation_skills
occupation_education
occupation_tasks
occupation_features
related_occupations
```

---

## 4.1. Nguyên tắc xử lý dữ liệu

Một số nguyên tắc được thống nhất trong quá trình xây dựng database:

### 1. Không merge bằng occupation title

Sử dụng:

```text
occupation_code
```

làm identifier chính.

Không sử dụng `occupation_title` để merge dữ liệu giữa các nguồn.

---

### 2. Không biến missing thành zero

Ví dụ:

```text
Không có OECD exposure
        ≠
AI exposure = 0
```

Do đó các occupation không có OECD data giữ:

```text
NULL
```

---

### 3. Không tự động xóa các giá trị Level = 0

`LV = 0` được giữ vì đây có thể là giá trị hợp lệ trong dữ liệu O\*NET.

---

### 4. Phân biệt raw data và engineered features

Ví dụ:

```text
importance_score
level_score
```

là dữ liệu nguồn.

Trong khi:

```text
normalized_weight
cognitive_intensity
social_intensity
physical_operational_intensity
information_digital_intensity
administrative_routine_intensity
```

là feature được project xây dựng.

---

### 5. AI Exposure không được dùng để tạo occupational clusters

`ai_exposure_score` được lưu trong `occupations` nhưng không được sử dụng làm input cho K-Means.

Thiết kế dự kiến:

```text
O*NET occupational features
          ↓
       K-Means
          ↓
 Occupational Clusters
          ↓
Compare / Analyze
          ↑
 OECD AI Exposure
```

Như vậy AI Exposure được sử dụng như một **external indicator** để phân tích các cluster sau khi clustering, thay vì trực tiếp quyết định cluster.

---

## 4.2 Data Validation

Sau khi insert dữ liệu vào SQLite, database cần được kiểm tra trước khi chuyển sang EDA và Machine Learning.

Các nhóm validation chính:

### Primary key

Kiểm tra:

- duplicate `occupation_code`
- NULL primary key

### Foreign key

Kiểm tra các bảng con có occupation tồn tại trong `occupations`.

### Duplicate records

Đặc biệt kiểm tra:

```text
occupation + skill
occupation + education_category
occupation + task
occupation + related occupation
```

### Numerical range

Kiểm tra các giá trị:

```text
importance_score
level_score
normalized_weight
ai_exposure_score
occupational features
```

### Missing values

Kiểm tra missing theo từng bảng và phân biệt:

```text
Missing do source coverage
```

với:

```text
Missing do data processing error
```

---

# 5. PHASE 4 — EXPLORATORY DATA ANALYSIS

EDA tập trung vào việc hiểu dữ liệu trước khi machine learning.

## 5.1. Occupation distribution

Phân tích:

- số lượng occupations
- education
- job zone
- skill distribution

## 5.2. AI Exposure

Phân tích:

- distribution của AI Exposure
- occupation có exposure cao/thấp
- AI Exposure theo occupational characteristics

## 5.3. Feature relationships

Phân tích mối quan hệ giữa:

```text
Cognitive
Social
Physical
Digital
Routine
```

và:

```text
AI Exposure
```

Lưu ý:

> Correlation không đồng nghĩa với causation.

## 5.4. Skill analysis

Phân tích:

- skills phổ biến
- skill importance
- skill distribution giữa các occupation

EDA sẽ giúp xác định feature nào phù hợp cho clustering.

---
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

## 1. Tổng quan dự án

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

Chuyển dữ liệu Work Activities từ O*NET thành biểu diễn số phù hợp cho bài toán phân cụm nghề nghiệp. Phiên bản hiện tại sử dụng trực tiếp **41 Work Activities Level (LV)** để giữ thông tin ở cấp từng hoạt động công việc.

## 7.2. Các phương án biểu diễn

| Representation | Số chiều | Vai trò |
|---|---:|---|
| 41 Work Activities LV | 41 | Biểu diễn chính cho K-Means |
| PCA từ 41 Work Activities LV | 14 | Phương án so sánh sau giảm chiều |

### Phương án 1 — 41 Work Activities LV

Mỗi occupation là một vector gồm 41 giá trị Level, tương ứng 41 hoạt động trong O*NET (ví dụ: Handling and Moving Objects, Making Decisions and Solving Problems, Analyzing Data or Information, Working with Computers, Controlling Machines and Processes).

Giá trị được pivot từ bảng `occupation_work_activities` (dùng `occupation_code` và `activity`), và được chuẩn hóa trước khi chạy K-Means để hạn chế ảnh hưởng của khác biệt thang đo.

Lý do chọn:

- Giữ nguyên ý nghĩa của từng hoạt động công việc.
- Không phụ thuộc vào cách nhóm hoạt động thủ công.
- Hỗ trợ diễn giải cụm dựa trên các hoạt động cụ thể của O*NET.

### Phương án 2 — PCA trên 41 Work Activities LV

Quy trình:

1. Chuẩn hóa 41 feature LV.
2. Fit PCA trên dữ liệu đã chuẩn hóa.
3. Giữ lại 14 principal components (giải thích khoảng **90,32%** tổng phương sai).
4. Dùng các components làm đầu vào cho K-Means.
5. Khi diễn giải, quay lại 41 Work Activities LV gốc để mô tả đặc trưng cụm.

14 components là một cấu hình tiềm năng, không phải mô hình tối ưu được xác định trước. PCA giúp kiểm tra việc giảm chiều có cải thiện cấu trúc phân cụm hay không, nhưng các component khó diễn giải trực tiếp hơn hoạt động công việc gốc.

## 7.3. Điều chỉnh so với feature baseline ban đầu

Giai đoạn đầu, dự án xây dựng 5 feature tổng hợp (mean LV của các nhóm activities, là feature do project định nghĩa, không phải category chính thức của O*NET):

- `cognitive_intensity`
- `social_intensity`
- `physical_operational_intensity`
- `information_digital_intensity`
- `administrative_routine_intensity`

Do EDA cho thấy các feature này có tương quan cao và mức độ phụ thuộc vào cách nhóm thủ công, dự án **không tiếp tục dùng 5 feature tổng hợp làm baseline trong pipeline clustering chính**. 41 Work Activities LV được chọn làm biểu diễn chính, PCA 14 components được giữ để đối chiếu. Phương án mở rộng 41 IM + 41 LV chưa được thử nghiệm.

Quyết định này không có nghĩa các feature tổng hợp ban đầu là sai, mà phản ánh lựa chọn biểu diễn phù hợp hơn với phạm vi phân tích hiện tại.

---

# 8. Unsupervised Learning

## 8.1. Bài toán

> Các nghề nghiệp có thể được chia thành những nhóm nào dựa trên đặc điểm hoạt động công việc, và các nhóm đó khác nhau như thế nào về AI Exposure?

Đây là bài toán phân cụm không giám sát, không có target `y`.

- **Input:** 41 Work Activities LV cho mỗi occupation, đã chuẩn hóa.
- **Output:** nhãn cụm cho từng occupation, hồ sơ đặc trưng từng cụm, phân phối AI Exposure của các cụm sau khi ghép dữ liệu OECD.

## 8.2. Phương pháp K-Means

Quy trình thực nghiệm:

1. Xây dựng ma trận đặc trưng từ 41 Work Activities LV.
2. Chuẩn hóa dữ liệu.
3. Chạy K-Means với K từ 2 đến 8.
4. Đánh giá bằng các clustering metrics.
5. Lưu nhãn cụm để phục vụ phân tích và diễn giải.
6. Phân tích kết quả với K = 4 như phương án ứng viên.

| Metric | Ý nghĩa |
|---|---|
| Inertia | Tổng bình phương khoảng cách từ các điểm đến tâm cụm; thường giảm khi tăng K |
| Silhouette Score | Mức độ một điểm phù hợp với cụm của nó so với cụm khác; càng cao càng tốt |
| Davies–Bouldin Index | Mức độ tương đồng giữa các cụm; càng thấp càng tốt |
| Calinski–Harabasz Index | Độ phân tán giữa các cụm so với trong cụm; càng cao càng tốt |

Không có chỉ số đơn lẻ nào đủ để xác định K tối ưu. Việc chọn K còn xem xét kích thước cụm và khả năng diễn giải theo đặc điểm nghề nghiệp.

## 8.3. So sánh feature representations (K = 4)

| Metric | 41 Work Activities LV | PCA 14 components |
|---|---:|---:|
| Silhouette Score | 0.1747 | 0.2019 |
| Davies–Bouldin Index | 1.7008 | 1.5320 |
| Calinski–Harabasz Index | 281.26 | 345.38 |

PCA có Silhouette và Calinski–Harabasz cao hơn, Davies–Bouldin thấp hơn trong thực nghiệm K = 4 này. Tuy nhiên điều đó chưa đủ để kết luận PCA tốt nhất cho toàn dự án, vì khả năng diễn giải cụm theo hoạt động công việc gốc cũng quan trọng. Dự án ưu tiên 41 Work Activities LV làm biểu diễn chính; PCA là phương án so sánh.

Lưu ý: Silhouette Score ở mức thấp (khoảng 0.17–0.20) cho thấy cấu trúc cụm không tách biệt mạnh. Các nhóm nên được hiểu là cách chia dữ liệu hữu ích cho diễn giải, không phải ranh giới tự nhiên rõ ràng giữa các nghề.

## 8.4. Kết quả phân cụm với K = 4

| Cluster | Số occupation | Đặc trưng chính | Tên mô tả đề xuất |
|---|---:|---|---|
| 0 | 204 | Xử lý vật thể và hoạt động thể chất nói chung | General Physical & Material-Handling Occupations |
| 1 | 221 | Giao tiếp bên ngoài, quan hệ liên cá nhân, máy tính và hành chính | Interpersonal, Administrative & Digital-Service Occupations |
| 2 | 258 | Phân tích thông tin, tư vấn, chiến lược và điều phối tổ chức | Analytical, Advisory & Organizational-Leadership Occupations |
| 3 | 228 | Sửa chữa, kiểm soát máy móc, thiết bị và vận hành quy trình | Mechanical, Equipment & Process-Operations Occupations |
| **Tổng** | **911** | | |

Tên cụm được đặt dựa trên các hoạt động có điểm Level trung bình nổi bật **tương đối** so với các cụm khác, kết hợp với các occupation tiêu biểu gần tâm cụm. Đây là nhãn mô tả do dự án xây dựng, không phải nhóm nghề chính thức của O*NET.

**Cluster 0.** Nổi bật ở `Handling and Moving Objects` và `Performing General Physical Activities`; các hoạt động phân tích thông tin và tư vấn thấp hơn đáng kể.

**Cluster 1.** Nổi bật ở giao tiếp với người ngoài tổ chức, làm việc với máy tính, thu thập thông tin và duy trì quan hệ liên cá nhân; xử lý vật thể và kiểm soát máy móc tương đối thấp.

**Cluster 2.** Nổi bật ở tư vấn, phân tích dữ liệu, bố trí nhân sự, định hướng cấp dưới và xử lý thông tin.

**Cluster 3.** Nổi bật ở sửa chữa thiết bị cơ khí, kiểm soát máy móc, xử lý vật thể, vận hành thiết bị và kiểm tra thiết bị hoặc vật liệu.

---

# 9. AI Exposure Analysis by Cluster

## 9.1. Mục tiêu

Tìm hiểu liệu các nhóm occupation có đặc trưng hoạt động công việc khác nhau có phân phối AI Exposure khác nhau hay không.

## 9.2. Ghép dữ liệu và xử lý giá trị thiếu

Nhãn cụm được ghép với `ai_exposure_score` trong bảng `occupations` bằng `occupation_code`. Các occupation không có AI Exposure từ OECD vẫn được giữ trong dữ liệu phân cụm; giá trị thiếu không bị thay bằng 0. Khi tính thống kê theo cụm, chỉ dùng occupation có điểm hợp lệ, và số lượng cùng tỷ lệ coverage được báo cáo.

## 9.3. Kết quả

| Cluster | Occupations có dữ liệu | Coverage | Mean | Median |
|---|---:|---:|---:|---:|
| 0 | 198/204 | 97.06% | 0.922 | 0.931 |
| 1 | 210/221 | 95.02% | 0.822 | 0.822 |
| 2 | 247/258 | 95.74% | 0.768 | 0.763 |
| 3 | 224/228 | 98.25% | 0.870 | 0.874 |

- **Cluster 0** có mean và median AI Exposure cao nhất.
- **Cluster 3** đứng thứ hai.
- **Cluster 1** ở mức trung gian.
- **Cluster 2** thấp nhất.

Chênh lệch mean giữa Cluster 0 và Cluster 2 khoảng 0.154 điểm; chênh lệch median khoảng 0.168 điểm.

Không nên suy luận rằng hoạt động thể chất tự thân có mức phơi nhiễm AI cao hơn hoạt động trí óc. Các cụm hình thành từ nhiều đặc trưng cùng lúc, và AI Exposure là chỉ số độc lập được phân tích sau phân cụm.

## 9.4. Trực quan hóa và kiểm định thống kê

Boxplot được dùng để so sánh trung vị, độ phân tán và quan sát ngoại lệ của AI Exposure giữa bốn cụm.

Kiểm định Kruskal–Wallis:

```text
H statistic: 329.5844
p-value:     3.9255 × 10⁻⁷¹
```

Ở mức ý nghĩa 0.05, có bằng chứng thống kê rằng phân phối AI Exposure khác biệt giữa ít nhất một số cụm. Các kiểm định Mann–Whitney U theo từng cặp, hiệu chỉnh Holm, cho thấy cả sáu cặp cụm đều khác biệt có ý nghĩa ở mức 0.05.

Các kiểm định hậu nghiệm được lưu làm kết quả bổ sung; phần trình bày chính tập trung vào thống kê mô tả, biểu đồ và phát hiện tổng quát. Các kết quả này không chứng minh quan hệ nhân quả giữa đặc điểm công việc và AI Exposure, và không cho biết trực tiếp nghề nào sẽ bị AI thay thế.

## 9.5. Kết luận giai đoạn Unsupervised Learning

1. Phân cụm 911 occupation có Work Activities LV thành bốn nhóm (K = 4).
2. Diễn giải các cụm bằng đặc trưng hoạt động công việc và occupation tiêu biểu.
3. Ghép OECD AI Exposure với nhãn cụm bằng `occupation_code`.
4. Tính thống kê mô tả và trực quan hóa phân phối AI Exposure.
5. Ghi nhận khác biệt có ý nghĩa thống kê về AI Exposure giữa các cụm.

Kết quả này là nền tảng cho giai đoạn tiếp theo: Skill Similarity và Career Transition Analysis.

---

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


