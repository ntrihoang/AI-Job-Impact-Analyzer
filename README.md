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
Cluster 0
Cluster 1
Cluster 2
Cluster 3
Cluster 4

chúng ta lấy OECD Exposure và gắn vào:
Occupation       Cluster       AI Exposure
------------------------------------------------
Accountant       Cluster 0       0.XX
Data Scientist   Cluster 0       0.XX
Nurse            Cluster 1       0.XX
Teacher          Cluster 1       0.XX
Electrician      Cluster 3       0.XX
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


# 4. Nguồn dữ liệu

## 4.1. O*NET

O*NET là nguồn dữ liệu nghề nghiệp chính của dự án.

Sử dụng O*NET để thu thập:

- Occupations
- Skills
- Knowledge
- Abilities
- Work Activities
- Tasks
- Education / Job Zone
- Related Occupations

O*NET-SOC Code được sử dụng làm occupation identifier.

---

## 4.2. OECD

OECD được sử dụng để lấy:

- AI Exposure
- Các AI capability indicators/gaps nếu dữ liệu phù hợp được sử dụng

AI Exposure được xem là một **external occupational indicator**, không phải target để train supervised model.

---

# 5. PHASE 1 — THU THẬP & THIẾT KẾ DỮ LIỆU

## 5.1. Dữ liệu cần thu thập

| Source | Dữ liệu | Mục đích |
|---|---|---|
| O*NET | Occupations | Định danh và mô tả occupation |
| O*NET | Skills | Skill similarity / career transition |
| O*NET | Tasks | Phân tích đặc điểm công việc |
| O*NET | Knowledge | Feature engineering / clustering |
| O*NET | Abilities | Feature engineering / clustering |
| O*NET | Work Activities | Xây dựng occupational features |
| O*NET | Education / Job Zone | Phân tích yêu cầu nghề |
| O*NET | Related Occupations | Baseline cho career recommendation |
| OECD | AI Exposure | Phân tích AI impact/exposure |

---

# 6. DATABASE DESIGN

MVP sử dụng 5 bảng chính.

```text
occupations
    │
    ├── occupation_skills
    │
    ├── occupation_tasks
    │
    ├── occupation_features
    │
    └── related_occupations
```

OECD AI Exposure được tích hợp vào `occupations`.

---

## 6.1. Bảng `occupations`

Đây là bảng master occupation.

```text
occupations
├── occupation_code          PK
├── occupation_title
├── description
├── education_level
├── job_zone
└── ai_exposure_score
```

### Nguồn dữ liệu

O*NET:

- occupation_code
- occupation_title
- description
- education_level
- job_zone

OECD:

- ai_exposure_score

### Vai trò

Mỗi record đại diện cho một occupation.

Ví dụ:

```text
occupation_code: 11-3031.00
occupation_title: Financial Managers
...
ai_exposure_score: ...
```

`occupation_code` là key dùng để liên kết dữ liệu.

---

## 6.2. Bảng `occupation_skills`

Lưu các skill của từng occupation.

```text
occupation_skills
├── occupation_code          FK
├── skill_id
├── skill_name
├── importance_score
├── level_score
└── normalized_weight
```

### Nguồn

O*NET Skills.

### Ý nghĩa

Một occupation có nhiều skills.

Ví dụ:

```text
Financial Manager
│
├── Critical Thinking
├── Mathematics
├── Reading Comprehension
├── Active Listening
├── Management of Financial Resources
└── ...
```

Các trường:

- `importance_score`: mức độ quan trọng của skill
- `level_score`: mức độ yêu cầu
- `normalized_weight`: feature do nhóm tự xây dựng từ dữ liệu O*NET

`normalized_weight` sẽ được sử dụng để tạo skill vector.

---

## 6.3. Bảng `occupation_tasks`

Lưu các task của occupation.

```text
occupation_tasks
├── task_id                  PK
├── occupation_code          FK
├── task_statement
└── task_type
```

### Nguồn

O*NET Tasks.

### Ví dụ

```text
Accountant
│
├── Prepare financial statements
├── Analyze financial data
├── Maintain accounting records
└── ...
```

`task_type` có thể lưu loại task tương ứng từ O*NET.

Bảng này chủ yếu phục vụ:

- EDA
- occupation analysis
- mô tả occupation

Không nhất thiết phải đưa toàn bộ task text trực tiếp vào K-Means trong MVP.

---

## 6.4. Bảng `occupation_features`

Đây là bảng chứa các occupational features được nhóm feature engineering.

```text
occupation_features
├── occupation_code          PK/FK
├── cognitive_intensity
├── social_intensity
├── physical_intensity
├── digital_intensity
└── routine_intensity
```

Các feature này không nhất thiết là raw field có sẵn trong O*NET.

Nhóm sẽ xây dựng chúng từ:

- Work Activities
- Abilities
- Skills
- Knowledge
- các occupational descriptors phù hợp

Ví dụ:

```text
O*NET
   ↓
Selected descriptors
   ↓
Feature Engineering
   ↓
cognitive_intensity
social_intensity
physical_intensity
digital_intensity
routine_intensity
```

Các feature này có thể được sử dụng làm input cho clustering.

---

## 6.5. Bảng `related_occupations`

Lưu các occupation liên quan được O*NET cung cấp.

```text
related_occupations
├── occupation_code             FK
├── related_occupation_code     FK
├── relatedness_tier
└── index_order
```

### Nguồn

O*NET Related Occupations.

### Vai trò

Không dùng trực tiếp làm recommendation chính.

Thay vào đó, sử dụng nó làm **baseline/reference** để đánh giá hệ thống career transition của nhóm.

Ví dụ:

```text
Accountant

O*NET Related Occupations
→ Financial Analyst
→ Financial Manager
→ Budget Analyst
```

So sánh với:

```text
Our Skill Similarity Model
→ Financial Analyst
→ Financial Manager
→ Data Analyst
```

Sau đó đánh giá mức độ overlap giữa hai kết quả.

---

# 7. DATA RELATIONSHIP

Quan hệ chính:

```text
occupations
     │
     ├──────────────< occupation_skills
     │
     ├──────────────< occupation_tasks
     │
     ├──────────────1 occupation_features
     │
     └──────────────< related_occupations
```

Trong đó:

```text
occupation_code
```

là key trung tâm.

Không merge occupation dựa trên title.

Nguyên tắc:

> Occupation Code = identifier/key  
> Occupation Title = display label

Nếu hai source sử dụng classification khác nhau thì phải có crosswalk hợp lệ trước khi merge.

---

# 8. PHASE 2 — DATA CLEANING & INTEGRATION

Các bước:

### Step 1 — Kiểm tra dữ liệu O*NET

- Missing values
- Duplicate occupations
- Duplicate skills
- Invalid occupation codes
- Invalid numerical values

### Step 2 — Chuẩn hóa occupation

Dùng:

```text
occupation_code
```

làm primary identifier.

### Step 3 — Merge OECD

Ghép:

```text
O*NET occupation
        +
OECD AI Exposure
```

dựa trên occupation code tương ứng.

### Step 4 — Kiểm tra coverage

Kiểm tra:

```text
Number of O*NET occupations
Number matched with OECD
Number unmatched
```

### Step 5 — Chuẩn hóa numerical features

Đặc biệt trước khi clustering cần scale các feature.

Ví dụ:

```text
StandardScaler
```

hoặc phương pháp phù hợp khác.

---

# 9. PHASE 3 — FEATURE ENGINEERING

## 9.1. Skill Weight

Từ:

```text
importance_score
level_score
```

xây dựng:

```text
normalized_weight
```

để biểu diễn mức độ quan trọng tương đối của skill đối với occupation.

---

## 9.2. Occupational Intensity Features

Xây dựng:

```text
cognitive_intensity
social_intensity
physical_intensity
digital_intensity
routine_intensity
```

Các feature này sẽ giúp mô tả occupation ở cấp độ tổng quát.

---

## 9.3. Skill Matrix

Chuyển dữ liệu dạng:

```text
Occupation | Skill | Weight
```

thành:

```text
             Skill A  Skill B  Skill C  Skill D ...
Occupation A   0.82     0.54     0.91     0.20
Occupation B   0.21     0.88     0.45     0.71
Occupation C   0.67     0.32     0.77     0.52
```

Đây là input cho career similarity.

---

# 10. PHASE 4 — EXPLORATORY DATA ANALYSIS

EDA tập trung vào việc hiểu dữ liệu trước khi machine learning.

## 10.1. Occupation distribution

Phân tích:

- số lượng occupations
- education
- job zone
- skill distribution

## 10.2. AI Exposure

Phân tích:

- distribution của AI Exposure
- occupation có exposure cao/thấp
- AI Exposure theo occupational characteristics

## 10.3. Feature relationships

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

## 10.4. Skill analysis

Phân tích:

- skills phổ biến
- skill importance
- skill distribution giữa các occupation

EDA sẽ giúp xác định feature nào phù hợp cho clustering.

---

# 11. PHASE 5 — UNSUPERVISED LEARNING

## 11.1. Bài toán

Câu hỏi:

> Các occupation có thể tự nhiên được chia thành những nhóm nào dựa trên đặc điểm nghề nghiệp?

Input:

```text
Occupation Features
+
Selected O*NET Features
```

Không có target `y`.

---

## 11.2. K-Means Clustering

Thử nhiều giá trị K:

```text
K = 3
K = 4
K = 5
K = 6
K = 7
```

Đánh giá bằng:

- Elbow Method
- Silhouette Score
- Davies-Bouldin Index

Không chọn K chỉ dựa vào một metric.

---

## 11.3. Cluster Interpretation

Sau khi clustering, phân tích từng cluster:

```text
Cluster 0
→ Cognitive + Digital intensive

Cluster 1
→ Social + Cognitive intensive

Cluster 2
→ Physical + Routine intensive

Cluster 3
→ Technical + Physical intensive
```

Tên trên chỉ là ví dụ.

Tên thực tế phải được xác định **sau khi xem dữ liệu và centroid**.

---

# 12. PHASE 6 — AI EXPOSURE ANALYSIS

Sau khi có cluster:

```text
Occupation
    ↓
Cluster
    ↓
OECD AI Exposure
```

Tính:

- Mean AI Exposure
    
- Median AI Exposure
    
- Distribution
    
- Range
    

theo từng cluster.

Ví dụ:

```text
Cluster 0 → Mean Exposure = 0.78
Cluster 1 → Mean Exposure = 0.61
Cluster 2 → Mean Exposure = 0.38
Cluster 3 → Mean Exposure = 0.31
```

Các con số trên chỉ là ví dụ minh họa.

Mục tiêu là trả lời câu hỏi:

> Các nhóm occupation có đặc điểm khác nhau có mức AI Exposure khác nhau như thế nào?

---

# 13. PHASE 7 — CAREER TRANSITION

## 13.1. Skill Similarity

Mỗi occupation được biểu diễn thành một skill vector.

Ví dụ:

```text
Accountant
[
  Mathematics = 0.82,
  Critical Thinking = 0.91,
  Reading = 0.88,
  ...
]
```

Tính:

```text
Cosine Similarity
```

giữa occupation hiện tại và toàn bộ occupation khác.

---

## 13.2. Top-K Similar Occupations

Ví dụ:

```text
Accountant

1. Financial Analyst       0.91
2. Financial Manager       0.86
3. Data Analyst            0.79
4. Business Analyst        0.77
5. ...
```

Không gọi trực tiếp đây là "nghề tốt hơn".

Đây là:

> Occupations có skill profile tương đồng.

---

# 14. PHASE 8 — SKILL GAP ANALYSIS

Sau khi tìm occupation tương đồng:

```text
Current Occupation
        ↓
Candidate Occupation
        ↓
Compare Skill Vectors
        ↓
Skill Gap
```

Ví dụ:

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

Mục tiêu:

> Giải thích tại sao hai occupation tương đồng hoặc khác nhau.

Không chỉ đưa ra một con số similarity.

---

# 15. PHASE 9 — EVALUATION

## 15.1. Clustering Evaluation

Sử dụng:

- Silhouette Score
    
- Davies-Bouldin Index
    
- Elbow Method
    
- Cluster interpretability
    

---

## 15.2. Career Recommendation Evaluation

Dùng O*NET Related Occupations làm baseline.

Ví dụ:

```text
O*NET Related Occupations
        VS
Our Skill Similarity
```

Đánh giá:

- Top-K overlap
    
- số occupation chung
    
- similarity distribution
    
- các trường hợp recommendation khác biệt
    

Mục tiêu không phải chứng minh hệ thống "đúng tuyệt đối", mà kiểm tra xem skill-based similarity có tạo ra các kết quả hợp lý và có mức tương đồng nhất định với dữ liệu tham chiếu hay không.

---

# 16. PHASE 10 — VISUALIZATION & STREAMLIT

Dashboard chỉ là lớp trình bày kết quả.

Có thể xây dựng một giao diện đơn giản gồm:

### Occupation Explorer

Người dùng chọn occupation.

Hiển thị:

- Occupation information
    
- Skills
    
- Education
    
- Job Zone
    
- AI Exposure
    
- Cluster
    

### Occupation Clusters

Hiển thị:

- cluster distribution
    
- PCA visualization
    
- cluster characteristics
    
- AI Exposure distribution
    

### Career Transition

Người dùng chọn occupation.

Hiển thị:

```text
Candidate Occupation
Skill Similarity
AI Exposure
Skill Gap
Education
```

Không cần xây dựng dashboard quá phức tạp.

---

# 17. TOÀN BỘ PIPELINE

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
        Occupation Features      Skill Matrix
             │                       │
             ↓                       ↓
          K-MEANS             COSINE SIMILARITY
             │                       │
             ↓                       ↓
     OCCUPATION CLUSTERS       CAREER TRANSITION
             │                       │
             └───────────┬───────────┘
                         ↓
                  AI EXPOSURE
                    ANALYSIS
                         ↓
                    EVALUATION
                         ↓
                  VISUALIZATION
                         ↓
                    STREAMLIT
```

---

# 18. Câu hỏi nghiên cứu chính

Dự án có thể xoay quanh các câu hỏi:

### RQ1

**Các occupation có thể được phân nhóm như thế nào dựa trên đặc điểm kỹ năng và công việc?**

### RQ2

**Các occupation cluster khác nhau có mức AI Exposure khác nhau như thế nào?**

### RQ3

**Có thể sử dụng skill similarity để xác định các occupation có tiềm năng chuyển đổi nghề nghiệp không?**

### RQ4

**Các recommendation dựa trên skill similarity có mức độ tương đồng như thế nào với O*NET Related Occupations?**

### RQ5

**Skill gap giữa occupation hiện tại và occupation mục tiêu có thể được sử dụng để giải thích career transition như thế nào?**

---

# 19. Kết quả đầu ra của MVP

Sau khi hoàn thành MVP, nhóm cần có:

### Dataset

```text
occupations
occupation_skills
occupation_tasks
occupation_features
related_occupations
```

### Data Science results

```text
EDA
+
Feature Engineering
+
K-Means Clustering
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

Người dùng chọn:

```text
Accountant
```

Hệ thống có thể trả về:

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

# 20. Hướng mở rộng sau MVP

Chỉ sau khi MVP hoàn thành mới cân nhắc mở rộng:

```text
MVP
│
├── Vietnam labor market
│
├── ILO
│
├── Job postings
│
├── Real-world demand
│
├── Salary
│
├── Employment trends
│
└── More advanced recommendation
```

Đặc biệt, nếu sau này thêm Việt Nam, có thể nghiên cứu:

```text
Global occupation analysis
        +
Vietnam labor market data
        ↓
Vietnam-specific career transition analysis
```
