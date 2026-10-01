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

# 3. Occupation Master Data

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

# 4. Occupation Education

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

# 5. Occupation Skills

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

## 5.1. Kiểm tra Skill Data

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

# 6. Normalized Skill Weight

Ngoài các giá trị gốc `importance_score` và `level_score`, project tạo thêm một feature:

```text
normalized_weight
```

Đây là **feature engineered của project**, không phải trường dữ liệu gốc.

Đầu tiên tính:

```text
raw_weight = importance_score × level_score
```

Sau đó chuẩn hóa trong phạm vi từng occupation:

```text
normalized_weight
=
raw_weight
/
Σ raw_weight của occupation
```

Mục đích là tạo một vector skill representation cho mỗi occupation.

Ví dụ về mặt khái niệm:

```text
Occupation A

Skill 1 → 0.30
Skill 2 → 0.25
Skill 3 → 0.20
Skill 4 → 0.15
Skill 5 → 0.10
             ────
              1.00
```

Vector này sẽ được sử dụng ở giai đoạn sau cho:

- Skill similarity
- Cosine similarity
- Skill gap analysis
- Career transition analysis

---

# 7. Occupation Work Activities

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

# 8. Work Activities: Importance và Level

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

# 9. Work Activity Feature Engineering

Từ O\*NET Work Activities, project xây dựng 5 occupational dimensions.

Các dimensions này là **feature do project định nghĩa**, không phải năm category chính thức của O\*NET.

Feature được tính bằng cách lấy mean `LV` của các activities thuộc cùng một nhóm.

## 9.1. Cognitive Intensity

```python
cognitive_intensity = mean(LV)
```

Activities gồm:

```text
Making Decisions and Solving Problems
Thinking Creatively
Developing Objectives and Strategies
Updating and Using Relevant Knowledge
Judging the Qualities of Objects, Services, or People
Evaluating Information to Determine Compliance with Standards
Estimating the Quantifiable Characteristics of Products, Events, or Information
Organizing, Planning, and Prioritizing Work
```

---

## 9.2. Social Intensity

```python
social_intensity = mean(LV)
```

Activities gồm:

```text
Communicating with Supervisors, Peers, or Subordinates
Communicating with People Outside the Organization
Establishing and Maintaining Interpersonal Relationships
Assisting and Caring for Others
Selling or Influencing Others
Resolving Conflicts and Negotiating with Others
Performing for or Working Directly with the Public
Coordinating the Work and Activities of Others
Developing and Building Teams
Training and Teaching Others
Guiding, Directing, and Motivating Subordinates
Coaching and Developing Others
Providing Consultation and Advice to Others
Interpreting the Meaning of Information for Others
Staffing Organizational Units
```

---

## 9.3. Physical / Operational Intensity

```python
physical_operational_intensity = mean(LV)
```

Activities gồm:

```text
Performing General Physical Activities
Handling and Moving Objects
Operating Vehicles, Mechanized Devices, or Equipment
Repairing and Maintaining Mechanical Equipment
Repairing and Maintaining Electronic Equipment
Controlling Machines and Processes
Inspecting Equipment, Structures, or Materials
```

---

## 9.4. Information / Digital Intensity

```python
information_digital_intensity = mean(LV)
```

Activities gồm:

```text
Working with Computers
Processing Information
Analyzing Data or Information
Getting Information
Drafting, Laying Out, and Specifying Technical Devices, Parts, and Equipment
```

Tên `information_digital_intensity` được sử dụng để tránh diễn giải tất cả activities trong nhóm này là inherently digital.

---

## 9.5. Administrative / Routine Intensity

```python
administrative_routine_intensity = mean(LV)
```

Activities gồm:

```text
Monitoring Processes, Materials, or Surroundings
Scheduling Work and Activities
Documenting/Recording Information
Performing Administrative Activities
Monitoring and Controlling Resources
Identifying Objects, Actions, and Events
```

Đây cũng là một dimension được project định nghĩa dựa trên semantic grouping của Work Activities.

---

# 10. Kiểm tra Coverage của Work Activities

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

# 11. Correlation Analysis của Occupational Features

Sau khi xây dựng 5 features, correlation matrix được kiểm tra để đánh giá mối quan hệ giữa các dimensions.

Kết quả hiện tại:

|                            | Cognitive | Social | Physical/Operational | Information/Digital | Administrative/Routine |
| -------------------------- | --------: | -----: | -------------------: | ------------------: | ---------------------: |
| **Cognitive**              |      1.00 |   0.81 |                -0.11 |                0.89 |                   0.89 |
| **Social**                 |      0.81 |   1.00 |                -0.20 |                0.66 |                   0.85 |
| **Physical/Operational**   |     -0.11 |  -0.20 |                 1.00 |               -0.12 |                  -0.02 |
| **Information/Digital**    |      0.89 |   0.66 |                -0.12 |                1.00 |                   0.82 |
| **Administrative/Routine** |      0.89 |   0.85 |                -0.02 |                0.82 |                   1.00 |

Kết quả cho thấy:

- Physical/Operational có tương quan thấp với các dimensions còn lại.
- Cognitive, Information/Digital và Administrative/Routine có tương quan cao.
- Social cũng có tương quan tương đối cao với Cognitive và Administrative/Routine.

Correlation cao không được xem ngay là lỗi dữ liệu.

Nó có thể phản ánh việc nhiều đặc điểm hoạt động nghề nghiệp cùng tăng ở các occupation có mức độ công việc phức tạp cao.

Việc đánh giá sâu hơn bằng PCA và các phương pháp khác sẽ được thực hiện ở giai đoạn feature analysis / Machine Learning.

---

# 12. OECD AI Exposure Integration

OECD data được đọc từ Excel workbook, trong đó sheet `Data` có nhiều dòng metadata/header trước phần dữ liệu occupation.

Dữ liệu thực tế bắt đầu từ row 4.



---

# 13. OECD - O\*NET Mapping Validation

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

# 14. Database Schema

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

# 15. Indexes

Các index được sử dụng để hỗ trợ truy vấn theo occupation:

```sql
```

---

# 16. Nguyên tắc xử lý dữ liệu

Một số nguyên tắc được thống nhất trong quá trình xây dựng database:

### 16.1. Không merge bằng occupation title

Sử dụng:

```text
occupation_code
```

làm identifier chính.

Không sử dụng `occupation_title` để merge dữ liệu giữa các nguồn.

---

### 16.2. Không biến missing thành zero

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

### 16.3. Không tự động xóa các giá trị Level = 0

`LV = 0` được giữ vì đây có thể là giá trị hợp lệ trong dữ liệu O\*NET.

---

### 16.4. Phân biệt raw data và engineered features

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

### 16.5. AI Exposure không được dùng để tạo occupational clusters

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

# 17. Data Validation

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

# 18. Current Data Pipeline Status

Tại thời điểm hiện tại, phần Data Preparation đã hoàn thành các bước chính:

```text
[✓] Collect O*NET data
[✓] Collect OECD AI Exposure
[✓] Clean occupation data
[✓] Establish occupation_code as central identifier
[✓] Process occupation skills
[✓] Calculate normalized skill weights
[✓] Process education distribution
[✓] Process Work Activities
[✓] Separate IM and LV
[✓] Engineer occupational activity features
[✓] Integrate OECD AI Exposure
[✓] Design SQLite database
[✓] Insert data into database
[ ] Complete automated database validation
[ ] EDA
[ ] Feature analysis / PCA
[ ] StandardScaler
[ ] K-Means
[ ] Skill similarity
[ ] Skill gap analysis
[ ] Career transition analysis
```

---

# 19. Next Step

Sau khi database validation hoàn tất, pipeline sẽ chuyển sang:

```text
SQLite Database
       ↓
      EDA
       ↓
Feature Validation
       ↓
StandardScaler
       ↓
K-Means Clustering
       ↓
Cluster Analysis
       ↓
OECD AI Exposure Analysis
       ↓
Skill Similarity
       ↓
Skill Gap
       ↓
Career Transition Analysis
```

Database được thiết kế theo hướng **giữ dữ liệu nguồn càng đầy đủ càng tốt**, trong khi các bước xử lý dành cho Machine Learning sẽ được thực hiện ở downstream pipeline. Điều này cho phép thay đổi cách feature engineering hoặc clustering mà không cần xây dựng lại toàn bộ dữ liệu gốc.


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
