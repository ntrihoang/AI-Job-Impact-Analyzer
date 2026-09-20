
# 1. Tổng quan dự án

### Tên

> **AI Job Impact & Career Transition Analyzer**

### Ý tưởng

Xây dựng một hệ thống phân tích nghề nghiệp nhằm:

1. Phân tích mức độ **tiếp xúc với AI (AI exposure)** của các nghề.
    
2. Tìm hiểu những **đặc điểm/kỹ năng/nhiệm vụ** liên quan đến mức AI exposure.
    
3. Khi người dùng chọn một nghề hiện tại, hệ thống tìm các nghề khác có **mức độ tương đồng về kỹ năng** để gợi ý các hướng chuyển đổi nghề nghiệp.
    

### MVP KHÔNG cố gắng:

- dự đoán nghề nào chắc chắn biến mất;
    
- dự đoán chính xác tương lai 5–10 năm;
    
- phân tích CV bằng NLP;
    
- xây dựng recommendation system phức tạp;
    
- xây dựng knowledge graph;
    
- chatbot AI;
    
- thu thập hàng trăm nghìn job postings.
    

Thay vào đó, MVP tập trung vào:

> **Occupation → AI Exposure Analysis → Skill Similarity → Career Transition**

---

# 2. Research Question

Đây là phần rất quan trọng vì nó quyết định dataset và model.

Mình đề xuất 2 câu hỏi:

### RQ1 — AI impact

> **What occupational characteristics are associated with higher levels of AI exposure?**

Tức là:

> Những đặc điểm nào của nghề liên quan đến mức độ tiếp xúc với AI cao?

### RQ2 — Career transition

> **Can transferable skills be used to identify occupations that are similar enough to support potential career transitions?**

Tức là:

> Có thể sử dụng transferable skills để tìm những nghề có mức độ tương đồng cao nhằm hỗ trợ chuyển đổi nghề nghiệp hay không?

Hai câu này **vừa sức với kiến thức hiện tại của nhóm**.

---

# 3. Kiến trúc MVP

```text
                   DATA SOURCES
                       │
          ┌────────────┼─────────────┐
          ↓            ↓             ↓
       O*NET          OECD          ILO
       Jobs/Skills    AI Exposure   AI Exposure
          │            │             │
          └────────────┼─────────────┘
                       ↓
                DATA INTEGRATION
                       ↓
                CLEAN DATASET
                       ↓
             ┌─────────┴──────────┐
             ↓                    ↓
       AI IMPACT ANALYSIS   SKILL ANALYSIS
             ↓                    ↓
       ML / Statistics      Similarity
             │                    │
             └─────────┬──────────┘
                       ↓
                  STREAMLIT
                       ↓
                 USER INTERFACE
```

---

1 thu thập data
2 tiền xử lí 
3 eda 
4 model
5 app

# 4. DATA — Phần quan trọng nhất

MVP sẽ **không tự thu thập mọi thứ từ Internet**.

Chúng ta sử dụng các nguồn có cấu trúc trước.

## 4.1. Nguồn 1 — O*NET

Đây là nguồn chính.

O*NET 31.0 cung cấp:

- Occupations
    
- Tasks
    
- Essential Skills
    
- Transferable Skills
    
- Work Activities
    
- Education
    
- Abilities
    
- Related Occupations
    
- Job titles
    
- Software skills
    

O*NET hiện cung cấp dữ liệu dưới CSV, Excel, JSON và nhiều format khác. ([O*NET Center](https://www.onetcenter.org/database.html?utm_source=chatgpt.com "O*NET Database at O*NET Resource Center"))

### Các bảng MVP cần lấy

Không lấy toàn bộ database.

Chỉ lấy:

### `Occupation Data`

```text
occupation_code
occupation_title
description
```

### `Task Statements`

```text
occupation_code
task_id
task
task_type
```

O*NET hiện có 18.838 task statements. ([O*NET Center](https://www.onetcenter.org/dictionary/31.0/csv/task_statements.html?utm_source=chatgpt.com "Task Statements - O*NET 31.0 Data Dictionary at O*NET Resource Center"))

### `Essential Skills`

```text
occupation_code
skill
importance
level
```

### `Transferable Skills`

```text
occupation_code
skill
importance
level
```

### `Related Occupations`

```text
occupation_code
related_occupation_code
```

---

# 5. Nguồn 2 — AI Exposure

MVP nên có **một nguồn AI exposure chính**, thay vì lấy 5 nguồn rồi trộn lung tung.

Mình đề xuất:

### OECD AI Exposure Measure

OECD xây dựng chỉ số bằng cách mapping AI capabilities với occupational requirements và tạo **AI Capability Gap**. Khoảng cách thấp hơn tương ứng với mức AI exposure tiềm năng cao hơn. OECD cũng cung cấp dataset XLSX. ([OECD](https://www.oecd.org/en/publications/the-oecd-ai-exposure-measure_f3da0f0a-en.html?utm_source=chatgpt.com "The OECD AI exposure measure | OECD"))

Dataset của chúng ta có thể cần:

```text
occupation
ai_exposure
source
```

Sau đó chuẩn hóa:

```text
AI Exposure
0 → 1
```

hoặc:

```text
0 → 100
```

---

# 6. ILO dùng để làm gì?

Không nhất thiết đưa ILO vào model ngay.

ILO 2025 có một phương pháp đánh giá GenAI exposure ở cấp occupational/task, dựa trên task-level data, expert input và AI predictions; nghiên cứu bao phủ gần 30.000 tasks trong hệ thống phân loại nghề nghiệp được nghiên cứu. ([International Labour Organization](https://www.ilo.org/publications/generative-ai-and-jobs-2025-update?utm_source=chatgpt.com "Generative AI and jobs: A 2025 update | International Labour Organization"))

Vì vậy trong MVP, ILO có thể dùng để:

### Option A

Làm **nguồn kiểm chứng/benchmark** cho OECD.

### Option B

Bổ sung AI exposure nếu mapping occupation phù hợp.

### Option C

Dùng trong phần Discussion:

> "Our findings are compared with existing occupational exposure estimates from ILO."

Điều này rất tốt cho report.

---

# 7. Dữ liệu Việt Nam

ILO đã công bố một brief năm 2026 áp dụng global occupational exposure index vào **Vietnam Labour Force Survey 2024**, phân tích exposure theo ngành, nghề và nhiều đặc điểm của lực lượng lao động Việt Nam. ([International Labour Organization](https://www.ilo.org/publications/generative-ai-and-jobs-viet-nam-labour-market-exposure-and-policy?utm_source=chatgpt.com "Generative AI and jobs in Viet Nam: Labour market exposure and policy considerations | International Labour Organization"))

Nhưng:

### MVP:

**Không cần biến toàn bộ dataset thành Việt Nam.**

Có thể dùng nguồn Việt Nam ở phần:

> Context / Discussion

Ví dụ:

> "The global occupation-level analysis is contextualized using ILO's 2026 assessment of GenAI exposure in Viet Nam."

Sau này nếu còn thời gian mới làm mapping:

```text
O*NET-SOC
      ↓
ISCO
      ↓
Vietnam occupation
```

---

# 8. Dataset cuối cùng

Sau khi merge, chúng ta muốn có **một bảng phân tích chính**:

### `occupation_master.csv`

Ví dụ:

|Column|Ý nghĩa|
|---|---|
|occupation_code|Mã nghề|
|occupation_name|Tên nghề|
|ai_exposure|AI exposure|
|education_level|Trình độ|
|skill_1|Skill|
|skill_2|Skill|
|...|...|
|related_occupation|Nghề liên quan|

Nhưng thực tế **không nên lưu skill thành hàng chục cột ngay từ đầu**.

Tốt hơn:

### `occupations.csv`

```text
occupation_code
occupation_name
ai_exposure
education
```

### `occupation_skills.csv`

```text
occupation_code
skill
importance
level
```

### `occupation_tasks.csv`

```text
occupation_code
task_id
task
task_type
```

### `related_occupations.csv`

```text
occupation_code
related_occupation_code
```

---

# 9. Data preprocessing

Đây là phần nhóm bạn đã học → tận dụng tối đa.

## Bước 1 — Remove duplicates

```text
occupation_code
```

phải unique ở bảng occupation.

---

## Bước 2 — Missing values

Kiểm tra:

```text
occupation
skills
AI exposure
education
```

---

## Bước 3 — Normalize occupation names

Ví dụ:

```text
"Accountants and Auditors"
"Accountant"
```

không được tự động coi là hai occupation khác nhau nếu chúng thực chất cùng mapping.

---

## Bước 4 — Normalize skills

Ví dụ:

```text
Data Analysis
Data Analysis
data analysis
DATA ANALYSIS
```

→ một skill.

---

## Bước 5 — Normalize AI exposure

Nếu nguồn có scale:

```text
0–1
```

thì giữ nguyên.

Nếu:

```text
0–100
```

thì normalize:

```python
df["ai_exposure_norm"] = df["ai_exposure"] / 100
```

---

# 10. EDA

Đây sẽ là một trong những phần **quan trọng nhất của project**.

## EDA 1 — Distribution

```text
AI Exposure distribution
```

Histogram.

Câu hỏi:

> AI exposure phân bố như thế nào giữa các occupation?

---

## EDA 2 — Top occupations

Ví dụ:

```text
Highest AI Exposure
────────────────────
Occupation A  0.89
Occupation B  0.87
Occupation C  0.85
...
```

Không nên kết luận:

> "Các nghề này sẽ bị thay thế."

Chỉ nên nói:

> "Các nghề này có mức exposure cao hơn theo chỉ số được sử dụng."

OECD cũng nhấn mạnh rằng exposure không phải là dự báo chắc chắn về việc làm bị thay thế; tác động thực tế còn phụ thuộc vào adoption, regulation, organizational change và social choices. ([OECD](https://www.oecd.org/en/publications/the-oecd-ai-exposure-measure_f3da0f0a-en.html?utm_source=chatgpt.com "The OECD AI exposure measure | OECD"))

---

# 11. EDA 3 — Skill vs AI exposure

Đây có thể là **biểu đồ quan trọng nhất**.

Ví dụ:

```text
AI Exposure
1.0 │                    ● ●
    │                ● ●
0.8 │           ● ●
    │
0.6 │       ●
    │
0.4 │   ●
    └────────────────────────
       Digital skill intensity
```

Câu hỏi:

> Có mối quan hệ giữa digital/technical skill và AI exposure không?

---

# 12. EDA 4 — Task characteristics

Nếu có dữ liệu task/work activity:

Phân tích:

```text
Occupation
       ↓
Tasks
       ↓
Task characteristics
       ↓
AI exposure
```

Ví dụ phân nhóm:

```text
Routine information processing
Administrative
Social interaction
Physical
Analytical
Creative
```

Nhưng **đừng tự gán hàng chục nghìn task vào category bằng tay**.

MVP có thể sử dụng các work activity/skill đã có trong O*NET thay vì tự xây task taxonomy.

---

# 13. Feature Engineering

Đây là bước chuyển từ raw dataset → ML dataset.

Ví dụ chúng ta tạo:

```text
digital_skill_score
analytical_skill_score
social_skill_score
creative_skill_score
management_skill_score
routine_activity_score
education_level
```

Sau đó:

```text
X =
[
 digital_skill_score,
 analytical_skill_score,
 social_skill_score,
 creative_skill_score,
 social_skill_score,
 education_level
]
```

Target:

```text
y = ai_exposure
```

---

# 14. Nhưng có một thay đổi mình muốn đề xuất

### Đừng biến ML thành mục tiêu chính.

Nếu OECD đã cung cấp AI exposure score, chúng ta không cần nói:

> "ML tiên đoán AI exposure của nghề."

Một MVP tốt hơn là:

## Experiment A

**Statistical analysis**

```text
Skill → AI exposure
```

## Experiment B

**ML regression**

```text
Occupation features
       ↓
ML
       ↓
Predicted AI exposure
```

Mục tiêu là xem:

> Những đặc trưng occupation có thể giải thích/ước lượng AI exposure đến mức nào?

---

# 15. Models

Không cần 10 model.

Chỉ:

### Baseline

**Linear Regression**

### Model 2

**Random Forest Regressor**

### Model 3

**Gradient Boosting / XGBoost** nếu nhóm đã quen.

Nếu chưa biết XGBoost:

> Không cần.

Random Forest là đủ.

---

# 16. Evaluation

Regression:

```text
MAE
RMSE
R²
```

Ví dụ:

|Model|MAE|RMSE|R²|
|---|--:|--:|--:|
|Linear Regression|0.14|0.18|0.62|
|Random Forest|0.09|0.12|0.81|

Các con số trên **chỉ là ví dụ**, chưa phải kết quả.

---

# 17. Feature Importance

Đây là phần rất đẹp để trình bày.

Random Forest:

```text
Feature Importance

Digital skills       ███████████████
Routine activities   ███████████
Analytical skills    ████████
Social skills        █████
Creative skills      ████
Education            ███
```

Sau đó giải thích:

> Model cho thấy những đặc trưng nào có đóng góp lớn hơn vào việc ước lượng AI exposure trong dataset.

Không nói:

> "Digital skills gây ra AI replacement."

**Correlation/predictive importance không đồng nghĩa causation.**

---

# 18. Career Transition

Đây là phần thứ hai của MVP.

Không dùng ML recommendation system.

## Skill Vector

Ví dụ:

```text
Occupation A

Accounting       0.95
Data Analysis    0.80
Communication    0.60
Management       0.40
```

Occupation B:

```text
Accounting       0.20
Data Analysis    0.90
Communication    0.70
Management       0.50
```

→ tạo vector.

---

# 19. Similarity

Dùng:

```python
cosine_similarity()
```

Ví dụ:

```text
Accountant
     ↓
Skill Vector
     ↓
Compare with all occupations
     ↓
Similarity
     ↓
Top 5
```

Output:

```text
Current occupation:
Accountant

Potentially similar occupations:

1. Financial Analyst       0.86
2. Budget Analyst          0.82
3. Business Analyst        0.76
4. Financial Examiner      0.74
5. Management Analyst      0.69
```

**Đây không phải lời khuyên nghề nghiệp cá nhân hóa.**

Nó chỉ là:

> occupations with high skill similarity.

---

# 20. Thêm một điều kiện rất hay

Nếu nghề hiện tại có:

```text
AI exposure = 0.85
```

thì không nên gợi ý toàn bộ nghề similarity cao.

Có thể filter:

```text
Similarity > 0.70
AND
AI exposure < current occupation
```

Ví dụ:

```text
Current:
Accountant
AI exposure = 0.80

Candidate:

Financial Analyst
Similarity = 0.84
AI exposure = 0.55

Business Analyst
Similarity = 0.78
AI exposure = 0.49
```

→ hiển thị:

> **Potential transition candidates**

Điều này tạo ra sự liên kết rất đẹp:

```text
AI Impact
     +
Skill Similarity
     ↓
Career Transition
```

---

# 21. Streamlit MVP

Chỉ cần **3 màn hình**.

## Page 1 — Overview

```text
AI Job Impact Analyzer

Total occupations: XXX

Average AI Exposure: XX

[Chart]
AI Exposure Distribution

[Chart]
Exposure by occupation
```

---

# 22. Page 2 — Occupation Analysis

User chọn:

```text
Select occupation:
[ Accountant ▼ ]
```

Hiển thị:

```text
ACCOUNTANT

AI Exposure
██████████████░░░░ 72%

Skills
──────────────
Accounting       95%
Data Analysis    82%
Communication    65%

Tasks
──────────────
• Prepare financial reports
• Analyze financial records
• Prepare tax documents
```

---

# 23. Page 3 — Career Transition

```text
Current occupation:
Accountant

Potentially similar occupations
```

|Occupation|Skill Similarity|AI Exposure|
|---|--:|--:|
|Financial Analyst|0.86|0.55|
|Budget Analyst|0.82|0.51|
|Business Analyst|0.78|0.48|

Sau đó có thể hiển thị:

> These occupations have high skill similarity and lower AI exposure in the reference dataset.

---

# 24. Technology Stack

MVP:

```text
Python
│
├── pandas
├── numpy
├── matplotlib
├── seaborn
├── scikit-learn
│
└── Streamlit
```

Không cần:

```text
❌ PyTorch
❌ TensorFlow
❌ Transformers
❌ LangChain
❌ Vector DB
❌ Neo4j
❌ FastAPI
❌ LLM API
```

Đây là **MVP**, đừng over-engineer.

---

# 25. Project structure

Mình đề xuất:

```text
AI_Job_Transition/
│
├── data/
│   ├── raw/
│   │   ├── onet/
│   │   ├── oecd/
│   │   └── ilo/
│   │
│   └── processed/
│       ├── occupations.csv
│       ├── skills.csv
│       ├── tasks.csv
│       └── occupation_master.csv
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   └── 04_modeling.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── similarity.py
│
├── models/
│   └── model.pkl
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

---

# 26. Phân chia công việc nhóm

Nếu nhóm 4 người, mình sẽ chia:

### Member 1 — Data

- O*NET
    
- OECD
    
- ILO
    
- download
    
- understand schema
    
- merge
    

### Member 2 — Cleaning + EDA

- missing values
    
- duplicate
    
- normalization
    
- visualization
    
- correlation
    

### Member 3 — ML

- feature engineering
    
- baseline
    
- Random Forest
    
- evaluation
    
- feature importance
    

### Member 4 — Career Transition + App

- skill vector
    
- cosine similarity
    
- Streamlit
    
- integration
    

### Cả nhóm

- research question
    
- interpretation
    
- report
    
- presentation
    

---

# 27. MVP Acceptance Criteria

Đây là phần rất quan trọng.

**MVP được coi là hoàn thành khi:**

### Data

-  Có occupation dataset
    
-  Có skill dataset
    
-  Có AI exposure
    
-  Merge thành dataset cuối
    
-  Document nguồn dữ liệu
    

### EDA

-  AI exposure distribution
    
-  Top/bottom exposure
    
-  Skill/exposure relationship
    
-  Một số occupation case studies
    

### ML

-  Baseline
    
-  Random Forest
    
-  MAE/RMSE/R²
    
-  Feature importance
    

### Career transition

-  Skill vector
    
-  Cosine similarity
    
-  Top 5 related occupations
    

### Application

-  Streamlit
    
-  Occupation selection
    
-  AI exposure display
    
-  Skill display
    
-  Career transition suggestions
    

### Documentation

-  Data sources
    
-  Methodology
    
-  Limitations
    
-  README
    

**Hoàn thành hết đây là MVP.**

---

# 28. Những thứ tuyệt đối để "Later"

Nếu còn thời gian, mới thêm:

### V1.1 — Job postings

```text
VietnamWorks / other job postings
        ↓
Job description
        ↓
NLP
        ↓
Skill extraction
```

### V1.2 — Vietnamese occupation mapping

```text
O*NET
 ↓
ISCO
 ↓
Vietnam occupation classification
```

### V1.3 — NLP

```text
Job description
       ↓
TF-IDF / embeddings
       ↓
Skill extraction
```

### V1.4 — Better recommendation

```text
Skill similarity
+
AI exposure
+
salary
+
job demand
+
education requirement
```

### V2 — Graph

```text
Occupation
    ↕
Skill
    ↕
Occupation
```

→ Neo4j.

### V3 — Personalization

```text
CV
 ↓
NLP
 ↓
Extract skills
 ↓
Current skill vector
 ↓
Career transition
```

Đây mới là lúc project trở thành **AI Job Transition Map** đúng nghĩa.

---

# 29. Một limitation cực kỳ quan trọng phải ghi ngay từ MVP

Tên project có chữ **"AI Job Impact"**, nhưng kết quả **không được diễn giải thành xác suất một nghề sẽ mất việc**.

Ví dụ không viết:

> ❌ Accountant có 72% khả năng bị AI thay thế.

Mà:

> **Accountant has an AI exposure score of 0.72 according to the reference exposure measure.**

Bởi OECD mô tả measure của họ là mức độ gần giữa capability của AI hiện tại và requirements của occupation, đồng thời nhấn mạnh rằng tác động thực tế còn phụ thuộc adoption, regulation, organizational change và social choices. ([OECD](https://www.oecd.org/en/publications/the-oecd-ai-exposure-measure_f3da0f0a-en.html?utm_source=chatgpt.com "The OECD AI exposure measure | OECD"))

ILO cũng phân biệt exposure với việc làm thực sự bị mất; nghiên cứu 2025 kết luận phần lớn jobs có khả năng được **transformed** hơn là hoàn toàn redundant. ([International Labour Organization](https://www.ilo.org/publications/generative-ai-and-jobs-2025-update?utm_source=chatgpt.com "Generative AI and jobs: A 2025 update | International Labour Organization"))

Điều này sẽ làm report của nhóm **chắc hơn rất nhiều**.

---

# 30. Toàn bộ MVP gói lại trong một câu

> **Build a data-driven system that analyzes occupational AI exposure using O*NET occupational characteristics and established AI-exposure measures, then identifies potentially relevant career-transition options based on transferable-skill similarity.**

Và pipeline cuối cùng chỉ cần nhớ:

```text
                 O*NET
                   │
        ┌──────────┼──────────┐
        ↓          ↓          ↓
      Tasks      Skills     Related Jobs
        │          │
        └────┬─────┘
             ↓
        Occupation Data
             │
             + ← OECD / ILO AI Exposure
             │
             ↓
        ┌─────────────┐
        │     EDA     │
        └──────┬──────┘
               ↓
       Feature Engineering
               ↓
        ┌──────┴──────┐
        ↓             ↓
    ML Analysis    Skill Vector
        ↓             ↓
    AI Impact      Similarity
        │             │
        └──────┬──────┘
               ↓
           Streamlit
               ↓
     AI Job Impact Analyzer
```



