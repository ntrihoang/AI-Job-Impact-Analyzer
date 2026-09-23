
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

## PHẦN 1: DỮ LIỆU & KIẾN TRÚC LƯU TRỮ (DATA & STORAGE)

Hệ thống sử dụng các bộ dữ liệu có cấu trúc từ O*NET, OECD, bảng ánh xạ chuẩn quốc tế và dữ liệu nền tảng về lực lượng lao động Việt Nam.

### 1.1 Dữ liệu thô đầu vào (Raw Data Sources)

1. **Bộ dữ liệu O*NET 31.0:**
    
    - `Occupation Data.csv`: Danh mục 832 nghề nghiệp (`O*NET-SOC Code`, `Title`, `Description`).
        
    - `Skills.csv` & `Abilities.csv`: Điểm `Data Value` theo `Scale ID` (`IM` - Importance, `LV` - Level).
        
    - `Work Context.csv`: Môi trường làm việc (tính lặp lại, khuôn mẫu, áp lực công việc).
        
    - `Education, Training, and Experience.csv`: Yêu cầu trình độ học vấn tối thiểu.
        
    - `Related Occupations.csv`: Các cặp nghề nghiệp chuyển đổi tham chiếu.
        
2. **OECD AI Exposure Dataset:**
    
    - File `.xlsx`/`.csv` chứa chỉ số `AI Capability Gap Index` (chuẩn hóa về dải $0.0 - 1.0$), đóng vai trò là nhãn Target ($y$).
        
3. **Crosswalk & Vietnam Baseline:**
    
    - `SOC_to_ISCO08.csv`: Bảng chuyển đổi mã SOC (Mỹ) sang mã ISCO-08 (Quốc tế).
        
    - `vn_isco_baseline.csv`: File CSV 9 dòng lưu trữ tổng số lao động và tỷ lệ lao động nữ tại Việt Nam phân theo 9 nhóm nghề chính ISCO-08 (trích xuất từ ILOSTAT / Tổng cục Thống kê).
        

### 1.2 Cấu trúc Dữ liệu Sạch (Clean Datasets / Relational Schema)

Dữ liệu sau khi xử lý được lưu trữ dưới dạng 4 file CSV sạch hoặc nạp vào SQLite/PostgreSQL:

#### Bảng 1: `occupations_master.csv` (Bảng Master Nghề nghiệp)

|**Tên cột**|**Kiểu dữ liệu**|**Mô tả**|
|---|---|---|
|`occupation_code` **(PK)**|VARCHAR(10)|Mã SOC (Ví dụ: `11-3031.00`)|
|`occupation_title`|VARCHAR(255)|Tên nghề nghiệp|
|`education_level_id`|INT|Trình độ học vấn ($1 = \text{High School}$, $4 = \text{Bachelor}$, $6 = \text{Master/PhD}$)|
|`cognitive_intensity`|FLOAT|Mức độ tư duy phức tạp ($0.0 - 1.0$)|
|`social_intensity`|FLOAT|Mức độ tương tác con người ($0.0 - 1.0$)|
|`physical_intensity`|FLOAT|Mức độ lao động chân tay ($0.0 - 1.0$)|
|`digital_intensity`|FLOAT|Mức độ sử dụng máy tính/công nghệ ($0.0 - 1.0$)|
|`routine_intensity`|FLOAT|Mức độ lặp đi lặp lại/khuôn mẫu ($0.0 - 1.0$)|
|`ai_exposure_score`|FLOAT|**Target ($y$):** Điểm tiếp xúc AI từ OECD ($0.0 - 1.0$)|

#### Bảng 2: `occupation_skills.csv` (Bảng Kỹ năng)

|**Tên cột**|**Kiểu dữ liệu**|**Mô tả**|
|---|---|---|
|`occupation_code` **(FK)**|VARCHAR(10)|Mã SOC liên kết `occupations_master`|
|`skill_id`|VARCHAR(50)|Mã Element ID trong O*NET (Ví dụ: `2.A.2.a`)|
|`skill_name`|VARCHAR(255)|Tên kỹ năng (_Critical Thinking_, _Negotiation_...)|
|`importance_score`|FLOAT|Điểm tầm quan trọng ($1.0 - 5.0$)|
|`level_score`|FLOAT|Điểm cấp độ yêu cầu ($0.0 - 7.0$)|
|`normalized_weight`|FLOAT|Trọng số kỹ năng chuẩn hóa $w_{i,j} \in [0, 1]$|

#### Bảng 3: `vn_isco_baseline.csv` (Bảng Bối cảnh Việt Nam)

|**Tên cột**|**Kiểu dữ liệu**|**Mô tả**|
|---|---|---|
|`isco_major_code` **(PK)**|INT|Mã nhóm nghề lớn ISCO-08 ($1 \to 9$)|
|`group_name_vn`|VARCHAR(255)|Tên nhóm nghề tiếng Việt|
|`vn_employment_thousands`|FLOAT|Tổng số lao động tại VN (đơn vị: nghìn người)|
|`female_share_pct`|FLOAT|Tỷ lệ lao động nữ (%)|

#### Bảng 4: `related_occupations.csv` (Bảng Nghề liên quan)

- **Cột:** `occupation_code`, `related_occupation_code`.
    

## ⚙️ PHẦN 2: TRÍCH XUẤT ĐẶC TRƯNG (FEATURE ENGINEERING)

Để loại bỏ hiện tượng thiên vị đối với các nghề có danh mục kỹ năng rộng (Scope Bias), phương pháp chuẩn hóa của _Felten et al. (2021)_ được áp dụng.

### 2.1 Công thức Tính toán 5 Nhóm Features ($X$)

Với mỗi nghề $k$, giá trị của một nhóm đặc trưng (Domain) được tính bằng tổng trọng số của nhóm đó chia cho tổng trọng số của tất cả năng lực mà nghề đó yêu cầu:

$$\text{Domain\_Intensity}_k = \frac{\sum_{j \in \text{Domain}} (\text{Importance}_{j,k} \times \text{Level}_{j,k})}{\sum_{\text{All } j} (\text{Importance}_{j,k} \times \text{Level}_{j,k})}$$

#### Danh mục Mapping O*NET Elements:

- **`cognitive_intensity`:** `2.A.2.a` (Critical Thinking), `2.A.2.b` (Complex Problem Solving), `2.D.1.e` (Judgment & Decision Making), `1.A.1.b.4` (Deductive Reasoning).
    
- **`social_intensity`:** `2.A.1.b` (Social Perceptiveness), `2.B.1.b` (Persuasion), `2.B.1.c` (Negotiation), `4.A.4.a.2` (Caring for Others).
    
- **`physical_intensity`:** `1.A.2.a` (Static Strength), `1.A.3.a` (Stamina), `1.A.1.a` (Arm-Hand Steadiness), `1.A.1.b` (Manual Dexterity).
    
- **`digital_intensity`:** Điểm $I \times L$ của Element `4.A.3.a.1` (_Interacting With Computers_) kết hợp số lượng công cụ công nghệ từ file `Technology Skills.csv`.
    
- **`routine_intensity`:** Trung bình cộng các chỉ số môi trường làm việc `4.C.3.b.7` (Repeat Same Tasks), `4.C.3.b.8` (Structured Work), `4.C.3.b.4` (Pace Speed Equipment).
    
- **`education_level_id`:** Lấy giá trị Yêu cầu Trình độ Học vấn có tỷ lệ % phân bổ cao nhất (Mode) trong file `Education, Training, and Experience.csv`.
    

### 2.2 Trọng số Ma trận Kỹ năng (Skill Matrix Weights)

Ô $w_{i,j}$ biểu diễn mức độ thành thạo của nghề $i$ đối với kỹ năng $j$:

$$w_{i,j} = \frac{\text{Importance}_{i,j} \times \text{Level}_{i,j}}{5.0 \times 7.0} = \frac{\text{Importance}_{i,j} \times \text{Level}_{i,j}}{35.0}$$

## 🤖 PHẦN 3: TRIỂN KHAI CÁC MÔ HÌNH & THUẬT TOÁN (MODELING & ALGORITHMS)

### 3.1 Bài toán 1: AI Impact Analysis (XGBoost + SHAP Explainability)

Mục tiêu là dự đoán điểm AI Exposure và giải thích các thuộc tính ảnh hưởng.

- **Thuật toán:** XGBoost Regressor (hoặc Random Forest Regressor).
    
- **Tập dữ liệu:**
    
    - $X = [\text{cognitive}, \text{social}, \text{physical}, \text{digital}, \text{routine}, \text{education}]$
        
    - $y = \text{ai\_exposure\_score}$
        
- **Chia tập dữ liệu:** $80\%$ Train, $20\%$ Test.
    
- **Đánh giá mô hình:** $R^2$ Score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
    
- **Giải thích mô hình (Explainability):**
    
    - Chạy **SHAP (SHapley Additive exPlanations)** để tính giá trị SHAP value cho từng sample.
        
    - Xuất top 2 yếu tố đẩy rủi ro AI lên cao nhất ($SHAP > 0$) và top 2 yếu tố giảm rủi ro AI ($SHAP < 0$).
        

### 3.2 Bài toán 2: Career Transition Pathway (Cosine Similarity + Filter)

Mục tiêu là tìm các nghề chuyển đổi có mức độ tương đồng kỹ năng cao, an toàn hơn trước AI và chỉ ra lỗ hổng kỹ năng.

1. **Ma trận Kỹ năng:** Tạo ma trận $M \in \mathbb{R}^{N \times K}$ ($N = 832$ nghề, $K = 35$ kỹ năng O*NET).
    
2. **Tính Độ tương đồng Cosine:**
    
    Với nghề hiện tại $V_A$ và nghề mục tiêu $V_B$:
    
    $$\text{Similarity}(A, B) = \frac{V_A \cdot V_B}{\Vert{}V_A\Vert{} \Vert{}V_B\Vert{}} = \frac{\sum_{k=1}^{K} w_{A,k} w_{B,k}}{\sqrt{\sum_{k=1}^{K} w_{A,k}^2} \sqrt{\sum_{k=1}^{K} w_{B,k}^2}}$$
    
3. **Bộ lọc Rủi ro AI (Safety Constraint):**
    
    Chỉ giữ lại các nghề $B$ thỏa mãn:
    
    $$\text{AI Exposure}(B) < \text{AI Exposure}(A)$$
    
4. **Trích xuất Khoảng cách Kỹ năng (Skill Gap Extraction):**
    
    Xác định danh sách các kỹ năng $s_k$ mà nghề $B$ đòi hỏi cao hơn nghề $A$:
    
    $$\text{Skill Gap}(A \to B) = \left\{ s_k \;\mid\; w_{B,k} - w_{A,k} > 0.20 \right\}$$
    

## 🇻🇳 PHẦN 4: TÍCH HỢP BỐI CẢNH THỊ TRƯỜNG VIỆT NAM (LOCALIZATION)

### 4.1 Luồng kết nối Dữ liệu (Crosswalk Routing)

Plaintext

```
[Mã SOC O*NET] ──► [SOC_to_ISCO08.csv] ──► [Mã ISCO-08] ──► [Ký tự đầu = isco_major_code] ──► [vn_isco_baseline.csv]
```

### 4.2 Phân tích Đối chiếu trong Jupyter Notebook (EDA)

Trong Notebook `EDA_and_Modeling.ipynb`, gộp chỉ số AI Exposure trung bình của mô hình theo 9 Nhóm nghề lớn ISCO-08 tại Việt Nam:

1. Tính `avg_ai_exposure` cho từng `isco_major_code`.
    
2. Merge với bảng `vn_isco_baseline.csv`.
    
3. Vẽ biểu đồ trục kép (Dual-axis Chart):
    
    - **Trục 1 (Cột):** Quy mô lao động Việt Nam (`vn_employment_thousands`).
        
    - **Trục 2 (Đường):** Điểm rủi ro AI Exposure trung bình (`avg_ai_exposure`).
        

## 🖥️ PHẦN 5: CHƯƠNG TRÌNH CLI TRÊN TERMINAL (`main.py`)

Chương trình chạy trực tiếp bằng dòng lệnh Python, không dùng giao diện Web UI, hiển thị báo cáo phân tích bằng thư viện `rich`.

### Kịch bản hiển thị Output trên Terminal:

Plaintext

```
================================================================================
                    CAREER AI EXPOSURE & TRANSITION REPORT                      
================================================================================
[Nghề nghiệp tra cứu]: Financial Managers (Mã SOC: 11-3031.00)

--------------------------------------------------------------------------------
1. AI IMPACT ANALYSIS (Mô hình XGBoost & SHAP)
--------------------------------------------------------------------------------
* Dự báo AI Exposure Score : 0.88 / 1.00 (MỨC ĐỘ RỦI RO CAO)
* Yếu tố làm tăng rủi ro   : Digital Intensity (+0.25), Routine Intensity (+0.12)
* Yếu tố bảo vệ (Giảm rủi ro): Social Intensity (-0.08)

--------------------------------------------------------------------------------
2. BỐI CẢNH THỊ TRƯỜNG VIỆT NAM (ILO Baseline & ISCO-08)
--------------------------------------------------------------------------------
* Nhóm nghề tương đương    : Nhóm 1 - Nhà quản lý (ISCO Code: 1211)
* Quy mô lao động tại VN   : ~1,250,000 người
* Tỷ lệ lao động nữ        : 42.5%
* Điểm rủi ro AI trung bình: 0.65 / 1.00 (Tác động ở mức khá)

--------------------------------------------------------------------------------
3. GỢI Ý LỘ TRÌNH CHUYỂN ĐỔI NGHỀ NGHIỆP AN TOÀN (Top 3 Recommendations)
--------------------------------------------------------------------------------
 [#1] Human Resources Managers (SOC: 11-3121.00)
      * Độ tương đồng kỹ năng : 87.5%
      * Chỉ số AI Exposure     : 0.62 (An toàn hơn: -0.26)
      * Kỹ năng cần bổ sung    : Personnel and Human Resources (+0.45), Negotiation (+0.30)

 [#2] Training and Development Managers (SOC: 11-3131.00)
      * Độ tương đồng kỹ năng : 82.1%
      * Chỉ số AI Exposure     : 0.58 (An toàn hơn: -0.30)
      * Kỹ năng cần bổ sung    : Instructing (+0.40), Learning Strategies (+0.35)
================================================================================
```

## 📅 PHẦN 6: LỘ TRÌNH THỰC HIỆN DỰ ÁN (PROJECT ROADMAP)

Plaintext

```
Phase 1: ETL & FE           Phase 2: Modeling         Phase 3: VN Context        Phase 4: CLI & Report
[Ngày 1 - 2]               [Ngày 3 - 4]               [Ngày 5 - 6]               [Ngày 7 - 8]
 ├── Thu thập O*NET/OECD    ├── Train XGBoost Model    ├── Tạo vn_isco_baseline   ├── Code main.py (Rich CLI)
 ├── Tính 6 Features (X)    ├── Chạy SHAP Analysis     ├── Map SOC -> ISCO-08     ├── Hoàn thiện Notebook
 └── Dựng Skill Matrix      └── Code Cosine Sim Alg    └── Vẽ biểu đồ EDA         └── Chuẩn bị Slide
```

### Danh mục Sản phẩm Bàn giao (Deliverables)

1. `data/`: Thư mục chứa 4 file CSV sạch (`occupations_master.csv`, `occupation_skills.csv`, `SOC_to_ISCO08.csv`, `vn_isco_baseline.csv`).
    
2. `notebooks/EDA_and_Modeling.ipynb`: Notebook hoàn chỉnh chứa pipeline ETL, Feature Engineering, EDA đối chiếu Mỹ - Việt Nam, huấn luyện mô hình XGBoost và SHAP plot.
    
3. `src/` & `main.py`: Chương trình Python CLI phục vụ tra cứu và chạy gợi ý chuyển nghề trực tiếp trên Terminal.
    
4. **Slide thuyết trình:** Tóm tắt phương pháp luận Data Science, kết quả mô hình ML, các biểu đồ EDA và thảo luận về tác động tới thị trường lao động Việt Nam.




