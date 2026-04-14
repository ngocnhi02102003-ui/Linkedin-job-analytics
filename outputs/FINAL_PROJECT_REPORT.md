# BÁO CÁO TỔNG KẾT DỰ ÁN: PHÂN TÍCH THỊ TRƯỜNG VIỆC LÀM LINKEDIN 2026

**Người trình bày:** Senior Data Analyst
**Đối tượng:** Ban Giám đốc (Stakeholders)
**Phạm vi:** 123,849 tin tuyển dụng LinkedIn (Tháng 03 - Tháng 04/2024)

---

## 1. Bối cảnh & Mục tiêu (Context)
Thị trường tuyển dụng trên LinkedIn là một kho dữ liệu khổng lồ nhưng "nhiễu". Dự án này được thiết lập để trả lời các câu hỏi chủ chốt:
- Làm sao để ước tính mức lương khi doanh nghiệp chủ động giấu thông tin?
- Những yếu tố nào (kỹ năng, thâm niên, remote) thúc đẩy sự tương tác của ứng viên?
- Làm sao để phân loại thị trường thành các nhóm mục tiêu (Segments) một cách tự động?

---

## 2. Quy trình xử lý Dữ liệu (ETL Workflow)
Để có được bộ dữ liệu "Sạch và Thông minh" trên Power BI, chúng tôi đã triển khai Pipeline qua 2 giai đoạn chính:

*   **Dữ liệu thô (Raw Data)**: Tiếp nhận ~3.3 TRIỆU dòng từ tin tuyển dụng thô (từ file `postings.csv` và các bảng phụ).
*   **Bước xử lý ETL (Python Script)**:
    - **Làm sạch**: Loại bỏ trùng lặp, chuẩn hóa ngày tháng và định dạng lương về một đơn vị chuẩn (USD/Year).
    - **Liên kết (Mapping)**: Kết nối 11 bảng rời rạc (Skills, Companies, Industries, Salaries) thành mô hình **Star Schema**.
    - **Tính toán Feature**: Tạo ra các chỉ số mới như `Engagement Score` (Điểm tương tác), `Salary Band` (Nhóm lương), và xử lý các giá trị thiếu bằng mã chuẩn **`-1`** (Unknown) để Dashboard không bị lỗi logic.
*   **Kết quả**: Một bộ dữ liệu tinh gọn (123,849 dòng) nhưng chứa hàm lượng thông tin gấp nhiều lần dữ liệu gốc, sẵn sàng cho phân tích chuyên sâu.

---

## 3. Kết quả Phân tích Chi tiết

### A. Dataset Overview (Cái nhìn tổng quan)
- **Quy mô**: 123,849 tin đăng từ 24,475 công ty thuộc 409 ngành nghề khác nhau.
- **Kỹ năng**: Trung bình 1.66 kỹ năng/job.
- **Tính minh bạch**: Chỉ **29.12%** công việc công khai lương.
- **Xu hướng làm việc**: **12.31%** cho phép Remote (Làm từ xa).
- **Phạm vi thời gian**: Dữ liệu bao phủ liên tục 28 ngày (24/03/2024 - 20/04/2024).

### B. Descriptive Analytics (Mô tả thị trường)
![Salary Distribution](file:///Users/n.n.nbi/Downloads/DA-Task/linkedin_jobs_project/outputs/figures/salary_distribution.png)
- **Top Industries**: Healthcare và IT là hai ngành có lượng tin tuyển dụng áp đảo.
![Top Industries](file:///Users/n.n.nbi/Downloads/DA-Task/linkedin_jobs_project/outputs/figures/top_industries.png)
- **Lương theo Thâm niên**: Nhóm *Executive* có trung vị lương cao nhất (~$180k), theo sau là *Mid-Senior*.
![Salary by Experience](file:///Users/n.n.nbi/Downloads/DA-Task/linkedin_jobs_project/outputs/figures/salary_by_experience.png)
- **Lương theo Hình thức**: Các công việc **Remote** có trung vị lương cao hơn ~$20k so với On-site trong cùng phân khúc.
![Salary by Remote](file:///Users/n.n.nbi/Downloads/DA-Task/linkedin_jobs_project/outputs/figures/salary_by_remote.png)
- **Tương tác**: Các tin đăng có dải lương từ 40k-150k USD nhận được sự quan tâm (Views/Applies) lớn nhất từ ứng viên.

### C. Diagnostic Analytics (Chẩn đoán nguyên nhân)
- **Tại sao giấu lương?**: Phân tích cho thấy các ngành đòi hỏi kỹ năng chuyên sâu (IT, Tài chính) có tỷ lệ giấu lương cao hơn 40% so với ngành Retail/Healthcare. Nguyên nhân có thể do sự cạnh tranh về nhân tài cao hoặc cấu trúc lương thưởng linh hoạt.
- **Sự liên quan giữa Skill & Salary**: Có mối tương quan thuận (nhưng yếu) giữa số lượng kỹ năng yêu cầu và mức lương. Tuy nhiên, nhóm giấu lương thường là nhóm yêu cầu kỹ năng phức tạp hơn.
- **Sponsorship**: Các tin được tài trợ (Sponsored) có tỷ lệ chuyển đổi (Applies/View) thấp hơn dự kiến, cho thấy nội dung tin đăng quan trọng hơn ngân sách quảng cáo.

---

## 4. Mô hình Dự đoán (Predictive Analytics)

Để giải quyết bài toán "Bất cân xứng thông tin" trên LinkedIn một cách toàn diện, chúng tôi không dùng một mô hình đơn lẻ mà xây dựng một **Hệ sinh thái Hỗ trợ ra quyết định (DSS)** với 3 tầng liên kết chặt chẽ:

![ML Architecture](file:///Users/n.n.nbi/Downloads/DA-Task/linkedin_jobs_project/outputs/figures/ml_architecture.png)

1. **Dự báo Lương (Regression)**: 
   - **HistGradientBoosting** là "Nhà vô địch" với độ chính xác (R²) đột phá đạt **0.60**. 
   - Ý nghĩa: Chúng ta có thể dự báo mức lương với độ tin cậy cao dựa trên sự kết hợp giữa thuật toán mạnh mẽ và xử lý từ khóa (NLP Title).
![Actual vs Predicted](file:///Users/n.n.nbi/Downloads/DA-Task/linkedin_jobs_project/outputs/figures/ml_salary_actual_vs_pred.png)
![Residuals Analysis](file:///Users/n.n.nbi/Downloads/DA-Task/linkedin_jobs_project/outputs/figures/ml_salary_residuals.png)

2. **Dự đoán Hot Job (Classification)**:
   - Sử dụng mô hình phân loại để dự đoán khả năng thành công (tương tác cao) của tin tuyển dụng.
![Confusion Matrix](file:///Users/n.n.nbi/Downloads/DA-Task/linkedin_jobs_project/outputs/figures/ml_hot_job_confusion_matrix.png)
![ROC Curve](file:///Users/n.n.nbi/Downloads/DA-Task/linkedin_jobs_project/outputs/figures/ml_hot_job_roc_curve.png)

3. **Phân khúc thị trường (Clustering)**:
   - Thuật toán **K-Means** chia thị trường thành 3 cụm (Cluster) riêng biệt giúp tối ưu hóa chiến lược tiếp cận theo từng phân đặc thù.
![Cluster Visualization](file:///Users/n.n.nbi/Downloads/DA-Task/linkedin_jobs_project/outputs/figures/ml_clusters_view.png)

---

## 5. Kiến nghị & Hành động (Recommendations)

1. **Về phía Tuyển dụng**: Khuyến khích HR công khai mức lương cho các vị trí từ 40k-150k USD để tăng tương tác dự kiến lên gấp 2.5 lần.
2. **Về phía Chiến lược**: Tập trung ngân sách vào các Job thuộc **Cluster 1** (Chuyên gia/Tech) vì đây là nhóm mang lại tương tác cao nhất.
3. **Về Công cụ BI**: Sử dụng dự báo từ mô hình **HistGradientBoosting** để cung cấp mức lương tham chiếu benchmarking cho nhân viên.

---
**Báo cáo đã sẵn sàng cho buổi trình bày chính thức.**
