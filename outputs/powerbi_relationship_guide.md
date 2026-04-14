# Power BI Relationship Guide: LinkedIn Jobs

This guide helps you set up the Star Schema for optimal dashboard performance.

## 1. Sơ đồ quan hệ (Star Schema Model)
Kết nối các bảng theo sơ đồ sau trong giao diện **Model View**:

| Bảng Nguồn (Many) | Bảng Đích (One) | Cột kết nối (Primary Key) | Cardinality | Filter Direction |
|---|---|---|---|---|
| `summary_for_dashboard` | `dim_company` | `company_id` | `*:1` | Single (One-way) |
| `summary_for_dashboard` | `dim_industry` | `primary_industry_id` | `*:1` | Single (One-way) |
| `summary_for_dashboard` | `dim_calendar` | `listed_date` | `*:1` | Single (One-way) |
| `fact_job_skill` | `summary_for_dashboard` | `job_id` | `*:1` | Single (One-way) |
| `fact_job_skill` | `dim_skill` | `skill_abr` | `*:1` | **Both (Two-way)** |
| `fact_job_benefit` | `summary_for_dashboard` | `job_id` | `*:1` | Single (One-way) |

> [!TIP]
> - Đối với **`fact_job_skill`**, bật **"Both directions"** cho quan hệ với `dim_skill` nếu bạn muốn chọn một kỹ năng và tự động lọc danh sách Jobs trong Fact table tương ứng.

## 2. Gợi ý KPIs & Slicers

### Cột dùng làm Slicer (Bộ lọc):
- **`salary_disclosure_status`**: Để lọc tin tuyển dụng công khai vs không công khai lương.
- **`salary_band_clean`**: Phân dải lương.
- **`experience_group_clean`**: Phân cấp bậc kinh nghiệm.
- **`is_remote`**: Lọc việc làm từ xa.

### Cột dùng làm KPIs:
- **`normalized_salary`**: Dùng Hàm `AVERAGE` hoặc `MEDIAN`.
- **`engagement_score`**: Dùng Hàm `SUM` để tính tổng độ thu hút thương hiệu.
- **`applies_per_view`**: Dùng Hàm `AVERAGE` để đánh giá hiệu quả tin đăng.

## 3. Cột không nên kéo trực tiếp
- **`job_id`**: Chỉ dùng để đếm (`COUNTDISTINCT`).
- **`company_id`**: Chứa ID thô, không có ý nghĩa phân tích trực quan. Hãy dùng `dim_company[name]`.

## 4. Best Practices cho DAX
Hãy tạo một bảng **`_Measures`** trống để lưu trữ các công thức:
- **`Hiring Efficiency`** = `DIVIDE(SUM(applies), SUM(views))`
- **`Salary Gap`** = `AVERAGE(normalized_salary) - AVERAGE(pred_salary)`
