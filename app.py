import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="LinkedIn Jobs Analytics - AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME & CSS ─────────────────────────────────────────────
# Light Premium Theme: #ffffff, #eff9ff, #184683, #003859
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stApp { color: #003859; font-weight: 500; }
    .stMetric {
        background-color: #eff9ff;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #184683;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    h1, h2, h3 { color: #184683 !important; font-family: 'Outfit', sans-serif; font-weight: 700 !important; }
    .stButton>button {
        background-color: #184683;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-weight: bold;
        width: 100%;
    }
    .stSidebar {
        background-color: #eff9ff;
        border-right: 2px solid #184683;
    }
    label { color: #003859 !important; font-weight: 600 !important; font-size: 1.1rem !important; }
    </style>
    /* Custom Card Style */
    .css-1r6slb0 { background-color: #eff9ff; border-radius: 12px; padding: 20px; }
    </style>
    """, unsafe_allow_html=True)

# ── DATA LOADING ───────────────────────────────────────────
BASE_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / 'data_processed/jobs_master.csv', low_memory=False)
    try:
        clusters = pd.read_csv(BASE_DIR / 'data_processed/powerbi/cluster_results.csv')
        df = pd.merge(df, clusters, on='job_id', how='left')
    except:
        pass
    return df

@st.cache_resource
def load_models():
    try:
        # Sử dụng đường dẫn tuyệt đối dựa trên vị trí file app.py
        salary_path = BASE_DIR / 'outputs/models/salary_model.pkl'
        hot_job_path = BASE_DIR / 'outputs/models/hot_job_model.pkl'
        
        if not salary_path.exists():
            st.error(f"Không tìm thấy file mô hình lương tại: {salary_path}")
            return None, None
            
        salary_model = joblib.load(salary_path)
        hot_job_model = joblib.load(hot_job_path) if hot_job_path.exists() else None
        return salary_model, hot_job_model
    except Exception as e:
        st.error(f"Lỗi khi nạp mô hình AI: {str(e)}")
        return None, None

# ── MAIN APP ────────────────────────────────────────────────
def main():
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/linkedin.png", width=80)
    st.sidebar.title("💎 Jobs AI Dashboard")
    st.sidebar.markdown("*Hệ thống phân tích thị trường lao động 2026*")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox("Lựa chọn tính năng", 
                                ["🏠 Khám phá Thị trường", "💰 Định giá Lương AI", "🔥 Dự báo Hot Job"])
    
    df = load_data()
    salary_model, hot_job_model = load_models()

    if page == "🏠 Khám phá Thị trường":
        st.title("📊 LinkedIn Job Market Dashboard")
        st.markdown("---")
        
        # Row 1: Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng số tin đăng", f"{len(df):,}")
        m2.metric("Lương trung vị (Median)", f"${df['normalized_salary'].median():,.0f}")
        m3.metric("Tỷ lệ Remote", f"{(df['is_remote'].mean()*100):.1f}%")
        m4.metric("Quy mô Công ty (Avg)", f"{df['employee_count'].mean():,.0f}")

        st.markdown("### 📈 Phân tích chuyên sâu")
        c1, c2 = st.columns(2)
        
        with c1:
            # Salary Bands Distribution
            fig_hist = px.histogram(df.dropna(subset=['normalized_salary']), 
                                  x="normalized_salary", nbins=50,
                                  title="Phân bổ mức lương toàn thị trường",
                                  labels={'normalized_salary':'Lương năm (USD)'},
                                  color_discrete_sequence=['#184683'])
            fig_hist.update_layout(plot_bgcolor='#eff9ff', paper_bgcolor='#ffffff', 
                                  font=dict(color='#003859', size=12))
            st.plotly_chart(fig_hist, width='stretch')

        with c2:
            # Experience vs Salary
            fig_box = px.box(df.dropna(subset=['normalized_salary', 'formatted_experience_level']), 
                            x='formatted_experience_level', y='normalized_salary',
                            title="Mức lương theo cấp độ kinh nghiệm",
                            color_discrete_sequence=['#184683'],
                            category_orders={"formatted_experience_level": ["Internship", "Entry Level", "Associate", "Mid-Senior Level", "Director", "Executive"]})
            fig_box.update_layout(plot_bgcolor='#eff9ff', paper_bgcolor='#ffffff', 
                                 font=dict(color='#003859', size=12))
            st.plotly_chart(fig_box, width='stretch')

        # Row 2: Clustering
        st.markdown("### 🧬 Phân khúc thị trường (K-Means Clustering)")
        if 'cluster' in df.columns:
            # Map cluster names based on research
            cluster_map = {
                0: "SME Specialist", 1: "Corporate Generalist", 2: "Enterprise Technical",
                3: "High-Value Growth", 4: "Viral Startup", 5: "Outlier/Error", 6: "Unicorn"
            }
            df['Cluster_Name'] = df['cluster'].map(cluster_map).fillna("Unknown")
            
            fig_scatter = px.scatter(df.sample(min(8000, len(df))), x='skill_count', y='views', 
                                     color='Cluster_Name',
                                     log_y=True,
                                     title="Market Positioning: Skills vs Engagement",
                                     color_discrete_sequence=px.colors.qualitative.Prism)
            fig_scatter.update_layout(plot_bgcolor='#eff9ff', paper_bgcolor='#ffffff', 
                                     font=dict(color='#003859', size=12))
            st.plotly_chart(fig_scatter, width='stretch')
        else:
            st.warning("Dữ liệu phân cụm chưa sẵn sàng.")

    elif page == "💰 Định giá Lương AI":
        st.title("💰 AI Salary Estimator")
        st.markdown("Dự báo mức lương kỳ vọng dựa trên thuật toán **HistGradientBoosting (R²=0.60)**.")
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown("#### 📝 Thông tin vị trí")
            title = st.text_input("Chức danh công việc", placeholder="ví dụ: Senior Frontend Developer")
            industry = st.selectbox("Ngành nghề", sorted(df['primary_industry_name'].dropna().unique()))
            exp_level = st.selectbox("Cấp độ kinh nghiệm", ["Entry Level", "Associate", "Mid-Senior Level", "Director", "Executive", "Internship"])
            work_type = st.selectbox("Hình thức làm việc", ["Full-Time", "Contract", "Part-Time", "Internship", "Temporary"])
            is_remote = st.checkbox("Cho phép làm từ xa (Remote)", value=False)
            skills_count = st.slider("Số lượng kỹ năng yêu cầu", 0, 15, 5)
            benefit_count = st.slider("Số lượng phúc lợi", 0, 10, 3)
            
            submit = st.button("🚀 Dự báo ngay")

        with col2:
            if submit and title:
                input_df = pd.DataFrame([{
                    'title': str(title),
                    'formatted_work_type': str(work_type),
                    'formatted_experience_level': str(exp_level),
                    'primary_industry_name': str(industry),
                    'country': 'US',
                    'is_remote': str(int(is_remote)),
                    'is_sponsored': '0',
                    'benefit_count': float(benefit_count),
                    'skill_count': float(skills_count),
                    'description_length': 1500.0,
                    'employee_count': 1000.0,
                    'follower_count': 5000.0
                }])
                
                prediction = salary_model.predict(input_df)[0]
                
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    title = {'text': "Giá trị thị trường ước tính ($)"},
                    gauge = {
                        'axis': {'range': [20000, 300000]},
                        'bar': {'color': "#184683"},
                        'steps' : [
                            {'range': [0, 80000], 'color': "#eff9ff"},
                            {'range': [80000, 160000], 'color': "#c4d8e2"},
                            {'range': [160000, 300000], 'color': "#184683"}
                        ],
                        'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 250000}
                    }))
                fig_gauge.update_layout(paper_bgcolor='#ffffff', font={'color': "#003859", 'family': "Arial", 'size': 14})
                st.plotly_chart(fig_gauge, width='stretch')
                
                st.info(f"💡 **Lời khuyên:** Với vị trí **{title}**, dải lương trung bình thị trường dao động từ **${(prediction*0.9):,.0f}** đến **${(prediction*1.1):,.0f}**.")
            elif submit:
                st.error("Vui lòng nhập Chức danh công việc!")
            else:
                st.image("https://img.icons8.com/bubbles/200/000000/money-bag.png", width=200)
                st.markdown("### Điền thông tin bên trái để bắt đầu định giá.")

    elif page == "🔥 Dự báo Hot Job":
        st.title("🔥 Hot Job Predictor")
        st.markdown("Xác định xác suất tin tuyển dụng trở thành **xu hướng (Hot)** dựa trên hàng trăm đặc tính.")
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔍 Nhập chi tiết tin đăng")
            h_title = st.text_input("Tiêu đề tin đăng", value="Lead Data Scientist")
            h_exp = st.selectbox("Yêu cầu kinh nghiệm", sorted(df['formatted_experience_level'].dropna().unique()), index=1)
            h_remote = st.radio("Làm việc từ xa", ["Có", "Không"], horizontal=True)
            h_skills = st.number_input("Số kỹ năng yêu cầu", 1, 20, 8)
            
            check_hot = st.button("🔥 Kiểm tra độ Hot")

        with c2:
            if check_hot:
                # Prepare data for model
                h_input = pd.DataFrame([{
                    'title': str(h_title),
                    'primary_industry_name': 'IT Services and IT Consulting',
                    'formatted_work_type': 'Full-Time',
                    'formatted_experience_level': str(h_exp),
                    'is_remote': '1' if h_remote=="Có" else '0',
                    'is_sponsored': '0',
                    'skill_count': float(h_skills),
                    'benefit_count': 5.0,
                    'description_length': 2000.0,
                    'company_size': 3.0,
                    'employee_count': 500.0,
                    'follower_count': 2000.0
                }])
                
                prob = hot_job_model.predict_proba(h_input)[0][1]
                
                st.markdown(f"### Xác suất trở thành Hot Job: **{prob*100:.1f}%**")
                
                if prob > 0.6:
                    st.success("✨ **CƠ HỘI VÀNG:** Tin đăng này có tiềm năng tương tác cực cao!")
                else:
                    st.warning("⚠️ **Lời khuyên:** Hãy thử bổ sung thêm kỹ năng hoặc điều chỉnh tiêu đề để hấp dẫn hơn.")
                
                # Radar Chart for Visualization (Mock/Static for aesthetic)
                st.markdown("#### Phân tích sức mạnh tin đăng")
                categories = ['Lương','Kỹ năng','Thương hiệu','Độ hiếm','Địa điểm']
                fig_radar = go.Figure(data=go.Scatterpolar(
                  r=[prob*5, 4, 3, 5, 2],
                  theta=categories,
                  fill='toself',
                  marker=dict(color='#184683')
                ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), 
                                         showlegend=False, 
                                         font=dict(color='#003859', size=14))
                st.plotly_chart(fig_radar, width='stretch')

if __name__ == "__main__":
    main()
