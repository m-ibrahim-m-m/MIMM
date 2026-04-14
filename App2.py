import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="plotly.express")

# ========================= THEME & STYLES =========================
COLORS = {
    "primary": "#1E3A5F", "accent": "#00C2CB", "success": "#00B37E",
    "warning": "#F59E0B", "danger": "#EF4444", "planned": "#3B82F6",
    "unplanned": "#F97316", "bg_dark": "#0F1923", "bg_card": "#1A2635",
    "text_light": "#E2E8F0", "text_muted": "#94A3B8",
}

STATUS_COLORS = {
    "Completed": "#00B37E", "In Progress": "#00C2CB", "Open": "#F59E0B",
    "Canceled": "#EF4444", "Not Executed & Deleted": "#6B7280",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'IBM Plex Sans', sans-serif", color="#E2E8F0", size=12),
    title=dict(font=dict(size=15, color="#E2E8F0"), x=0.01, xanchor="left"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    margin=dict(l=16, r=16, t=48, b=16),
    colorway=["#00C2CB", "#3B82F6", "#00B37E", "#F59E0B", "#EF4444", "#F97316"],
)

LEGEND_DEFAULT = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1)

def apply_layout(fig, legend=None):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(legend={**LEGEND_DEFAULT, **(legend or {})})
    return fig

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif !important; }
.stApp { background: #0F1923; color: #E2E8F0; }
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #131E2B !important; border-right: 1px solid rgba(255,255,255,0.06); }
.dash-hero { background: linear-gradient(135deg, #1E3A5F 0%, #0F1923 60%); border: 1px solid rgba(0,194,203,0.18); border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; position: relative; overflow: hidden; }
.dash-hero h1 { font-size:1.9rem!important; font-weight:700!important; color:#E2E8F0!important; margin:0 0 0.25rem!important; letter-spacing:-0.02em; }
.section-header { display:flex; align-items:center; gap:0.6rem; margin:2rem 0 1rem; padding-bottom:0.6rem; border-bottom:1px solid rgba(255,255,255,0.06); }
.kpi-card { background:#1A2635; border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:1.2rem 1.4rem; position:relative; overflow:hidden; }
.kpi-label { font-size:0.72rem; color:#64748B; font-weight:600; letter-spacing:0.07em; text-transform:uppercase; margin-bottom:0.5rem; }
.kpi-value { font-size:1.6rem; font-weight:700; color:#E2E8F0; font-family:'IBM Plex Mono',monospace; line-height:1; }
.chart-card { background:#1A2635; border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:1.25rem; margin-bottom:1rem; }
</style>
"""

MONTH_ORDER = ['January','February','March','April','May','June','July','August','September','October','November','December']

# ========================= DATA =========================
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    try:
        return pd.read_excel("D:/Dash Board/Maintenance Orders.xlsx")
    except FileNotFoundError:
        st.error("❌ Default file not found. Please upload an Excel file.")
        return pd.DataFrame()

def process_data(data):
    if data.empty: return data
    for col in ['Basic start date', 'Basic finish date']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
    if 'Basic start date' in data.columns:
        data['Year'] = data['Basic start date'].dt.year
        data['Month'] = data['Basic start date'].dt.month_name()
        data['Quarter'] = data['Basic start date'].dt.quarter

    def determine_status(row):
        statuses = f"{row.get('System status','')} {row.get('User Status','')}".upper().split()
        for code, name in {'CNCL':'Canceled','CNF':'Completed','JIPR':'In Progress','NCMP':'Not Executed & Deleted'}.items():
            if code in statuses: return name
        return 'Open'
    
    data['Order Status'] = data.apply(determine_status, axis=1)
    data['Cost Deviation'] = data.get('Total sum (actual)', 0) - data.get('Total planned costs', 0)
    data['Cost Variance %'] = data['Cost Deviation'] / data.get('Total planned costs', 0).replace(0, np.nan) * 100
    data['Plan Type'] = np.where(data.get('Maintenance Plan').notna(), 'Planned', 'Unplanned')
    return data

# ========================= SIDEBAR (Safer) =========================
def create_filters(data):
    st.sidebar.markdown(
        "<div style='padding:0.5rem 0 1.2rem'>"
        "<p style='color:#00C2CB;font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin:0 0 0.2rem'>Maintenance Dashboard</p>"
        "<p style='color:#64748B;font-size:0.78rem;margin:0'>Filter & explore data</p></div>",
        unsafe_allow_html=True
    )

    all_plants = sorted(data['Plant'].dropna().unique())
    all_months = sorted(data['Month'].dropna().unique(), key=lambda x: MONTH_ORDER.index(x) if x in MONTH_ORDER else 99)
    all_plan_types = sorted(data['Plan Type'].dropna().unique())
    all_statuses = sorted(data['Order Status'].dropna().unique())
    all_order_types = sorted(data['Order Type'].dropna().unique())
    all_work_centers = sorted(data['Main Work Center'].dropna().unique())
    all_groups = sorted(data['Group'].dropna().unique())
    year_min = int(data['Year'].min()) if not data.empty else 2020
    year_max = int(data['Year'].max()) if not data.empty else 2026

    ss = st.session_state

    for key, val in [("f_plants", all_plants), ("f_years", (year_min, year_max)), ("f_months", all_months),
                     ("f_plan_type", all_plan_types), ("f_statuses", all_statuses), ("f_order_types", all_order_types),
                     ("f_work_centers", all_work_centers), ("f_groups", all_groups)]:
        if key not in ss:
            ss[key] = val

    # Safer sanitise with fallback
    for key, options in [("f_plants", all_plants), ("f_months", all_months), ("f_plan_type", all_plan_types),
                         ("f_statuses", all_statuses), ("f_order_types", all_order_types),
                         ("f_work_centers", all_work_centers), ("f_groups", all_groups)]:
        if not ss.get(key):
            ss[key] = list(options)
        else:
            valid = [v for v in ss[key] if v in options]
            ss[key] = valid if valid else list(options)

    sy = ss["f_years"]
    ss["f_years"] = (max(year_min, min(int(sy[0]), year_max)), max(year_min, min(int(sy[1]), year_max)))

    with st.sidebar.expander("🏭 Plants", expanded=True):
        plants = st.multiselect("Plants", options=all_plants, key="f_plants", label_visibility="collapsed")
    with st.sidebar.expander("📅 Time Range", expanded=True):
        years = st.slider("Years", min_value=year_min, max_value=year_max, key="f_years", label_visibility="collapsed")
        months = st.multiselect("Months", options=all_months, key="f_months", label_visibility="collapsed")
    with st.sidebar.expander("⚙️ Order Attributes", expanded=False):
        plan_type = st.multiselect("Plan Type", options=all_plan_types, key="f_plan_type")
        statuses = st.multiselect("Order Status", options=all_statuses, key="f_statuses")
        order_types = st.multiselect("Work Order Type", options=all_order_types, key="f_order_types")
    with st.sidebar.expander("🏗️ Department / Group", expanded=False):
        work_centers = st.multiselect("Main Work Center", options=all_work_centers, key="f_work_centers")
        groups = st.multiselect("Task List Code (Group)", options=all_groups, key="f_groups")

    if st.sidebar.button("↺ Reset All Filters", use_container_width=True):
        for key in list(ss.keys()):
            if key.startswith("f_"): del ss[key]
        st.rerun()

    return plants, years, months, plan_type, statuses, work_centers, order_types, groups

# ========================= MAIN =========================
def main():
    st.set_page_config(page_title="Maintenance Analytics", page_icon="🔧", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown('<div class="dash-hero"><h1>🏭 Maintenance <span class="accent">Analytics</span> Dashboard</h1><p>Real-time insight into work orders, costs, and team performance</p></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload maintenance data (Excel .xlsx)", type=["xlsx"], label_visibility="collapsed")

    cached_load = st.cache_data(load_data)
    with st.spinner("Loading and processing data…"):
        raw_data = cached_load(uploaded_file)
        data = process_data(raw_data)

    if data.empty:
        st.warning("⚠️ No data available. Please upload an Excel file.")
        st.stop()

    plants, years, months, plan_type, statuses, work_centers, order_types, groups = create_filters(data)

    # Safe filtering with fallback
    fd = data.copy()
    if plants: fd = fd[fd['Plant'].isin(plants)]
    if months: fd = fd[fd['Month'].isin(months)]
    if plan_type: fd = fd[fd['Plan Type'].isin(plan_type)]
    if statuses: fd = fd[fd['Order Status'].isin(statuses)]
    if order_types: fd = fd[fd['Order Type'].isin(order_types)]
    if work_centers: fd = fd[fd['Main Work Center'].isin(work_centers)]
    if groups: fd = fd[fd['Group'].isin(groups)]
    if years: fd = fd[fd['Year'].between(years[0], years[1])]

    if fd.empty:
        st.error("⚠️ No records match the current filters. Try resetting all filters.")
        st.stop()

    display_filter_summary(fd, years, months)
    section_header("📊", "Key Performance Indicators")
    display_kpis(fd)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["✅ Completion", "📦 Order Status", "🏗️ Departments", "💰 Cost Analysis", "📄 Raw Data"])

    with tab1:
        section_header("✅", "Completion Overview", "Planned & Overall")
        plot_completion_donuts(fd)
        section_header("📈", "Trends Over Time")
        plot_status_trends(fd)
    with tab2:
        section_header("📊", "Status Distribution")
        plot_status_distribution(fd)
        section_header("📦", "Order Type Breakdown")
        plot_order_type_analysis(fd)
    with tab3:
        section_header("🏗️", "Department Performance")
        plot_department_orders(fd)
    with tab4:
        section_header("💰", "Cost Analysis")
        plot_cost_analysis(fd)
    with tab5:
        section_header("📄", "Raw Data Explorer")
        show_raw_data(fd)

if __name__ == "__main__":
    main()
