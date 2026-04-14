import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import io
import warnings

# Suppress specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly.express")

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="Maintenance Analytics Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI/UX
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
    }
    .stPlotlyChart {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .sidebar .sidebar-content {
        background-color: #F1F5F9;
    }
    </style>
""", unsafe_allow_html=True)

# ========================= DATA LOADING & PROCESSING =========================
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    try:
        return pd.read_excel("D:/Dash Board/Maintenance Orders.xlsx")
    except FileNotFoundError:
        st.error("❌ Default file not found at: D:/Dash Board/Maintenance Orders.xlsx")
        return pd.DataFrame()

def process_data(data):
    if data.empty:
        return data

    # Convert dates
    date_cols = ['Basic start date', 'Basic finish date']
    for col in date_cols:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')

    # Temporal features
    data['Year'] = data['Basic start date'].dt.year
    data['Month'] = data['Basic start date'].dt.month_name()
    data['Quarter'] = data['Basic start date'].dt.quarter

    # Order Status
    def determine_status(row):
        statuses = f"{row.get('System status', '')} {row.get('User Status', '')}".upper().split()
        priority = {
            'CNCL': 'Canceled',
            'CNF': 'Completed',
            'JIPR': 'In Progress',
            'NCMP': 'Not Executed & Deleted'
        }
        for code, name in priority.items():
            if code in statuses:
                return name
        return 'Open'

    data['Order Status'] = data.apply(determine_status, axis=1)

    # Cost metrics
    data['Cost Deviation'] = data['Total sum (actual)'] - data['Total planned costs']
    data['Cost Variance %'] = (
        (data['Cost Deviation'] / data['Total planned costs'].replace(0, np.nan)) * 100
    )

    # Plan Type
    data['Plan Type'] = np.where(data['Maintenance Plan'].notna(), 'Planned', 'Unplanned')

    return data

# ========================= FILTER CREATION =========================
def create_filters(data):
    st.sidebar.header("🔍 Filter Options")
    st.sidebar.markdown("---")

    plants = st.sidebar.multiselect(
        "🌱 Select Plants:",
        options=sorted(data['Plant'].dropna().unique()),
        default=sorted(data['Plant'].dropna().unique())[:2]
    )

    years = st.sidebar.slider(
        "📅 Select Year Range:",
        min_value=int(data['Year'].min()),
        max_value=int(data['Year'].max()),
        value=(int(data['Year'].min()), int(data['Year'].max()))
    )

    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    available_months = sorted(data['Month'].dropna().unique(), 
                            key=lambda x: month_order.index(x) if x in month_order else 99)

    months = st.sidebar.multiselect(
        "📆 Select Months:",
        options=available_months,
        default=available_months
    )

    plan_type = st.sidebar.multiselect(
        "📋 Plan Type:",
        options=data['Plan Type'].unique(),
        default=data['Plan Type'].unique()
    )

    statuses = st.sidebar.multiselect(
        "📌 Order Statuses:",
        options=data['Order Status'].unique(),
        default=data['Order Status'].unique()
    )

    work_centers = st.sidebar.multiselect(
        "🏭 Department (Main Work Center):",
        options=sorted(data['Main Work Center'].dropna().unique()),
        default=sorted(data['Main Work Center'].dropna().unique())
    )

    order_types = st.sidebar.multiselect(
        "🔧 Work Order Type:",
        options=sorted(data['Order Type'].dropna().unique()),
        default=sorted(data['Order Type'].dropna().unique())
    )

    groups = st.sidebar.multiselect(
        "📂 Task List Code (Group):",
        options=sorted(data['Group'].dropna().unique()),
        default=sorted(data['Group'].dropna().unique())
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        st.rerun()

    return plants, years, months, plan_type, statuses, work_centers, order_types, groups

# ========================= DISPLAY FUNCTIONS =========================
def display_filter_summary(filtered_data):
    with st.expander("🔎 Active Filters Summary", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Plants", len(filtered_data['Plant'].unique()))
        with col2:
            st.metric("Months", len(filtered_data['Month'].unique()))
        with col3:
            st.metric("Years", len(filtered_data['Year'].unique()))
        with col4:
            st.metric("Total Orders", f"{len(filtered_data):,}")

def display_kpis(filtered_data):
    st.header("📊 Key Performance Indicators")

    planned_orders = filtered_data[filtered_data['Plan Type'] == 'Planned'].copy()
    total_planned = len(planned_orders)
    completed_planned = len(planned_orders[planned_orders['Order Status'] == 'Completed'])
    planned_completion_pct = (completed_planned / total_planned * 100) if total_planned > 0 else 0

    total_orders = len(filtered_data)
    completed_orders = len(filtered_data[filtered_data['Order Status'] == 'Completed'])
    overall_completion_pct = (completed_orders / total_orders * 100) if total_orders > 0 else 0

    metrics = [
        ("Total Orders", len(filtered_data), "All maintenance orders", f"{len(filtered_data):,}"),
        ("Planned Orders", total_planned, "Orders linked to maintenance plans", f"{total_planned:,}"),
        ("Planned Completion", planned_completion_pct, "Planned orders completion rate", f"{planned_completion_pct:.1f}%"),
        ("Overall Completion", overall_completion_pct, "All orders completion rate", f"{overall_completion_pct:.1f}%"),
        ("Actual Cost (EGP)", filtered_data['Total sum (actual)'].sum(), "Total actual spending", f"{filtered_data['Total sum (actual)'].sum():,.2f}"),
        ("Avg Cost Deviation", filtered_data['Cost Deviation'].mean(), "Average deviation from plan", f"{filtered_data['Cost Deviation'].mean():,.2f}")
    ]

    cols = st.columns(len(metrics))
    for i, (label, value, help_text, formatted) in enumerate(metrics):
        with cols[i]:
            st.metric(label=label, value=formatted, help=help_text, delta=None)

    # Completion Progress Bars
    st.subheader("Completion Rates")
    col1, col2 = st.columns(2)
    with col1:
        st.progress(planned_completion_pct / 100, text=f"**Planned Orders Completion**: {planned_completion_pct:.1f}%")
    with col2:
        st.progress(overall_completion_pct / 100, text=f"**Overall Completion**: {overall_completion_pct:.1f}%")

def main():
    # ====================== HEADER ======================
    st.markdown('<h1 class="main-header">🏭 Maintenance Operations Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time insights into your maintenance performance, costs, and efficiency</p>', unsafe_allow_html=True)

    # File Uploader with better styling
    uploaded_file = st.file_uploader(
        "📤 Upload your Maintenance Orders Excel file",
        type=["xlsx"],
        help="Upload your data or the system will use the default file"
    )

    with st.spinner("🔄 Loading and processing maintenance data..."):
        raw_data = load_data(uploaded_file)
        data = process_data(raw_data)

    if data.empty:
        st.warning("⚠️ No data available. Please upload an Excel file to continue.")
        st.stop()

    # ====================== FILTERS ======================
    plants, years, months, plan_type, statuses, work_centers, order_types, groups = create_filters(data)

    # Apply filters
    filtered_data = data[
        (data['Plant'].isin(plants)) &
        (data['Year'].between(years[0], years[1])) &
        (data['Month'].isin(months)) &
        (data['Plan Type'].isin(plan_type)) &
        (data['Order Status'].isin(statuses)) &
        (data['Main Work Center'].isin(work_centers)) &
        (data['Order Type'].isin(order_types)) &
        (data['Group'].isin(groups))
    ].copy()

    if filtered_data.empty:
        st.error("❌ No data matches the selected filters. Please adjust your filters.")
        st.stop()

    # ====================== TABBED NAVIGATION ======================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview & KPIs",
        "📈 Order Analysis",
        "🏭 Department & Trends",
        "💰 Cost Analysis",
        "📋 Raw Data"
    ])

    with tab1:
        display_filter_summary(filtered_data)
        st.divider()
        display_kpis(filtered_data)

    with tab2:
        plot_order_status_distribution(filtered_data)
        plot_order_type_analysis(filtered_data)

    with tab3:
        plot_department_orders(filtered_data)
        plot_status_trends(filtered_data)

    with tab4:
        plot_cost_analysis(filtered_data)

    with tab5:
        show_raw_data(filtered_data)

    # Footer
    st.caption("🔧 Maintenance Analytics Dashboard | Built with Streamlit & Plotly")

# ====================== Keep all your existing plot functions ======================
# (Copy-paste all your original plot functions here without any change)

def plot_order_status_distribution(filtered_data):
    st.subheader("📊 Order Status Distribution (Planned vs Unplanned)")
    status_counts = (filtered_data.groupby(['Order Status', 'Plan Type'], observed=True)
                     .size().reset_index(name='Count'))
    fig = px.bar(
        status_counts, x='Order Status', y='Count', color='Plan Type',
        barmode='group', text='Count',
        color_discrete_map={'Planned': '#636EFA', 'Unplanned': '#EF553B'}
    )
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig.update_layout(legend_title="Plan Type", legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # Plant breakdown
    plant_status = (filtered_data.groupby(['Plant', 'Order Status'], observed=True)
                    .size().reset_index(name='Count'))
    fig2 = px.bar(plant_status, x='Plant', y='Count', color='Order Status', barmode='group')
    st.plotly_chart(fig2, use_container_width=True)

# ... (Keep all other plot functions exactly as they were: plot_department_orders, plot_status_trends, etc.)

def plot_department_orders(filtered_data):
    st.subheader("🏗️ Department-wise Order Distribution")
    dept_counts = (filtered_data.groupby(['Main Work Center', 'Order Status'], observed=True)
                   .size().reset_index(name='Count'))
    dept_counts.rename(columns={'Main Work Center': 'Department'}, inplace=True)
    fig = px.bar(dept_counts, x='Department', y='Count', color='Order Status', text='Count')
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# (Add the rest of your plot functions here - they remain unchanged)

def show_raw_data(filtered_data):
    st.subheader("📄 Raw Data Explorer")
    st.dataframe(
        filtered_data.sort_values('Basic start date', ascending=False),
        use_container_width=True,
        height=500
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        filtered_data.to_excel(writer, index=False, sheet_name='Filtered Data')
    buffer.seek(0)
    st.download_button(
        label="📥 Download Filtered Data as Excel",
        data=buffer,
        file_name="filtered_maintenance_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# ========================= RUN APP =========================
if __name__ == "__main__":
    main()
