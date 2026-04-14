import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import io
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="plotly.express")

# ========================= PAGE CONFIG & STYLING =========================
st.set_page_config(
    page_title="Maintenance Analytics Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern UI Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stPlotlyChart {
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# ========================= DATA LOADING =========================
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    try:
        return pd.read_excel("D:/Dash Board/Maintenance Orders.xlsx")
    except FileNotFoundError:
        st.error("❌ Default file not found. Please upload an Excel file.")
        return pd.DataFrame()

def process_data(data):
    if data.empty:
        return data

    # Date conversion
    for col in ['Basic start date', 'Basic finish date']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')

    # Temporal features
    data['Year'] = data['Basic start date'].dt.year
    data['Month'] = data['Basic start date'].dt.month_name()
    data['Quarter'] = data['Basic start date'].dt.quarter

    # Order Status
    def determine_status(row):
        statuses = f"{row.get('System status', '')} {row.get('User Status', '')}".upper().split()
        priority = {'CNCL': 'Canceled', 'CNF': 'Completed', 'JIPR': 'In Progress', 'NCMP': 'Not Executed & Deleted'}
        for code, name in priority.items():
            if code in statuses:
                return name
        return 'Open'
    
    data['Order Status'] = data.apply(determine_status, axis=1)

    # Cost metrics
    data['Cost Deviation'] = data['Total sum (actual)'] - data['Total planned costs']
    data['Cost Variance %'] = (data['Cost Deviation'] / data['Total planned costs'].replace(0, np.nan)) * 100

    # Plan Type
    data['Plan Type'] = np.where(data['Maintenance Plan'].notna(), 'Planned', 'Unplanned')

    return data

# ========================= FILTERS =========================
def create_filters(data):
    st.sidebar.header("🔍 Filter Options")
    st.sidebar.markdown("---")

    plants = st.sidebar.multiselect("🌱 Plants", 
        options=sorted(data['Plant'].dropna().unique()), 
        default=sorted(data['Plant'].dropna().unique())[:2])

    years = st.sidebar.slider("📅 Year Range", 
        min_value=int(data['Year'].min()), 
        max_value=int(data['Year'].max()), 
        value=(int(data['Year'].min()), int(data['Year'].max())))

    month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
    available_months = sorted(data['Month'].dropna().unique(), 
                              key=lambda x: month_order.index(x) if x in month_order else 99)

    months = st.sidebar.multiselect("📆 Months", options=available_months, default=available_months)

    plan_type = st.sidebar.multiselect("📋 Plan Type", 
        options=data['Plan Type'].unique(), default=data['Plan Type'].unique())

    statuses = st.sidebar.multiselect("📌 Order Status", 
        options=data['Order Status'].unique(), default=data['Order Status'].unique())

    work_centers = st.sidebar.multiselect("🏭 Department", 
        options=sorted(data['Main Work Center'].dropna().unique()), 
        default=sorted(data['Main Work Center'].dropna().unique()))

    order_types = st.sidebar.multiselect("🔧 Order Type", 
        options=sorted(data['Order Type'].dropna().unique()), 
        default=sorted(data['Order Type'].dropna().unique()))

    groups = st.sidebar.multiselect("📂 Task Group", 
        options=sorted(data['Group'].dropna().unique()), 
        default=sorted(data['Group'].dropna().unique()))

    if st.sidebar.button("🔄 Reset Filters", use_container_width=True):
        st.rerun()

    return plants, years, months, plan_type, statuses, work_centers, order_types, groups

# ========================= PLOT FUNCTIONS (All Fixed) =========================
def plot_order_status_distribution(filtered_data):
    st.subheader("📊 Order Status Distribution")
    status_counts = filtered_data.groupby(['Order Status', 'Plan Type'], observed=True).size().reset_index(name='Count')
    
    fig = px.bar(status_counts, x='Order Status', y='Count', color='Plan Type',
                 barmode='group', text='Count',
                 color_discrete_map={'Planned': '#636EFA', 'Unplanned': '#EF553B'})
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig.update_layout(legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # Plant breakdown
    plant_status = filtered_data.groupby(['Plant', 'Order Status'], observed=True).size().reset_index(name='Count')
    fig2 = px.bar(plant_status, x='Plant', y='Count', color='Order Status', barmode='group')
    st.plotly_chart(fig2, use_container_width=True)

def plot_department_orders(filtered_data):
    st.subheader("🏗️ Department-wise Orders")
    dept_counts = filtered_data.groupby(['Main Work Center', 'Order Status'], observed=True).size().reset_index(name='Count')
    dept_counts.rename(columns={'Main Work Center': 'Department'}, inplace=True)
    
    fig = px.bar(dept_counts, x='Department', y='Count', color='Order Status', text='Count')
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

def plot_status_trends(filtered_data):
    st.subheader("📈 Status Trends Over Time")
    month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
    
    trend_data = filtered_data.groupby(['Year', 'Month', 'Order Status'], observed=True).size().reset_index(name='Count')
    trend_data['Month'] = pd.Categorical(trend_data['Month'], categories=month_order, ordered=True)
    trend_data = trend_data.sort_values(['Year', 'Month'])
    
    fig = px.line(trend_data, x='Month', y='Count', color='Order Status', 
                  facet_col='Year', markers=True)
    fig.update_xaxes(categoryorder='array', categoryarray=month_order)
    st.plotly_chart(fig, use_container_width=True)

def plot_order_type_analysis(filtered_data):
    st.subheader("📦 Order Type Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        type_dist = filtered_data['Order Type'].value_counts().reset_index()
        type_dist.columns = ['Order Type', 'Count']
        fig = px.pie(type_dist, names='Order Type', values='Count', title="Order Type Distribution")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        trend_data = filtered_data.groupby(['Year', 'Month', 'Order Type'], observed=True).size().reset_index(name='Count')
        month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
        trend_data['Month'] = pd.Categorical(trend_data['Month'], categories=month_order, ordered=True)
        trend_data = trend_data.sort_values(['Year', 'Month'])
        
        fig = px.line(trend_data, x='Month', y='Count', color='Order Type', 
                      facet_col='Year', markers=True, title="Monthly Order Type Trends")
        st.plotly_chart(fig, use_container_width=True)

    # Cost by Order Type
    st.subheader("💸 Planned vs Actual Cost by Order Type")
    cost_data = filtered_data.groupby('Order Type', observed=True).agg({
        'Total planned costs': 'mean',
        'Total sum (actual)': 'mean'
    }).reset_index()
    fig = px.bar(cost_data, x='Order Type', y=['Total planned costs', 'Total sum (actual)'],
                 barmode='group', title="Average Planned vs Actual Costs")
    st.plotly_chart(fig, use_container_width=True)

def plot_cost_analysis(filtered_data):
    st.subheader("💵 Cost Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        dev_data = filtered_data.nsmallest(10, 'Cost Deviation')
        fig = px.bar(dev_data, x='Order Type', y='Cost Deviation', color='Plant',
                     title="Top 10 Cost Savings", text_auto='.2f')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(filtered_data, x='Order Status', y='Cost Variance %', 
                     color='Plant', title="Cost Variance Distribution")
        st.plotly_chart(fig, use_container_width=True)

def display_kpis(filtered_data):
    st.header("📊 Key Metrics")
    
    planned = filtered_data[filtered_data['Plan Type'] == 'Planned']
    total_planned = len(planned)
    completed_planned = len(planned[planned['Order Status'] == 'Completed'])
    planned_pct = (completed_planned / total_planned * 100) if total_planned > 0 else 0
    
    total_orders = len(filtered_data)
    completed = len(filtered_data[filtered_data['Order Status'] == 'Completed'])
    overall_pct = (completed / total_orders * 100) if total_orders > 0 else 0

    cols = st.columns(6)
    metrics = [
        ("Total Orders", f"{total_orders:,}", ""),
        ("Planned Orders", f"{total_planned:,}", "🔹"),
        ("Planned Completion", f"{planned_pct:.1f}%", "✅"),
        ("Overall Completion", f"{overall_pct:.1f}%", "🏆"),
        ("Total Actual Cost", f"{filtered_data['Total sum (actual)'].sum():,.0f} EGP", "💰"),
        ("Avg Cost Deviation", f"{filtered_data['Cost Deviation'].mean():,.0f} EGP", "📉")
    ]
    
    for i, (label, value, emoji) in enumerate(metrics):
        with cols[i]:
            st.metric(label=f"{emoji} {label}", value=value)

def show_raw_data(filtered_data):
    st.subheader("📄 Raw Data")
    st.dataframe(filtered_data.sort_values('Basic start date', ascending=False), 
                 use_container_width=True, height=550)
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        filtered_data.to_excel(writer, index=False)
    buffer.seek(0)
    
    st.download_button(
        label="📥 Download Filtered Data (Excel)",
        data=buffer,
        file_name="maintenance_filtered_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# ========================= MAIN APP =========================
def main():
    st.markdown('<h1 class="main-header">🔧 Maintenance Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powerful insights for smarter maintenance decisions</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Maintenance Orders Excel file", type=["xlsx"])

    with st.spinner("🔄 Processing your maintenance data..."):
        raw_data = load_data(uploaded_file)
        data = process_data(raw_data)

    if data.empty:
        st.warning("⚠️ No data loaded. Please upload an Excel file.")
        st.stop()

    # Filters
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
        st.error("❌ No records match your current filters. Please adjust the filters.")
        st.stop()

    # Tabs - Clean Navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Order Analysis", 
        "🏭 Departments & Trends", 
        "💰 Cost Analysis", 
        "📋 Raw Data"
    ])

    with tab1:
        display_kpis(filtered_data)
        st.divider()
        plot_order_status_distribution(filtered_data)

    with tab2:
        plot_order_type_analysis(filtered_data)

    with tab3:
        plot_department_orders(filtered_data)
        st.divider()
        plot_status_trends(filtered_data)

    with tab4:
        plot_cost_analysis(filtered_data)

    with tab5:
        show_raw_data(filtered_data)

    st.caption("🔧 Maintenance Analytics Dashboard • Clean & Professional UI")

if __name__ == "__main__":
    main()
