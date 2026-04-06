import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import io
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="plotly.express")

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="Maintenance Analytics Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

    date_cols = ['Basic start date', 'Basic finish date']
    for col in date_cols:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')

    data['Year'] = data['Basic start date'].dt.year
    data['Month'] = data['Basic start date'].dt.month_name()
    data['Quarter'] = data['Basic start date'].dt.quarter

    def determine_status(row):
        statuses = f"{row.get('System status', '')} {row.get('User Status', '')}".upper().split()
        priority = {'CNCL': 'Canceled', 'CNF': 'Completed', 'JIPR': 'In Progress', 'NCMP': 'Not Executed & Deleted'}
        for code, name in priority.items():
            if code in statuses:
                return name
        return 'Open'

    data['Order Status'] = data.apply(determine_status, axis=1)
    data['Cost Deviation'] = data['Total sum (actual)'] - data['Total planned costs']
    data['Cost Variance %'] = (data['Cost Deviation'] / data['Total planned costs'].replace(0, np.nan)) * 100
    data['Plan Type'] = np.where(data['Maintenance Plan'].notna(), 'Planned', 'Unplanned')

    return data


# ========================= MAIN APP =========================
def main():
    st.title("🏭 Maintenance Operations Analytics Dashboard")

    uploaded_file = st.file_uploader("📤 Upload maintenance data (Excel)", type=["xlsx"])

    # Load and process data FIRST (before any sidebar widgets)
    with st.spinner("🔄 Loading and processing data..."):
        raw_data = load_data(uploaded_file)
        data = process_data(raw_data)

    if data.empty:
        st.warning("⚠️ No data available. Please upload an Excel file.")
        st.stop()

    # ====================== FILTERS (NOW SAFE) ======================
    st.sidebar.header("🔍 Filter Options")

    plants = st.sidebar.multiselect(
        "Select Plants:",
        options=sorted(data['Plant'].dropna().unique()),
        default=sorted(data['Plant'].dropna().unique())[:2]
    )

    years = st.sidebar.slider(
        "Select Year Range:",
        min_value=int(data['Year'].min()),
        max_value=int(data['Year'].max()),
        value=(int(data['Year'].min()), int(data['Year'].max()))
    )

    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    available_months = sorted(data['Month'].dropna().unique(),
                              key=lambda x: month_order.index(x) if x in month_order else 99)

    months = st.sidebar.multiselect(
        "Select Months:",
        options=available_months,
        default=available_months
    )

    plan_type = st.sidebar.multiselect(
        "Plan Type:", options=data['Plan Type'].unique(), default=data['Plan Type'].unique()
    )

    statuses = st.sidebar.multiselect(
        "Order Statuses:", options=data['Order Status'].unique(), default=data['Order Status'].unique()
    )

    work_centers = st.sidebar.multiselect(
        "Department (Main Work Center):",
        options=sorted(data['Main Work Center'].dropna().unique()),
        default=sorted(data['Main Work Center'].dropna().unique())
    )

    order_types = st.sidebar.multiselect(
        "Work Order Type:",
        options=sorted(data['Order Type'].dropna().unique()),
        default=sorted(data['Order Type'].dropna().unique())
    )

    groups = st.sidebar.multiselect(
        "Task List Code (Group):",
        options=sorted(data['Group'].dropna().unique()),
        default=sorted(data['Group'].dropna().unique())
    )

    # ====================== APPLY FILTERS ======================
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

    # ====================== DASHBOARD CONTENT ======================
    st.divider()
    st.header("🔎 Active Filters Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Selected Plants")
        st.write(", ".join(map(str, filtered_data['Plant'].unique())))
    with col2:
        st.subheader("Selected Months")
        st.write(", ".join(filtered_data['Month'].unique()))
    with col3:
        st.subheader("Selected Years")
        st.write(", ".join(map(str, sorted(filtered_data['Year'].unique()))))

    st.divider()

    # Reuse your display functions (kept the same as before)
    display_kpis(filtered_data)
    st.divider()
    plot_order_status_distribution(filtered_data)
    st.divider()
    plot_department_orders(filtered_data)
    st.divider()
    plot_status_trends(filtered_data)
    plot_order_type_analysis(filtered_data)
    st.divider()
    plot_cost_analysis(filtered_data)
    st.divider()
    show_raw_data(filtered_data)


# ========================= DISPLAY FUNCTIONS (unchanged) =========================
def display_kpis(filtered_data):
    st.header("📊 Key Metrics (Sorted Ascending)")
    planned_orders = filtered_data[filtered_data['Plan Type'] == 'Planned'].copy()
    total_planned = len(planned_orders)
    completed_planned = len(planned_orders[planned_orders['Order Status'] == 'Completed'])
    planned_completion_pct = (completed_planned / total_planned * 100) if total_planned > 0 else 0

    total_orders = len(filtered_data)
    completed_orders = len(filtered_data[filtered_data['Order Status'] == 'Completed'])
    overall_completion_pct = (completed_orders / total_orders * 100) if total_orders > 0 else 0

    metrics = [
        ("Total Orders", len(filtered_data), "", lambda x: f"{x:,}"),
        ("Planned Orders", total_planned, "", lambda x: f"{x:,}"),
        ("Planned Completion %", planned_completion_pct, "", lambda x: f"{x:.1f}%"),
        ("Overall Completion %", overall_completion_pct, "", lambda x: f"{x:.1f}%"),
        ("Actual Cost (EGP)", filtered_data['Total sum (actual)'].sum(), "", lambda x: f"{x:,.2f}"),
        ("Avg Cost Deviation (EGP)", filtered_data['Cost Deviation'].mean(), "", lambda x: f"{x:,.2f}")
    ]

    sorted_metrics = sorted([{"label": l, "value": v, "formatter": f} for l, v, _, f in metrics], key=lambda x: x["value"])

    cols = st.columns(len(sorted_metrics))
    for i, m in enumerate(sorted_metrics):
        with cols[i]:
            st.metric(label=m["label"], value=m["formatter"](m["value"]))

    st.subheader("📈 Completion Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        if total_planned > 0:
            fig = px.pie(names=['Completed', 'Not Completed'],
                         values=[completed_planned, total_planned - completed_planned],
                         title="Planned Orders", hole=0.4)
            fig.update_traces(textinfo='percent+label', textposition='inside')
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if total_orders > 0:
            fig = px.pie(names=['Completed', 'Not Completed'],
                         values=[completed_orders, total_orders - completed_orders],
                         title="Overall Orders", hole=0.4)
            fig.update_traces(textinfo='percent+label', textposition='inside')
            st.plotly_chart(fig, use_container_width=True)


def plot_order_status_distribution(filtered_data):
    st.subheader("📊 Order Status Distribution")
    status_counts = filtered_data.groupby(['Order Status', 'Plan Type'], observed=True).size().reset_index(name='Count')
    fig = px.bar(status_counts, x='Order Status', y='Count', color='Plan Type', barmode='group', text='Count')
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)


def plot_department_orders(filtered_data):
    st.subheader("🏗️ Department-wise Order Distribution")
    dept_counts = filtered_data.groupby(['Main Work Center', 'Order Status'], observed=True).size().reset_index(name='Count')
    dept_counts = dept_counts.rename(columns={'Main Work Center': 'Department'})
    fig = px.bar(dept_counts, x='Department', y='Count', color='Order Status', text='Count')
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)


def plot_status_trends(filtered_data):
    st.subheader("📈 Status Trends Over Time")
    month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
    trend_data = filtered_data.groupby(['Year', 'Month', 'Order Status'], observed=True).size().reset_index(name='Count')
    trend_data['Month'] = pd.Categorical(trend_data['Month'], categories=month_order, ordered=True)
    trend_data = trend_data.sort_values(['Year', 'Month'])
    fig = px.line(trend_data, x='Month', y='Count', color='Order Status', facet_col='Year', markers=True)
    st.plotly_chart(fig, use_container_width=True)


def plot_order_type_analysis(filtered_data):
    st.subheader("📦 Order Type Analysis")
    col1, col2 = st.columns(2)
    with col1:
        type_dist = filtered_data['Order Type'].value_counts().reset_index()
        type_dist.columns = ['Order Type', 'Count']
        fig = px.pie(type_dist, names='Order Type', values='Count', title="Order Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        trend_data = filtered_data.groupby(['Year', 'Month', 'Order Type'], observed=True).size().reset_index(name='Count')
        month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
        trend_data['Month'] = pd.Categorical(trend_data['Month'], categories=month_order, ordered=True)
        trend_data = trend_data.sort_values(['Year', 'Month'])
        fig = px.line(trend_data, x='Month', y='Count', color='Order Type', facet_col='Year', markers=True)
        st.plotly_chart(fig, use_container_width=True)


def plot_cost_analysis(filtered_data):
    st.subheader("💵 Cost Analysis")
    cols = st.columns(2)
    with cols[0]:
        fig = px.bar(filtered_data.nsmallest(10, 'Cost Deviation'), x='Order Type', y='Cost Deviation', color='Plant',
                     title="Top 10 Cost Savings")
        st.plotly_chart(fig, use_container_width=True)
    with cols[1]:
        fig = px.box(filtered_data, x='Order Status', y='Cost Variance %', color='Plant',
                     title="Cost Variance Distribution")
        st.plotly_chart(fig, use_container_width=True)


def show_raw_data(filtered_data):
    st.subheader("📄 Raw Data Explorer")
    st.dataframe(filtered_data.sort_values('Basic start date', ascending=False), use_container_width=True, height=400)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        filtered_data.to_excel(writer, index=False)
    buffer.seek(0)
    st.download_button("📥 Download Filtered Data", buffer, "filtered_maintenance_data.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()
