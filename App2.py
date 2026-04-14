import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import io

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Maintenance Analytics Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    try:
        return pd.read_excel("D:/Dash Board/Maintenance Orders.xlsx")
    except:
        return pd.DataFrame()

# ------------------ PROCESS DATA ------------------
def process_data(data):
    if data.empty:
        return data

    for col in ['Basic start date', 'Basic finish date']:
        data[col] = pd.to_datetime(data[col], errors='coerce')

    data['Year'] = data['Basic start date'].dt.year
    data['Month'] = data['Basic start date'].dt.month_name()
    data['Quarter'] = data['Basic start date'].dt.quarter

    def determine_status(row):
        statuses = f"{row['System status']} {row['User Status']}".split()
        for s in ['CNCL', 'CNF', 'JIPR', 'NCMP']:
            if s in statuses:
                return {
                    'CNCL': 'Canceled',
                    'CNF': 'Completed',
                    'JIPR': 'In Progress',
                    'NCMP': 'Not Executed & Deleted'
                }[s]
        return 'Open'

    data['Order Status'] = data.apply(determine_status, axis=1)

    data['Plan Type'] = np.where(
        data['Maintenance Plan'].notna(),
        'Planned',
        'Unplanned'
    )

    data['Cost Deviation'] = data['Total sum (actual)'] - data['Total planned costs']
    data['Cost Variance %'] = (data['Cost Deviation'] / data['Total planned costs']).replace([np.inf, -np.inf], np.nan) * 100

    return data

# ------------------ SIDEBAR FILTERS ------------------
def create_filters(data):
    st.sidebar.title("🔍 Filters")

    with st.sidebar.expander("🏭 Plant", True):
        plants = st.multiselect("Plant", data['Plant'].unique(), default=data['Plant'].unique())

    with st.sidebar.expander("📅 Time", True):
        years = st.slider("Year",
                          int(data['Year'].min()),
                          int(data['Year'].max()),
                          (int(data['Year'].min()), int(data['Year'].max())))
        months = st.multiselect("Month", data['Month'].unique(), default=data['Month'].unique())

    with st.sidebar.expander("⚙️ Operations"):
        work_center = st.multiselect("Department", data['Main Work Center'].unique(), default=data['Main Work Center'].unique())
        order_type = st.multiselect("Order Type", data['Order Type'].unique(), default=data['Order Type'].unique())
        group = st.multiselect("Task List", data['Group'].unique(), default=data['Group'].unique())

    with st.sidebar.expander("📊 Status"):
        status = st.multiselect("Order Status", data['Order Status'].unique(), default=data['Order Status'].unique())
        plan_type = st.multiselect("Plan Type", data['Plan Type'].unique(), default=data['Plan Type'].unique())

    if st.sidebar.button("🔄 Reset Filters"):
        st.experimental_rerun()

    return plants, years, months, status, work_center, order_type, group, plan_type

# ------------------ KPI ------------------
def display_kpis(df):
    total = len(df)
    completed = len(df[df['Order Status'] == 'Completed'])
    planned = df[df['Plan Type'] == 'Planned']
    planned_completed = len(planned[planned['Order Status'] == 'Completed'])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Orders", f"{total:,}")
    col2.metric("Completed", f"{completed:,}")
    col3.metric("Completion %", f"{(completed/total*100 if total else 0):.1f}%")
    col4.metric("Planned Completion %", f"{(planned_completed/len(planned)*100 if len(planned) else 0):.1f}%")

# ------------------ MAIN ------------------
def main():

    st.markdown("<h1 style='text-align:center;'>🏭 Maintenance Analytics Dashboard</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

    data = load_data(uploaded_file)
    df = process_data(data)

    if df.empty:
        st.warning("Please upload valid data.")
        return

    filters = create_filters(df)

    filtered = df[
        (df['Plant'].isin(filters[0])) &
        (df['Year'].between(filters[1][0], filters[1][1])) &
        (df['Month'].isin(filters[2])) &
        (df['Order Status'].isin(filters[3])) &
        (df['Main Work Center'].isin(filters[4])) &
        (df['Order Type'].isin(filters[5])) &
        (df['Group'].isin(filters[6])) &
        (df['Plan Type'].isin(filters[7]))
    ]

    if filtered.empty:
        st.error("No data matches filters")
        return

    # ------------------ TABS ------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Performance",
        "🏭 Operations",
        "💰 Cost",
        "📄 Data"
    ])

    # ------------------ OVERVIEW ------------------
    with tab1:
        display_kpis(filtered)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(filtered, names='Order Status', title="Order Status")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(filtered, names='Plan Type', title="Planned vs Unplanned")
            st.plotly_chart(fig, use_container_width=True)

    # ------------------ PERFORMANCE ------------------
    with tab2:
        fig = px.bar(filtered, x='Order Status', color='Plan Type', title="Status Distribution")
        st.plotly_chart(fig, use_container_width=True)

        trend = filtered.groupby(['Month', 'Order Status']).size().reset_index(name='Count')
        fig = px.line(trend, x='Month', y='Count', color='Order Status')
        st.plotly_chart(fig, use_container_width=True)

    # ------------------ OPERATIONS ------------------
    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            dept = filtered['Main Work Center'].value_counts().reset_index()
            dept.columns = ['Dept', 'Count']
            fig = px.bar(dept, x='Dept', y='Count')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            typ = filtered['Order Type'].value_counts().reset_index()
            typ.columns = ['Type', 'Count']
            fig = px.pie(typ, names='Type', values='Count')
            st.plotly_chart(fig)

    # ------------------ COST ------------------
    with tab4:
        fig = px.box(filtered, x='Order Status', y='Cost Variance %')
        st.plotly_chart(fig, use_container_width=True)

        cost = filtered.groupby('Order Type')[['Total planned costs', 'Total sum (actual)']].mean().reset_index()
        fig = px.bar(cost, x='Order Type', y=['Total planned costs', 'Total sum (actual)'], barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    # ------------------ DATA ------------------
    with tab5:
        st.dataframe(filtered, use_container_width=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            filtered.to_excel(writer, index=False)

        st.download_button("Download Excel", buffer, "data.xlsx")

# ------------------ RUN ------------------
if __name__ == "__main__":
    main()
