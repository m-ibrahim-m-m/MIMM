# maintenance_dashboard.py

import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np


# Configure page settings
st.set_page_config(
    page_title="Maintenance Analytics Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading and processing
@st.cache_data
def load_data(uploaded_file=None):
    """Load data from file or uploaded source"""
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    try:
        return pd.read_excel("D:/Dash Board/Maintenance Orders.xlsx")
    except FileNotFoundError:
        return pd.DataFrame()

def process_data(data):
    """Clean and transform raw data"""
    if data.empty:
        return data
    
    # Convert dates and extract temporal features
    date_cols = ['Basic start date', 'Basic finish date']
    for col in date_cols:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    
    data['Year'] = data['Basic start date'].dt.year
    data['Month'] = data['Basic start date'].dt.month_name()
    data['Quarter'] = data['Basic start date'].dt.quarter
    
    # Calculate status categories
    def determine_status(row):
        statuses = f"{row['System status']} {row['User Status']}".split()
        priority_order = [
            'CNCL',  # Canceled
            'CNF',   # Completed
            'EXEC',  # Execution
            'NCMP'   # Not Completed
        ]
        for status in priority_order:
            if status in statuses:
                return {
                    'CNCL': 'Canceled',
                    'CNF': 'Completed',
                    'EXEC': 'In Progress',
                    'NCMP': 'Not Executed & Deleted'
                }[status]
        return 'Open'
    
    data['Order Status'] = data.apply(determine_status, axis=1)
    
    # Calculate cost metrics
    data['Cost Deviation'] = data['Total sum (actual)'] - data['Total planned costs']
    data['Cost Variance %'] = (data['Cost Deviation'] / data['Total planned costs']).replace([np.inf, -np.inf], np.nan) * 100
    
    return data

def create_filters(data):
    """Generate interactive filters in sidebar"""
    st.sidebar.header("🔍 Filter Options")
    
    # Plant multi-select
    plants = st.sidebar.multiselect(
        "Select Plants:",
        options=data['Plant'].unique(),
        default=data['Plant'].unique()[:2],
        help="Filter by plant location"
    )
    
    # Year selector
    years = st.sidebar.slider(
        "Select Year Range:",
        min_value=int(data['Year'].min()),
        max_value=int(data['Year'].max()),
        value=(int(data['Year'].min()), int(data['Year'].max()))
    )
    
    # Month selector
    months = st.sidebar.multiselect(
        "Select Month:",
        options=data['Month'].unique(),
        default=data['Month'].unique()
    )
    
    # Status selector
    statuses = st.sidebar.multiselect(
        "Order Statuses:",
        options=data['Order Status'].unique(),
        default=data['Order Status'].unique()
    )
    
    # Work center selector
    work_center = st.sidebar.multiselect(
        "Department:",
        options=data['Main Work Center'].unique(),
        default=data['Main Work Center'].unique()
    )
     # Work center selector
    Order_type = st.sidebar.multiselect(
        "Work Order Type:",
        options=data['Order Type'].unique(),
        default=data['Order Type'].unique()
    )
    
    return plants, years, months, statuses, work_center ,Order_type

def display_filter_summary(filtered_data):
    """Show the selected filters summary"""
    st.header("🔎 Active Filters Summary")
    col1, col2 ,col3= st.columns(3)
    
    with col1:
        st.subheader("Selected Plants:")
        st.write(", ".join(filtered_data['Plant'].unique()))
    
    with col2:
        st.subheader("Selected Months:")
        st.write(", ".join(filtered_data['Month'].unique()))
    with col3:
        st.subheader("Selected Years:")
        st.write(", ".join(filtered_data['Year'].astype('str').unique()))

def display_kpis(filtered_data):
    """Show key performance indicators sorted by value ascending"""
    st.header("📊 Key Metrics (Ascending Order)")
    
    # Calculate completion percentages
    planned_orders = filtered_data[filtered_data['Maintenance Plan'].notna()]
    total_planned = len(planned_orders)
    completed_planned = len(planned_orders[planned_orders['Order Status'] == 'Completed'])
    planned_completion_pct = (completed_planned / total_planned * 100) if total_planned > 0 else 0
    
    total_orders = len(filtered_data)
    completed_orders = len(filtered_data[filtered_data['Order Status'] == 'Completed'])
    overall_completion_pct = (completed_orders / total_orders * 100) if total_orders > 0 else 0

    # Update metrics list
    metrics = [
        ("Total Orders", len(filtered_data), "", lambda x: f"{x:,}"),
        ("Planned Orders", total_planned, 
         "Orders with maintenance plans", lambda x: f"{x:,}"),
        ("Planned Completion %", planned_completion_pct,
         "Percentage of completed planned orders", lambda x: f"{x:.1f}%"),
        ("Overall Completion %", overall_completion_pct,
         "Percentage of all completed orders", lambda x: f"{x:.1f}%"),
        ("Actual Cost", filtered_data['Total sum (actual)'].sum(), 
         "EGP", lambda x: f"{x:,.2f}"),
        ("Avg Cost Deviation", filtered_data['Cost Deviation'].mean(), 
         "EGP", lambda x: f"{x:,.2f}")
    ]
    
    # Create sortable data structure
    metric_objects = [
        {
            "label": label,
            "value": value,
            "help": help_text,
            "formatter": formatter
        } 
        for (label, value, help_text, formatter) in metrics
    ]
    
    # Sort metrics by numeric value ascending
    sorted_metrics = sorted(metric_objects, key=lambda x: x["value"])
    
    # Create columns dynamically based on sorted metrics
    cols = st.columns(len(sorted_metrics))
    
    for i, metric in enumerate(sorted_metrics):
        with cols[i]:
            st.metric(
                label=metric["label"],
                value=metric["formatter"](metric["value"]),
                help=metric["help"]
            )
    
    # Add completion breakdown pie charts
    st.subheader("📈 Completion Breakdown")
    col1, col2 = st.columns(2)
    
    with col1:
        if total_planned > 0:
            labels = ['Completed', 'Not Completed']
            values = [completed_planned, total_planned - completed_planned]
            fig = px.pie(
                names=labels,
                values=values,
                title="Planned Orders Completion",
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(
                textinfo='percent+label',
                pull=[0.1, 0],
                textposition='inside'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No planned orders available for selected filters.")
    
    with col2:
        if total_orders > 0:
            labels = ['Completed', 'Not Completed']
            values = [completed_orders, total_orders - completed_orders]
            fig = px.pie(
                names=labels,
                values=values,
                title="Overall Orders Completion",
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig.update_traces(
                textinfo='percent+label',
                pull=[0.1, 0],
                textposition='inside'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No orders available for selected filters.")

def plot_order_status_distribution(filtered_data):
    """Visualize order status distribution with planned/unplanned breakdown"""
    st.subheader("📊 Order Status Distribution (Planned vs Unplanned)")
    
    # Create planned/unplanned categorization
    filtered_data['Plan Type'] = np.where(
        filtered_data['Maintenance Plan'].notna(),
        'Planned',
        'Unplanned'
    )
    
    # Create grouped status counts
    status_counts = filtered_data.groupby(
        ['Order Status', 'Plan Type']
    ).size().reset_index(name='Count')
    
    fig = px.bar(
        status_counts,
        x='Order Status',
        y='Count',
        color='Plan Type',
        barmode='group',
        text='Count',
        title="Orders by Status with Planned/Unplanned Breakdown",
        color_discrete_map={
            'Planned': '#636EFA',
            'Unplanned': '#EF553B'
        }
    )
    
    fig.update_traces(
        texttemplate='%{text:,}',
        textposition='outside'
    )
    
    fig.update_layout(
        xaxis_title="Order Status",
        yaxis_title="Number of Orders",
        legend_title="Plan Type",
        uniformtext_minsize=10,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1 ))
    
    st.plotly_chart(fig, use_container_width=True)

def plot_department_orders(filtered_data):
    """Visualize department-wise order counts"""
    st.subheader("🏗️ Department-wise Order Distribution")
    
    department_counts = filtered_data['Main Work Center'].value_counts().reset_index()
    department_counts.columns = ['Department', 'Count']
    
    fig = px.bar(
        department_counts,
        x='Department',
        y='Count',
        color='Department',
        text='Count',
        title="Orders by Department"
    )
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig.update_layout(
        xaxis_title="Department",
        yaxis_title="Number of Orders",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_status_trends(filtered_data):
    """Visualize order status trends with proper month order"""
    st.subheader("📈 Status Trends Over Time")
    
    # Create temporal aggregation
    trend_data = filtered_data.groupby(
        ['Year', 'Month', 'Order Status']
    ).size().reset_index(name='Count')
    
    # Define correct month order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    # Convert to categorical for proper sorting
    trend_data['Month'] = pd.Categorical(trend_data['Month'], 
                                       categories=month_order, 
                                       ordered=True)
    
    # Sort data chronologically
    trend_data = trend_data.sort_values(['Year', 'Month'])
    
    fig = px.line(
        trend_data,
        x='Month',
        y='Count',
        color='Order Status',
        facet_col='Year',
        markers=True,
        title="Monthly Order Status Trends",
        category_orders={"Month": month_order}
    )
    
    # Ensure proper x-axis ordering
    fig.update_xaxes(type='category', categoryorder='array', categoryarray=month_order)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create temporal aggregation
    trend_data2 = filtered_data.groupby(
        ['Year', 'Month', 'Order Status', 'Plan Type']
        ).size().reset_index(name='Count1')
    fig2 = px.line(
        trend_data2,
        x='Plan Type',
        y='Count1',
        color='Order Status',
        facet_col='Year',
        markers=True,
       category_orders={"Month": month_order},
       hover_data=['Month', 'Plan Type', 'Order Status'])

    st.plotly_chart(fig2, use_container_width=True)


def plot_order_type_analysis(filtered_data):
    """Visualize order type analysis"""
    st.subheader("📦 Order Type Analysis")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Order Type Distribution
        type_dist = filtered_data['Order Type'].value_counts().reset_index()
        type_dist.columns = ['Order Type', 'Count']
        
        fig = px.pie(
            type_dist,
            names='Order Type',
            values='Count',
            title="Order Type Distribution"
        )
        fig.update_traces(textposition='inside', textinfo='percent')
        st.plotly_chart(fig)
    
    with col2:
        # Order Type Trends Over Time
        trend_data = filtered_data.groupby(
            ['Year', 'Month', 'Order Type']
        ).size().reset_index(name='Count')
        
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        trend_data['Month'] = pd.Categorical(trend_data['Month'], 
                                           categories=month_order, 
                                           ordered=False)
        trend_data = trend_data.sort_values(['Year', 'Month'])
        
        fig = px.line(
            trend_data,
            x='Month',
            y='Count',
            color='Order Type',
            facet_col='Year',
            markers=True,
            title="Monthly Order Type Trends"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Order Type vs Cost Analysis
    st.subheader("💸 Order Type Cost Analysis")
    cost_data = filtered_data.groupby('Order Type').agg({
        'Total planned costs': 'mean',
        'Total sum (actual)': 'mean',
        'Cost Deviation': 'mean'
    }).reset_index()
    
    fig = px.bar(
        cost_data,
        x='Order Type',
        y=['Total planned costs', 'Total sum (actual)'],
        barmode='group',
        title="Planned vs Actual Costs by Order Type",
        labels={'value': 'Cost (EGP)', 'variable': 'Cost Type'}
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_cost_analysis(filtered_data):
    """Visualize cost-related metrics with proper sorting"""
    st.subheader("💵 Cost Analysis (Ascending Order)")
    
    cols = st.columns(2)
    with cols[0]:
        # Top cost savings
        dev_data = filtered_data.nsmallest(10, 'Cost Deviation')
        fig = px.bar(
            dev_data,
            x='Order Type',
            y='Cost Deviation',
            color='Plant',
            title="Top 10 Cost Savings by Order Type",
            labels={'Cost Deviation': 'Cost Deviation (EGP)'},
            text_auto='.2f'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Order Type",
            yaxis_title="Cost Deviation"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with cols[1]:
        # Cost variance distribution
        fig = px.box(
            filtered_data,
            x='Order Status',
            y='Cost Variance %',
            color='Plant',
            title="Cost Variance Distribution",
            category_orders={"Order Status": filtered_data['Order Status'].value_counts().index.sort_values()}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_raw_data(filtered_data):
    """Display interactive data table"""
    st.subheader("📄 Raw Data Explorer")
    st.dataframe(
        filtered_data.sort_values('Basic start date', ascending=False),
        column_config={
            "Cost Variance %": st.column_config.NumberColumn(
                format="%.2f%%"
            )
        },
        use_container_width=True,
        height=400
    )

def main():
    """Main application flow"""
    st.title("🏭 Maintenance Operations Analytics Dashboard")
    
    # Data loading and processing
    uploaded_file = st.file_uploader(
        "📤 Upload maintenance data (Excel)", 
        type=["xlsx"]
    )
    
    with st.spinner("🔄 Loading and processing data..."):
        data = load_data(uploaded_file)
        processed_data = process_data(data)
    
    if processed_data.empty:
        st.warning("⚠️ No data loaded! Please upload a valid file.")
        return
    
    # Create filters and filter data
    plants, years, months, statuses, work_center, Order_type = create_filters(processed_data)
    
    filtered_data = processed_data[
        (processed_data['Plant'].isin(plants)) &
        (processed_data['Year'].between(years[0], years[1])) &
        (processed_data['Month'].isin(months)) &
        (processed_data['Order Status'].isin(statuses)) &
        (processed_data['Main Work Center'].isin(work_center))&
        (processed_data['Order Type'].isin(Order_type))
    ]
    
    # Dashboard layout
    display_filter_summary(filtered_data)
    st.divider()
    
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

if __name__ == "__main__":
    main()
