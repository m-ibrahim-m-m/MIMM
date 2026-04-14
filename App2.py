import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io

# =============================================================================
# PAGE CONFIGURATION & CUSTOM STYLING
# =============================================================================

st.set_page_config(
    page_title="Maintenance Analytics Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom CSS - Dark Industrial Theme
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main > div {
        padding: 2rem 3rem;
    }
    
    /* Header styling */
    h1 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    h3 {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 1.1rem !important;
    }
    
    /* Metric cards container */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
        border-color: #3b82f6;
    }
    
    /* Metric label styling */
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Metric value styling */
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    
    [data-testid="stSidebar"] h1 {
        color: #f8fafc !important;
        font-size: 1.3rem !important;
        padding: 1rem 0;
        text-align: center;
        border-bottom: 1px solid #334155;
        margin-bottom: 1rem;
    }
    
    /* Custom divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #3b82f6, transparent);
        margin: 2rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 8px -1px rgba(59, 130, 246, 0.4);
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: #1e293b;
        border: 2px dashed #475569;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
        background: #1e293b;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        border: 1px solid #334155;
    }
    
    /* Success/Info/Warning message styling */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    /* Custom card for filter summary */
    .filter-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #3b82f6 !important;
    }
    
    /* Selectbox/Multiselect styling */
    .stMultiSelect label, .stSlider label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    
    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 12px;
        background: #1e293b !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #3b82f6;
    }
    
    /* Status badges */
    .status-completed { color: #22c55e; font-weight: 600; }
    .status-progress { color: #3b82f6; font-weight: 600; }
    .status-canceled { color: #ef4444; font-weight: 600; }
    .status-open { color: #f59e0b; font-weight: 600; }
    .status-not-executed { color: #6b7280; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING & PROCESSING (Preserved Features)
# =============================================================================

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
            'JIPR',  # Execution
            'NCMP'   # Not Completed
        ]
        for status in priority_order:
            if status in statuses:
                return {
                    'CNCL': 'Canceled',
                    'CNF': 'Completed',
                    'JIPR': 'In Progress',
                    'NCMP': 'Not Executed & Deleted'
                }[status]
        return 'Open'
    
    data['Order Status'] = data.apply(determine_status, axis=1)
    
    # Calculate cost metrics
    data['Cost Deviation'] = data['Total sum (actual)'] - data['Total planned costs']
    data['Cost Variance %'] = (data['Cost Deviation'] / data['Total planned costs']).replace([np.inf, -np.inf], np.nan) * 100
    
    # Create Plan Type
    data['Plan Type'] = np.where(
        data['Maintenance Plan'].notna(),
        'Planned',
        'Unplanned'
    )
    
    return data

# =============================================================================
# FILTER COMPONENTS (Enhanced UI)
# =============================================================================

def create_filters(data):
    """Generate interactive filters in sidebar with improved organization"""
    
    # Sidebar header with icon
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #3b82f6; margin-bottom: 1.5rem;">
            <h1 style="margin: 0; font-size: 1.4rem;">🔧 Filter Panel</h1>
            <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;">Configure your analytics view</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Plant selection with search
    st.sidebar.markdown("### 🏭 Plant Selection")
    plants = st.sidebar.multiselect(
        "Select Plants",
        options=sorted(data['Plant'].unique()),
        default=sorted(data['Plant'].unique())[:2],
        help="Filter by plant location",
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Time period filters
    st.sidebar.markdown("### 📅 Time Period")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        year_start = st.selectbox(
            "From Year",
            options=sorted(data['Year'].unique()),
            index=0
        )
    with col2:
        year_end = st.selectbox(
            "To Year",
            options=sorted(data['Year'].unique()),
            index=len(data['Year'].unique()) - 1
        )
    
    months = st.sidebar.multiselect(
        "Months",
        options=data['Month'].unique(),
        default=data['Month'].unique(),
        help="Select specific months"
    )
    
    st.sidebar.markdown("---")
    
    # Order characteristics
    st.sidebar.markdown("### 📋 Order Characteristics")
    
    plan_type = st.sidebar.multiselect(
        "Plan Type",
        options=['Planned', 'Unplanned'],
        default=['Planned', 'Unplanned'],
        help="Planned vs Unplanned maintenance"
    )
    
    statuses = st.sidebar.multiselect(
        "Order Status",
        options=sorted(data['Order Status'].unique()),
        default=sorted(data['Order Status'].unique()),
        help="Filter by current status"
    )
    
    order_type = st.sidebar.multiselect(
        "Work Order Type",
        options=sorted(data['Order Type'].unique()),
        default=sorted(data['Order Type'].unique())
    )
    
    st.sidebar.markdown("---")
    
    # Organizational filters
    st.sidebar.markdown("### 🏢 Organization")
    
    work_center = st.sidebar.multiselect(
        "Department",
        options=sorted(data['Main Work Center'].unique()),
        default=sorted(data['Main Work Center'].unique())
    )
    
    group = st.sidebar.multiselect(
        "Task List Code",
        options=sorted(data['Group'].unique()),
        default=sorted(data['Group'].unique())
    )
    
    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        st.rerun()
    
    return plants, (year_start, year_end), months, statuses, work_center, order_type, group, plan_type

# =============================================================================
# DISPLAY COMPONENTS (Enhanced Visualizations)
# =============================================================================

def display_filter_summary(filtered_data, total_records):
    """Show the selected filters summary in a professional card layout"""
    
    # Calculate filter statistics
    filter_pct = (len(filtered_data) / total_records * 100) if total_records > 0 else 0
    
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                    border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem;
                    border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: #f8fafc; font-size: 1.2rem;">📊 Active Data View</h3>
                <span style="background: #3b82f6; color: white; padding: 0.25rem 0.75rem; 
                           border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                    {len(filtered_data):,} of {total_records:,} records ({filter_pct:.1f}%)
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div class="filter-card">
                    <strong style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Plants</strong><br>
                    <span style="color: #f8fafc; font-size: 0.95rem;">{", ".join(filtered_data['Plant'].unique())}</span>
                </div>
                <div class="filter-card">
                    <strong style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Period</strong><br>
                    <span style="color: #f8fafc; font-size: 0.95rem;">
                        {filtered_data['Year'].min():.0f} - {filtered_data['Year'].max():.0f}
                    </span>
                </div>
                <div class="filter-card">
                    <strong style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Months</strong><br>
                    <span style="color: #f8fafc; font-size: 0.95rem;">{len(filtered_data['Month'].unique())} selected</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_kpis(filtered_data):
    """Show key performance indicators with professional styling"""
    
    st.header("📈 Key Performance Indicators")
    
    # Calculate metrics
    planned_orders = filtered_data[filtered_data['Maintenance Plan'].notna()]
    total_planned = len(planned_orders)
    completed_planned = len(planned_orders[planned_orders['Order Status'] == 'Completed'])
    planned_completion_pct = (completed_planned / total_planned * 100) if total_planned > 0 else 0
    
    total_orders = len(filtered_data)
    completed_orders = len(filtered_data[filtered_data['Order Status'] == 'Completed'])
    overall_completion_pct = (completed_orders / total_orders * 100) if total_orders > 0 else 0
    
    actual_cost = filtered_data['Total sum (actual)'].sum()
    planned_cost = filtered_data['Total planned costs'].sum()
    cost_efficiency = (planned_cost / actual_cost * 100) if actual_cost > 0 else 0
    
    # Create metric cards in a grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Orders",
            value=f"{total_orders:,}",
            delta=f"{total_orders - completed_orders} pending" if total_orders > 0 else None
        )
    
    with col2:
        st.metric(
            label="Planned Completion",
            value=f"{planned_completion_pct:.1f}%",
            delta=f"{completed_planned}/{total_planned}" if total_planned > 0 else "N/A"
        )
    
    with col3:
        st.metric(
            label="Overall Completion",
            value=f"{overall_completion_pct:.1f}%",
            delta=f"{completed_orders}/{total_orders}" if total_orders > 0 else "N/A"
        )
    
    with col4:
        st.metric(
            label="Actual Cost",
            value=f"{actual_cost:,.0f} EGP",
            delta=f"{cost_efficiency:.1f}% efficiency" if actual_cost > 0 else None
        )
    
    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    avg_deviation = filtered_data['Cost Deviation'].mean()
    unplanned_count = len(filtered_data[filtered_data['Plan Type'] == 'Unplanned'])
    in_progress = len(filtered_data[filtered_data['Order Status'] == 'In Progress'])
    canceled = len(filtered_data[filtered_data['Order Status'] == 'Canceled'])
    
    with col1:
        st.metric(
            label="Avg Cost Deviation",
            value=f"{avg_deviation:,.0f} EGP",
            delta="under budget" if avg_deviation < 0 else "over budget"
        )
    
    with col2:
        st.metric(
            label="Unplanned Orders",
            value=f"{unplanned_count:,}",
            delta=f"{unplanned_count/total_orders*100:.1f}%" if total_orders > 0 else "N/A"
        )
    
    with col3:
        st.metric(
            label="In Progress",
            value=f"{in_progress:,}",
            delta=f"{in_progress/total_orders*100:.1f}%" if total_orders > 0 else "N/A"
        )
    
    with col4:
        st.metric(
            label="Canceled",
            value=f"{canceled:,}",
            delta=f"{canceled/total_orders*100:.1f}%" if total_orders > 0 else "N/A"
        )
    
    # Completion breakdown charts
    st.subheader("Completion Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if total_planned > 0:
            labels = ['Completed', 'Pending']
            values = [completed_planned, total_planned - completed_planned]
            colors = ['#22c55e', '#3b82f6']
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker_colors=colors,
                textinfo='percent+label',
                textposition='outside',
                pull=[0.05, 0]
            )])
            
            fig.update_layout(
                title=dict(text="Planned Orders Completion", font=dict(size=16, color='#e2e8f0')),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                margin=dict(t=50, b=30, l=30, r=30),
                annotations=[dict(text=f'{completed_planned}<br>Done', x=0.5, y=0.5, 
                                font_size=20, showarrow=False, font_color='#f8fafc')]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No planned orders in current selection")
    
    with col2:
        if total_orders > 0:
            # Status distribution donut
            status_counts = filtered_data['Order Status'].value_counts()
            colors_map = {
                'Completed': '#22c55e',
                'In Progress': '#3b82f6',
                'Open': '#f59e0b',
                'Canceled': '#ef4444',
                'Not Executed & Deleted': '#6b7280'
            }
            colors = [colors_map.get(status, '#94a3b8') for status in status_counts.index]
            
            fig = go.Figure(data=[go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=0.55,
                marker_colors=colors,
                textinfo='percent',
                textposition='inside'
            )])
            
            fig.update_layout(
                title=dict(text="Overall Status Distribution", font=dict(size=16, color='#e2e8f0')),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5,
                           font=dict(color='#e2e8f0')),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                margin=dict(t=50, b=60, l=30, r=30),
                annotations=[dict(text=f'{total_orders}<br>Total', x=0.5, y=0.5, 
                                font_size=20, showarrow=False, font_color='#f8fafc')]
            )
            st.plotly_chart(fig, use_container_width=True)

def plot_order_status_distribution(filtered_data):
    """Visualize order status distribution with professional styling"""
    
    st.subheader("📊 Order Status & Plan Type Analysis")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📋 Plan Type Breakdown", "🏭 Plant Breakdown"])
    
    with tab1:
        # Grouped bar chart: Status vs Plan Type
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
            color_discrete_map={
                'Planned': '#3b82f6',
                'Unplanned': '#f97316'
            },
            template='plotly_dark'
        )
        
        fig.update_traces(
            texttemplate='%{text:,}',
            textposition='outside',
            marker_line_width=0
        )
        
        fig.update_layout(
            xaxis_title="Order Status",
            yaxis_title="Number of Orders",
            legend_title="Plan Type",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            margin=dict(t=30, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Plant breakdown
        plant_status = filtered_data.groupby(
            ['Plant', 'Order Status']
        ).size().reset_index(name='Count')
        
        fig = px.bar(
            plant_status,
            x='Plant',
            y='Count',
            color='Order Status',
            barmode='group',
            text='Count',
            template='plotly_dark',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_traces(
            texttemplate='%{text:,}',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_title="Plant",
            yaxis_title="Number of Orders",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            margin=dict(t=30, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def plot_department_orders(filtered_data):
    """Visualize department-wise order counts with enhanced styling"""
    
    st.subheader("🏭 Department Performance")
    
    department_counts = filtered_data['Main Work Center'].value_counts().reset_index()
    department_counts.columns = ['Department', 'Count']
    
    # Add completion rate per department
    dept_stats = filtered_data.groupby('Main Work Center').agg({
        'Order Status': lambda x: (x == 'Completed').sum() / len(x) * 100
    }).reset_index()
    dept_stats.columns = ['Department', 'Completion Rate']
    
    department_counts = department_counts.merge(dept_stats, on='Department')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            department_counts,
            x='Department',
            y='Count',
            color='Completion Rate',
            text='Count',
            template='plotly_dark',
            color_continuous_scale='RdYlGn',
            range_color=[0, 100]
        )
        
        fig.update_traces(
            texttemplate='%{text:,}',
            textposition='outside',
            marker_line_width=0
        )
        
        fig.update_layout(
            xaxis_title="Department",
            yaxis_title="Number of Orders",
            coloraxis_colorbar=dict(title="Completion %"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            margin=dict(t=30, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Department summary table
        st.markdown("#### Department Summary")
        summary_df = department_counts[['Department', 'Count', 'Completion Rate']].copy()
        summary_df['Completion Rate'] = summary_df['Completion Rate'].round(1).astype(str) + '%'
        summary_df = summary_df.sort_values('Count', ascending=False)
        
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Department": st.column_config.TextColumn("Department"),
                "Count": st.column_config.NumberColumn("Orders", format="%d"),
                "Completion Rate": st.column_config.TextColumn("Completion")
            }
        )

def plot_status_trends(filtered_data):
    """Visualize order status trends with professional time series styling"""
    
    st.subheader("📈 Temporal Trends Analysis")
    
    # Month ordering
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    # Create temporal aggregation
    trend_data = filtered_data.groupby(
        ['Year', 'Month', 'Order Status']
    ).size().reset_index(name='Count')
    
    trend_data['Month'] = pd.Categorical(trend_data['Month'], 
                                       categories=month_order, 
                                       ordered=True)
    trend_data = trend_data.sort_values(['Year', 'Month'])
    
    # Line chart with markers
    fig = px.line(
        trend_data,
        x='Month',
        y='Count',
        color='Order Status',
        facet_col='Year',
        markers=True,
        template='plotly_dark',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey'))
    )
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Orders",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Plan type trends
    st.subheader("📊 Planned vs Unplanned Trends")
    
    trend_data2 = filtered_data.groupby(
        ['Year', 'Month', 'Plan Type']
    ).size().reset_index(name='Count')
    
    trend_data2['Month'] = pd.Categorical(trend_data2['Month'], 
                                         categories=month_order, 
                                         ordered=True)
    trend_data2 = trend_data2.sort_values(['Year', 'Month'])
    
    fig2 = px.area(
        trend_data2,
        x='Month',
        y='Count',
        color='Plan Type',
        facet_col='Year',
        template='plotly_dark',
        color_discrete_map={'Planned': '#3b82f6', 'Unplanned': '#f97316'}
    )
    
    fig2.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Orders",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80, b=50)
    )
    
    st.plotly_chart(fig2, use_container_width=True)

def plot_order_type_analysis(filtered_data):
    """Visualize order type analysis with professional layouts"""
    
    st.subheader("📦 Order Type Deep Dive")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Order Type Distribution with improved donut chart
        type_dist = filtered_data['Order Type'].value_counts().reset_index()
        type_dist.columns = ['Order Type', 'Count']
        
        fig = go.Figure(data=[go.Pie(
            labels=type_dist['Order Type'],
            values=type_dist['Count'],
            hole=0.6,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(line=dict(color='#1e293b', width=2))
        )])
        
        fig.update_layout(
            title=dict(text="Distribution by Type", font=dict(size=16, color='#e2e8f0')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            showlegend=False,
            margin=dict(t=50, b=30, l=30, r=30),
            annotations=[dict(text='Types', x=0.5, y=0.5, font_size=16, 
                            showarrow=False, font_color='#94a3b8')]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Order Type Trends Over Time
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        
        trend_data = filtered_data.groupby(
            ['Year', 'Month', 'Order Type']
        ).size().reset_index(name='Count')
        
        trend_data['Month'] = pd.Categorical(trend_data['Month'], 
                                           categories=month_order, 
                                           ordered=True)
        trend_data = trend_data.sort_values(['Year', 'Month'])
        
        fig = px.line(
            trend_data,
            x='Month',
            y='Count',
            color='Order Type',
            facet_col='Year',
            markers=True,
            template='plotly_dark'
        )
        
        fig.update_layout(
            title=dict(text="Monthly Trends", font=dict(size=16, color='#e2e8f0')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(t=50, b=80)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Cost analysis by order type
    st.subheader("💰 Cost Analysis by Order Type")
    
    cost_data = filtered_data.groupby('Order Type').agg({
        'Total planned costs': 'mean',
        'Total sum (actual)': 'mean',
        'Cost Deviation': 'mean',
        'Order Type': 'count'
    }).rename(columns={'Order Type': 'Count'}).reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Planned vs Actual Costs', 'Cost Deviation'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Planned vs Actual
    fig.add_trace(
        go.Bar(name='Planned', x=cost_data['Order Type'], y=cost_data['Total planned costs'],
               marker_color='#3b82f6', opacity=0.8),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Actual', x=cost_data['Order Type'], y=cost_data['Total sum (actual)'],
               marker_color='#f97316', opacity=0.8),
        row=1, col=1
    )
    
    # Deviation
    colors = ['#22c55e' if x < 0 else '#ef4444' for x in cost_data['Cost Deviation']]
    fig.add_trace(
        go.Bar(name='Deviation', x=cost_data['Order Type'], y=cost_data['Cost Deviation'],
               marker_color=colors, opacity=0.8, showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(
        template='plotly_dark',
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80, b=50)
    )
    
    fig.update_yaxes(title_text="Cost (EGP)", row=1, col=1)
    fig.update_yaxes(title_text="Deviation (EGP)", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_cost_analysis(filtered_data):
    """Visualize cost-related metrics with professional styling"""
    
    st.subheader("💵 Detailed Cost Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Top 10 Cost Savings")
        # Top cost savings (negative deviation = under budget)
        savings_data = filtered_data.nsmallest(10, 'Cost Deviation').copy()
        
        fig = px.bar(
            savings_data,
            x='Cost Deviation',
            y='Order Type',
            color='Plant',
            orientation='h',
            template='plotly_dark',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(
            texttemplate='%{x:,.0f}',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_title="Cost Savings (EGP)",
            yaxis_title="Order Type",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            margin=dict(t=30, b=50, l=100),
            yaxis=dict(autorange="reversed")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Cost Variance Distribution")
        # Box plot for cost variance
        fig = px.box(
            filtered_data,
            x='Order Status',
            y='Cost Variance %',
            color='Plant',
            template='plotly_dark',
            points='outliers'
        )
        
        fig.update_layout(
            xaxis_title="Order Status",
            yaxis_title="Cost Variance (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            margin=dict(t=30, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Cost heatmap by month and status
    st.subheader("🔥 Cost Intensity Heatmap")
    
    heatmap_data = filtered_data.groupby(['Month', 'Order Status'])['Total sum (actual)'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Order Status', columns='Month', values='Total sum (actual)').fillna(0)
    
    # Reorder columns by month
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    existing_months = [m for m in month_order if m in heatmap_pivot.columns]
    heatmap_pivot = heatmap_pivot[existing_months]
    
    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x="Month", y="Status", color="Cost (EGP)"),
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        color_continuous_scale='Viridis',
        template='plotly_dark',
        aspect='auto'
    )
    
    fig.update_traces(text=heatmap_pivot.values, texttemplate="%{text:,.0f}")
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        margin=dict(t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_raw_data(filtered_data):
    """Display interactive data table with professional styling"""
    
    st.subheader("📄 Data Explorer")
    
    # Summary stats above table
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Displayed Records", len(filtered_data))
    with col2:
        st.metric("Total Cost", f"{filtered_data['Total sum (actual)'].sum():,.0f} EGP")
    with col3:
        date_range = f"{filtered_data['Basic start date'].min().strftime('%Y-%m-%d') if not filtered_data['Basic start date'].isna().all() else 'N/A'} to {filtered_data['Basic start date'].max().strftime('%Y-%m-%d') if not filtered_data['Basic start date'].isna().all() else 'N/A'}"
        st.metric("Date Range", date_range)
    with col4:
        st.metric("Planned %", f"{(filtered_data['Plan Type'] == 'Planned').sum() / len(filtered_data) * 100:.1f}%")
    
    # Data table
    st.dataframe(
        filtered_data.sort_values('Basic start date', ascending=False),
        use_container_width=True,
        height=400,
        column_config={
            "Cost Variance %": st.column_config.NumberColumn(
                "Cost Var %",
                format="%.2f%%",
                help="Percentage deviation from planned cost"
            ),
            "Total sum (actual)": st.column_config.NumberColumn(
                "Actual Cost",
                format="%.2f EGP"
            ),
            "Total planned costs": st.column_config.NumberColumn(
                "Planned Cost",
                format="%.2f EGP"
            ),
            "Cost Deviation": st.column_config.NumberColumn(
                "Deviation",
                format="%.2f EGP"
            ),
            "Basic start date": st.column_config.DateColumn(
                "Start Date",
                format="YYYY-MM-DD"
            )
        }
    )
    
    # Export functionality
    col1, col2 = st.columns([3, 1])
    with col2:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            filtered_data.to_excel(writer, index=False, sheet_name='Filtered Data')
            # Add summary sheet
            summary = pd.DataFrame({
                'Metric': ['Total Records', 'Total Cost', 'Avg Cost', 'Completion Rate'],
                'Value': [
                    len(filtered_data),
                    filtered_data['Total sum (actual)'].sum(),
                    filtered_data['Total sum (actual)'].mean(),
                    (filtered_data['Order Status'] == 'Completed').sum() / len(filtered_data) * 100
                ]
            })
            summary.to_excel(writer, index=False, sheet_name='Summary')
        buffer.seek(0)
        
        st.download_button(
            label="📥 Export to Excel",
            data=buffer,
            file_name="maintenance_analytics.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application flow with professional layout"""
    
    # App header with branding
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <h1 style="margin: 0; background: linear-gradient(135deg, #3b82f6, #8b5cf6); 
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🔧 Maintenance Operations Analytics
            </h1>
            <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem;">
                Comprehensive dashboard for maintenance order insights and cost analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Data loading section
    st.markdown("### 📤 Data Source")
    
    uploaded_file = st.file_uploader(
        "Upload maintenance data (Excel format)", 
        type=["xlsx"],
        help="Upload your Maintenance Orders Excel file. If no file is uploaded, the app will attempt to load from the default path.",
        label_visibility="collapsed"
    )
    
    with st.spinner("🔄 Loading and processing data..."):
        data = load_data(uploaded_file)
        processed_data = process_data(data)
    
    if processed_data.empty:
        st.error("⚠️ No data loaded! Please upload a valid Excel file containing maintenance orders.")
        st.stop()
    
    # Store total records for percentage calculations
    total_records = len(processed_data)
    
    # Create filters
    plants, years, months, statuses, work_center, order_type, group, plan_type = create_filters(processed_data)
    
    # Apply filters
    filtered_data = processed_data[
        (processed_data['Plant'].isin(plants)) &
        (processed_data['Year'].between(years[0], years[1])) &
        (processed_data['Month'].isin(months)) &
        (processed_data['Order Status'].isin(statuses)) &
        (processed_data['Main Work Center'].isin(work_center)) &
        (processed_data['Order Type'].isin(order_type)) &
        (processed_data['Group'].isin(group)) &
        (processed_data['Plan Type'].isin(plan_type))
    ].copy()
    
    if filtered_data.empty:
        st.warning("⚠️ No records match your current filter selection. Please adjust filters.")
        st.stop()
    
    # Dashboard sections
    display_filter_summary(filtered_data, total_records)
    st.divider()
    
    display_kpis(filtered_data)
    st.divider()
    
    plot_order_status_distribution(filtered_data)
    st.divider()
    
    plot_department_orders(filtered_data)
    st.divider()
    
    plot_status_trends(filtered_data)
    st.divider()
    
    plot_order_type_analysis(filtered_data)
    st.divider()
    
    plot_cost_analysis(filtered_data)
    st.divider()
    
    show_raw_data(filtered_data)
    
    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0; color: #64748b; font-size: 0.9rem; border-top: 1px solid #334155; margin-top: 2rem;">
            <p>Maintenance Analytics Dashboard • Built with Streamlit • Data last updated: {}</p>
        </div>
    """.format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
