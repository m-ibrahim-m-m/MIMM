import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import io
from datetime import datetime
import plotly.graph_objects as go

# Configure page settings with enhanced theme
st.set_page_config(
    page_title="Maintenance Analytics Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium UI/UX
def load_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Color Variables */
        :root {
            --primary: #0f172a;
            --secondary: #1e293b;
            --accent: #3b82f6;
            --accent-alt: #06b6d4;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg-light: #f8fafc;
            --bg-lighter: #f1f5f9;
            --border: #e2e8f0;
            --text-primary: #0f172a;
            --text-secondary: #64748b;
        }
        
        /* Main container */
        .main {
            padding: 0;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            min-height: 100vh;
        }
        
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        }
        
        /* Header styling */
        .header-container {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 2.5rem 2rem;
            border-bottom: 2px solid rgba(59, 130, 246, 0.2);
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        
        .header-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: white;
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(135deg, #3b82f6, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header-subtitle {
            color: #cbd5e1;
            font-size: 0.95rem;
            margin: 0.5rem 0 0 0;
            font-weight: 500;
        }
        
        .header-timestamp {
            color: #94a3b8;
            font-size: 0.85rem;
            margin-top: 1rem;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: white;
            border-right: 2px solid var(--border);
        }
        
        .sidebar-section-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--primary);
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Metric Cards */
        .metric-card {
            background: white;
            border: 2px solid var(--border);
            padding: 1.5rem;
            border-radius: 0.875rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, var(--accent), var(--accent-alt));
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-8px);
            border-color: var(--accent);
            box-shadow: 0 20px 40px rgba(59, 130, 246, 0.15);
        }
        
        .metric-card:hover::before {
            transform: scaleX(1);
        }
        
        .metric-card h4 {
            color: var(--text-secondary);
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0 0 0.75rem 0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-card p {
            color: var(--primary);
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background-color: transparent;
            padding: 0;
            border-bottom: 2px solid var(--border);
            margin-bottom: 2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 0;
            padding: 1rem 1.5rem;
            font-weight: 600;
            color: var(--text-secondary);
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
            font-size: 0.95rem;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--accent);
            background-color: rgba(59, 130, 246, 0.05);
        }
        
        .stTabs [aria-selected="true"] [data-baseweb="tab"] {
            color: var(--accent);
            border-bottom-color: var(--accent);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, var(--accent), var(--accent-alt));
            color: white;
            border: none;
            padding: 0.625rem 1.5rem;
            border-radius: 0.75rem;
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.25);
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.2);
            transition: left 0.4s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(59, 130, 246, 0.35);
        }
        
        .stButton > button:hover::before {
            left: 100%;
        }
        
        /* Input and select styling */
        .stMultiSelect > div > div {
            border: 2px solid var(--border);
            border-radius: 0.75rem;
            transition: all 0.3s ease;
        }
        
        .stMultiSelect > div > div:hover {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        
        .stSlider > div > div {
            padding: 1rem 0;
        }
        
        /* Text input styling */
        .stTextInput > div > div > input {
            border: 2px solid var(--border);
            border-radius: 0.75rem;
            padding: 0.75rem 1rem;
            transition: all 0.3s ease;
            font-size: 0.9rem;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
            outline: none;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: white;
            border: 2px solid var(--border);
            border-radius: 0.75rem;
            padding: 1rem;
            transition: all 0.3s ease;
        }
        
        .streamlit-expanderHeader:hover {
            border-color: var(--accent);
            background-color: rgba(59, 130, 246, 0.03);
        }
        
        /* Divider styling */
        hr {
            border: none;
            height: 2px;
            background: var(--border);
            margin: 2rem 0;
        }
        
        /* Info/Warning boxes */
        .stAlert {
            border-radius: 0.875rem;
            border-left: 4px solid;
            padding: 1rem 1.25rem;
        }
        
        .stAlert[data-baseweb="notification"] {
            background-color: rgba(59, 130, 246, 0.1);
            border-left-color: var(--accent);
        }
        
        /* Data table styling */
        .stDataFrame {
            border: 2px solid var(--border);
            border-radius: 0.875rem;
            overflow: hidden;
        }
        
        /* Progress bar */
        .stProgress > div {
            background: linear-gradient(90deg, var(--success), var(--accent));
            border-radius: 1rem;
            height: 8px;
        }
        
        /* Markdown sections */
        h1 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--primary);
            font-size: 2rem;
            font-weight: 800;
            margin: 1.5rem 0 0.5rem 0;
        }
        
        h2 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--primary);
            font-size: 1.5rem;
            font-weight: 700;
            margin: 1.25rem 0 0.75rem 0;
        }
        
        h3 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--primary);
            font-size: 1.25rem;
            font-weight: 700;
            margin: 1rem 0 0.5rem 0;
        }
        
        h4 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--primary);
            font-size: 1rem;
            font-weight: 600;
            margin: 0.75rem 0 0.25rem 0;
        }
        
        p {
            color: var(--text-secondary);
            line-height: 1.6;
            font-size: 0.95rem;
        }
        
        /* Footer styling */
        .footer-text {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.85rem;
            padding: 2rem 1rem;
            border-top: 2px solid var(--border);
            background: white;
            margin-top: 2rem;
        }
        
        /* Filter summary cards grid */
        .filter-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main {
                padding: 0;
            }
            
            .header-container {
                padding: 1.5rem 1rem;
            }
            
            .header-title {
                font-size: 1.75rem;
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 0.75rem 1rem;
                font-size: 0.85rem;
            }
            
            h1 { font-size: 1.5rem; }
            h2 { font-size: 1.25rem; }
            h3 { font-size: 1rem; }
        }
    </style>
    """, unsafe_allow_html=True)

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
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
    
    if 'Basic start date' in data.columns:
        data['Year'] = data['Basic start date'].dt.year
        data['Month'] = data['Basic start date'].dt.month_name()
        data['Quarter'] = data['Basic start date'].dt.quarter
        data['Week'] = data['Basic start date'].dt.isocalendar().week
    
    # Calculate status categories
    def determine_status(row):
        statuses = f"{row.get('System status', '')} {row.get('User Status', '')}".split()
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
    if 'Total sum (actual)' in data.columns and 'Total planned costs' in data.columns:
        data['Cost Deviation'] = data['Total sum (actual)'] - data['Total planned costs']
        data['Cost Variance %'] = (data['Cost Deviation'] / data['Total planned costs']).replace([np.inf, -np.inf], np.nan) * 100
    
    # Calculate duration
    if 'Basic finish date' in data.columns and 'Basic start date' in data.columns:
        data['Duration Days'] = (data['Basic finish date'] - data['Basic start date']).dt.days
    
    return data

def create_enhanced_filters(data):
    """Generate enhanced interactive filters in sidebar"""
    with st.sidebar:
        st.markdown("""
        <div style='margin-bottom: 1.5rem;'>
            <h2 style='margin: 0; color: #0f172a; font-size: 1.5rem;'>🔧 Navigation</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats in sidebar
        st.markdown("### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Orders", f"{len(data):,}", delta=None)
        with col2:
            st.metric("Plants", f"{data['Plant'].nunique() if 'Plant' in data.columns else 'N/A'}", delta=None)
        
        st.divider()
        
        st.markdown("### 🔍 Filter Options")
        
        # Date range filter
        if 'Year' in data.columns:
            st.markdown("#### 📅 Date Range")
            min_year = int(data['Year'].min())
            max_year = int(data['Year'].max())
            years = st.slider(
                "Select Year Range:",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                format="%d"
            )
        else:
            years = (2020, 2024)
        
        # Plant selector
        st.markdown("#### 🏭 Plants")
        plants = st.multiselect(
            "Select Plants:",
            options=data['Plant'].unique() if 'Plant' in data.columns else [],
            default=data['Plant'].unique() if 'Plant' in data.columns else [],
            help="Filter by plant location"
        )
        
        # Month selector
        if 'Month' in data.columns:
            st.markdown("#### 📆 Months")
            months = st.multiselect(
                "Select Months:",
                options=data['Month'].unique(),
                default=data['Month'].unique()
            )
        else:
            months = []
        
        # Plan Type Filter
        data['Plan Type'] = np.where(
            data['Maintenance Plan'].notna() if 'Maintenance Plan' in data.columns else False,
            'Planned',
            'Unplanned'
        )
        st.markdown("#### 📋 Plan Type")
        plan_type = st.multiselect(
            "Plan Type:",
            options=data['Plan Type'].unique(),
            default=data['Plan Type'].unique()
        )
        
        # Status selector
        st.markdown("#### ✅ Order Status")
        statuses = st.multiselect(
            "Order Statuses:",
            options=data['Order Status'].unique(),
            default=data['Order Status'].unique()
        )
        
        # Advanced filters (collapsible)
        with st.expander("🔧 Advanced Filters"):
            work_center = st.multiselect(
                "Department:",
                options=data['Main Work Center'].unique() if 'Main Work Center' in data.columns else [],
                default=data['Main Work Center'].unique() if 'Main Work Center' in data.columns else []
            )
            
            Order_type = st.multiselect(
                "Work Order Type:",
                options=data['Order Type'].unique() if 'Order Type' in data.columns else [],
                default=data['Order Type'].unique() if 'Order Type' in data.columns else []
            )
            
            Group = st.multiselect(
                "Task List Code:",
                options=data['Group'].unique() if 'Group' in data.columns else [],
                default=data['Group'].unique() if 'Group' in data.columns else []
            )
        
        st.divider()
        
        # Reset filters button
        if st.button("🔄 Reset All Filters", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    return plants, years, months, statuses, work_center, Order_type, Group, plan_type

def display_enhanced_header():
    """Display enhanced dashboard header"""
    st.markdown("""
    <div class='header-container'>
        <h1 class='header-title'>🏭 Maintenance Operations Analytics</h1>
        <p class='header-subtitle'>Real-time insights and analytics for maintenance operations</p>
    </div>
    """, unsafe_allow_html=True)

def display_filter_summary(filtered_data):
    """Show the selected filters summary with enhanced UI"""
    st.markdown("### 🔎 Active Filters Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>🏭 Plants</h4>
            <p>{len(filtered_data['Plant'].unique()) if 'Plant' in filtered_data.columns else 0} selected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>📆 Time Period</h4>
            <p>{filtered_data['Year'].min() if 'Year' in filtered_data.columns else 'N/A'} - {filtered_data['Year'].max() if 'Year' in filtered_data.columns else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>✅ Status Types</h4>
            <p>{len(filtered_data['Order Status'].unique())} statuses</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_orders = len(filtered_data)
        st.markdown(f"""
        <div class='metric-card'>
            <h4>📊 Total Records</h4>
            <p>{total_orders:,} orders</p>
        </div>
        """, unsafe_allow_html=True)

def display_enhanced_kpis(filtered_data):
    """Show key performance indicators with enhanced visualization"""
    st.markdown("### 📊 Key Performance Indicators")
    
    # Calculate KPIs
    if 'Maintenance Plan' in filtered_data.columns:
        planned_orders = filtered_data[filtered_data['Maintenance Plan'].notna()]
        total_planned = len(planned_orders)
        completed_planned = len(planned_orders[planned_orders['Order Status'] == 'Completed'])
        planned_completion_pct = (completed_planned / total_planned * 100) if total_planned > 0 else 0
    else:
        planned_completion_pct = 0
    
    total_orders = len(filtered_data)
    completed_orders = len(filtered_data[filtered_data['Order Status'] == 'Completed'])
    overall_completion_pct = (completed_orders / total_orders * 100) if total_orders > 0 else 0
    
    # Calculate additional KPIs
    avg_duration = filtered_data['Duration Days'].mean() if 'Duration Days' in filtered_data.columns else 0
    total_cost = filtered_data['Total sum (actual)'].sum() if 'Total sum (actual)' in filtered_data.columns else 0
    avg_cost_deviation = filtered_data['Cost Deviation'].mean() if 'Cost Deviation' in filtered_data.columns else 0
    
    # Display KPIs in a grid
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            label="📋 Total Orders",
            value=f"{total_orders:,}",
            delta=f"{((completed_orders/total_orders)*100 if total_orders>0 else 0):.1f}% completion"
        )
    
    with col2:
        st.metric(
            label="✅ Completion Rate",
            value=f"{overall_completion_pct:.1f}%",
            delta=f"{planned_completion_pct:.1f}% planned"
        )
    
    with col3:
        st.metric(
            label="⏱️ Avg Duration",
            value=f"{avg_duration:.1f} days" if avg_duration > 0 else "N/A"
        )
    
    with col4:
        st.metric(
            label="💰 Total Cost",
            value=f"EGP {total_cost:,.0f}" if total_cost > 0 else "N/A"
        )
    
    with col5:
        st.metric(
            label="📊 Avg Cost Deviation",
            value=f"EGP {avg_cost_deviation:,.0f}" if avg_cost_deviation != 0 else "N/A"
        )
    
    with col6:
        st.metric(
            label="🏭 Active Plants",
            value=f"{filtered_data['Plant'].nunique() if 'Plant' in filtered_data.columns else 0}"
        )
    
    # Progress bars for completion rates
    st.markdown("#### Completion Progress")
    col1, col2 = st.columns(2)
    
    with col1:
        st.progress(overall_completion_pct / 100, text=f"Overall Completion: {overall_completion_pct:.1f}%")
    
    with col2:
        if 'Maintenance Plan' in filtered_data.columns:
            st.progress(planned_completion_pct / 100, text=f"Planned Orders Completion: {planned_completion_pct:.1f}%")

def plot_status_overview(filtered_data):
    """Simplified status overview for the Overview tab"""
    st.markdown("### 📊 Quick Status Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Simple status distribution bar chart
        status_counts = filtered_data['Order Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        fig = px.bar(
            status_counts,
            x='Status',
            y='Count',
            color='Status',
            text='Count',
            title="Order Status Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True, key="overview_status_chart")
    
    with col2:
        # Simple completion pie chart
        completion_data = pd.DataFrame({
            'Status': ['Completed', 'Not Completed'],
            'Count': [
                len(filtered_data[filtered_data['Order Status'] == 'Completed']),
                len(filtered_data[filtered_data['Order Status'] != 'Completed'])
            ]
        })
        
        fig = px.pie(
            completion_data,
            values='Count',
            names='Status',
            title="Completion Overview",
            hole=0.3,
            color_discrete_sequence=['#10b981', '#ef4444']
        )
        
        st.plotly_chart(fig, use_container_width=True, key="overview_completion_chart")

def plot_detailed_status_analysis(filtered_data):
    """Detailed status analysis for the Status tab"""
    st.markdown("### 📊 Detailed Status Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📈 Status Distribution", "🏭 By Plant", "📊 Trend Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Create grouped status counts with plan type
            if 'Maintenance Plan' in filtered_data.columns:
                filtered_data['Plan Type'] = np.where(
                    filtered_data['Maintenance Plan'].notna(),
                    'Planned',
                    'Unplanned'
                )
            else:
                filtered_data['Plan Type'] = 'Unknown'
            
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
                    'Unplanned': '#EF553B',
                    'Unknown': '#00CC96'
                }
            )
            
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(
                xaxis_title="Order Status",
                yaxis_title="Number of Orders",
                legend_title="Plan Type",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key="detailed_status_bar")
        
        with col2:
            # Pie chart for overall status distribution
            status_pie = filtered_data['Order Status'].value_counts().reset_index()
            status_pie.columns = ['Status', 'Count']
            
            fig_pie = px.pie(
                status_pie,
                values='Count',
                names='Status',
                title="Overall Status Distribution",
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True, key="detailed_status_pie")
    
    with tab2:
        # Plant-wise breakdown
        if 'Plant' in filtered_data.columns:
            plant_status = filtered_data.groupby(
                ['Plant', 'Order Status']
            ).size().reset_index(name='Count')
            
            fig_plant = px.bar(
                plant_status,
                x='Plant',
                y='Count',
                color='Order Status',
                barmode='stack',
                text='Count',
                title="Status Distribution by Plant",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig_plant.update_traces(texttemplate='%{text:,}', textposition='inside')
            fig_plant.update_layout(
                xaxis_title="Plant",
                yaxis_title="Number of Orders",
                legend_title="Order Status",
                height=400
            )
            
            st.plotly_chart(fig_plant, use_container_width=True, key="detailed_plant_status")
        else:
            st.warning("Plant column not available for this analysis")
    
    with tab3:
        # Status trends over time
        if 'Year' in filtered_data.columns and 'Month' in filtered_data.columns:
            trend_data = filtered_data.groupby(
                ['Year', 'Month', 'Order Status']
            ).size().reset_index(name='Count')
            
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
            
            trend_data['Month'] = pd.Categorical(trend_data['Month'], 
                                               categories=month_order, 
                                               ordered=True)
            
            trend_data = trend_data.sort_values(['Year', 'Month'])
            
            fig_trend = px.area(
                trend_data,
                x='Month',
                y='Count',
                color='Order Status',
                facet_col='Year',
                title="Monthly Status Trends (Area Chart)",
                category_orders={"Month": month_order},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            
            fig_trend.update_layout(height=400)
            st.plotly_chart(fig_trend, use_container_width=True, key="detailed_trend_chart")
        else:
            st.warning("Date information not available for trend analysis")

def plot_department_analysis(filtered_data):
    """Visualize department-wise analysis with enhanced metrics"""
    st.markdown("### 🏗️ Department Performance Analysis")
    
    if 'Main Work Center' not in filtered_data.columns:
        st.warning("Department information not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        department_counts = filtered_data['Main Work Center'].value_counts().reset_index()
        department_counts.columns = ['Department', 'Count']
        department_counts = department_counts.head(10)
        
        fig = px.bar(
            department_counts,
            x='Department',
            y='Count',
            color='Count',
            text='Count',
            title="Top 10 Departments by Orders",
            color_continuous_scale='Viridis'
        )
        
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(xaxis_title="Department", yaxis_title="Number of Orders", showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True, key="dept_orders")
    
    with col2:
        # Department completion rates
        dept_completion = filtered_data.groupby('Main Work Center').agg({
            'Order Status': lambda x: (x == 'Completed').sum(),
            'Order': 'count'
        }).reset_index()
        
        dept_completion['Completion Rate'] = (dept_completion['Order Status'] / dept_completion['Order'] * 100).round(1)
        dept_completion = dept_completion.nlargest(10, 'Completion Rate')
        
        fig = px.bar(
            dept_completion,
            x='Main Work Center',
            y='Completion Rate',
            color='Completion Rate',
            text='Completion Rate',
            title="Top 10 Departments by Completion Rate",
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(xaxis_title="Department", yaxis_title="Completion Rate (%)", showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True, key="dept_completion")

def plot_cost_analysis(filtered_data):
    """Visualize cost-related metrics with enhanced insights"""
    st.markdown("### 💰 Financial Analytics")
    
    if 'Total sum (actual)' not in filtered_data.columns or 'Total planned costs' not in filtered_data.columns:
        st.warning("Cost information not available")
        return
    
    tab1, tab2, tab3 = st.tabs(["📊 Cost Overview", "💰 Cost Savings", "📈 Variance Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Order Type' in filtered_data.columns:
                cost_by_type = filtered_data.groupby('Order Type').agg({
                    'Total planned costs': 'sum',
                    'Total sum (actual)': 'sum'
                }).reset_index()
                
                fig = px.bar(
                    cost_by_type,
                    x='Order Type',
                    y=['Total planned costs', 'Total sum (actual)'],
                    barmode='group',
                    title="Planned vs Actual Costs by Order Type",
                    labels={'value': 'Cost (EGP)', 'variable': 'Cost Type'},
                    color_discrete_sequence=['#2E86AB', '#A23B72']
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="cost_by_type")
        
        with col2:
            if 'Plant' in filtered_data.columns:
                cost_by_plant = filtered_data.groupby('Plant')['Total sum (actual)'].sum().reset_index()
                
                fig = px.pie(
                    cost_by_plant,
                    values='Total sum (actual)',
                    names='Plant',
                    title="Cost Distribution by Plant",
                    hole=0.3,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                st.plotly_chart(fig, use_container_width=True, key="cost_by_plant")
    
    with tab2:
        filtered_data['Cost Savings'] = filtered_data['Total planned costs'] - filtered_data['Total sum (actual)']
        savings_data = filtered_data.nlargest(10, 'Cost Savings')
        
        fig = px.bar(
            savings_data,
            x='Order Type' if 'Order Type' in filtered_data.columns else 'Order Status',
            y='Cost Savings',
            color='Plant' if 'Plant' in filtered_data.columns else None,
            title="Top 10 Cost Savings",
            labels={'Cost Savings': 'Cost Savings (EGP)'},
            text_auto='.2f'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, key="cost_savings")
    
    with tab3:
        fig = px.box(
            filtered_data,
            x='Order Status',
            y='Cost Variance %',
            color='Plant' if 'Plant' in filtered_data.columns else None,
            title="Cost Variance Distribution by Status",
            points="all"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, key="variance_analysis")

def plot_temporal_analysis(filtered_data):
    """Visualize temporal trends and patterns"""
    st.markdown("### 📈 Temporal Analysis")
    
    if 'Year' not in filtered_data.columns or 'Month' not in filtered_data.columns:
        st.warning("Date information not available for temporal analysis")
        return
    
    tab1, tab2 = st.tabs(["📅 Monthly Trends", "📊 Quarterly Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_orders = filtered_data.groupby(['Year', 'Month']).size().reset_index(name='Count')
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_orders['Month'] = pd.Categorical(monthly_orders['Month'], categories=month_order, ordered=True)
            monthly_orders = monthly_orders.sort_values(['Year', 'Month'])
            
            fig = px.line(
                monthly_orders,
                x='Month',
                y='Count',
                color='Year',
                markers=True,
                title="Monthly Order Volume",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="monthly_volume")
        
        with col2:
            if 'Total sum (actual)' in filtered_data.columns:
                monthly_cost = filtered_data.groupby(['Year', 'Month']).agg({
                    'Total sum (actual)': 'sum'
                }).reset_index()
                monthly_cost['Month'] = pd.Categorical(monthly_cost['Month'], categories=month_order, ordered=True)
                monthly_cost = monthly_cost.sort_values(['Year', 'Month'])
                
                fig = px.area(
                    monthly_cost,
                    x='Month',
                    y='Total sum (actual)',
                    color='Year',
                    title="Monthly Cost Trends",
                    labels={'Total sum (actual)': 'Total Cost (EGP)'}
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="monthly_cost")
    
    with tab2:
        quarterly_data = filtered_data.groupby(['Year', 'Quarter']).agg({
            'Order': 'count',
            'Total sum (actual)': 'sum' if 'Total sum (actual)' in filtered_data.columns else 'count'
        }).reset_index()
        
        fig = px.bar(
            quarterly_data,
            x='Quarter',
            y='Order',
            color='Year',
            barmode='group',
            title="Quarterly Order Volume",
            text='Order'
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="quarterly_volume")

def show_enhanced_raw_data(filtered_data):
    """Display interactive data table with enhanced features"""
    st.markdown("### 📄 Data Explorer")
    
    # Add search functionality
    search_term = st.text_input("🔍 Search across all columns", placeholder="Enter search term...", key="search_box")
    
    if search_term:
        mask = filtered_data.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        display_data = filtered_data[mask]
        st.info(f"Found {len(display_data)} matching records")
    else:
        display_data = filtered_data
    
    # Column selector
    available_columns = display_data.columns.tolist()
    default_columns = ['Order', 'Order Type', 'Order Status', 'Plant', 'Main Work Center', 
                      'Total planned costs', 'Total sum (actual)', 'Basic start date', 'Basic finish date']
    
    selected_columns = st.multiselect(
        "Select columns to display",
        available_columns,
        default=[col for col in default_columns if col in available_columns],
        key="column_selector"
    )
    
    if selected_columns:
        display_data = display_data[selected_columns]
    
    # Display dataframe
    st.dataframe(
        display_data,
        use_container_width=True,
        height=500
    )
    
    # Export functionality
    col1, col2 = st.columns(2)
    
    with col1:
        csv = display_data.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"maintenance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_csv"
        )
    
    with col2:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            display_data.to_excel(writer, index=False, sheet_name='Maintenance Data')
        buffer.seek(0)
        
        st.download_button(
            label="📊 Download as Excel",
            data=buffer,
            file_name=f"maintenance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="download_excel"
        )

def main():
    """Main application flow with enhanced UI/UX"""
    
    # Load custom CSS
    load_custom_css()
    
    # Display enhanced header
    display_enhanced_header()
    
    # Data loading section
    with st.expander("📤 Data Source", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload maintenance data (Excel)", 
            type=["xlsx", "xls"],
            help="Upload an Excel file containing maintenance order data"
        )
    
    # Data loading spinner
    with st.spinner("🔄 Loading and processing data..."):
        data = load_data(uploaded_file)
        processed_data = process_data(data)
    
    if processed_data.empty:
        st.warning("⚠️ No data loaded! Please upload a valid Excel file or check the default file path.")
        st.info("💡 Tip: You can download a sample template or upload your own maintenance data file.")
        return
    
    # Create filters and filter data
    plants, years, months, statuses, work_center, Order_type, Group, Plan_Type = create_enhanced_filters(processed_data)
    
    # Apply filters
    filter_conditions = True
    
    if 'Plant' in processed_data.columns and plants:
        filter_conditions = filter_conditions & (processed_data['Plant'].isin(plants))
    if 'Year' in processed_data.columns:
        filter_conditions = filter_conditions & (processed_data['Year'].between(years[0], years[1]))
    if 'Month' in processed_data.columns and months:
        filter_conditions = filter_conditions & (processed_data['Month'].isin(months))
    if statuses:
        filter_conditions = filter_conditions & (processed_data['Order Status'].isin(statuses))
    if 'Main Work Center' in processed_data.columns and work_center:
        filter_conditions = filter_conditions & (processed_data['Main Work Center'].isin(work_center))
    if 'Order Type' in processed_data.columns and Order_type:
        filter_conditions = filter_conditions & (processed_data['Order Type'].isin(Order_type))
    if 'Group' in processed_data.columns and Group:
        filter_conditions = filter_conditions & (processed_data['Group'].isin(Group))
    if Plan_Type:
        filter_conditions = filter_conditions & (processed_data['Plan Type'].isin(Plan_Type))
    
    filtered_data = processed_data[filter_conditions]
    
    if filtered_data.empty:
        st.warning("⚠️ No data matches the selected filters. Please adjust your filter criteria.")
        return
    
    # Display filter summary
    display_filter_summary(filtered_data)
    st.divider()
    
    # Main dashboard tabs
    tab_overview, tab_status, tab_department, tab_financial, tab_temporal, tab_data = st.tabs([
        "📊 Overview", "📈 Status Analytics", "🏭 Department Analysis", "💰 Financial Analytics", "📅 Temporal Analysis", "📄 Data Explorer"
    ])
    
    with tab_overview:
        display_enhanced_kpis(filtered_data)
        st.divider()
        plot_status_overview(filtered_data)
    
    with tab_status:
        plot_detailed_status_analysis(filtered_data)
    
    with tab_department:
        plot_department_analysis(filtered_data)
    
    with tab_financial:
        plot_cost_analysis(filtered_data)
    
    with tab_temporal:
        plot_temporal_analysis(filtered_data)
    
    with tab_data:
        show_enhanced_raw_data(filtered_data)
    
    # Footer
    st.divider()
    st.markdown("""
    <div class='footer-text'>
        <p>🔧 Maintenance Analytics Dashboard | Built with Streamlit | Data updates in real-time</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
