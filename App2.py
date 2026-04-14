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

# Custom CSS for enhanced UI/UX
def load_custom_css():
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            padding: 0rem 1rem;
        }
        
        /* Custom card styling */
        .custom-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Metric card styling */
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
        
        /* Header styling */
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 2rem;
            color: white;
        }
        
        /* Status badges */
        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 2rem;
            font-size: 0.875rem;
            font-weight: 500;
            display: inline-block;
        }
        
        .status-completed {
            background-color: #10b981;
            color: white;
        }
        
        .status-progress {
            background-color: #3b82f6;
            color: white;
        }
        
        .status-canceled {
            background-color: #ef4444;
            color: white;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8fafc;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background-color: #f1f5f9;
            padding: 0.5rem;
            border-radius: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        
        /* Animation for loading */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading {
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .metric-card {
                margin-bottom: 1rem;
            }
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
        data[col] = pd.to_datetime(data[col], errors='coerce')
    
    data['Year'] = data['Basic start date'].dt.year
    data['Month'] = data['Basic start date'].dt.month_name()
    data['Quarter'] = data['Basic start date'].dt.quarter
    data['Week'] = data['Basic start date'].dt.isocalendar().week
    
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
    
    # Calculate duration
    data['Duration Days'] = (data['Basic finish date'] - data['Basic start date']).dt.days
    
    return data

def create_enhanced_filters(data):
    """Generate enhanced interactive filters in sidebar"""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/maintenance.png", width=80)
        st.markdown("## 🔧 Navigation")
        
        # Quick stats in sidebar
        st.markdown("### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Orders", f"{len(data):,}", delta=None)
        with col2:
            st.metric("Plants", f"{data['Plant'].nunique()}", delta=None)
        
        st.divider()
        
        st.markdown("### 🔍 Filter Options")
        
        # Date range filter
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
        
        # Plant selector with icons
        st.markdown("#### 🏭 Plants")
        plants = st.multiselect(
            "Select Plants:",
            options=data['Plant'].unique(),
            default=data['Plant'].unique(),
            help="Filter by plant location"
        )
        
        # Month selector
        st.markdown("#### 📆 Months")
        months = st.multiselect(
            "Select Months:",
            options=data['Month'].unique(),
            default=data['Month'].unique()
        )
        
        # Plan Type Filter
        data['Plan Type'] = np.where(
            data['Maintenance Plan'].notna(),
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
            # Work center selector
            work_center = st.multiselect(
                "Department:",
                options=data['Main Work Center'].unique(),
                default=data['Main Work Center'].unique()
            )
            
            # Order Type selector
            Order_type = st.multiselect(
                "Work Order Type:",
                options=data['Order Type'].unique(),
                default=data['Order Type'].unique()
            )
            
            # Task list selector
            Group = st.multiselect(
                "Task List Code:",
                options=data['Group'].unique(),
                default=data['Group'].unique()
            )
        
        st.divider()
        
        # Reset filters button
        if st.button("🔄 Reset All Filters", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    return plants, years, months, statuses, work_center, Order_type, Group, plan_type

def display_enhanced_header():
    """Display enhanced dashboard header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.image("https://img.icons8.com/fluency/96/maintenance.png", width=80)
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <h1>🏭 Maintenance Operations Analytics Dashboard</h1>
            <p style='color: #666;'>Real-time insights and analytics for maintenance operations</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.markdown(f"""
        <div style='text-align: right;'>
            <p style='color: #666; font-size: 0.875rem;'>Last Updated</p>
            <p style='font-weight: bold;'>{current_time}</p>
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
            <p><strong>{len(filtered_data['Plant'].unique())}</strong> selected</p>
            <small>{', '.join(filtered_data['Plant'].unique()[:2])}{'...' if len(filtered_data['Plant'].unique()) > 2 else ''}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>📆 Time Period</h4>
            <p><strong>{filtered_data['Year'].min()} - {filtered_data['Year'].max()}</strong></p>
            <small>{len(filtered_data['Month'].unique())} months selected</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>✅ Status</h4>
            <p><strong>{len(filtered_data['Order Status'].unique())}</strong> statuses</p>
            <small>{', '.join(filtered_data['Order Status'].unique())}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_orders = len(filtered_data)
        st.markdown(f"""
        <div class='metric-card'>
            <h4>📊 Total Records</h4>
            <p><strong>{total_orders:,}</strong> orders</p>
            <small>After all filters</small>
        </div>
        """, unsafe_allow_html=True)

def display_enhanced_kpis(filtered_data):
    """Show key performance indicators with enhanced visualization"""
    st.markdown("### 📊 Key Performance Indicators")
    
    # Calculate KPIs
    planned_orders = filtered_data[filtered_data['Maintenance Plan'].notna()]
    total_planned = len(planned_orders)
    completed_planned = len(planned_orders[planned_orders['Order Status'] == 'Completed'])
    planned_completion_pct = (completed_planned / total_planned * 100) if total_planned > 0 else 0
    
    total_orders = len(filtered_data)
    completed_orders = len(filtered_data[filtered_data['Order Status'] == 'Completed'])
    overall_completion_pct = (completed_orders / total_orders * 100) if total_orders > 0 else 0
    
    # Calculate additional KPIs
    avg_duration = filtered_data['Duration Days'].mean() if 'Duration Days' in filtered_data.columns else 0
    total_cost = filtered_data['Total sum (actual)'].sum()
    avg_cost_deviation = filtered_data['Cost Deviation'].mean()
    
    # Display KPIs in a grid
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            label="📋 Total Orders",
            value=f"{total_orders:,}",
            delta=f"{((completed_orders/total_orders)*100 if total_orders>0 else 0):.1f}% completion",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="✅ Completion Rate",
            value=f"{overall_completion_pct:.1f}%",
            delta=f"{planned_completion_pct:.1f}% planned",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="⏱️ Avg Duration",
            value=f"{avg_duration:.1f} days",
            delta=None
        )
    
    with col4:
        st.metric(
            label="💰 Total Cost",
            value=f"EGP {total_cost:,.0f}",
            delta=None
        )
    
    with col5:
        st.metric(
            label="📊 Avg Cost Deviation",
            value=f"EGP {avg_cost_deviation:,.0f}",
            delta_color="inverse"
        )
    
    with col6:
        st.metric(
            label="🏭 Active Plants",
            value=f"{filtered_data['Plant'].nunique()}",
            delta=None
        )
    
    # Progress bars for completion rates
    st.markdown("#### Completion Progress")
    col1, col2 = st.columns(2)
    
    with col1:
        st.progress(overall_completion_pct / 100, text=f"Overall Completion: {overall_completion_pct:.1f}%")
    
    with col2:
        if total_planned > 0:
            st.progress(planned_completion_pct / 100, text=f"Planned Orders Completion: {planned_completion_pct:.1f}%")

def plot_enhanced_order_status_distribution(filtered_data):
    """Visualize order status distribution with enhanced design"""
    st.markdown("### 📊 Order Status Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📈 Status Distribution", "🏭 By Plant", "📊 Trend Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Create grouped status counts
            filtered_data['Plan Type'] = np.where(
                filtered_data['Maintenance Plan'].notna(),
                'Planned',
                'Unplanned'
            )
            
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
                    x=1
                ),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
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
            
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                pull=[0.05 if i == 0 else 0 for i in range(len(status_pie))]
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        # Plant-wise breakdown
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
        
        fig_plant.update_traces(
            texttemplate='%{text:,}',
            textposition='inside'
        )
        
        fig_plant.update_layout(
            xaxis_title="Plant",
            yaxis_title="Number of Orders",
            legend_title="Order Status",
            height=400
        )
        
        st.plotly_chart(fig_plant, use_container_width=True)
    
    with tab3:
        # Status trends over time
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
        st.plotly_chart(fig_trend, use_container_width=True)

def plot_enhanced_department_analysis(filtered_data):
    """Visualize department-wise analysis with enhanced metrics"""
    st.markdown("### 🏗️ Department Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        department_counts = filtered_data['Main Work Center'].value_counts().reset_index()
        department_counts.columns = ['Department', 'Count']
        department_counts = department_counts.head(10)  # Top 10 departments
        
        fig = px.bar(
            department_counts,
            x='Department',
            y='Count',
            color='Count',
            text='Count',
            title="Top 10 Departments by Orders",
            color_continuous_scale='Viridis'
        )
        
        fig.update_traces(
            texttemplate='%{text:,}',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_title="Department",
            yaxis_title="Number of Orders",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
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
        
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_title="Department",
            yaxis_title="Completion Rate (%)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def plot_enhanced_cost_analysis(filtered_data):
    """Visualize cost-related metrics with enhanced insights"""
    st.markdown("### 💰 Financial Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Cost Overview", "💰 Cost Savings", "📈 Variance Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost by order type
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
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cost distribution by plant
            cost_by_plant = filtered_data.groupby('Plant')['Total sum (actual)'].sum().reset_index()
            
            fig = px.pie(
                cost_by_plant,
                values='Total sum (actual)',
                names='Plant',
                title="Cost Distribution by Plant",
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Cost savings analysis
        filtered_data['Cost Savings'] = filtered_data['Total planned costs'] - filtered_data['Total sum (actual)']
        savings_data = filtered_data.nlargest(10, 'Cost Savings')
        
        fig = px.bar(
            savings_data,
            x='Order Type',
            y='Cost Savings',
            color='Plant',
            title="Top 10 Cost Savings by Order Type",
            labels={'Cost Savings': 'Cost Savings (EGP)'},
            text_auto='.2f'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Variance analysis
        fig = px.box(
            filtered_data,
            x='Order Status',
            y='Cost Variance %',
            color='Plant',
            title="Cost Variance Distribution by Status",
            points="all"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

def plot_temporal_analysis(filtered_data):
    """Visualize temporal trends and patterns"""
    st.markdown("### 📈 Temporal Analysis")
    
    tab1, tab2 = st.tabs(["📅 Monthly Trends", "📊 Quarterly Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly order volume
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
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Monthly cost trends
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
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Quarterly analysis
        quarterly_data = filtered_data.groupby(['Year', 'Quarter']).agg({
            'Order': 'count',
            'Total sum (actual)': 'sum',
            'Duration Days': 'mean'
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
        st.plotly_chart(fig, use_container_width=True)

def show_enhanced_raw_data(filtered_data):
    """Display interactive data table with enhanced features"""
    st.markdown("### 📄 Data Explorer")
    
    # Add search functionality
    search_term = st.text_input("🔍 Search across all columns", placeholder="Enter search term...")
    
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
        default=[col for col in default_columns if col in available_columns]
    )
    
    if selected_columns:
        display_data = display_data[selected_columns]
    
    # Display dataframe with custom formatting
    st.dataframe(
        display_data.sort_values('Basic start date', ascending=False) if 'Basic start date' in display_data.columns else display_data,
        column_config={
            "Total planned costs": st.column_config.NumberColumn(format="EGP %.2f"),
            "Total sum (actual)": st.column_config.NumberColumn(format="EGP %.2f"),
            "Cost Variance %": st.column_config.NumberColumn(format="%.2f%%"),
            "Basic start date": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Basic finish date": st.column_config.DateColumn(format="YYYY-MM-DD"),
        },
        use_container_width=True,
        height=500
    )
    
    # Export functionality
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Export to CSV
        csv = display_data.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"maintenance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export to Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            display_data.to_excel(writer, index=False, sheet_name='Maintenance Data')
        buffer.seek(0)
        
        st.download_button(
            label="📊 Download as Excel",
            data=buffer,
            file_name=f"maintenance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        # Show row count
        st.metric("Total Rows", f"{len(display_data):,}")

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
    filtered_data = processed_data[
        (processed_data['Plant'].isin(plants)) &
        (processed_data['Year'].between(years[0], years[1])) &
        (processed_data['Month'].isin(months)) &
        (processed_data['Order Status'].isin(statuses)) &
        (processed_data['Main Work Center'].isin(work_center)) &
        (processed_data['Order Type'].isin(Order_type)) &
        (processed_data['Group'].isin(Group)) &
        (processed_data['Plan Type'].isin(Plan_Type))
    ]
    
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
        plot_enhanced_order_status_distribution(filtered_data)
    
    with tab_status:
        plot_enhanced_order_status_distribution(filtered_data)
    
    with tab_department:
        plot_enhanced_department_analysis(filtered_data)
    
    with tab_financial:
        plot_enhanced_cost_analysis(filtered_data)
    
    with tab_temporal:
        plot_temporal_analysis(filtered_data)
    
    with tab_data:
        show_enhanced_raw_data(filtered_data)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🔧 Maintenance Analytics Dashboard | Built with Streamlit | Data updates in real-time</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
