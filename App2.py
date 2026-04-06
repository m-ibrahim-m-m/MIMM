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
    "primary": "#1E3A5F",
    "accent": "#00C2CB",
    "success": "#00B37E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "planned": "#3B82F6",
    "unplanned": "#F97316",
    "bg_dark": "#0F1923",
    "bg_card": "#1A2635",
    "text_light": "#E2E8F0",
    "text_muted": "#94A3B8",
}

STATUS_COLORS = {
    "Completed": "#00B37E",
    "In Progress": "#00C2CB",
    "Open": "#F59E0B",
    "Canceled": "#EF4444",
    "Not Executed & Deleted": "#6B7280",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'IBM Plex Sans', sans-serif", color="#E2E8F0", size=12),
    title=dict(font=dict(size=15, color="#E2E8F0"), x=0.01, xanchor="left"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    margin=dict(l=16, r=16, t=48, b=16),
    colorway=["#00C2CB", "#3B82F6", "#00B37E", "#F59E0B", "#EF4444", "#F97316"],
)

LEGEND_DEFAULT = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1)

def apply_layout(fig, legend=None):
    """Apply base dark theme then optionally override legend."""
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
[data-testid="stSidebar"] label { color: #94A3B8 !important; font-size: 0.78rem !important; }
.dash-hero {
    background: linear-gradient(135deg, #1E3A5F 0%, #0F1923 60%);
    border: 1px solid rgba(0,194,203,0.18);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.dash-hero::before {
    content:'';
    position:absolute;top:0;right:0;width:320px;height:100%;
    background:radial-gradient(ellipse at 80% 50%,rgba(0,194,203,0.12) 0%,transparent 70%);
    pointer-events:none;
}
.dash-hero h1 { font-size:1.9rem!important;font-weight:700!important;color:#E2E8F0!important;margin:0 0 0.25rem!important;letter-spacing:-0.02em; }
.dash-hero p { color:#64748B;font-size:0.9rem;margin:0; }
.dash-hero .accent { color:#00C2CB; }
.section-header {
    display:flex;align-items:center;gap:0.6rem;
    margin:2rem 0 1rem;padding-bottom:0.6rem;
    border-bottom:1px solid rgba(255,255,255,0.06);
}
.section-header span { font-size:1.05rem;font-weight:600;color:#E2E8F0;letter-spacing:-0.01em; }
.section-header .pill {
    background:rgba(0,194,203,0.12);color:#00C2CB;
    font-size:0.7rem;font-weight:600;padding:2px 8px;
    border-radius:20px;letter-spacing:0.06em;text-transform:uppercase;
}
.kpi-card {
    background:#1A2635;border:1px solid rgba(255,255,255,0.06);
    border-radius:12px;padding:1.2rem 1.4rem;
    position:relative;overflow:hidden;
    transition:border-color 0.2s,transform 0.2s;
}
.kpi-card:hover { border-color:rgba(0,194,203,0.3);transform:translateY(-2px); }
.kpi-card::after {
    content:'';position:absolute;bottom:0;left:0;right:0;height:3px;
    background:var(--kpi-accent,#00C2CB);border-radius:0 0 12px 12px;
}
.kpi-label { font-size:0.72rem;color:#64748B;font-weight:600;letter-spacing:0.07em;text-transform:uppercase;margin-bottom:0.5rem; }
.kpi-value { font-size:1.6rem;font-weight:700;color:#E2E8F0;font-family:'IBM Plex Mono',monospace;line-height:1; }
.chart-card {
    background:#1A2635;border:1px solid rgba(255,255,255,0.06);
    border-radius:14px;padding:1.25rem;margin-bottom:1rem;
}
.stTabs [data-baseweb="tab-list"] {
    background:#131E2B!important;border-radius:10px!important;
    padding:4px!important;gap:4px!important;border:1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    background:transparent!important;color:#64748B!important;
    border-radius:7px!important;font-size:0.85rem!important;
    font-weight:500!important;padding:0.5rem 1.2rem!important;
}
.stTabs [aria-selected="true"] { background:#1E3A5F!important;color:#00C2CB!important; }
[data-testid="stFileUploader"] {
    border:1.5px dashed rgba(0,194,203,0.3)!important;
    border-radius:12px!important;background:rgba(0,194,203,0.03)!important;
}
[data-baseweb="tag"] { background:rgba(0,194,203,0.15)!important;color:#00C2CB!important; }
.filter-pill {
    display:inline-block;background:rgba(0,194,203,0.1);
    border:1px solid rgba(0,194,203,0.2);color:#00C2CB;
    font-size:0.75rem;padding:3px 10px;border-radius:20px;
    margin:2px;font-family:'IBM Plex Mono',monospace;
}
.filter-group-label { font-size:0.7rem;color:#64748B;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px; }
.stDownloadButton > button {
    background:linear-gradient(135deg,#1E3A5F,#164e63)!important;
    color:#00C2CB!important;border:1px solid rgba(0,194,203,0.3)!important;
    border-radius:8px!important;font-weight:600!important;
}
hr { border-color:rgba(255,255,255,0.06)!important;margin:1.5rem 0!important; }
::-webkit-scrollbar { width:6px;height:6px; }
::-webkit-scrollbar-track { background:#0F1923; }
::-webkit-scrollbar-thumb { background:#1E3A5F;border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#00C2CB; }
</style>
"""

MONTH_ORDER = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']

# ========================= DATA =========================
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    try:
        return pd.read_excel("D:/Dash Board/Maintenance Orders.xlsx")
    except FileNotFoundError:
        st.error("❌ Default file not found. Please upload an Excel file above.")
        return pd.DataFrame()

def process_data(data):
    if data.empty:
        return data
    for col in ['Basic start date', 'Basic finish date']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
    data['Year'] = data['Basic start date'].dt.year
    data['Month'] = data['Basic start date'].dt.month_name()
    data['Quarter'] = data['Basic start date'].dt.quarter

    def determine_status(row):
        statuses = f"{row.get('System status','')} {row.get('User Status','')}".upper().split()
        for code, name in {'CNCL':'Canceled','CNF':'Completed','JIPR':'In Progress','NCMP':'Not Executed & Deleted'}.items():
            if code in statuses:
                return name
        return 'Open'
    data['Order Status'] = data.apply(determine_status, axis=1)
    data['Cost Deviation'] = data['Total sum (actual)'] - data['Total planned costs']
    data['Cost Variance %'] = data['Cost Deviation'] / data['Total planned costs'].replace(0, np.nan) * 100
    data['Plan Type'] = np.where(data['Maintenance Plan'].notna(), 'Planned', 'Unplanned')
    return data

# ========================= SIDEBAR =========================
def create_filters(data):
    st.sidebar.markdown(
        "<div style='padding:0.5rem 0 1.2rem'>"
        "<p style='color:#00C2CB;font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin:0 0 0.2rem'>Maintenance Dashboard</p>"
        "<p style='color:#64748B;font-size:0.78rem;margin:0'>Filter & explore data</p></div>",
        unsafe_allow_html=True
    )
    with st.sidebar.expander("🏭 Plants", expanded=True):
        plants = st.multiselect("Plants", options=sorted(data['Plant'].dropna().unique()),
                                default=sorted(data['Plant'].dropna().unique())[:2],
                                label_visibility="collapsed")
    with st.sidebar.expander("📅 Time Range", expanded=True):
        years = st.slider("Years", min_value=int(data['Year'].min()),
                          max_value=int(data['Year'].max()),
                          value=(int(data['Year'].min()), int(data['Year'].max())),
                          label_visibility="collapsed")
        available_months = sorted(data['Month'].dropna().unique(),
                                  key=lambda x: MONTH_ORDER.index(x) if x in MONTH_ORDER else 99)
        months = st.multiselect("Months", options=available_months, default=available_months,
                                label_visibility="collapsed")
    with st.sidebar.expander("⚙️ Order Attributes", expanded=False):
        plan_type = st.multiselect("Plan Type", options=data['Plan Type'].unique(),
                                     default=data['Plan Type'].unique())
        statuses = st.multiselect("Order Status", options=data['Order Status'].unique(),
                                     default=data['Order Status'].unique())
        order_types = st.multiselect("Work Order Type",
                                     options=sorted(data['Order Type'].dropna().unique()),
                                     default=sorted(data['Order Type'].dropna().unique()))
    with st.sidebar.expander("🏗️ Department / Group", expanded=False):
        work_centers = st.multiselect("Main Work Center",
                                      options=sorted(data['Main Work Center'].dropna().unique()),
                                      default=sorted(data['Main Work Center'].dropna().unique()))
        groups = st.multiselect("Task List Code (Group)",
                                options=sorted(data['Group'].dropna().unique()),
                                default=sorted(data['Group'].dropna().unique()))
    st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:1rem 0'>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='color:#64748B;font-size:0.72rem;text-align:center'>Maintenance Analytics · v2.0</p>", unsafe_allow_html=True)
    return plants, years, months, plan_type, statuses, work_centers, order_types, groups

# ========================= HELPERS =========================
def section_header(icon, title, badge=None):
    badge_html = f'<span class="pill">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="section-header"><span style="font-size:1.1rem">{icon}</span>'
        f'<span>{title}</span>{badge_html}</div>',
        unsafe_allow_html=True
    )

# ========================= KPIs =========================
def display_kpis(fd):
    planned = fd[fd['Plan Type'] == 'Planned']
    total_p = len(planned)
    comp_p = len(planned[planned['Order Status'] == 'Completed'])
    pct_p = comp_p / total_p * 100 if total_p else 0
    total_o = len(fd)
    comp_o = len(fd[fd['Order Status'] == 'Completed'])
    pct_o = comp_o / total_o * 100 if total_o else 0
    act_cost = fd['Total sum (actual)'].sum()
    avg_dev = fd['Cost Deviation'].mean()

    kpis = [
        ("Total Orders", f"{total_o:,}", COLORS["accent"]),
        ("Planned Orders", f"{total_p:,}", COLORS["planned"]),
        ("Planned Completion", f"{pct_p:.1f}%", COLORS["success"]),
        ("Overall Completion", f"{pct_o:.1f}%", COLORS["success"]),
        ("Actual Cost (EGP)", f"{act_cost:,.0f}", COLORS["warning"]),
        ("Avg Cost Deviation", f"{avg_dev:+,.0f}", COLORS["danger"] if avg_dev > 0 else COLORS["success"]),
    ]
    cols = st.columns(len(kpis))
    for col, (label, value, accent) in zip(cols, kpis):
        with col:
            st.markdown(
                f'<div class="kpi-card" style="--kpi-accent:{accent}">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div></div>',
                unsafe_allow_html=True
            )

# ========================= FILTER SUMMARY =========================
def display_filter_summary(fd, years):
    plants_str = " ".join(f'<span class="filter-pill">{p}</span>' for p in sorted(fd['Plant'].unique()))
    st.markdown(
        f'<div style="background:#1A2635;border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:1rem 1.4rem;margin-bottom:0.5rem;">'
        f'<div style="display:flex;flex-wrap:wrap;gap:1.5rem;align-items:center;">'
        f'<div><div class="filter-group-label">Plants</div>{plants_str}</div>'
        f'<div><div class="filter-group-label">Period</div><span class="filter-pill">{years[0]} – {years[1]}</span></div>'
        f'<div><div class="filter-group-label">Dataset</div><span class="filter-pill">{len(fd):,} records</span></div>'
        f'</div></div>',
        unsafe_allow_html=True
    )

# ========================= CHARTS =========================
def plot_completion_donuts(fd):
    planned = fd[fd['Plan Type'] == 'Planned']
    total_p = len(planned); comp_p = len(planned[planned['Order Status'] == 'Completed'])
    total_o = len(fd); comp_o = len(fd[fd['Order Status'] == 'Completed'])

    def donut(vals, labels, title, hit_color):
        fig = go.Figure(go.Pie(
            values=vals, labels=labels, hole=0.62,
            marker_colors=[hit_color, "rgba(255,255,255,0.06)"],
            textinfo='none',
            hovertemplate='%{label}: %{value:,} (%{percent})<extra></extra>'
        ))
        pct = vals[0] / sum(vals) * 100 if sum(vals) else 0
        fig.add_annotation(text=f"<b>{pct:.1f}%</b>", x=0.5, y=0.55,
                           font=dict(size=22, color="#E2E8F0", family="IBM Plex Mono"), showarrow=False)
        fig.add_annotation(text="complete", x=0.5, y=0.38,
                           font=dict(size=11, color="#64748B"), showarrow=False)
        fig.update_layout(
            title_text=title, showlegend=True,
            legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                        bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="'IBM Plex Sans', sans-serif", color="#E2E8F0", size=12),
            margin=dict(l=16, r=16, t=48, b=16),
        )
        return fig

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        if total_p:
            st.plotly_chart(donut([comp_p, total_p-comp_p], ['Completed','Remaining'],
                                  "Planned Orders Completion", COLORS["success"]), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        if total_o:
            st.plotly_chart(donut([comp_o, total_o-comp_o], ['Completed','Remaining'],
                                  "Overall Orders Completion", COLORS["accent"]), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

def plot_status_distribution(fd):
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        sc = fd.groupby(['Order Status','Plan Type'], observed=True).size().reset_index(name='Count')
        fig = px.bar(sc, x='Order Status', y='Count', color='Plan Type', barmode='group', text='Count',
                     title="Orders by Status — Planned vs Unplanned",
                     color_discrete_map={'Planned': COLORS["planned"], 'Unplanned': COLORS["unplanned"]})
        fig.update_traces(texttemplate='%{text:,}', textposition='outside', marker_line_width=0, opacity=0.9)
        apply_layout(fig, legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        ps = fd.groupby(['Plant','Order Status'], observed=True).size().reset_index(name='Count')
        fig2 = px.bar(ps, x='Plant', y='Count', color='Order Status', barmode='stack',
                      title="Status Split per Plant", color_discrete_map=STATUS_COLORS)
        fig2.update_traces(marker_line_width=0)
        apply_layout(fig2)
        st.plotly_chart(fig2, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

def plot_department_orders(fd):
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    dc = fd.groupby(['Main Work Center','Order Status'], observed=True).size().reset_index(name='Count')
    dc.rename(columns={'Main Work Center':'Department'}, inplace=True)
    fig = px.bar(dc, x='Department', y='Count', color='Order Status', text='Count',
                 title="Orders by Department", color_discrete_map=STATUS_COLORS)
    fig.update_traces(texttemplate='%{text:,}', textposition='outside', marker_line_width=0)
    apply_layout(fig)
    st.plotly_chart(fig, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

def plot_status_trends(fd):
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    td = fd.groupby(['Year','Month','Order Status'], observed=True).size().reset_index(name='Count')
    td['Month'] = pd.Categorical(td['Month'], categories=MONTH_ORDER, ordered=True)
    td = td.sort_values(['Year','Month'])
    fig = px.line(td, x='Month', y='Count', color='Order Status', facet_col='Year',
                  markers=True, title="Monthly Order Status Trends",
                  color_discrete_map=STATUS_COLORS, line_shape='spline')
    fig.update_traces(line_width=2.5, marker_size=6)
    fig.update_xaxes(categoryorder='array', categoryarray=MONTH_ORDER, tickangle=45, tickfont_size=10)
    apply_layout(fig)
    st.plotly_chart(fig, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

def plot_order_type_analysis(fd):
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        td = fd['Order Type'].value_counts().reset_index()
        td.columns = ['Order Type','Count']
        fig = px.pie(td, names='Order Type', values='Count', title="Order Type Distribution", hole=0.5)
        fig.update_traces(textposition='inside', textinfo='percent+label',
                          marker=dict(line=dict(color='#0F1923', width=2)))
        apply_layout(fig)
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        trd = fd.groupby(['Year','Month','Order Type'], observed=True).size().reset_index(name='Count')
        trd['Month'] = pd.Categorical(trd['Month'], categories=MONTH_ORDER, ordered=True)
        trd = trd.sort_values(['Year','Month'])
        fig = px.line(trd, x='Month', y='Count', color='Order Type', facet_col='Year',
                      markers=True, title="Monthly Order Type Trends", line_shape='spline')
        fig.update_traces(line_width=2, marker_size=5)
        fig.update_xaxes(tickangle=45, tickfont_size=10)
        apply_layout(fig)
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    cd = fd.groupby('Order Type', observed=True).agg(
        Planned=('Total planned costs','mean'), Actual=('Total sum (actual)','mean')).reset_index()
    fig = go.Figure()
    fig.add_bar(x=cd['Order Type'], y=cd['Planned'], name='Planned',
                marker_color=COLORS["planned"], marker_line_width=0, opacity=0.85)
    fig.add_bar(x=cd['Order Type'], y=cd['Actual'], name='Actual',
                marker_color=COLORS["accent"], marker_line_width=0, opacity=0.85)
    apply_layout(fig)
    fig.update_layout(barmode='group', title_text="Avg Planned vs Actual Cost by Order Type")
    st.plotly_chart(fig, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

def plot_cost_analysis(fd):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        dev_data = fd.nsmallest(10, 'Cost Deviation')
        fig = px.bar(dev_data, x='Order Type', y='Cost Deviation', color='Plant',
                     title="Top 10 Cost Savings (by Order Type)", text_auto='.2s')
        fig.update_traces(marker_line_width=0)
        apply_layout(fig)
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig = px.box(fd, x='Order Status', y='Cost Variance %', color='Plant',
                     title="Cost Variance % Distribution by Status",
                     color_discrete_sequence=[COLORS["accent"], COLORS["planned"],
                                              COLORS["warning"], COLORS["success"]])
        fig.update_traces(line_width=1.5)
        apply_layout(fig)
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

def show_raw_data(fd):
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    search = st.text_input("🔎 Search within data", placeholder="Type to filter any column…")
    if search:
        mask = fd.apply(lambda col: col.astype(str).str.contains(search, case=False, na=False))
        display_df = fd[mask.any(axis=1)]
    else:
        display_df = fd
    st.markdown(
        f"<p style='color:#64748B;font-size:0.78rem;margin-bottom:0.5rem'>"
        f"Showing <b style='color:#E2E8F0'>{len(display_df):,}</b> of "
        f"<b style='color:#E2E8F0'>{len(fd):,}</b> records</p>",
        unsafe_allow_html=True
    )
    st.dataframe(display_df.sort_values('Basic start date', ascending=False),
                 width="stretch", height=380)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        fd.to_excel(writer, index=False, sheet_name='Filtered Data')
    buffer.seek(0)
    st.download_button("📥 Download Filtered Data as Excel", data=buffer,
                       file_name="filtered_maintenance_data.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown('</div>', unsafe_allow_html=True)

# ========================= MAIN =========================
def main():
    st.set_page_config(page_title="Maintenance Analytics", page_icon="🔧",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="dash-hero">'
        '<h1>🏭 Maintenance <span class="accent">Analytics</span> Dashboard</h1>'
        '<p>Real-time insight into work orders, costs, and team performance</p>'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload maintenance data (Excel .xlsx)",
                                     type=["xlsx"], label_visibility="collapsed")

    cached_load = st.cache_data(load_data)
    with st.spinner("Loading and processing data…"):
        raw_data = cached_load(uploaded_file)
        data = process_data(raw_data)

    if data.empty:
        st.warning("⚠️ No data available. Please upload an Excel file.")
        st.stop()

    plants, years, months, plan_type, statuses, work_centers, order_types, groups = create_filters(data)

    fd = data[
        (data['Plant'].isin(plants)) &
        (data['Year'].between(years[0], years[1])) &
        (data['Month'].isin(months)) &
        (data['Plan Type'].isin(plan_type)) &
        (data['Order Status'].isin(statuses)) &
        (data['Main Work Center'].isin(work_centers)) &
        (data['Order Type'].isin(order_types)) &
        (data['Group'].isin(groups))
    ].copy()

    if fd.empty:
        st.warning("⚠️ No records match the current filters. Please adjust your selections.")
        st.stop()

    display_filter_summary(fd, years)
    section_header("📊", "Key Performance Indicators")
    display_kpis(fd)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "✅ Completion", "📦 Order Status", "🏗️ Departments", "💰 Cost Analysis", "📄 Raw Data"
    ])

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
