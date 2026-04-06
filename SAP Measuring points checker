import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import calendar
import plotly.express as px
import io

# --- Page Config ---
st.set_page_config(page_title="SAP Measuring Points Checker", layout="wide")
st.title("🔧 SAP Measuring Points & Work Order Checker")
pd.set_option("styler.render.max_elements", 506547)
# --- Helper Functions ---
@st.cache_data
def load_data(file):
    if file is not None:
        if file.name.endswith(".csv"):
            return pd.read_csv(file,encoding='ISO-8859-1' or 'utf-8',sep=None,engine='python')
        else:
            return pd.read_excel(file)
    return pd.DataFrame()

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

@st.cache_data
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Final Results')
    return output.getvalue()


def color_limit_status(val):
    if isinstance(val, str):
        if "Above" in val or "Below" in val:
            return 'background-color: #ffcccc'  # red
        elif "Within" in val:
            return 'background-color: #d4edda'  # green
        elif "Not Measured" in val:
            return 'background-color: #f8d7da'  # pink
    return ''

# --- Sidebar Inputs ---
st.sidebar.header("📁 Upload Files")
mp_file = st.sidebar.file_uploader("Measuring Points Readings file", type=["csv", "xlsx"])
wo_file = st.sidebar.file_uploader("Work Orders file", type=["csv", "xlsx"])
expected_mp_file = st.sidebar.file_uploader("Expected to be measured Measuring Points file", type=["csv", "xlsx"])

st.sidebar.header("📅 Select Month")
month = st.sidebar.selectbox("Month", range(1, 13), format_func=lambda m: calendar.month_name[m])
year = st.sidebar.selectbox("Year", range(2023, datetime.today().year + 1))
selected_month = datetime(year, month, 1)
start_date = selected_month
end_date = selected_month + pd.offsets.MonthEnd(1)

# --- Load Data ---
mp_df = load_data(mp_file)
wo_df = load_data(wo_file)
expected_df = load_data(expected_mp_file)

if not mp_df.empty and not wo_df.empty and not expected_df.empty:
    required_mp_cols = ['Measuring point', 'Functional Location', 'Meas/TotCountrRdg   _',
                        'Upper range limit', 'Lower range limit', 'Date', 'Order']
    required_wo_cols = ['Order', 'Basic start date', 'Functional Location', 'Group']
    required_expected_cols = ['Measuring point']

    if not all(col in mp_df.columns for col in required_mp_cols):
        st.error(f"❌ Measuring Points file must include: {', '.join(required_mp_cols)}")
    elif not all(col in wo_df.columns for col in required_wo_cols):
        st.error(f"❌ Work Orders file must include: {', '.join(required_wo_cols)}")
    elif not all(col in expected_df.columns for col in required_expected_cols):
        st.error(f"❌ Expected Measuring Points file must include: {', '.join(required_expected_cols)}")
    else:
        mp_df['Date'] = pd.to_datetime(mp_df['Date'], errors='coerce')
        wo_df['Basic start date'] = pd.to_datetime(wo_df['Basic start date'], errors='coerce')
        mp_df['Meas/TotCountrRdg   _'] = pd.to_numeric(mp_df['Meas/TotCountrRdg   _'], errors='coerce')
        mp_df['Lower range limit'] = pd.to_numeric(mp_df.get('Lower range limit'), errors='coerce')
        mp_df['Upper range limit'] = pd.to_numeric(mp_df.get('Upper range limit'), errors='coerce')

        # mp_filtered = mp_df[(mp_df['Date'] >= start_date) & (mp_df['Date'] <= end_date)].copy()
        mp_filtered = mp_df
        wo_filtered = wo_df[(wo_df['Basic start date'] >= start_date) & (wo_df['Basic start date'] <= end_date)].copy()

        wo_filtered['GroupNormalized'] = wo_filtered['Group'].astype(str).str[2:]
        allowed_groups = ['SJ0030', 'SJ0031', 'SJ0032','-E-028','-E-029','-E-012','SJ0078']
        wo_filtered = wo_filtered[wo_filtered['GroupNormalized'].isin(allowed_groups)]

        mp_filtered['LimitStatus'] = np.where(
            mp_filtered['Meas/TotCountrRdg   _'].isna(), 'Not Measured',
            np.where(mp_filtered['Meas/TotCountrRdg   _'] < mp_filtered['Lower range limit'], 'Below Lower Limit',
            np.where(mp_filtered['Meas/TotCountrRdg   _'] > mp_filtered['Upper range limit'], 'Above Upper Limit', 'Within Limits'))
        )

        mp_with_orders = mp_filtered[mp_filtered['Order'].notna() & (mp_filtered['Order'].astype(str).str.strip() != '')]
        mp_without_orders = mp_filtered[mp_filtered['Order'].isna() | (mp_filtered['Order'].astype(str).str.strip() == '')]

        merged_with_orders = pd.merge(mp_with_orders, wo_filtered, on='Order', how='left', suffixes=('_MP', '_WO'))
        matched = merged_with_orders[merged_with_orders['Basic start date'].notna()]

        for col in matched.columns:
            if col not in mp_without_orders.columns:
                mp_without_orders[col] = np.nan

        merged_df = pd.concat([matched, mp_without_orders], ignore_index=True)
        merged_df.drop(columns='GroupNormalized', inplace=True, errors='ignore')
        merged_df['Measurement document'] = pd.to_numeric(merged_df['Measurement document'], errors='coerce')
        merged_df = merged_df.sort_values(by='Measurement document', ascending=False)
        merged_df = merged_df.drop_duplicates(subset=['Measuring point'])

        expected_df['Expected'] = True
        final_df = pd.merge(expected_df, merged_df, on='Measuring point', how='left')

        # --- Display Outputs ---
        st.subheader("📌 Measuring Points with Limit Check")
        st.dataframe(mp_filtered.style.applymap(color_limit_status, subset=['LimitStatus']), use_container_width=True)

        st.subheader("📌 Filtered Work Orders in Selected Month & Group")
        st.dataframe(wo_filtered, use_container_width=True)

        st.subheader("🔍 Final Merged Measuring Points with Work Orders")
        st.dataframe(final_df, use_container_width=True)

        # --- Readings Status Chart ---
        if 'LimitStatus' in final_df.columns:
            final_df['LimitStatus'] = final_df['LimitStatus'].fillna('Not Measured')
            limit_counts = final_df['LimitStatus'].value_counts().reset_index()
            limit_counts.columns = ['LimitStatus', 'Count']
            fig = px.bar(limit_counts, x='LimitStatus', y='Count', color='LimitStatus',
                         text='Count',
                         title='Readings Status Overview',
                         color_discrete_map={
                             'Below Lower Limit': 'red',
                             'Above Upper Limit': 'orange',
                             'Within Limits': 'green',
                             'Not Measured': 'gray'
                         })
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

            # Pie Chart
            pie_fig = px.pie(limit_counts, values='Count', names='LimitStatus',
                             title='Readings Status Distribution')
            st.plotly_chart(pie_fig, use_container_width=True)

            # Trend over the selected month
            if 'Date' in final_df.columns:
                daily_counts = final_df[final_df['Date'].notna()].groupby(final_df['Date'].dt.date).size().reset_index(name='Readings')
                trend_fig = px.line(daily_counts, x='Date', y='Readings', title='Readings Over Time')
                st.plotly_chart(trend_fig, use_container_width=True)

            # Top 10 measuring points with violations
            violations = final_df[final_df['LimitStatus'].isin(['Below Lower Limit', 'Above Upper Limit'])]
            top_violations = violations['Measuring point'].value_counts().head(10).reset_index()
            top_violations.columns = ['Measuring point', 'Violation Count']
            st.subheader("🚨 Top 10 Measuring Points with Limit Violations")
            st.dataframe(top_violations, use_container_width=True)

            # Completion Rate
            total_expected = expected_df.shape[0]
            measured_count = final_df[final_df['LimitStatus'] != 'Not Measured'].shape[0]
            completion_rate = round((measured_count / total_expected) * 100, 2) if total_expected > 0 else 0
            st.metric("📊 Completion Rate", f"{completion_rate}%")

            # --- Limit Status per Plant ---
            if 'Measurement position_x' in final_df.columns:
                st.subheader("🏭 Limit Status Matrix per Plant")

                status_by_plant = final_df.groupby(['Measurement position_x', 'LimitStatus']).size().reset_index(name='Count')
                plant_matrix = status_by_plant.pivot(index='Measurement position_x', columns='LimitStatus', values='Count').fillna(0).astype(int)
                plant_matrix['Total'] = plant_matrix.sum(axis=1)
                for status in ['Below Lower Limit', 'Above Upper Limit', 'Within Limits', 'Not Measured']:
                    if status not in plant_matrix.columns:
                        plant_matrix[status] = 0
                    plant_matrix[status + ' %'] = round((plant_matrix[status] / plant_matrix['Total']) * 100, 2)
                st.dataframe(plant_matrix, use_container_width=True)

                matrix_fig = px.bar(
                    status_by_plant,
                    x='Measurement position_x',
                    y='Count',
                    color='LimitStatus',
                    barmode='group',
                    title='Limit Status Distribution by Plant',
                    text='Count',
                    color_discrete_map={
                        'Below Lower Limit': 'red',
                        'Above Upper Limit': 'orange',
                        'Within Limits': 'green',
                        'Not Measured': 'gray'
                    }
                )
                matrix_fig.update_traces(textposition='outside')
                st.plotly_chart(matrix_fig, use_container_width=True)

            # --- Limit Status by Category and Plant ---
            if 'MeasPointCategory' in final_df.columns:
                st.subheader("📊 Matrix: Limit Status by Measuring Point Category and Plant")

                # Grouping the data
                category_plant_status = final_df.groupby(
                    ['Measurement position_x', 'MeasPointCategory', 'LimitStatus']
                ).size().reset_index(name='Count')

                # Pivoting to create matrix
                matrix_cp = category_plant_status.pivot_table(
                    index=['Measurement position_x', 'MeasPointCategory'],
                    columns='LimitStatus',
                    values='Count',
                    fill_value=0
                )

                # Add total and percentage columns
                matrix_cp['Total'] = matrix_cp.sum(axis=1)
                for status in ['Below Lower Limit', 'Above Upper Limit', 'Within Limits', 'Not Measured']:
                    if status not in matrix_cp.columns:
                        matrix_cp[status] = 0
                    matrix_cp[status + ' %'] = round((matrix_cp[status] / matrix_cp['Total']) * 100, 2)

                # Display the matrix
                st.dataframe(matrix_cp, use_container_width=True)

                st.subheader("🧮 Limit Status Matrix by Measuring Point Category and Plant")
                    
                # Group and pivot data
                category_status = final_df.groupby(
                    ['Measurement position_x', 'MeasPointCategory', 'LimitStatus']
                ).size().reset_index(name='Count')

                pivot_matrix = category_status.pivot_table(
                    index=['Measurement position_x', 'MeasPointCategory'],
                    columns='LimitStatus',
                    values='Count',
                    fill_value=0
                )

                # Calculate total and percentage columns
                pivot_matrix['Total'] = pivot_matrix.sum(axis=1)
                for status in ['Below Lower Limit', 'Above Upper Limit', 'Within Limits', 'Not Measured']:
                    if status not in pivot_matrix.columns:
                        pivot_matrix[status] = 0
                    pivot_matrix[status + ' %'] = round((pivot_matrix[status] / pivot_matrix['Total']) * 100, 2)
                # --- Missing Readings per Main PL/SS/FLOC Location ---
            if all(col in final_df.columns for col in ['Main PL/SS/FLOC', 'Measurement position_x', 'MeasPointCategory']):
                expected_by_floc = final_df.groupby(
                    ['Main PL/SS/FLOC', 'Measurement position_x', 'MeasPointCategory']
                ).size().reset_index(name='ExpectedCount')

                if all(col in final_df.columns for col in ['Main PL/SS/FLOC', 'Measurement position_x', 'MeasPointCategory']):
                    measured_by_floc = final_df[final_df['LimitStatus'] != 'Not Measured'].groupby(
                        ['Main PL/SS/FLOC', 'Measurement position_x', 'MeasPointCategory']
                    ).size().reset_index(name='MeasuredCount')

                    missing_summary = pd.merge(
                        expected_by_floc,
                        measured_by_floc,
                        on=['Main PL/SS/FLOC', 'Measurement position_x', 'MeasPointCategory'],
                        how='left'
                    )

                    missing_summary['MeasuredCount'] = missing_summary['MeasuredCount'].fillna(0).astype(int)
                    missing_summary['MissingCount'] = missing_summary['ExpectedCount'] - missing_summary['MeasuredCount']

                    st.subheader("❗ Missing Readings Summary by Main PL/SS/FLOC")
                    st.dataframe(missing_summary, use_container_width=True)
                else:
                    st.warning("⚠️ Some required columns are missing in the merged data for measured readings.")
            else:
                st.warning("⚠️ Some required columns are missing in the expected file to analyze missing readings.")

        excel_data = convert_df_to_excel(final_df)
        st.download_button(
            "⬇️ Download Final Results (Excel)",
            data=excel_data,
            file_name="final_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


else:
    st.info("📄 Please upload Measuring Points Readings, Work Orders, and Expected Points files to proceed.")
