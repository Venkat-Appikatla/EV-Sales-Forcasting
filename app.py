import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="EV Sales Forecasting",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f5f7f9;
        padding: 2rem;
    }
    
    /* Headers */
    .css-10trblm {
        color: #1f4287;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    /* Metrics */
    .css-1r6slb0 {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        transition: transform 0.3s ease;
    }
    .css-1r6slb0:hover {
        transform: translateY(-5px);
    }
    
    /* Charts container */
    .chart-container {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1f4287;
        padding: 2rem 1rem;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1f4287;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #163364;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🚗 Global EV Sales Forecasting Dashboard")
st.markdown("### Machine Learning-Based Sales Prediction and Analysis")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\appik\OneDrive\Desktop\DSAIML2255-Machine Learning-Based Sales Forecasting Model with Automated Data Processing Using Pandas and Visualization\Code\IEA-EV-dataEV salesHistoricalCars.csv")
    return df

df = load_data()

# Sidebar
with st.sidebar:
    st.header("Dashboard Controls")
    selected_region = st.selectbox(
        "Select Region",
        options=df['region'].unique()
    )
    
    # New feature: Multi-region comparison
    st.markdown("### Multi-Region Comparison")
    selected_regions = st.multiselect(
        "Select Regions to Compare",
        options=df['region'].unique(),
        default=[selected_region]
    )
    
    # New feature: Year range filter
    min_year, max_year = int(df['year'].min()), int(df['year'].max())
    year_range = st.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=2050,
        value=(min_year, max_year)
    )
    
    # New feature: Forecast horizon
    forecast_years = max(7, year_range[1] - 2023)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This dashboard provides insights into global EV sales trends and forecasts
    using advanced machine learning techniques.
    """)

    # -----------------------------
    st.markdown("---")
    st.markdown("### Developed By")
    st.markdown("""
    APPIKATLA VENKATA RAMANA
    """)
    # ------------------------------

# Main content
col1, col2, col3 = st.columns(3)

# Filter data by year range
df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

# Key metrics
primary_region = selected_regions[0]
filtered_df = df_filtered[df_filtered['region'] == primary_region]
if not filtered_df.empty:
    latest_year = filtered_df['year'].max()
    latest_sales = filtered_df[filtered_df['year'] == latest_year]['value'].values[0]
    prev_year = filtered_df[filtered_df['year'] == latest_year-1]
    if not prev_year.empty:
        growth = ((latest_sales - prev_year['value'].values[0]) / prev_year['value'].values[0] * 100)
    else:
        growth = 0

    with col1:
        st.metric("Latest Annual Sales", f"{int(latest_sales):,}", f"{growth:.1f}% YoY")
        
    with col2:
        avg_sales = filtered_df['value'].mean()
        st.metric("Average Annual Sales", f"{int(avg_sales):,}")
        
    with col3:
        total_sales = filtered_df['value'].sum()
        st.metric("Total Historical Sales", f"{int(total_sales):,}")
else:
    st.warning("No data available for the selected filters.")

# Sales Trend Chart
st.markdown("### Historical Sales Trend")
if len(selected_regions) == 1:
    filtered_df = df_filtered[df_filtered['region'] == selected_regions[0]]
    fig_trend = px.line(
        filtered_df,
        x='year',
        y='value',
        title=f'EV Sales Trend in {selected_regions[0]} ({year_range[0]}-{year_range[1]})',
        template='plotly_white'
    )
else:
    filtered_df = df_filtered[df_filtered['region'].isin(selected_regions)]
    fig_trend = px.line(
        filtered_df,
        x='year',
        y='value',
        color='region',
        title=f'EV Sales Trend Comparison ({year_range[0]}-{year_range[1]})',
        template='plotly_white'
    )

fig_trend.update_traces(line_width=3)
fig_trend.update_layout(
    xaxis_title="Year",
    yaxis_title="Sales Volume",
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)
st.plotly_chart(fig_trend, use_container_width=True)

# Year-over-Year Growth
st.markdown("### Year-over-Year Growth Analysis")
primary_region = selected_regions[0]
growth_df = df_filtered[df_filtered['region'] == primary_region].copy()
if not growth_df.empty:
    growth_df['YoY_Growth'] = growth_df['value'].pct_change() * 100

    fig_growth = px.bar(
        growth_df[growth_df['year'] > growth_df['year'].min()],
        x='year',
        y='YoY_Growth',
        title=f'Year-over-Year Growth Rate in {primary_region} ({year_range[0]}-{year_range[1]})',
        template='plotly_white'
    )
    fig_growth.update_traces(marker_color='#1f4287')
    fig_growth.update_layout(
        xaxis_title="Year",
        yaxis_title="Growth Rate (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_growth, use_container_width=True)
else:
    st.warning("No data for growth analysis.")

# Regional Comparison
st.markdown("### Regional Market Share Analysis")
latest_year_data = df_filtered[df_filtered['year'] == df_filtered['year'].max()]
if not latest_year_data.empty:
    fig_pie = px.pie(
        latest_year_data,
        values='value',
        names='region',
        title=f'Regional Market Share Distribution ({df_filtered["year"].max()})',
        template='plotly_white'
    )
    fig_pie.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.warning("No data for market share.")

# LSTM-Based Future Sales Forecast
st.markdown("### LSTM-Based Future Sales Forecast")

# Prepare data for LSTM (total global sales per year)
dl_df = df[df['parameter'] == 'EV sales'].groupby('year')['value'].sum().reset_index()
dl_df.rename(columns={'value': 'total_sales'}, inplace=True)

# Normalize data
df_years = dl_df['year']
scaler = MinMaxScaler()
scaled_sales = scaler.fit_transform(dl_df[['total_sales']])

# Load LSTM model
lstm_model = load_model('lstm_model.keras')

# Create last sequence (window=3)
window = 3
last_sequence = scaled_sales[-window:]
predictions_scaled = []

for _ in range(forecast_years):  # Predict based on slider
    input_seq = last_sequence[-window:].reshape(1, window, 1)
    pred = lstm_model.predict(input_seq, verbose=0)
    predictions_scaled.append(pred[0][0])
    last_sequence = np.append(last_sequence, pred[0][0])

# Inverse transform to get real values
predicted_values = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
forecast_years_list = list(range(2024, 2024 + forecast_years))
actual_years = dl_df['year']
actual_values = dl_df['total_sales']

# Plot actual vs forecast using Plotly
fig_lstm = go.Figure()
fig_lstm.add_trace(go.Scatter(x=actual_years, y=actual_values, mode='lines+markers', name='Actual Sales', line=dict(color='#1f4287', width=3)))
fig_lstm.add_trace(go.Scatter(x=forecast_years_list, y=predicted_values, mode='lines+markers', name='LSTM Forecast', line=dict(color='#e94560', width=3, dash='dash')))
fig_lstm.add_vline(x=2023.5, line_dash='dash', line_color='red', annotation_text='Forecast Start', annotation_position='top right')
fig_lstm.update_layout(
    title=f"EV Sales Forecast using LSTM (2024–{2023 + forecast_years})",
    xaxis_title="Year",
    yaxis_title="EV Sales (Vehicles)",
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig_lstm, use_container_width=True)

# Data Export Feature
st.markdown("### Data Export")
col1, col2 = st.columns(2)
with col1:
    if st.button("Download Filtered Data (CSV)"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f'ev_sales_{"_".join(selected_regions)}.csv',
            mime='text/csv'
        )
with col2:
    if st.button("Download Forecast Data (CSV)"):
        forecast_df = pd.DataFrame({
            'Year': forecast_years_list,
            'Predicted_Sales': predicted_values
        })
        csv_forecast = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast CSV",
            data=csv_forecast,
            file_name='ev_sales_forecast.csv',
            mime='text/csv'
        )

# Additional Insights
st.markdown("### Additional Insights: Sales Distribution by Region")
fig_box = px.box(
    df_filtered,
    x='region',
    y='value',
    title='EV Sales Distribution Across Regions',
    template='plotly_white'
)
fig_box.update_layout(
    xaxis_title="Region",
    yaxis_title="Sales Volume",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)
st.plotly_chart(fig_box, use_container_width=True)

# # Statistical Analysis
# st.markdown("### Statistical Analysis")
# st.markdown("#### Normality Test (Shapiro-Wilk) for Sales Data")
# test_region = st.selectbox("Select Region for Test", options=df['region'].unique(), key='normality')
# test_data = df_filtered[df_filtered['region'] == test_region]['value']
# if not test_data.empty:
#     stat, p_value = stats.shapiro(test_data)
#     st.write(f"**Statistic:** {stat:.4f}")
#     st.write(f"**p-value:** {p_value:.4f}")
#     if p_value > 0.05:
#         st.success("Data appears to be normally distributed (fail to reject H0)")
#     else:
#         st.warning("Data does not appear to be normally distributed (reject H0)")

#     # Additional Visualization with Matplotlib
#     st.markdown("### Sales Histogram")
#     fig, ax = plt.subplots()
#     ax.hist(test_data, bins=10, alpha=0.7, color='blue', edgecolor='black')
#     ax.set_title(f'Sales Distribution in {test_region}')
#     ax.set_xlabel('Sales Volume')
#     ax.set_ylabel('Frequency')
#     st.pyplot(fig)
# else:
#     st.warning("No data for statistical analysis.")

# # Correlation Heatmap
# st.markdown("### Correlation Heatmap of Regional Sales")
# pivot_df = df_filtered.pivot_table(values='value', index='year', columns='region', aggfunc='sum')
# if not pivot_df.empty:
#     corr = pivot_df.corr()
#     fig, ax = plt.subplots(figsize=(10, 8))
#     sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
#     ax.set_title('Correlation between Regional EV Sales')
#     st.pyplot(fig)
# else:
#     st.warning("Not enough data for correlation analysis.")

# Summary Statistics
with st.expander("View Summary Statistics"):
    if not df_filtered.empty:
        st.dataframe(df_filtered.describe())
    else:
        st.warning("No data available.")

# Simple Linear Regression Forecast
st.markdown("### Simple Linear Regression Forecast")
lr_region = st.selectbox("Select Region for Linear Regression", options=df['region'].unique(), key='lr')
lr_years = st.slider("Years to predict ahead", min_value=1, max_value=30, value=5, key='lr_years')
lr_data = df_filtered[df_filtered['region'] == lr_region]
if not lr_data.empty and len(lr_data) > 1:
    X = lr_data['year'].values.reshape(-1, 1)
    y = lr_data['value'].values
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future years
    last_year = lr_data['year'].max()
    future_years = list(range(last_year + 1, last_year + 1 + lr_years))
    future_preds = model.predict(np.array(future_years).reshape(-1, 1))
    
    st.write(f"**Predicted sales for {lr_region}:**")
    for year, pred in zip(future_years, future_preds):
        st.write(f"- {year}: {int(pred):,}")
    
    # Plot
    fig_lr = go.Figure()
    fig_lr.add_trace(go.Scatter(x=lr_data['year'], y=lr_data['value'], mode='markers', name='Actual'))
    fig_lr.add_trace(go.Scatter(x=future_years, y=future_preds, mode='markers+lines', name='Prediction', marker=dict(color='red')))
    # Add trend line
    trend_years = np.linspace(lr_data['year'].min(), last_year + lr_years, 100)
    trend_values = model.predict(trend_years.reshape(-1, 1))
    fig_lr.add_trace(go.Scatter(x=trend_years, y=trend_values, mode='lines', name='Trend Line', line=dict(dash='dash')))
    fig_lr.update_layout(title=f'Linear Regression for {lr_region}', xaxis_title='Year', yaxis_title='Sales')
    st.plotly_chart(fig_lr)
# else:
#     st.warning("Not enough data for linear regression.")

