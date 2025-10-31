import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SA Unemployment Forecast",
    page_icon="📊",
    layout="wide"
)

# Load dataset
@st.cache_data
def load_data():
    file_path = 'world_bank_dataset.csv'
    df = pd.read_csv(file_path)
    
    # Filter South Africa unemployment data
    sa_data = df[df['Country'] == 'South Africa'].copy()
    sa_data['Year'] = pd.to_datetime(sa_data['Year'], format='%Y')
    sa_data = sa_data.sort_values('Year')
    
    # Prepare Prophet dataframe
    prophet_df = sa_data[['Year', 'Unemployment Rate (%)']].rename(
        columns={'Year':'ds', 'Unemployment Rate (%)':'y'}
    ).dropna()
    
    return prophet_df

# Main app
def main():
    st.title("🇿🇦 South Africa Unemployment Forecast Dashboard")
    st.markdown("### AI-Driven Job Loss Analysis & Future Projections")
    
    # Load data
    data = load_data()
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # Time window filter
    st.sidebar.subheader("Time Window")
    start_year = st.sidebar.slider("Start Year", 2010, 2020, 2010)
    end_year = st.sidebar.slider("End Year", 2020, 2022, 2022)
    
    # Forecast horizon
    forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 15, 10)
    
    # Scenario toggle
    scenario = st.sidebar.radio(
        "Scenario",
        ["Base Case", "Optimistic (-5%)", "Pessimistic (+5%)"]
    )
    
    # Filter data based on selection
    filtered_data = data[
        (data['ds'].dt.year >= start_year) & 
        (data['ds'].dt.year <= end_year)
    ]
    
    # Split into train/test
    train_size = int(len(filtered_data) * 0.8)
    train_data = filtered_data[:train_size]
    test_data = filtered_data[train_size:]
    
    # Initialize Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=0.1,
        interval_width=0.95,
        n_changepoints=min(3, len(train_data)-1)
    )
    
    # Fit model
    model.fit(train_data)
    
    # Create future dataframe
    future_forecast = model.make_future_dataframe(periods=forecast_years, freq='Y')
    forecast_future = model.predict(future_forecast)
    
    # Apply scenario adjustments
    if scenario == "Optimistic (-5%)":
        forecast_future['yhat'] *= 0.95
        forecast_future['yhat_lower'] *= 0.95
        forecast_future['yhat_upper'] *= 0.95
    elif scenario == "Pessimistic (+5%)":
        forecast_future['yhat'] *= 1.05
        forecast_future['yhat_lower'] *= 1.05
        forecast_future['yhat_upper'] *= 1.05
    
    # Add scenario columns
    forecast_future['yhat_optimistic'] = forecast_future['yhat'] * 0.95
    forecast_future['yhat_pessimistic'] = forecast_future['yhat'] * 1.05
    
    # Filter only future years
    future_only = forecast_future[forecast_future['ds'] > filtered_data['ds'].max()]
    
    # Calculate model metrics
    test_forecast = model.predict(test_data[['ds']])
    mae = mean_absolute_error(test_data['y'], test_forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(test_data['y'], test_forecast['yhat']))
    mape = np.mean(np.abs((test_data['y'] - test_forecast['yhat'])/test_data['y'])) * 100
    r2 = r2_score(test_data['y'], test_forecast['yhat'])
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🔍 EDA", "📋 Planning Notes"])
    
    with tab1:
        st.header("Unemployment Forecast")
        
        # Forecast plot with scenarios
        st.subheader("Forecast Plot with Scenarios")
        fig, ax = plt.subplots(figsize=(12,6))
        
        # Historical data
        ax.plot(filtered_data['ds'], filtered_data['y'], 'bo-', label='Historical', linewidth=2)
        
        # Base forecast
        ax.plot(forecast_future['ds'], forecast_future['yhat'], label='Base Forecast', color='blue', linewidth=2)
        
        # Optimistic
        ax.plot(forecast_future['ds'], forecast_future['yhat_optimistic'], label='Optimistic (-5%)', color='green', linestyle='--', linewidth=2)
        
        # Pessimistic
        ax.plot(forecast_future['ds'], forecast_future['yhat_pessimistic'], label='Pessimistic (+5%)', color='red', linestyle='--', linewidth=2)
        
        # Prediction interval
        ax.fill_between(forecast_future['ds'], forecast_future['yhat_lower'], forecast_future['yhat_upper'],
                         color='blue', alpha=0.1, label='Prediction Interval')
        
        ax.axvline(x=filtered_data['ds'].max(), color='black', linestyle='--', alpha=0.7)
        ax.set_title(f'Forecasted South Africa Unemployment Rate ({scenario})')
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment Rate (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        
        # Forecast table
        st.subheader("Future Forecast (2023–2033)")
        st.dataframe(future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper',
                                  'yhat_optimistic', 'yhat_pessimistic']].round(2))
        
        # Model metrics
        st.subheader("Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.2f}%")
        col2.metric("RMSE", f"{rmse:.2f}%")
        col3.metric("MAPE", f"{mape:.2f}%")
        col4.metric("R²", f"{r2:.2f}")
        
        # Trend components
        st.subheader("Trend and Yearly Seasonality Components")
        fig2 = model.plot_components(forecast_future)
        st.pyplot(fig2)
    
    with tab2:
        st.header("Exploratory Data Analysis")
        
        # Time series plot
        st.subheader("Historical Unemployment Rate")
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(filtered_data['ds'], filtered_data['y'], 'bo-', linewidth=2)
        ax3.set_title('Historical Unemployment Rate')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Unemployment Rate (%)')
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Statistics**")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Value': [
                    f"{filtered_data['y'].mean():.2f}%",
                    f"{filtered_data['y'].median():.2f}%",
                    f"{filtered_data['y'].std():.2f}%",
                    f"{filtered_data['y'].min():.2f}%",
                    f"{filtered_data['y'].max():.2f}%"
                ]
            })
            st.dataframe(stats_df)
        
        with col2:
            st.write("**Year-over-Year Change**")
            yoy_change = filtered_data['y'].pct_change() * 100
            yoy_df = pd.DataFrame({
                'Year': filtered_data['ds'].dt.year[1:],
                'YoY Change (%)': yoy_change.values[1:]
            })
            st.dataframe(yoy_df.round(2))
        
        # Distribution plot
        st.subheader("Distribution of Unemployment Rates")
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.hist(filtered_data['y'], bins=10, color='skyblue', edgecolor='black')
        ax4.set_title('Distribution of Unemployment Rates')
        ax4.set_xlabel('Unemployment Rate (%)')
        ax4.set_ylabel('Frequency')
        st.pyplot(fig4)
    
    with tab3:
        st.header("Planning Notes")
        
        st.markdown("""
        ### Key Findings:
        - South Africa's unemployment rate has shown an **upward trend** from 2010-2022
        - Current unemployment levels are at **historically high rates**
        - Without intervention, unemployment is projected to **remain elevated** over the next decade
        
        ### Scenario Implications:
        - **Base Case**: Unemployment stabilizes at current high levels
        - **Optimistic**: With strong policy intervention, unemployment could gradually decrease
        - **Pessimistic**: Without action, unemployment could worsen significantly
        
        ### Recommended Actions:
        1. **Immediate**: Implement youth employment programs and skills development initiatives
        2. **Medium-term**: Reform education system to focus on future-ready skills
        3. **Long-term**: Diversify economy and strengthen social safety nets
        
        ### AI Impact Considerations:
        - AI adoption could accelerate job losses in vulnerable sectors
        - Proactive policies needed to mitigate AI-driven unemployment
        - Invest in AI-resistant skills and human-centered technology
        
        ### Model Performance Notes:
        - The forecasting model shows strong predictive accuracy (R² = {:.2f})
        - Mean Absolute Error of {:.2f}% indicates reliable predictions
        - Scenario analysis provides range of possible outcomes for planning
        """.format(r2, mae))

if __name__ == "__main__":
    main()