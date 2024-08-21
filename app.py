import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("Forecasting Tool")
st.write("---")
st.sidebar.header("Input Parameters")
uploaded_file = st.sidebar.file_uploader("Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df)
    st.write("---")
    st.header('Select Columns')

    x_col = st.selectbox('Select X-axis Column', df.columns)
    y_col = st.selectbox('Select Y-axis Column', df.columns)
    st.write("---")
    
    placeholder = st.empty()
    
    if x_col != y_col:
        ### Block 1

        if pd.api.types.is_datetime64_any_dtype(df[x_col]) or pd.api.types.is_timedelta64_dtype(df[x_col]):
            df[x_col] = pd.to_datetime(df[x_col])
        else:
            df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
    
        st.header(f"{y_col} over {x_col}")
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x=x_col, y=y_col, marker='o', color='blue')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{y_col} Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
        placeholder.empty()
        st.write("---")

        ### Block 2

        st.header("Rolling Mean")

        window_size = st.number_input("Select Window", min_value=1, value=12)

        bl2 = pd.DataFrame(index=df.index)
        
        bl2['Rolling_Mean'] = df[y_col].rolling(window=window_size).mean()
        bl2['Rolling_Std'] = df[y_col].rolling(window=window_size).std()
    
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.lineplot(data=df, x=df.index, y=y_col, label=y_col, color='blue', ax=ax)
        sns.lineplot(data=bl2, x=bl2.index, y='Rolling_Mean', label=f'{window_size}-Month Rolling Mean', color='red', ax=ax)
        ax.fill_between(bl2.index, bl2['Rolling_Mean'] - bl2['Rolling_Std'], bl2['Rolling_Mean'] + bl2['Rolling_Std'], color='gray', alpha=0.3, label=f'{window_size}-Month Rolling Std')
        
        ax.set_xlabel('Date')
        ax.set_ylabel(y_col)
        ax.set_title('Sales with Rolling Mean and Standard Deviation')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        st.write("---")

        ### Block 3

        st.header("Decomposition (Separating Trend, Seasonality, and Residuals)")

        df[x_col] = pd.to_datetime(df[x_col])
        df.set_index(x_col, inplace=True)
        df = df[[y_col]].copy()
        df = df.rename(columns={y_col: y_col})
        df_resampled = df.resample('M').sum()

        stl = STL(df_resampled[y_col], seasonal=13)
        result = stl.fit()

        fig, axs = plt.subplots(4, 1, figsize=(12, 8), dpi=200)
        
        axs[0].plot(df_resampled.index, result.observed)
        axs[0].set_title('Observed')
        
        axs[1].plot(df_resampled.index, result.trend)
        axs[1].set_title('Trend')
        
        axs[2].plot(df_resampled.index, result.seasonal)
        axs[2].set_title('Seasonality')
        
        axs[3].plot(df_resampled.index, result.resid)
        axs[3].set_title('Residuals')

        plt.tight_layout()
        st.pyplot(fig)
        st.write("---")

        ### Block 4

        st.header('Seasonal Plots: Comparing Sales Across Different Time Periods')

        bl4 = df.copy()
        bl4['Month'] = bl4.index.month
        bl4['Year'] = bl4.index.year

        plt.figure(figsize=(12, 6), dpi=200)
        sns.lineplot(data=bl4, x='Month', y=y_col, hue='Year', marker='o', palette='tab10')

        plt.xlabel('Month')
        plt.ylabel(y_col)
        plt.title('Monthly Sales Across Different Years')
        plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        st.pyplot(plt)
        st.write("---")

        ### Block 5

        st.header('Stationarity Check - Dickey-Fuller Test')

        if y_col in df.columns:
            adf_result = adfuller(df[y_col])
            
            adf_statistic = adf_result[0]
            p_value = adf_result[1]
            
            st.markdown(f"""
            <p style="font-size: 20px;"><strong>ADF Statistic:</strong> {adf_statistic}</p>
            <p style="font-size: 20px;"><strong>p-value:</strong> {p_value}</p>
            """, unsafe_allow_html=True)

            if p_value <= 0.05:
                conclusion = ("The time series is stationary, meaning it has no unit root, "
                            "and its statistical properties such as mean, variance, and autocorrelation are constant over time. "
                            "This is a key assumption for many time series models.")
            else:
                conclusion = ("The time series is non-stationary, indicating the presence of a unit root. "
                            "This implies that the time series' statistical properties change over time, "
                            "which may affect the performance of certain time series models. "
                            "Differencing or other transformations may be needed to achieve stationarity.")
            
            st.markdown(f"<p style='font-size: 20px;'><strong>Conclusion:</strong> {conclusion}</p>", unsafe_allow_html=True)

            if p_value >= 0.05:
                apply_log_transform = st.checkbox("Do you want to apply First-Order Differencing?", value=False)
            
                if apply_log_transform:
                    df[f'{y_col}_diff'] = df[y_col].diff().dropna()
                    y_col_diff = f'{y_col}_diff'
                    
                    adf_result_diff = adfuller(df[y_col_diff].dropna())
                    
                    adf_statistic_diff = adf_result_diff[0]
                    p_value_diff = adf_result_diff[1]
                    
                    st.markdown(f"""
                    <p style="font-size: 20px;"><strong>ADF Statistic after First-Order Differencing:</strong> {adf_statistic_diff}</p>
                    <p style="font-size: 20px;"><strong>p-value after First-Order Differencing:</strong> {p_value_diff}</p>
                    """, unsafe_allow_html=True)

                    if p_value_diff <= 0.05:
                        df[f'{y_col}_diff'] = df[y_col].diff()
                        conclusion_diff = ("After applying First-Order Differencing, the time series is now stationary.")
                    else:
                        conclusion_diff = ("Even after differencing, the time series remains non-stationary.")
                    
                    st.markdown(f"<p style='font-size: 20px;'><strong>Conclusion after Differencing:</strong> {conclusion_diff}</p>", unsafe_allow_html=True)
        else:
            st.write(f"The {y_col} column is not available in the dataset.")
        
        st.write("---")

        ### Block 6
        
        st.header('ACF and PACF')

        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')

        i = st.number_input("Enter the differencing order (I) for ARIMA: ", min_value=0, value=0)

        if i > 0:
            df[f'{y_col}_diff'] = df[y_col].diff().dropna()
            differenced_column = f'{y_col}_diff'
        else:
            differenced_column = y_col

        max_lags = st.number_input("Select the number of lags for ACF/PACF:", min_value=1, max_value=min(len(df) // 2 - 1, 100), value=12)

        fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=300)

        plot_acf(df[differenced_column].dropna(), ax=ax[0], lags=max_lags)
        ax[0].set_title('ACF Plot')

        plot_pacf(df[differenced_column].dropna(), ax=ax[1], lags=max_lags)
        ax[1].set_title('PACF Plot')

        plt.tight_layout()

        st.pyplot(fig)
        st.write("---")

        ### Block 7

        st.header('Select forecast steps')

        forecast_steps = st.number_input("Enter the the forcast steps number: ", min_value=1, value=12)

        st.write("---")

        ### Block 8

        train_df = df.iloc[:len(df) - forecast_steps]
        test_df = df.iloc[len(df) - forecast_steps:forecast_steps + len(df)]

        st.header('Train Data and Test Data')

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(train_df.index, train_df[y_col], label='Training Data')
        ax.plot(test_df.index, test_df[y_col], label='Test Data', color='orange')
        vertical_line_position = train_df.index[-1]
        ax.axvline(x=vertical_line_position, color='red', linestyle='--', label='Train-Test Split')
        ax.set_xlabel('Date')
        ax.set_ylabel(y_col)
        ax.set_title('Training and Test Set Split')
        ax.legend()

        st.pyplot(fig)
        st.write("---")

        ### Block 9
        
        st.header("ARIMA Model")

        p = st.number_input("Enter the autoregressive order (p) for ARIMA: ", min_value=0, value=1)
        d = st.number_input("Enter the differencing order (d) for ARIMA: ", min_value=0, value=0)
        q = st.number_input("Enter the moving average order (q) for ARIMA: ", min_value=0, value=1)
        frequency_options = ['D', 'M', 'W', 'Y']
        freq_code = st.selectbox("Select Frequency:", frequency_options)

        arima_model = ARIMA(train_df[y_col], order=(p, d, q))
        arima_results = arima_model.fit()

        forecast = arima_results.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()
        forecast_index = pd.date_range(start=train_df.index[-1] + pd.DateOffset(1), periods=forecast_steps, freq=freq_code)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df[y_col], label='Historical Sales')
        ax.plot(forecast_index, forecast_mean, label='Forecast', color='red')
        ax.fill_between(forecast_index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
        ax.set_xlim(df.index.min(), df.index.max())
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title('Forecast with Confidence Intervals')
        ax.legend()

        st.pyplot(fig)

        mae = mean_absolute_error(test_df[y_col], forecast_mean[:len(test_df)])
        mse = mean_squared_error(test_df[y_col], forecast_mean[:len(test_df)])
        rmse = np.sqrt(mse)
        r2 = r2_score(test_df[y_col], forecast_mean[:len(test_df)])

        st.markdown(f"""
        <h2 style='font-size:20px;'>Mean Absolute Error (MAE): {mae:.2f}</h2>
        <h2 style='font-size:20px;'>Mean Squared Error (MSE): {mse:.2f}</h2>
        <h2 style='font-size:20px;'>Root Mean Squared Error (RMSE): {rmse:.2f}</h2>
        <h2 style='font-size:20px;'>R-squared (R²): {r2:.2f}</h2>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <h2 style='font-size:20px;'>Test Data</h2>
            """, unsafe_allow_html=True)

            st.dataframe(test_df)

        with col2:
            st.markdown(f"""
            <h2 style='font-size:20px;'>Predicted Data</h2>
            """, unsafe_allow_html=True)

            forecast_df = pd.DataFrame({
                'Forecasted_mean': forecast_mean
            })

            st.dataframe(forecast_df)

        st.write("---")
        
        ### Block 10

        st.header("SARIMA Model")

        ps = 0 
        ds = 0 
        qs = 0 
        Ps_ = st.number_input("Enter the seasonal autoregressive order (P): ", min_value=0, value=1)
        Ds_ = st.number_input("Enter the seasonal differencing order (D): ", min_value=0, value=0)
        Qs_ = st.number_input("Enter the seasonal moving average order (Q): ", min_value=0, value=1)
        ss = st.number_input("Enter the seasonal period (s): ", min_value=1, value=12)

        frequency_options = ['D', 'M', 'W', 'Y']
        freq_code_sarima = st.selectbox("Select Frequency:", frequency_options, key="freq_code_sarima")
        sarima_model = SARIMAX(train_df[y_col], order=(ps, ds, qs), seasonal_order=(Ps_, Ds_, Qs_, ss))
        sarima_results = sarima_model.fit()

        forecast = sarima_results.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()
        forecast_index = pd.date_range(start=train_df.index[-1] + pd.DateOffset(1), periods=forecast_steps, freq=freq_code_sarima)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df[y_col], label='Historical Sales')
        ax.plot(forecast_index, forecast_mean, label='Forecast', color='red')
        ax.fill_between(forecast_index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
        ax.set_xlim(df.index.min(), df.index.max())
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title('Forecast with Confidence Intervals')
        ax.legend()

        st.pyplot(fig)

        mae = mean_absolute_error(test_df[y_col], forecast_mean[:len(test_df)])
        mse = mean_squared_error(test_df[y_col], forecast_mean[:len(test_df)])
        rmse = np.sqrt(mse)
        r2 = r2_score(test_df[y_col], forecast_mean[:len(test_df)])

        st.markdown(f"""
        <h2 style='font-size:20px;'>Mean Absolute Error (MAE): {mae:.2f}</h2>
        <h2 style='font-size:20px;'>Mean Squared Error (MSE): {mse:.2f}</h2>
        <h2 style='font-size:20px;'>Root Mean Squared Error (RMSE): {rmse:.2f}</h2>
        <h2 style='font-size:20px;'>R-squared (R²): {r2:.2f}</h2>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <h2 style='font-size:20px;'>Test Data</h2>
            """, unsafe_allow_html=True)

            st.dataframe(test_df)

        with col2: 
            st.markdown(f"""
            <h2 style='font-size:20px;'>Predicted Data</h2>
            """, unsafe_allow_html=True)

            forecast_df = pd.DataFrame({
                'Forecasted_mean': forecast_mean
            })

            st.dataframe(forecast_df)

        st.write("---")

        ### Block 11

        st.header("FB-Prophet Model")

        df_prophet = df.reset_index()
        df_prophet = df_prophet[[x_col, y_col]]
        df_prophet.columns = ['ds', 'y']

        train_prophet = df_prophet.iloc[:len(df) - forecast_steps]
        test_prophet = df_prophet.iloc[len(df) - forecast_steps:]

        model = Prophet()
        model.fit(train_prophet)

        future = test_prophet[['ds']]
        forecast = model.predict(future)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df[y_col], label='Historical Sales')
        ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='pink', alpha=0.3)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title('Prophet Forecast with Confidence Intervals')
        ax.legend()

        st.pyplot(fig)

        actual = test_prophet['y'].values
        forecasted = forecast['yhat'].values

        mae = mean_absolute_error(actual, forecasted)
        mse = mean_squared_error(actual, forecasted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, forecasted)

        st.markdown(f"""
            <h2 style='font-size:20px;'>Mean Absolute Error (MAE): {mae:.2f}</h2>
            <h2 style='font-size:20px;'>Mean Squared Error (MSE): {mse:.2f}</h2>
            <h2 style='font-size:20px;'>Root Mean Squared Error (RMSE): {rmse:.2f}</h2>
            <h2 style='font-size:20px;'>R-squared (R²): {r2:.2f}</h2>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <h2 style='font-size:20px;'>Test Data</h2>
            """, unsafe_allow_html=True)

            st.dataframe(test_df)

        with col2: 
            st.markdown(f"""
            <h2 style='font-size:20px;'>Predicted Data</h2>
            """, unsafe_allow_html=True)

            forecast_df = pd.DataFrame({
                'Forecasted_mean': forecasted
            })

            forecast_df.index = test_df.index

            st.dataframe(forecast_df)

        st.write("---")
else:
    st.markdown(f"""
            <h2 style='font-size:20px;'>Please upload a dataset to get started. Ensure your dataset is cleaned and formatted correctly, with columns suitable for time series analysis.</h2>
            """, unsafe_allow_html=True)
