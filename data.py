import pandas as pd
import numpy as np

def generate_big_synthetic_data(output_csv='big_data_2014_2023.csv'):
    """
    Generates a large synthetic dataset (daily data from 2014-01-01 to 2023-12-31)
    with a strongly correlated 'Gross_Revenue' target. Saves the result to CSV.
    """

    # 1. Create a daily date range
    date_range = pd.date_range(start='2014-01-01', end='2023-12-31', freq='D')
    n = len(date_range)  # ~3652 days

    np.random.seed(42)  # For reproducibility

    # 2. Generate synthetic predictors

    # (A) Net_Gas_Price: random but slightly trending upward
    # We'll add a small trend from 3.0 to 8.0 across the time range
    net_gas_price = np.linspace(3.0, 8.0, n) + np.random.normal(0, 0.3, n)

    # (B) Corn_Price: random in 3 to 6, with small random variation
    corn_price = np.random.uniform(3.0, 6.0, n) + np.sin(np.linspace(0, 6*np.pi, n)) * 0.2

    # (C) CPI: from about 120 to 140 across the 10 years, plus minor noise
    cpi_trend = np.linspace(120, 140, n)
    cpi_noise = np.random.normal(0, 1.0, n)
    cpi = cpi_trend + cpi_noise

    # (D) Exchange_Rate_JPY_USD: random in ~ [90, 130], plus mild sinusoidal variation
    exch_base = np.random.uniform(90, 130, n)
    exch_variation = 5.0 * np.sin(np.linspace(0, 4*np.pi, n))
    exchange_rate = exch_base + exch_variation

    # 3. Construct a "true" relationship for Gross_Revenue
    #    We'll make a strong linear combination so the model can easily achieve ~95% R²
    #    + some mild seasonality + small random noise

    # Time-based index for mild trend from 0 to ~300 across n days
    time_index = np.linspace(0, 1, n)  # normalized from 0 to 1
    mild_trend = 300 * time_index      # up to +300 across the time range

    # Seasonality (annual sinusoid ~ amplitude 100)
    # Each year has ~365 days => 2π * t / 365
    seasonality = 100 * np.sin(2 * np.pi * np.arange(n) / 365)

    # We'll define a "true" formula for Gross_Revenue:
    #   GR = 1000 + 0.5 * net_gas_price + 30 * corn_price + 2 * (cpi - 120)
    #         - 0.2 * (exchange_rate - 110) + mild_trend + seasonality + small noise
    # This yields a strong correlation with each predictor.
    noise = np.random.normal(0, 30, n)  # standard dev 30
    gross_revenue = (
        1000
        + 0.5 * net_gas_price
        + 30.0 * corn_price
        + 2.0 * (cpi - 120)
        - 0.2 * (exchange_rate - 110)
        + mild_trend
        + seasonality
        + noise
    )

    # 4. Create DataFrame
    df = pd.DataFrame({
        'Date': date_range,
        'Gross_Revenue': gross_revenue,
        'Net_Gas_Price': net_gas_price,
        'Corn_Price': corn_price,
        'CPI': cpi,
        'Exchange_Rate_JPY_USD': exchange_rate
    })

    # 5. Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Synthetic dataset saved to {output_csv} with {n} rows.")

if __name__ == "__main__":
    generate_big_synthetic_data()
