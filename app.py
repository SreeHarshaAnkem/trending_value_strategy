import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from st_aggrid import AgGrid, GridOptionsBuilder

# Objective function to minimize: sum of squared differences
def objective(w, x, target):
    return (np.dot(x, w) - target) ** 2

# Constraints: w must be positive integers and x_i * w_i >= target/25
def constraint_natural_numbers(w):
    return w - 1  # ensures w are natural numbers (i.e., greater than 0)

def constraint_product(x, w, target):
    return x * w - target // 25  # ensures x_i * w_i >= target/25

# Round the weights to the nearest natural number after optimization
def round_to_natural_numbers(w):
    return np.clip(np.round(w), 1, np.inf).astype(int)

# Solve the equation for w1, ..., w25 given x1, ..., x25 with constraints
def find_weights(x, target):
    w0 = np.ones(len(x))  # Initial guess for weights
    bounds = [(1, None)] * len(x)  # Bounds for weights

    cons = [
        {'type': 'ineq', 'fun': constraint_natural_numbers},  # Natural numbers constraint
        {'type': 'ineq', 'fun': lambda w: constraint_product(x, w, target)}  # x_i * w_i >= target/25
    ]

    result = minimize(objective, w0, args=(x, target), bounds=bounds, constraints=cons)
    w_natural = round_to_natural_numbers(result.x)
    return w_natural

# Streamlit UI
st.title('Trending Value Momentum Strategy')

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Let the user select columns for decile calculations
    all_columns = df.columns.tolist()
    decile_columns = st.multiselect("Select decile columns", options=all_columns, default=[
        'PE Ratio', 'PB Ratio', 'Price / CFO', 'PS Ratio', 'EV/EBITDA Ratio', 'Dividend Yield'])

    # Add a slider to filter by Market Cap
    min_market_cap = int(df["Market Cap"].min())
    max_market_cap = int(df["Market Cap"].max())


    # Filter dataframe by selected market cap
    df = df.loc[(df["Market Cap"] >= 500)]

    # Data preprocessing
    for col in df.columns:
        if col not in ["Name", "Ticker", "Sub-Sector"]:
            df[col] = df[col].astype("float")
    df = df.loc[~df["PE Ratio"].isnull()]

    # Decile calculation
    for col in decile_columns:
        mask = (df[col].isnull()) | (df[col] < 0)
        if col != "Dividend Yield":
            decile = pd.qcut(df.loc[~mask, col], q=10, labels=np.arange(1, 11))
        else:
            decile = pd.qcut(df.loc[~mask, col], q=10, labels=np.arange(10, 0, -1))
        df.loc[~mask, col] = decile.astype("float")
        df.loc[mask, col] = 10
    
    df["composite_score"] = df.loc[:, decile_columns].sum(axis=1)
    df["composite_decile"] = pd.qcut(df["composite_score"], q=10, labels=np.arange(0, 10))
    
    df_value = df.loc[df["composite_decile"] == 0]
    df_value_momentum = df_value.sort_values(by=["6M Return"], ascending=False).head(25)

    # Get user input for lumpsum investment amount
    target = st.number_input("Enter your lumpsum investment amount", value=50000)

    # Optimize weights
    df_value_momentum["qty"] = find_weights(df_value_momentum["Close Price"], target=target)

    # Add "Allocated Amount" column based on qty * Close Price
    df_value_momentum["Allocated Amount"] = df_value_momentum["qty"] * df_value_momentum["Close Price"]

    # Create AgGrid configuration to freeze the first column
    gb = GridOptionsBuilder.from_dataframe(df_value_momentum)
    gb.configure_pagination(enabled=True)
    gb.configure_side_bar()  # Enables sidebar (filters, etc.)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=False)
    gb.configure_column('Ticker', pinned='left')  # Freeze the Ticker column
    grid_options = gb.build()

    # Display AgGrid with frozen first column
    st.subheader("Optimized Portfolio")
    AgGrid(df_value_momentum, gridOptions=grid_options, enable_enterprise_modules=True)

    # Display total allocation in footer
    total_allocation = df_value_momentum["Allocated Amount"].sum()
    st.markdown(f"**Total Allocation: ${total_allocation:,.2f}**")

    # Display sector-wise counts
    df_sectorwise_count = df_value_momentum.groupby("Sub-Sector").agg({"Ticker": "nunique", "Allocated Amount":"sum"}).sort_values(by="Allocated Amount", ascending=False)
    st.subheader("Sector-wise Distribution")
    st.dataframe(df_sectorwise_count)

    # Add disclaimer
    st.markdown("""
    **Disclaimer**: 
    It is recommended to rebalance your portfolio semi-annually or quarterly for more aggressive strategies.
    """)
