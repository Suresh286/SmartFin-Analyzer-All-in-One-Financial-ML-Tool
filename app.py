import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import random
import pandas as pd
import yfinance as yf
from datetime import date
import time
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

# Set Streamlit dark theme
st.set_page_config(page_title="SmartFin Analyzer", page_icon="💹", layout="wide", initial_sidebar_state="expanded")

# Lottie animation support
try:
    from streamlit_lottie import st_lottie
except ImportError:
    st.warning("Install streamlit-lottie for animated graphics: pip install streamlit-lottie")
    st_lottie = None
import json

# --- Finance GIF for Home Page ---
def get_finance_gif():
    url = "https://media.giphy.com/media/3o7TKtnuHOHHUjR38Y/giphy.gif"  # Finance-themed GIF
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# --- Sidebar Navigation ---
option = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["🏠 Home", "📈 Stock Price Prediction", "📊 Movement Classification", "🔍 Investor Clustering"]
)

# --- Sidebar Logo & Help ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2331/2331943.png", width=60)
    st.markdown("<h4 style='color:#00c7b6;'>SmartFin Analyzer</h4>", unsafe_allow_html=True)
    with st.expander("ℹ️ Help & About"):
        st.markdown("""
        **SmartFin Analyzer** is an all-in-one financial ML tool for:
        - Stock price prediction
        - Stock movement classification
        - Investor segmentation
        
        **How to use:**
        1. Select a module from the sidebar
        2. Follow the step-by-step workflow
        3. Download your results
        
        [GitHub](https://github.com/) | [Contact](mailto:your@email.com)
        """)

# --- Home Page ---
def get_random_finance_quote():
    quotes = [
        ("An investment in knowledge pays the best interest.", "Benjamin Franklin"),
        ("The stock market is filled with individuals who know the price of everything, but the value of nothing.", "Philip Fisher"),
        ("Risk comes from not knowing what you are doing.", "Warren Buffett"),
        ("In investing, what is comfortable is rarely profitable.", "Robert Arnott"),
        ("The four most dangerous words in investing are: 'this time it's different.'", "Sir John Templeton"),
        ("Know what you own, and know why you own it.", "Peter Lynch"),
    ]
    return random.choice(quotes)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def home_page():
    st.title("💹 SmartFin Analyzer – An All-in-One Financial ML Tool")
    st.markdown("""
    <h3 style='color:#00c7b6;'>Your Gateway to Smarter Financial Decisions</h3>
    """, unsafe_allow_html=True)

    # --- Stock/Finance Lottie Animation ---
    lottie_url = "https://assets10.lottiefiles.com/packages/lf20_2glqweqs.json"  # Modern stock chart animation
    lottie_json = load_lottieurl(lottie_url)
    if st_lottie and lottie_json:
        st_lottie(lottie_json, height=220, key="stock_lottie")
    else:
        st.info("[Animated stock chart would appear here if streamlit-lottie is installed]")

    # --- Finance Emoji Bar ---
    st.markdown("""
    <div style='font-size:2rem; text-align:center;'>💹 💰 📊 📈 🏦 💵</div>
    """, unsafe_allow_html=True)

    # --- Custom Divider with Icon ---
    st.markdown("""
    <hr style='border: none; border-top: 2px solid #00c7b6; margin: 1em 0; position: relative;'>
    <div style='text-align:center; margin-top:-2.5em; margin-bottom:1em;'>
        <span style='background:#0e1117; padding:0 1em; color:#00c7b6; font-size:1.5rem;'>💡</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Quote of the Day ---
    quote, author = get_random_finance_quote()
    st.markdown(f"<div style='text-align:center; font-style:italic; color:#b2dfdb;'>\"{quote}\"<br><span style='font-size:1rem;'>– {author}</span></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        Welcome to **SmartFin Analyzer**! This tool lets you:
        - <b>Predict stock prices</b> with advanced regression models
        - <b>Classify stock movements</b> using technical indicators
        - <b>Segment investors</b> for personalized strategies
        <br>
        <span style='color:#00c7b6;'>Select a module from the sidebar to get started.</span>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📦 Modules Overview")
        st.markdown("""
        - <b>📈 Stock Price Prediction:</b> Upload or fetch stock data, select features, and predict future prices.
        - <b>📊 Movement Classification:</b> Use technical indicators to classify price movements as up or down.
        - <b>🔍 Investor Clustering:</b> Upload portfolio data and discover your investor type.
        """, unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🚀 Get Started"):
            st.sidebar.success("Choose a module from the sidebar!")
        with st.expander("❓ How to Use This App"):
            st.markdown("""
            1. Select a module from the sidebar.
            2. Upload your data or fetch it using the provided tools.
            3. Follow the step-by-step instructions for each analysis.
            4. Download your results and visualizations.
            """)
        # --- Sample Data Download ---
        sample_csv = """Date,Close,Volume\n2023-01-01,100,10000\n2023-01-02,102,12000\n2023-01-03,101,11000\n"""
        st.download_button(
            label="📥 Download Sample Stock Data",
            data=sample_csv,
            file_name="sample_stock_data.csv",
            mime="text/csv"
        )
        st.info("Pro Tip: You can always return to this home page from the sidebar.")
    with col2:
        st.image(get_finance_gif(), caption="Finance in Motion!", use_container_width=True)
    st.markdown("---")
    st.markdown("<center><small>SmartFin Analyzer v1.0 | Made with ❤️ using Streamlit</small></center>", unsafe_allow_html=True)

# --- Module Stubs ---
def stock_price_prediction():
    st.header("📈 Stock Price Prediction (Linear Regression)")
    step = st.session_state.get('sp_step', 1)
    st.markdown(f"<b>Step {step} of 3</b>", unsafe_allow_html=True)
    st.markdown("""
    <span style='color:#00c7b6;'>Step 1: Load your stock data (CSV upload, Yahoo Finance, or Finnhub)</span>
    """, unsafe_allow_html=True)

    # Step 1: Data Upload or Fetch
    data_source = st.radio("Select Data Source:", ["Upload CSV", "Fetch with yfinance", "Fetch with Finnhub"], horizontal=True)
    df = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your historical stock data (CSV)", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
    elif data_source == "Fetch with yfinance":
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)")
        period = st.selectbox("Select Period", ["1y", "2y", "5y", "max"], index=0)
        if st.button("Fetch Data (yfinance)") and ticker:
            df = yf.download(ticker, period=period)
            df.reset_index(inplace=True)
    elif data_source == "Fetch with Finnhub":
        st.markdown("""
        <small>Get a free API key at <a href='https://finnhub.io/register' target='_blank'>finnhub.io</a></small>
        """, unsafe_allow_html=True)
        finnhub_api_key = st.text_input("Enter your Finnhub API Key", type="password")
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", key="finnhub_ticker")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date(2023, 1, 1), key="finnhub_start")
        with col2:
            end_date = st.date_input("End Date", value=date.today(), key="finnhub_end")
        if st.button("Fetch Data (Finnhub)"):
            if not finnhub_api_key:
                st.warning("Please enter your Finnhub API key.")
            elif not ticker:
                st.warning("Please enter a stock ticker.")
            else:
                start_unix = int(time.mktime(start_date.timetuple()))
                end_unix = int(time.mktime(end_date.timetuple()))
                url = f"https://finnhub.io/api/v1/stock/candle?symbol={ticker.upper()}&resolution=D&from={start_unix}&to={end_unix}&token={finnhub_api_key}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("s") == "ok":
                        df = pd.DataFrame({
                            "Date": pd.to_datetime(data["t"], unit="s"),
                            "Open": data["o"],
                            "High": data["h"],
                            "Low": data["l"],
                            "Close": data["c"],
                            "Volume": data["v"]
                        })
                    else:
                        st.warning("No data found for this ticker and date range.")
                else:
                    st.warning("Failed to fetch data from Finnhub. Check your API key and ticker.")

    # Data Preview and Warning
    if df is not None and not df.empty:
        target_col = 'Close' if 'Close' in df.columns else st.session_state.get('sp_target_col_widget')
        if not target_col:
            st.warning(f"Your data does not contain a 'Close' column. Please select the target column for prediction.")
            numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
            target_col = st.selectbox("Select target column for prediction:", options=numeric_cols, key="sp_target_col_widget")
            if not target_col:
                st.error("Please select a target column to proceed.")
                return
        st.success(f"Data loaded successfully! Using '{target_col}' as the target column.")
        st.dataframe(df.head())
        # Step 2: Feature Selection
        st.markdown("""
        <span style='color:#00c7b6;'>Step 2: Select Features for Prediction</span>
        """, unsafe_allow_html=True)
        default_features = [col for col in df.columns if col not in ["Date", "Adj Close", target_col]]
        features = st.multiselect(
            "Select features:",
            options=[col for col in df.columns if col not in ["Date", "Adj Close", target_col]],
            default=default_features[:2]
        )
        # Option to add lag features and moving average
        add_lag = st.checkbox("Include Previous Day (Lag Feature)", value=True)
        add_ma = st.checkbox("Include 5-Day Moving Average", value=True)
        if len(features) < 1:
            st.warning("Please select at least one feature.")
        if st.button("Next: Train Model"):
            st.session_state['sp_df'] = df
            st.session_state['sp_features'] = features
            st.session_state['sp_add_lag'] = add_lag
            st.session_state['sp_add_ma'] = add_ma
            st.session_state['sp_step'] = 2
    else:
        st.warning("Please upload a CSV file or fetch data to proceed.")

    # Step 2: Preprocessing, Model Training, Results
    if st.session_state.get('sp_step', 1) == 2:
        st.markdown("""
        <span style='color:#00c7b6;'>Step 3: Model Training & Results</span>
        """, unsafe_allow_html=True)
        df = st.session_state['sp_df'].copy()
        features = st.session_state['sp_features']
        add_lag = st.session_state['sp_add_lag']
        add_ma = st.session_state['sp_add_ma']
        target_col = 'Close' if 'Close' in df.columns else st.session_state.get('sp_target_col_widget')
        # Preprocessing
        if 'Date' in df.columns:
            df = df.sort_values('Date')
        if add_lag and target_col in df.columns:
            df['Prev_Target'] = df[target_col].shift(1)
            features = features + ['Prev_Target'] if 'Prev_Target' not in features else features
        if add_ma and target_col in df.columns:
            df['MA_5'] = df[target_col].rolling(window=5).mean()
            features = features + ['MA_5'] if 'MA_5' not in features else features
        df = df.dropna()
        # Model
        X = df[features]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.success(f"Model trained! MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
        st.image(get_finance_gif(), width=200)
        # Visualization
        fig = go.Figure()
        x_axis = df.iloc[y_test.index]['Date'] if 'Date' in df.columns else y_test.index
        fig.add_trace(go.Scatter(x=x_axis, y=y_test, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=x_axis, y=y_pred, mode='lines', name='Predicted'))
        fig.update_layout(title=f'Actual vs Predicted {target_col}', xaxis_title='Date', yaxis_title=target_col, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        # Download predictions
        results_df = df.iloc[y_test.index].copy()
        results_df[f'Predicted_{target_col}'] = y_pred
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=results_df.to_csv(index=False),
            file_name="stock_predictions.csv",
            mime="text/csv"
        )
        # Navigation
        if st.button("⬅️ Back to Feature Selection"):
            st.session_state['sp_step'] = 1
    if st.button("🔄 Reset Module", key="sp_reset"):
        for k in list(st.session_state.keys()):
            if k.startswith('sp_'):
                del st.session_state[k]
        st.rerun()

def stock_movement_classification():
    st.header("📊 Stock Movement Classification (Logistic Regression)")
    step = st.session_state.get('smc_step', 1)
    st.markdown(f"<b>Step {step} of 3</b>", unsafe_allow_html=True)
    st.markdown("""
    <span style='color:#00c7b6;'>Step 1: Load your stock data (CSV upload, Yahoo Finance, or Finnhub)</span>
    """, unsafe_allow_html=True)

    # Step 1: Data Upload or Fetch (reuse logic)
    data_source = st.radio("Select Data Source:", ["Upload CSV", "Fetch with yfinance", "Fetch with Finnhub"], horizontal=True, key="smc_source")
    df = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your historical stock data (CSV)", type=["csv"], key="smc_upload")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
    elif data_source == "Fetch with yfinance":
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", key="smc_yf_ticker")
        period = st.selectbox("Select Period", ["1y", "2y", "5y", "max"], index=0, key="smc_yf_period")
        if st.button("Fetch Data (yfinance)", key="smc_yf_btn") and ticker:
            df = yf.download(ticker, period=period)
            df.reset_index(inplace=True)
    elif data_source == "Fetch with Finnhub":
        st.markdown("""
        <small>Get a free API key at <a href='https://finnhub.io/register' target='_blank'>finnhub.io</a></small>
        """, unsafe_allow_html=True)
        finnhub_api_key = st.text_input("Enter your Finnhub API Key", type="password", key="smc_finnhub_key")
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", key="smc_finnhub_ticker")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date(2023, 1, 1), key="smc_finnhub_start")
        with col2:
            end_date = st.date_input("End Date", value=date.today(), key="smc_finnhub_end")
        if st.button("Fetch Data (Finnhub)", key="smc_finnhub_btn"):
            if not finnhub_api_key:
                st.warning("Please enter your Finnhub API key.")
            elif not ticker:
                st.warning("Please enter a stock ticker.")
            else:
                start_unix = int(time.mktime(start_date.timetuple()))
                end_unix = int(time.mktime(end_date.timetuple()))
                url = f"https://finnhub.io/api/v1/stock/candle?symbol={ticker.upper()}&resolution=D&from={start_unix}&to={end_unix}&token={finnhub_api_key}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("s") == "ok":
                        df = pd.DataFrame({
                            "Date": pd.to_datetime(data["t"], unit="s"),
                            "Open": data["o"],
                            "High": data["h"],
                            "Low": data["l"],
                            "Close": data["c"],
                            "Volume": data["v"]
                        })
                    else:
                        st.warning("No data found for this ticker and date range.")
                else:
                    st.warning("Failed to fetch data from Finnhub. Check your API key and ticker.")

    # Data Preview and Warning
    if df is not None and not df.empty:
        target_col = 'Close' if 'Close' in df.columns else st.session_state.get('smc_target_col_widget')
        if not target_col:
            st.warning(f"Your data does not contain a 'Close' column. Please select the target column for classification.")
            numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
            target_col = st.selectbox("Select target column for classification:", options=numeric_cols, key="smc_target_col_widget")
            if not target_col:
                st.error("Please select a target column to proceed.")
                return
        st.success(f"Data loaded successfully! Using '{target_col}' as the target column.")
        st.dataframe(df.head())
        # Step 2: Calculate technical indicators and label target
        st.markdown("""
        <span style='color:#00c7b6;'>Step 2: Calculate Technical Indicators & Label Target</span>
        """, unsafe_allow_html=True)
        df = df.sort_values('Date') if 'Date' in df.columns else df
        # Return %
        df['Return_%'] = df[target_col].pct_change() * 100
        # RSI
        def compute_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        df['RSI'] = compute_rsi(df[target_col])
        # MACD
        exp12 = df[target_col].ewm(span=12, adjust=False).mean()
        exp26 = df[target_col].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        # Target label: 1 if next day target_col > today, else 0
        df['Target'] = (df[target_col].shift(-1) > df[target_col]).astype(int)
        df = df.dropna()
        st.dataframe(df[['Date', target_col, 'Return_%', 'RSI', 'MACD', 'Target']].head())
        # Feature selection
        features = st.multiselect(
            "Select features for classification:",
            options=['Return_%', 'RSI', 'MACD'],
            default=['Return_%', 'RSI', 'MACD']
        )
        if len(features) < 1:
            st.warning("Please select at least one feature.")
        # Navigation button to next step (model)
        if st.button("Next: Train Classifier"):
            st.session_state['smc_df'] = df
            st.session_state['smc_features'] = features
            st.session_state['smc_step'] = 2
    else:
        st.warning("Please upload a CSV file or fetch data to proceed.")

    # Step 2: Model Training, Evaluation, Visualization
    if st.session_state.get('smc_step', 1) == 2:
        st.markdown("""
        <span style='color:#00c7b6;'>Step 3: Model Training & Results</span>
        """, unsafe_allow_html=True)
        df = st.session_state['smc_df'].copy()
        features = st.session_state['smc_features']
        target_col = 'Close' if 'Close' in df.columns else st.session_state.get('smc_target_col_widget')
        # Model
        X = df[features]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Model trained! Accuracy: {acc:.2%}")
        st.image(get_finance_gif(), width=200)
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = ff.create_annotated_heatmap(cm, x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'], colorscale='blues', showscale=True)
        fig_cm.update_layout(title_text='Confusion Matrix', template='plotly_dark')
        st.plotly_chart(fig_cm, use_container_width=True)
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve (AUC = {:.2f})'.format(roc_auc)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', template='plotly_dark')
        st.plotly_chart(fig_roc, use_container_width=True)
        # Download results
        results_df = df.iloc[y_test.index].copy()
        results_df['Predicted'] = y_pred
        results_df['Probability'] = y_prob
        st.download_button(
            label="📥 Download Classification Results as CSV",
            data=results_df.to_csv(index=False),
            file_name="stock_classification_results.csv",
            mime="text/csv"
        )
        # Navigation
        if st.button("⬅️ Back to Feature Selection", key="smc_back"):
            st.session_state['smc_step'] = 1
    if st.button("🔄 Reset Module", key="smc_reset"):
        for k in list(st.session_state.keys()):
            if k.startswith('smc_'):
                del st.session_state[k]
        st.rerun()

def investor_segmentation():
    st.header("🔍 Investor Segmentation (K-Means Clustering)")
    step = st.session_state.get('invseg_step', 1)
    st.markdown(f"<b>Step {step} of 3</b>", unsafe_allow_html=True)
    st.markdown("""
    <span style='color:#00c7b6;'>Step 1: Upload your portfolio data (CSV)</span>
    """, unsafe_allow_html=True)

    # Step 1: Data Upload
    uploaded_file = st.file_uploader("Upload your portfolio data (CSV)", type=["csv"], key="invseg_upload")
    df = None
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

    # Data Preview and Warning
    if df is not None and not df.empty:
        st.success("Data loaded successfully!")
        st.dataframe(df.head())
        # Step 2: Feature Selection and Cluster Number
        st.markdown("""
        <span style='color:#00c7b6;'>Step 2: Select Features & Number of Clusters</span>
        """, unsafe_allow_html=True)
        features = st.multiselect(
            "Select features for clustering:",
            options=[col for col in df.columns if df[col].dtype in ['float64', 'int64']],
            default=[col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        )
        n_clusters = st.slider("Select number of clusters:", min_value=2, max_value=5, value=3)
        if len(features) < 2:
            st.warning("Please select at least two features for meaningful clustering.")
        # Navigation button to next step (clustering)
        if st.button("Next: Run Clustering"):
            st.session_state['invseg_df'] = df
            st.session_state['invseg_features'] = features
            st.session_state['invseg_n_clusters'] = n_clusters
            st.session_state['invseg_step'] = 2
    else:
        st.warning("Please upload a CSV file to proceed.")

    # Step 2: Clustering, Visualization, Download
    if st.session_state.get('invseg_step', 1) == 2:
        st.markdown("""
        <span style='color:#00c7b6;'>Step 3: Clustering & Results</span>
        """, unsafe_allow_html=True)
        df = st.session_state['invseg_df'].copy()
        features = st.session_state['invseg_features']
        n_clusters = st.session_state['invseg_n_clusters']
        # K-Means
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        df['Cluster'] = clusters
        st.success(f"Clustering complete! {n_clusters} clusters identified.")
        st.image(get_finance_gif(), width=200)
        st.dataframe(df.head())
        # Visualization
        if len(features) == 2:
            fig = px.scatter(
                df, x=features[0], y=features[1], color=df['Cluster'].astype(str),
                title='Investor Clusters (2D)',
                labels={'color': 'Cluster'},
                template='plotly_dark',
                symbol_sequence=['circle']*n_clusters
            )
            st.plotly_chart(fig, use_container_width=True)
        elif len(features) >= 3:
            fig = px.scatter_3d(
                df, x=features[0], y=features[1], z=features[2], color=df['Cluster'].astype(str),
                title='Investor Clusters (3D)',
                labels={'color': 'Cluster'},
                template='plotly_dark',
                symbol_sequence=['circle']*n_clusters
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add more features for better cluster visualization.")
        # Download results
        st.download_button(
            label="📥 Download Cluster Assignments as CSV",
            data=df.to_csv(index=False),
            file_name="investor_clusters.csv",
            mime="text/csv"
        )
        # Navigation
        if st.button("⬅️ Back to Feature Selection", key="invseg_back"):
            st.session_state['invseg_step'] = 1
    if st.button("🔄 Reset Module", key="invseg_reset"):
        for k in list(st.session_state.keys()):
            if k.startswith('invseg_'):
                del st.session_state[k]
        st.rerun()

# --- Main App Logic ---
if option == "🏠 Home":
    home_page()
elif option == "📈 Stock Price Prediction":
    stock_price_prediction()
elif option == "📊 Movement Classification":
    stock_movement_classification()
elif option == "🔍 Investor Clustering":
    investor_segmentation()

# --- Footer ---
st.markdown("---")
st.markdown("<center><small>SmartFin Analyzer v1.0 | Made with ❤️ using Streamlit | <a href='mailto:your@email.com'>Contact</a></small></center>", unsafe_allow_html=True) 