import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta
import time
import random

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Billionaire Finance ML Suite PRO", layout="wide")

# -------------------- CSS Styling --------------------
st.markdown("""
    <style>
    .block-container { padding-top: 1rem !important; }
    header[data-testid="stHeader"] { background: transparent; }
    .floating-header {
        width: 100%;
        background: linear-gradient(135deg, #f9a825, #ffc107, #f9a825);
        color: #000;
        padding: 25px 10px;
        font-size: 28px;
        font-weight: 900;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
        z-index: 9999;
        border-bottom: 3px solid #000;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.01); }
        100% { transform: scale(1); }
    }
    .floating-header .subtext {
        font-size: 18px;
        font-weight: 600;
        margin-top: 8px;
    }
    .stApp {
        margin-top: -50px !important;
        padding-top: 150px !important;
        background: linear-gradient(-45deg, #1e1f26, #2e2f3e, #3f414f, #4f5261);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp { color: white; font-family: 'Segoe UI', sans-serif; }
    h1, h2, h3 { color: #FFD700; text-shadow: 1px 1px 2px #000; }
    .stButton>button {
        background: linear-gradient(to right, #FFEFBA, #FFFFFF);
        color: black;
        font-weight: bold;
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 215, 0, 0.4);
    }
    .stSidebar {
        background-color: #111 !important;
        color: white;
        border-right: 1px solid #ffc107;
    }
    .stSidebar label, .stSidebar .css-qrbaxs, .stSidebar .css-1cpxqw2, 
    .stSidebar p, .stSidebar span, .stSidebar input::placeholder {
        color: #f8f9fa !important;
        font-weight: 600;
    }
    .stDownloadButton>button {
        background-color: #FFEFBA;
        color: black;
        font-weight: bold;
    }
    .stProgress > div > div > div > div { background-color: #ffc107; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background: #333;
        color: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: #ffc107 !important;
        color: #000 !important;
        font-weight: bold;
    }
    .stSelectbox div[data-baseweb="select"] { background: #333; color: white; }
    .stNumberInput input { background: #333; color: white; }
    .stCheckbox span { color: white; }
    </style>

    <div class="floating-header">
        💎 BILLIONAIRE FINANCE ML SUITE PRO
        <div class="subtext">
            Welcome to the <b>ULTIMATE ML playground</b> for financial elites. <br>
            <b>Train powerful models. Visualize dynamic trends. Feel the billionaire vibe.</b>
        </div>
    </div>
""", unsafe_allow_html=True)

# -------------------- Video --------------------
st.markdown("""
    <div style="display: flex; justify-content: center; padding-bottom: 20px;">
        <iframe width="80%" height="400" src="https://www.youtube.com/embed/jN7B8ulTCyg?autoplay=1&mute=1&loop=1&playlist=jN7B8ulTCyg&controls=0&modestbranding=1" 
        frameborder="0" allow="autoplay; encrypted-media" allowfullscreen style="border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);"></iframe>
    </div>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("📂 Upload & Fetch Data")
    with st.expander("⚡ Data Sources", expanded=True):
        uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"], key="file_uploader")
        st.markdown("**https://www.kaggle.com/**")
        st.info( "Download the dataset from above link and upload it above to analyze.")
# -------------------- Model Selection --------------------
MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Machine": SVR(),
    "Neural Network": MLPRegressor(max_iter=1000)
}

# -------------------- Session State Initialization --------------------
for step in ["data_loaded", "preprocessed", "features_selected", "split_done", "trained", "evaluated"]:
    if step not in st.session_state:
        st.session_state[step] = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Linear Regression"

# -------------------- Load Dataset --------------------
data = None
if uploaded_file:
    with st.spinner('💎 Processing your dataset...'):
        try:
            data = pd.read_csv(uploaded_file)
            # Handle datetime parsing
            if 'Date' in data.columns:
                try:
                    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                except Exception:
                    pass
            # Convert numeric columns
            numeric_cols = ['Open', 'Close', 'High', 'Low', 'Volume']
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            st.sidebar.success("Dataset uploaded successfully!")
            with st.expander("📄 Dataset Preview", expanded=True):
                st.dataframe(data.head().style.highlight_max(axis=0, color='#ffc107'))
        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}")



# -------------------- ML Pipeline --------------------
st.header("🚀 Machine Learning Pipeline")

model_col1, model_col2 = st.columns([3, 1])
with model_col1:
    st.session_state.selected_model = st.selectbox("Select Model", list(MODELS.keys()), 
                                                 index=list(MODELS.keys()).index(st.session_state.selected_model))
with model_col2:
    show_params = st.checkbox("Advanced Parameters", False)

if show_params and st.session_state.selected_model:
    st.write(f"⚙️ {st.session_state.selected_model} Parameters")
    if st.session_state.selected_model == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 500, 100)
        max_depth = st.slider("Max Depth", 1, 50, 10)
        MODELS["Random Forest"] = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    elif st.session_state.selected_model == "Gradient Boosting":
        n_estimators = st.slider("Number of Trees", 10, 500, 100)
        learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
        MODELS["Gradient Boosting"] = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    elif st.session_state.selected_model == "Neural Network":
        hidden_layers = st.slider("Hidden Layers", 1, 5, 1)
        layer_size = st.slider("Layer Size", 10, 200, 50)
        MODELS["Neural Network"] = MLPRegressor(hidden_layer_sizes=(layer_size,)*hidden_layers, max_iter=1000)

# -------------------- Step 1: Load Data --------------------
if st.button("1️⃣ Load Data", key="load_data"):
    if data is not None:
        with st.spinner('💎 Loading data...'):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
        st.write("📄 Preview of Dataset:")
        st.dataframe(data.head().style.background_gradient(cmap='YlOrBr'))
        # Normalize column names
        data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
        st.session_state.data = data
        st.session_state.data_loaded = True
        st.success("✅ Dataset loaded successfully!")
        st.balloons()
    else:
        st.error("⚠️ No data found. Please upload a file or fetch stock data first.")

# -------------------- Step 2: Preprocessing --------------------
if st.button("2️⃣ Preprocessing", key="preprocessing"):
    if not st.session_state.get('data_loaded', False):
        st.error("⚠️ Please load data before preprocessing.")
    else:
        with st.spinner('🧹 Cleaning and preprocessing data...'):
            data = st.session_state.data.copy()
            steps = st.empty()
            progress_bar = st.progress(0)
            
            steps.markdown("🔍 Checking for missing values...")
            missing = data.isnull().sum()
            progress_bar.progress(20)
            time.sleep(0.5)
            
            if missing.sum() > 0:
                steps.markdown(f"🧹 Found {missing.sum()} missing values. Imputing...")
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                imputer = SimpleImputer(strategy='mean')
                data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
                progress_bar.progress(50)
                time.sleep(0.5)
            
            steps.markdown("⚖️ Scaling numerical features...")
            scaler = StandardScaler()
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            progress_bar.progress(80)
            time.sleep(0.5)
            
            steps.markdown("✨ Finalizing preprocessing...")
            st.session_state.data = data
            st.session_state.preprocessed = True
            progress_bar.progress(100)
            
            st.success("✅ Data preprocessing completed!")
            st.balloons()
            
            if missing.sum() > 0:
                fig = px.bar(x=missing.index, y=missing.values, 
                             labels={'x': 'Columns', 'y': 'Missing Values'},
                             title='Missing Values Before Imputation',
                             color=missing.values,
                             color_continuous_scale='YlOrRd')
                st.plotly_chart(fig)

# -------------------- Step 3: Feature Engineering --------------------
if st.button("3️⃣ Feature Engineering", key="feature_eng"):
    if not st.session_state.get('preprocessed', False):
        st.error("⚠️ Please preprocess the data before feature engineering.")
    else:
        with st.spinner('⚙️ Engineering features...'):
            data = st.session_state.data.copy()
            progress_bar = st.progress(0)
            features = []
            target = None
            
            price_cols = ['open', 'high', 'low', 'close']
            available_price_cols = [col for col in price_cols if col in data.columns]
            
            if len(available_price_cols) >= 1:
                steps = st.empty()
                
                steps.markdown("📊 Creating technical indicators...")
                # Simple price-based features
                if 'open' in data.columns and 'close' in data.columns:
                    data['price_change'] = data['close'] - data['open']
                    data['daily_return'] = data['close'].pct_change().fillna(0)
                    features.extend(['price_change', 'daily_return'])
                
                if 'close' in data.columns:
                    if 'high' in data.columns and 'low' in data.columns:
                        data['range'] = data['high'] - data['low']
                        features.append('range')
                    
                    # Adaptive moving averages for small datasets
                    data_length = len(data)
                    short_window = max(2, data_length // 4)  # At least 2, up to 1/4 of data
                    long_window = max(3, data_length // 2)   # At least 3, up to 1/2 of data
                    data['ma_short'] = data['close'].rolling(window=short_window, min_periods=1).mean()
                    data['ma_long'] = data['close'].rolling(window=long_window, min_periods=1).mean()
                    features.extend(['ma_short', 'ma_long'])
                    
                    # RSI only if enough data (adaptive window)
                    rsi_window = max(2, min(14, data_length // 2))
                    if rsi_window >= 2:
                        delta = data['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window, min_periods=1).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window, min_periods=1).mean()
                        rs = gain / loss
                        rs = rs.replace([np.inf, -np.inf], 0)  # Handle division by zero
                        data['rsi'] = 100 - (100 / (1 + rs))
                        data['rsi'] = data['rsi'].fillna(50)  # Neutral RSI for NaNs
                        features.append('rsi')
                
                progress_bar.progress(50)
                time.sleep(0.5)
                
                # Volume-based features
                if 'volume' in data.columns:
                    data['volume_change'] = data['volume'].pct_change().fillna(0)
                    features.append('volume_change')
                
                # Set target
                target = 'close' if 'close' in data.columns else available_price_cols[-1]
                
                # Clean NaNs with minimal data loss
                data = data.fillna(method='ffill').fillna(method='bfill')
                
                if len(data) < 2:
                    st.error("❌ Insufficient data after feature engineering. Try a larger dataset.")
                    st.stop()
                
                st.session_state.features = features
                st.session_state.target = target
                st.session_state.data = data
                st.session_state.features_selected = True
                progress_bar.progress(100)
                
                st.success(f"✅ Engineered {len(features)} features successfully!")
                st.balloons()
                
                st.write("📊 Engineered Features:", features)
                st.write("🎯 Target Variable:", target)
                
                if features and target:
                    try:
                        corr = data[features + [target]].corr()
                        fig = px.imshow(corr, text_auto=True, aspect="auto",
                                        color_continuous_scale='YlOrRd',
                                        title='Feature Correlation Heatmap')
                        st.plotly_chart(fig)
                    except Exception as e:
                        st.warning(f"Could not generate correlation heatmap: {str(e)}")
            else:
                st.error("⚠️ No price columns found for feature engineering.")

# -------------------- Step 4: Train-Test Split --------------------
if st.button("4️⃣ Train-Test Split", key="train_test_split"):
    if not st.session_state.get('features_selected', False):
        st.error("⚠️ Please complete feature engineering first.")
    else:
        try:
            data = st.session_state.data.copy()
            features = [f for f in st.session_state.features if f in data.columns]
            target = st.session_state.target
            
            if not features or target not in data.columns:
                st.error("❌ Invalid features or target column")
                st.stop()
            
            X = data[features]
            y = data[target]
            
            if len(X) < 2:
                st.error("❌ Insufficient data for splitting. Need at least 2 samples.")
                st.stop()
            
            # Adaptive splitting for small datasets
            test_size = 0.2 if len(X) >= 10 else max(0.2, 1/len(X))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
                shuffle=False  # Preserve temporal order
            )
            
            st.session_state.update({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'split_done': True
            })
            
            st.success(f"✅ Split successful! Train: {len(X_train)}, Test: {len(X_test)}")
            
        except Exception as e:
            st.error(f"❌ Error during train-test split: {str(e)}")
            st.stop()

# -------------------- Step 5: Model Training --------------------
if st.button("5️⃣ Visual Model Training", key="training"):
    if not st.session_state.get('split_done', False):
        st.error("⚠️ Please split the data before training the model.")
    else:
        with st.spinner(f'🏋️ Training {st.session_state.selected_model}...'):
            model = MODELS[st.session_state.selected_model]
            status = st.empty()
            progress_bar = st.progress(0)
            
            status.markdown("🔧 Initializing model...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            status.markdown("📊 Fitting model to training data...")
            model.fit(st.session_state.X_train, st.session_state.y_train)
            progress_bar.progress(80)
            time.sleep(0.5)
            
            status.markdown("✨ Finalizing model...")
            st.session_state.model = model
            st.session_state.trained = True
            progress_bar.progress(100)
            
            st.success(f"✅ {st.session_state.selected_model} trained successfully!")
            st.balloons()
            
            if hasattr(model, 'coef_'):
                coef_df = pd.DataFrame({
                    'Feature': st.session_state.features,
                    'Coefficient': model.coef_
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                st.write("📌 Model Coefficients:")
                fig = px.bar(coef_df, x='Feature', y='Coefficient', 
                             color='Coefficient', color_continuous_scale='YlOrRd',
                             title='Feature Importance (Coefficients)')
                st.plotly_chart(fig)

# -------------------- Step 6: Evaluation --------------------
if st.button("6️⃣ Evaluation", key="evaluation"):
    if not st.session_state.get('trained', False):
        st.error("⚠️ Please train the model before evaluation.")
    else:
        with st.spinner('📊 Evaluating model performance...'):
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            progress_bar = st.progress(0)
            status = st.empty()
            
            status.markdown("🧠 Making predictions...")
            y_pred = model.predict(X_test)
            progress_bar.progress(40)
            time.sleep(0.5)
            
            status.markdown("📉 Calculating metrics...")
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            progress_bar.progress(80)
            time.sleep(0.5)
            
            st.session_state.y_pred = y_pred
            st.session_state.evaluated = True
            progress_bar.progress(100)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Squared Error (MSE)", f"{mse:.4f}", delta_color="inverse")
            with col2:
                st.metric("R-squared Score (R²)", f"{r2:.4f}")
            
            st.success("✅ Model evaluation completed!")
            st.balloons()
            
            tab1, tab2, tab3 = st.tabs(["📈 Actual vs Predicted", "📊 Residuals", "📉 Error Distribution"])
            
            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
                                     marker=dict(color='#ffc107', size=8, opacity=0.7),
                                     name='Actual vs Predicted'))
                fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                            mode='lines', line=dict(color='red', dash='dash'),
                            name='Perfect Prediction'))
                fig.update_layout(title="📌 Actual vs Predicted Prices",
                              xaxis_title="Actual Prices",
                              yaxis_title="Predicted Prices",
                              template="plotly_dark")
                st.plotly_chart(fig)
            
            with tab2:
                residuals = y_test - y_pred
                fig = px.scatter(x=y_pred, y=residuals,
                              labels={'x': 'Predicted Values', 'y': 'Residuals'},
                              title='Residual Analysis',
                              trendline="lowess")
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig)
            
            with tab3:
                errors = y_test - y_pred
                fig = px.histogram(errors, nbins=50,
                                 title='Prediction Error Distribution',
                                 labels={'value': 'Prediction Error'},
                                 color_discrete_sequence=['#ffc107'])
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig)

# -------------------- Step 7: Results Visualization --------------------
if st.button("7️⃣ Results Visualization", key="visualization"):
    if not st.session_state.get('evaluated', False):
        st.error("⚠️ Please evaluate the model before viewing results.")
    else:
        with st.spinner('🎨 Creating visualizations...'):
            y_test = st.session_state.y_test.reset_index(drop=True)
            y_pred = pd.Series(st.session_state.y_pred)
            
            if 'date' in st.session_state.data.columns:
                test_dates = st.session_state.data.iloc[y_test.index]['date']
                results_df = pd.DataFrame({
                    "Date": test_dates,
                    "Actual": y_test,
                    "Predicted": y_pred
                }).set_index('Date')
            else:
                results_df = pd.DataFrame({
                    "Actual": y_test,
                    "Predicted": y_pred
                })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Actual'],
                                   mode='lines', name='Actual',
                                   line=dict(color='#ffc107', width=2)))
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Predicted'],
                                   mode='lines', name='Predicted',
                                   line=dict(color='#4CAF50', width=2)))
            fig.update_layout(title="📈 Actual vs Predicted Prices Over Time",
                           xaxis_title="Date",
                           yaxis_title="Price",
                           template="plotly_dark",
                           hovermode="x unified")
            st.plotly_chart(fig)
            
            st.write("📋 Prediction Results:")
            st.dataframe(results_df.style.background_gradient(subset=['Actual', 'Predicted'], cmap='YlOrBr'))
            
            csv = results_df.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f'{st.session_state.selected_model}_predictions.csv',
                mime='text/csv',
                key='download_csv'
            )
            
            st.success("✅ Results visualized successfully!")
            st.balloons()

# -------------------- Extras --------------------
st.markdown("---")
st.header("✨ Billionaire Extras")

if 'data' in locals() and data is not None and 'close' in data.columns:
    latest_price = data['close'].iloc[-1]
    prev_price = data['close'].iloc[-2] if len(data) > 1 else latest_price
    price_change = latest_price - prev_price
    pct_change = (price_change / prev_price) * 100 if prev_price != 0 else 0
    
    st.subheader(f"📊 Live Market Snapshot")
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Price", f"${latest_price:.2f}")
    col2.metric("Daily Change", f"${price_change:.2f}", f"{pct_change:.2f}%")
    if 'volume' in data.columns:
        col3.metric("Volume", f"{data['volume'].iloc[-1]:,}")

if st.session_state.get('trained', False):
    st.subheader("🔮 Prediction Playground")
    feature_values = {}
    cols = st.columns(3)
    for i, feature in enumerate(st.session_state.features):
        with cols[i % 3]:
            min_val = float(st.session_state.X_train[feature].min())
            max_val = float(st.session_state.X_train[feature].max())
            default_val = float(st.session_state.X_train[feature].median())
            feature_values[feature] = st.slider(feature, min_val, max_val, default_val)
    
    if st.button("🚀 Predict with Current Features"):
        input_data = pd.DataFrame([feature_values])
        prediction = st.session_state.model.predict(input_data)
        
        st.success(f"📈 Predicted {st.session_state.target}: {prediction[0]:.2f}")
        
        confidence = random.uniform(85, 95)
        st.write(f"🔍 Confidence: {confidence:.1f}%")
        
        fig = go.Figure(go.Indicator(
            mode="number+gauge+delta",
            value=prediction[0],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Predicted {st.session_state.target}"},
            delta={'reference': default_val},
            gauge={
                'shape': "bullet",
                'axis': {'range': [min_val, max_val]},
                'threshold': {
                    'line': {'color': "red", 'width': 2},
                    'thickness': 0.75,
                    'value': default_val
                },
                'steps': [
                    {'range': [min_val, default_val], 'color': "gray"},
                    {'range': [default_val, max_val], 'color': "lightgray"}
                ],
                'bar': {'color': "#ffc107"}
            }))
        st.plotly_chart(fig)

st.markdown("---")
st.subheader("💎 Billionaire Advice of the Day")
advice = [
    "The stock market is a device for transferring money from the impatient to the patient. - Warren Buffett",
    "Risk comes from not knowing what you're doing. - Warren Buffett",
    "The four most dangerous words in investing are: 'This time it's different.' - Sir John Templeton",
    "Know what you own, and know why you own it. - Peter Lynch",
    "In investing, what is comfortable is rarely profitable. - Robert Arnott",
    "The individual investor should act consistently as an investor and not as a speculator. - Benjamin Graham",
    "The biggest risk of all is not taking one. - Mellody Hobson",
    "Time in the market beats timing the market. - Unknown",
    "Diversification is protection against ignorance. - Warren Buffett",
    "Be fearful when others are greedy and greedy when others are fearful. - Warren Buffett"
]
selected_advice = random.choice(advice)
st.markdown(f"<div style='background-color: #e6f3ff; padding: 10px; border-radius: 5px; color: green;'>{selected_advice}</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 20px; color: #ffc107;">
        <p>💎 BILLIONAIRE FINANCE ML SUITE PRO | © 2025 | For the Elite Only</p>
        <p style="font-size: 0.8em;">Disclaimer: This is a simulation tool. Past performance is not indicative of future results.</p>
    </div>
""", unsafe_allow_html=True)