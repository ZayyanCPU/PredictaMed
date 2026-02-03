import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="PredictaMed - AI Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
    }
    .risk-low {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
    st.session_state.models = {}
    st.session_state.scalers = {}
    st.session_state.feature_sets = {}

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    data = pd.read_csv('Multiple Disease Data.csv')
    
    encode = {
        'Smoking': {'Yes': 1, 'No': 0},
        'AlcoholDrinking': {'Yes': 1, 'No': 0},
        'Stroke': {'Yes': 1, 'No': 0},
        'DiffWalking': {'Yes': 1, 'No': 0},
        'Sex': {'Male': 1, 'Female': 0},
        'AgeCategory': {'18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4, 
                        '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9, 
                        '70-74': 10, '75-79': 11, '80 or older': 12},
        'Race': {'White': 0, 'Black': 1, 'Asian': 2, 'Hispanic': 3, 'Other': 4},
        'PhysicalActivity': {'Yes': 1, 'No': 0},
        'GenHealth': {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4},
        'HeartDisease': {'Yes': 1, 'No': 0},
        'Diabetic': {'Yes': 1, 'No': 0},
        'Asthma': {'Yes': 1, 'No': 0},
        'KidneyDisease': {'Yes': 1, 'No': 0},
        'SkinCancer': {'Yes': 1, 'No': 0}
    }
    
    for column, encodeDict in encode.items():
        if column in data.columns:
            data[column] = data[column].map(encodeDict)
    
    return data

@st.cache_resource
def train_models(data):
    """Train the best models for each disease"""
    matrix = data.corr()
    
    # Threshold values from original notebook
    threshold = {
        'HeartDisease': 0.2, 
        'Stroke': 0.13, 
        'Diabetic': 0.2, 
        'Asthma': 0.1, 
        'KidneyDisease': 0.13, 
        'SkinCancer': 0.06
    }
    
    # Get correlated features for each disease
    greater = {}
    for condition, thresh in threshold.items():
        corrTarget = matrix[condition]
        greater[condition] = corrTarget[corrTarget > thresh].index.tolist()
    
    # Best models based on original analysis
    best_models = {
        'HeartDisease': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Stroke': DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42),
        'Diabetic': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Asthma': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'KidneyDisease': LogisticRegression(C=1, max_iter=500, random_state=42),
        'SkinCancer': LogisticRegression(C=1, max_iter=500, random_state=42)
    }
    
    trained_models = {}
    trained_scalers = {}
    feature_sets = {}
    
    target = ['HeartDisease', 'Stroke', 'Diabetic', 'Asthma', 'KidneyDisease', 'SkinCancer']
    
    for disease in target:
        selectFeat = greater.get(disease, [])
        dataset = data.dropna()
        x = dataset[selectFeat]
        y = dataset[disease]
        x = x.drop(disease, axis=1)
        
        feature_sets[disease] = list(x.columns)
        
        imputer = SimpleImputer(strategy='mean')
        xImpute = imputer.fit_transform(x)
        scaler = StandardScaler()
        xScaled = scaler.fit_transform(xImpute)
        
        xtrain, xtest, ytrain, ytest = train_test_split(xScaled, y, test_size=0.2, random_state=42)
        
        model = best_models[disease]
        model.fit(xtrain, ytrain)
        
        trained_models[disease] = model
        trained_scalers[disease] = scaler
    
    return trained_models, trained_scalers, feature_sets

def predict_disease(models, scalers, feature_sets, user_data):
    """Make predictions for all diseases"""
    predictions = {}
    
    for disease in models.keys():
        features = feature_sets[disease]
        input_data = []
        
        for feat in features:
            if feat in user_data:
                input_data.append(user_data[feat])
            else:
                input_data.append(0)
        
        input_array = np.array(input_data).reshape(1, -1)
        scaled_input = scalers[disease].transform(input_array)
        
        pred = models[disease].predict(scaled_input)[0]
        prob = models[disease].predict_proba(scaled_input)[0]
        
        predictions[disease] = {
            'prediction': pred,
            'probability': prob[1] if len(prob) > 1 else prob[0]
        }
    
    return predictions

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 PredictaMed</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Multi-Disease Prediction System</p>', unsafe_allow_html=True)
    
    # Load data and train models
    with st.spinner('Loading models...'):
        data = load_and_preprocess_data()
        models, scalers, feature_sets = train_models(data)
    
    # Sidebar for user input
    st.sidebar.markdown("## 📋 Patient Information")
    st.sidebar.markdown("---")
    
    # Personal Information
    st.sidebar.markdown("### 👤 Personal Details")
    
    age_category = st.sidebar.selectbox(
        "Age Category",
        options=['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', 
                 '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'],
        index=5
    )
    age_map = {'18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4, 
               '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9, 
               '70-74': 10, '75-79': 11, '80 or older': 12}
    
    sex = st.sidebar.radio("Sex", options=['Male', 'Female'], horizontal=True)
    
    race = st.sidebar.selectbox(
        "Race/Ethnicity",
        options=['White', 'Black', 'Asian', 'Hispanic', 'Other']
    )
    race_map = {'White': 0, 'Black': 1, 'Asian': 2, 'Hispanic': 3, 'Other': 4}
    
    # Health Metrics
    st.sidebar.markdown("### 📊 Health Metrics")
    
    bmi = st.sidebar.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0, 0.1)
    
    physical_health = st.sidebar.slider(
        "Physical Health (days with issues in last 30 days)", 
        0, 30, 0
    )
    
    mental_health = st.sidebar.slider(
        "Mental Health (days with issues in last 30 days)", 
        0, 30, 0
    )
    
    sleep_time = st.sidebar.slider("Average Sleep Time (hours)", 1, 24, 7)
    
    gen_health = st.sidebar.selectbox(
        "General Health",
        options=['Excellent', 'Very good', 'Good', 'Fair', 'Poor'],
        index=2
    )
    gen_health_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}
    
    # Lifestyle
    st.sidebar.markdown("### 🏃 Lifestyle")
    
    smoking = st.sidebar.radio("Smoking", options=['No', 'Yes'], horizontal=True)
    alcohol = st.sidebar.radio("Alcohol Drinking", options=['No', 'Yes'], horizontal=True)
    physical_activity = st.sidebar.radio("Physical Activity", options=['Yes', 'No'], horizontal=True)
    diff_walking = st.sidebar.radio("Difficulty Walking", options=['No', 'Yes'], horizontal=True)
    
    # Compile user data
    user_data = {
        'BMI': bmi,
        'Smoking': 1 if smoking == 'Yes' else 0,
        'AlcoholDrinking': 1 if alcohol == 'Yes' else 0,
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'DiffWalking': 1 if diff_walking == 'Yes' else 0,
        'Sex': 1 if sex == 'Male' else 0,
        'AgeCategory': age_map[age_category],
        'Race': race_map[race],
        'PhysicalActivity': 1 if physical_activity == 'Yes' else 0,
        'GenHealth': gen_health_map[gen_health],
        'SleepTime': sleep_time
    }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔬 Disease Risk Analysis")
        st.markdown("Click the button below to analyze your health data and predict disease risks.")
        
        predict_button = st.button("🔍 Analyze Health Risks", use_container_width=True)
        
        if predict_button:
            with st.spinner('Analyzing your health data...'):
                predictions = predict_disease(models, scalers, feature_sets, user_data)
            
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            # Display predictions in a grid
            cols = st.columns(3)
            
            disease_icons = {
                'HeartDisease': '❤️',
                'Stroke': '🧠',
                'Diabetic': '🩸',
                'Asthma': '🫁',
                'KidneyDisease': '🫘',
                'SkinCancer': '🔬'
            }
            
            disease_names = {
                'HeartDisease': 'Heart Disease',
                'Stroke': 'Stroke',
                'Diabetic': 'Diabetes',
                'Asthma': 'Asthma',
                'KidneyDisease': 'Kidney Disease',
                'SkinCancer': 'Skin Cancer'
            }
            
            for idx, (disease, result) in enumerate(predictions.items()):
                with cols[idx % 3]:
                    risk_level = "High Risk" if result['prediction'] == 1 else "Low Risk"
                    prob_percent = result['probability'] * 100
                    
                    if result['prediction'] == 1:
                        st.error(f"""
                        **{disease_icons[disease]} {disease_names[disease]}**
                        
                        🔴 **{risk_level}**
                        
                        Risk Probability: **{prob_percent:.1f}%**
                        """)
                    else:
                        st.success(f"""
                        **{disease_icons[disease]} {disease_names[disease]}**
                        
                        🟢 **{risk_level}**
                        
                        Risk Probability: **{prob_percent:.1f}%**
                        """)
            
            # Risk summary
            st.markdown("---")
            st.markdown("### 📈 Risk Summary")
            
            high_risks = [disease_names[d] for d, r in predictions.items() if r['prediction'] == 1]
            
            if high_risks:
                st.warning(f"⚠️ **Elevated risk detected for:** {', '.join(high_risks)}")
                st.info("""
                **Recommendations:**
                - Consult with a healthcare professional for proper diagnosis
                - Consider lifestyle modifications
                - Regular health checkups are advised
                """)
            else:
                st.success("✅ **Good news!** No elevated disease risks detected based on the provided information.")
                st.info("""
                **Keep up the healthy lifestyle!**
                - Continue regular exercise
                - Maintain a balanced diet
                - Get regular health checkups
                """)
    
    with col2:
        st.markdown("### 📋 Your Input Summary")
        
        with st.expander("Personal Details", expanded=True):
            st.write(f"**Age:** {age_category}")
            st.write(f"**Sex:** {sex}")
            st.write(f"**Race:** {race}")
        
        with st.expander("Health Metrics", expanded=True):
            st.write(f"**BMI:** {bmi}")
            st.write(f"**General Health:** {gen_health}")
            st.write(f"**Sleep:** {sleep_time} hours")
            st.write(f"**Physical Health Issues:** {physical_health} days")
            st.write(f"**Mental Health Issues:** {mental_health} days")
        
        with st.expander("Lifestyle", expanded=True):
            st.write(f"**Smoking:** {smoking}")
            st.write(f"**Alcohol:** {alcohol}")
            st.write(f"**Physical Activity:** {physical_activity}")
            st.write(f"**Difficulty Walking:** {diff_walking}")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **PredictaMed** uses machine learning models trained on 59,068 patient records 
        to predict disease risks.
        
        **Disclaimer:** This is for educational purposes only and should not replace 
        professional medical advice.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🏥 PredictaMed - AI-Powered Multi-Disease Prediction System</p>
        <p>Developed by Zayyan | For Educational Purposes Only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
