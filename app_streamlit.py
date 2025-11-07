import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="Financial Risk Assessment",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load the trained risk model"""
    try:
        with open('models/saved_models/risk_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please train the model first.")
        return None

# Load data from database
@st.cache_data
def load_data():
    """Load customer data from database"""
    try:
        from sqlalchemy import create_engine
        from config import get_db_connection_string
        
        engine = create_engine(get_db_connection_string())
        
        # Load customers
        customers = pd.read_sql("SELECT * FROM staging.customers LIMIT 100", engine)
        
        # Load credit history
        credit = pd.read_sql("SELECT * FROM staging.credit_history LIMIT 100", engine)
        
        # Load existing predictions if available
        try:
            predictions = pd.read_sql("""
                SELECT * FROM predictions.risk_scores 
                ORDER BY prediction_date DESC 
                LIMIT 100
            """, engine)
        except:
            predictions = pd.DataFrame()
        
        return customers, credit, predictions
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def calculate_risk_score(model, features):
    """Calculate risk score from model predictions"""
    try:
        prediction_proba = model.predict_proba(features)[0]
        default_probability = prediction_proba[1]
        risk_score = default_probability * 100
        
        if risk_score < 30:
            risk_category = 'Low'
            color = 'green'
            recommendation = '✅ Approve - Low risk of default'
        elif risk_score < 60:
            risk_category = 'Medium'
            color = 'orange'
            recommendation = '⚠️ Review - Manual underwriting recommended'
        else:
            risk_category = 'High'
            color = 'red'
            recommendation = '❌ Reject - High risk of default'
        
        return {
            'risk_score': float(risk_score),
            'risk_category': risk_category,
            'default_probability': float(default_probability),
            'recommendation': recommendation,
            'color': color
        }
    except Exception as e:
        st.error(f"Error calculating risk: {str(e)}")
        return None

def create_gauge_chart(risk_score, risk_category):
    """Create a gauge chart for risk score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score", 'font': {'size': 24}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 60], 'color': '#fff3cd'},
                {'range': [60, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'size': 16}
    )
    
    return fig

def create_feature_importance_chart(features_dict):
    """Create horizontal bar chart of top features"""
    # Mock feature importance - in production, load from model
    feature_importance = {
        'Credit Score': 0.25,
        'Debt to Income': 0.20,
        'Annual Income': 0.15,
        'Credit Utilization': 0.12,
        'Delinquent Accounts': 0.10,
        'Employment Stability': 0.08,
        'Years Employed': 0.05,
        'Loan Amount': 0.05
    }
    
    df_importance = pd.DataFrame(list(feature_importance.items()), 
                                 columns=['Feature', 'Importance'])
    df_importance = df_importance.sort_values('Importance', ascending=True)
    
    fig = px.bar(df_importance, 
                 x='Importance', 
                 y='Feature',
                 orientation='h',
                 title='Top Risk Factors',
                 color='Importance',
                 color_continuous_scale='Blues')
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Importance Score",
        yaxis_title=""
    )
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">💰 Financial Risk Assessment Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Select Page", 
                            ["Single Prediction", "Batch Analysis", "Model Insights"])
    
    if page == "Single Prediction":
        show_single_prediction(model)
    elif page == "Batch Analysis":
        show_batch_analysis(model)
    else:
        show_model_insights(model)

def show_single_prediction(model):
    """Page for single customer risk prediction"""
    st.header("🎯 Single Customer Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Information")
        
        age = st.slider("Age", 18, 80, 35)
        annual_income = st.number_input("Annual Income ($)", 
                                       min_value=10000, 
                                       max_value=500000, 
                                       value=60000,
                                       step=5000)
        years_employed = st.slider("Years Employed", 0, 40, 5)
        num_dependents = st.slider("Number of Dependents", 0, 10, 2)
        
        employment_status = st.selectbox("Employment Status", 
                                        ["Employed", "Self-Employed", "Unemployed"])
        home_ownership = st.selectbox("Home Ownership", 
                                     ["Own", "Rent", "Mortgage"])
    
    with col2:
        st.subheader("Credit Information")
        
        credit_score = st.slider("Credit Score", 300, 850, 700)
        num_credit_accounts = st.slider("Number of Credit Accounts", 1, 20, 5)
        credit_utilization = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.3)
        num_delinquent = st.slider("Number of Delinquent Accounts", 0, 10, 0)
        total_debt = st.number_input("Total Debt ($)", 
                                     min_value=0, 
                                     max_value=500000, 
                                     value=20000,
                                     step=1000)
        years_credit_history = st.slider("Years of Credit History", 0, 50, 10)
    
    st.subheader("Loan Information")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        loan_amount = st.number_input("Loan Amount ($)", 
                                     min_value=1000, 
                                     max_value=500000, 
                                     value=50000,
                                     step=1000)
    with col4:
        loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72])
    with col5:
        # Calculate interest rate based on credit score
        if credit_score >= 750:
            interest_rate = np.random.uniform(0.03, 0.06)
        elif credit_score >= 700:
            interest_rate = np.random.uniform(0.06, 0.10)
        elif credit_score >= 650:
            interest_rate = np.random.uniform(0.10, 0.15)
        else:
            interest_rate = np.random.uniform(0.15, 0.25)
        
        st.metric("Estimated Interest Rate", f"{interest_rate*100:.2f}%")
    
    # Calculate derived features
    employment_stability = (years_employed / age) * 100 if age > 0 else 0
    delinquency_score = (num_delinquent * 20 + (1 - credit_utilization) * 30)
    loan_to_income = loan_amount / annual_income if annual_income > 0 else 0
    
    # Calculate DTI
    monthly_rate = interest_rate / 12
    if monthly_rate > 0:
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**loan_term) / \
                         ((1 + monthly_rate)**loan_term - 1)
    else:
        monthly_payment = loan_amount / loan_term
    
    annual_payment = monthly_payment * 12
    dti = annual_payment / annual_income if annual_income > 0 else 0
    
    # Encode categorical
    employment_mapping = {'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2}
    home_mapping = {'Own': 0, 'Rent': 1, 'Mortgage': 2}
    
    # Prepare features for prediction
    features = pd.DataFrame({
        'age': [age],
        'annual_income': [annual_income],
        'years_employed': [years_employed],
        'num_dependents': [num_dependents],
        'credit_score': [credit_score],
        'num_credit_accounts': [num_credit_accounts],
        'credit_utilization_ratio': [credit_utilization],
        'num_delinquent_accounts': [num_delinquent],
        'total_debt': [total_debt],
        'years_credit_history': [years_credit_history],
        'loan_amount': [loan_amount],
        'loan_term_months': [loan_term],
        'interest_rate': [interest_rate],
        'debt_to_income_ratio': [dti],
        'employment_stability_score': [employment_stability],
        'delinquency_score': [delinquency_score],
        'loan_to_income_ratio': [loan_to_income],
        'employment_status_encoded': [employment_mapping[employment_status]],
        'home_ownership_encoded': [home_mapping[home_ownership]]
    })
    
    # Predict button
    if st.button("🔍 Assess Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing risk..."):
            result = calculate_risk_score(model, features)
            
            if result:
                st.success("Risk assessment completed!")
                
                # Display results
                st.markdown("---")
                st.header("📊 Risk Assessment Results")
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Risk Score", f"{result['risk_score']:.1f}/100")
                with col2:
                    risk_class = f"risk-{result['risk_category'].lower()}"
                    st.markdown(f"<div class='metric-card'><h4>Risk Category</h4>"
                              f"<p class='{risk_class}'>{result['risk_category']}</p></div>", 
                              unsafe_allow_html=True)
                with col3:
                    st.metric("Default Probability", 
                             f"{result['default_probability']*100:.1f}%")
                with col4:
                    st.metric("Monthly Payment", f"${monthly_payment:,.2f}")
                
                # Gauge chart
                st.plotly_chart(create_gauge_chart(result['risk_score'], 
                                                   result['risk_category']),
                              use_container_width=True)
                
                # Recommendation
                st.markdown("### 💡 Recommendation")
                if result['risk_category'] == 'Low':
                    st.success(result['recommendation'])
                elif result['risk_category'] == 'Medium':
                    st.warning(result['recommendation'])
                else:
                    st.error(result['recommendation'])
                
                # Key factors
                st.markdown("### 🔑 Key Risk Factors")
                factors_col1, factors_col2 = st.columns(2)
                
                with factors_col1:
                    st.metric("Debt-to-Income Ratio", f"{dti:.2%}", 
                             delta="Good" if dti < 0.43 else "High",
                             delta_color="normal" if dti < 0.43 else "inverse")
                    st.metric("Credit Utilization", f"{credit_utilization:.1%}",
                             delta="Good" if credit_utilization < 0.3 else "High",
                             delta_color="normal" if credit_utilization < 0.3 else "inverse")
                
                with factors_col2:
                    st.metric("Loan-to-Income Ratio", f"{loan_to_income:.2f}",
                             delta="Good" if loan_to_income < 3 else "High",
                             delta_color="normal" if loan_to_income < 3 else "inverse")
                    st.metric("Employment Stability", f"{employment_stability:.1f}%",
                             delta="Good" if employment_stability > 10 else "Low",
                             delta_color="normal" if employment_stability > 10 else "inverse")

def show_batch_analysis(model):
    """Page for batch predictions"""
    st.header("📊 Batch Risk Analysis")
    
    # Load data
    customers, credit, predictions = load_data()
    
    if customers.empty:
        st.warning("No customer data available. Please generate data first.")
        return
    
    # Display existing predictions
    if not predictions.empty:
        st.subheader("Recent Risk Assessments")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Assessments", len(predictions))
        with col2:
            low_risk = len(predictions[predictions['risk_category'] == 'Low'])
            st.metric("Low Risk", low_risk, 
                     delta=f"{low_risk/len(predictions)*100:.1f}%")
        with col3:
            medium_risk = len(predictions[predictions['risk_category'] == 'Medium'])
            st.metric("Medium Risk", medium_risk,
                     delta=f"{medium_risk/len(predictions)*100:.1f}%")
        with col4:
            high_risk = len(predictions[predictions['risk_category'] == 'High'])
            st.metric("High Risk", high_risk,
                     delta=f"{high_risk/len(predictions)*100:.1f}%")
        
        # Risk distribution chart
        st.subheader("Risk Distribution")
        risk_counts = predictions['risk_category'].value_counts()
        
        fig = px.pie(values=risk_counts.values, 
                    names=risk_counts.index,
                    title='Risk Category Distribution',
                    color=risk_counts.index,
                    color_discrete_map={'Low': '#28a745', 
                                       'Medium': '#ffc107', 
                                       'High': '#dc3545'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Score distribution histogram
        st.subheader("Risk Score Distribution")
        fig = px.histogram(predictions, x='risk_score', 
                          nbins=30,
                          title='Distribution of Risk Scores',
                          labels={'risk_score': 'Risk Score', 'count': 'Frequency'},
                          color_discrete_sequence=['#1f77b4'])
        fig.add_vline(x=30, line_dash="dash", line_color="green", 
                     annotation_text="Low/Medium")
        fig.add_vline(x=60, line_dash="dash", line_color="orange", 
                     annotation_text="Medium/High")
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("Detailed Predictions")
        display_predictions = predictions[['customer_id', 'risk_score', 
                                          'risk_category', 'default_probability', 
                                          'prediction_date']].copy()
        display_predictions['risk_score'] = display_predictions['risk_score'].round(2)
        display_predictions['default_probability'] = (
            display_predictions['default_probability'] * 100
        ).round(2).astype(str) + '%'
        
        st.dataframe(display_predictions, use_container_width=True, height=400)
    
    else:
        st.info("No predictions available yet. Run the Airflow DAG or train the model.")

def show_model_insights(model):
    """Page for model performance and insights"""
    st.header("🤖 Model Performance Insights")
    
    # Model information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Information")
        st.write("**Model Type:** Random Forest Classifier")
        st.write("**Version:** v1.0.0")
        st.write("**Training Date:** " + datetime.now().strftime("%Y-%m-%d"))
        st.write("**Features:** 19 input features")
    
    with col2:
        st.subheader("Performance Metrics")
        # Mock metrics - in production, load from database
        st.metric("Accuracy", "87.3%")
        st.metric("AUC-ROC", "0.91")
        st.metric("Precision", "84.6%")
        st.metric("Recall", "79.2%")
    
    # Feature importance
    st.subheader("Feature Importance")
    st.write("The most influential factors in risk prediction:")
    
    features_dict = {}  # Placeholder
    fig = create_feature_importance_chart(features_dict)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model explanation
    st.subheader("📖 How the Model Works")
    
    with st.expander("Understanding Risk Categories"):
        st.write("""
        **Low Risk (0-30):** 
        - Strong credit profile
        - Low debt-to-income ratio
        - Stable employment history
        - **Recommendation:** Approve with standard terms
        
        **Medium Risk (30-60):**
        - Average credit profile
        - Moderate debt levels
        - Some risk factors present
        - **Recommendation:** Manual review required
        
        **High Risk (60-100):**
        - Poor credit history
        - High debt burden
        - Multiple risk factors
        - **Recommendation:** Decline or require collateral
        """)
    
    with st.expander("Key Risk Factors Explained"):
        st.write("""
        1. **Credit Score:** Historical creditworthiness (300-850)
        2. **Debt-to-Income Ratio:** Monthly debt payments vs. income
        3. **Credit Utilization:** Percentage of available credit used
        4. **Delinquent Accounts:** Past-due payment history
        5. **Employment Stability:** Job tenure relative to age
        6. **Loan-to-Income Ratio:** Loan amount vs. annual income
        """)

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Financial Risk Assessment System v1.0.0</p>
        <p>⚠️ For educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()