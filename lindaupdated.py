# ----------------------------------------------------------
# 🔐 Token Validation (for Render deployment)
# ----------------------------------------------------------
import os
import sys
import streamlit as st
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

# Fetch secret and toggle
TOKEN_SECRET = os.getenv("TOKEN_SECRET", "dev-secret")  # must match token_server.py
TOKEN_PROTECT = os.getenv("TOKEN_PROTECT", "0") == "1"
TOKEN_DEFAULT_MAX_AGE = int(os.getenv("TOKEN_DEFAULT_MAX_AGE", "3600"))  # 1 hour default

def validate_token():
    """Validate ?token=... in Streamlit app URL"""
    query_params = st.query_params
    token = query_params.get("token", [None])[0] if isinstance(query_params.get("token"), list) else query_params.get("token")

    if not TOKEN_PROTECT:
        return True  # protection disabled (local/dev)

    if not token:
        st.error("❌ Missing access token. Please use a valid expiring link.")
        st.stop()

    serializer = URLSafeTimedSerializer(TOKEN_SECRET)
    try:
        data = serializer.loads(token, max_age=TOKEN_DEFAULT_MAX_AGE)
        st.session_state["token_payload"] = data
        return True
    except SignatureExpired:
        st.error("⏰ Token has expired. Please request a new link.")
        st.stop()
    except BadSignature:
        st.error("🚫 Invalid or tampered token.")
        st.stop()

validate_token()

import pandas as pd
import numpy as np
import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import uuid
import os
import ast
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix



recognizer = sr.Recognizer()
def get_voice_input():
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        audio = recognizer.listen(source, timeout=5)
    try:
        query = recognizer.recognize_google(audio)
        st.success(f"You said: {query}")
        return query
    except Exception as e:
        st.error(f"Voice error: {e}")
        return ""

def speak_text(text):
    tts = gTTS(text=text, lang="en")
    tts.save("result.mp3")
    os.system("start result.mp3")

def ask_ai(question, df):
    genai.configure(api_key="AIzaSyADkL6_0GAaUgT9MzWoEGbTJkZfLgWtSE0")
    model = genai.GenerativeModel("models/gemini-2.0-flash")

    prompt = f"""
    You are a pandas assistant.
    The dataframe has these columns: {list(df.columns)}.
    Translate the user question into a valid single-line Pandas expression
    using 'df'. Be case-insensitive with column names.
    Only output the code, no explanation.
    Question: {question}
    """

    response = model.generate_content(prompt)
    code = response.text.strip()

    if code.startswith("```"):
        code = code.strip("`")
    code_lines = [line.strip() for line in code.splitlines() 
                  if line.strip() and not line.strip().startswith('#')]
    valid_code = ''
    for line in code_lines:
        if line.lower() in ['python', 'pandas', 'code']:
            continue
        valid_code = line
        break
    return valid_code

def fix_column_names(df, code):
    mapping = {col.lower(): col for col in df.columns}
    for low, real in mapping.items():
        code = code.replace(f"['{low}']", f"['{real}']")
        code = code.replace(f'["{low}"]', f'["{real}"]')
    return code

def train_recommender(df, target_col):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate columns if any

    assert isinstance(target_col, str), f"{target_col} is not a string!"
    assert target_col in df.columns, f"{target_col} not in columns: {df.columns.tolist()}"
    assert df.columns.is_unique, f"Duplicate columns found: {df.columns.tolist()}"

    # -----------------------
    # Debug: initial info
    # -----------------------
    st.write("🧩 DEBUG INFO (train_recommender start)")
    st.write("Columns:", df.columns.tolist())
    st.write("Target column name:", target_col)
    st.write("Target exists:", target_col in df.columns)
    st.write("Target shape (raw):", getattr(df[target_col], "shape", "N/A"))
    st.write("Target sample:")
    st.dataframe(df[[target_col]].head())

    # -----------------------
    # Prepare X and y (ensure y is 1-D)
    # -----------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # If y is a DataFrame (accidentally), take the first column
    if isinstance(y, pd.DataFrame):
        st.write("Note: target was a DataFrame — taking first column.")
        y = y.iloc[:, 0]

    # Convert to numpy and force 1-D
    y = np.asarray(y)
    st.write("y np.asarray() shape before ravel:", getattr(y, "shape", None))
    y = y.ravel()
    st.write("y shape after ravel (should be 1-D):", y.shape)
    st.write("y sample (first 5):", y[:5])

    # -----------------------
    # Feature lists
    # -----------------------
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    st.write("num_cols:", num_cols)
    st.write("cat_cols:", cat_cols)
    st.write("X shape:", X.shape)

    # -----------------------
    # Preprocessor, model, pipeline
    # -----------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", model)
    ])

    # -----------------------
    # Encode target with LabelEncoder -> ensure 1-D
    # -----------------------
    le = LabelEncoder()
    # LabelEncoder expects 1D input; ensure it:
    y_for_le = np.ravel(y)
    y_enc = le.fit_transform(y_for_le)
    y_enc = np.asarray(y_enc).ravel()
    st.write("y_enc shape (after LabelEncoder):", y_enc.shape)
    st.write("le.classes_:", le.classes_)

    # -----------------------
    # Split - show shapes
    # -----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )
    st.write("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
    st.write("y_train shape:", getattr(y_train, "shape", None), "y_test shape:", getattr(y_test, "shape", None))

    # -----------------------
    # Fit and predict
    # -----------------------
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)

    st.write("y_pred shape:", getattr(y_pred, "shape", None))
    st.write("y_probs shape (n_samples, n_classes):", getattr(y_probs, "shape", None))

    # -----------------------
    # Metrics - handle binary vs multiclass ROC AUC correctly
    # -----------------------
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }

    # Add roc_auc if possible
    try:
        # For binary classification, pass y_probs[:,1] and y_test 1-D
        if y_probs.shape[1] == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, y_probs[:, 1])
        else:
            # multiclass
            metrics["roc_auc"] = roc_auc_score(y_test, y_probs, multi_class="ovr")
    except Exception as e:
        st.write("Could not compute roc_auc:", e)
        metrics["roc_auc"] = None

    return {"pipeline": pipeline, "le": le, "metrics": metrics, "y_test": y_test, "y_probs": y_probs}


def prepare_input(user_input, df, target_col):
    # Drop the target column only if it exists in df
    if target_col in df.columns:
        full_cols = df.drop(columns=[target_col]).columns
    else:
        full_cols = df.columns

    input_df = pd.DataFrame([user_input])
    
    for col in full_cols:
        if col not in input_df.columns or pd.isna(input_df.at[0, col]):
            if col in df.columns and df[col].dtype == object:
                # Use mode if column exists and is categorical
                input_df[col] = df[col].mode()[0] if not df[col].mode().empty else np.nan
            else:
                # Use mean or NaN if numeric or no mode
                input_df[col] = df[col].mean() if col in df.columns else np.nan

    # Reorder columns to match df
    return input_df[full_cols]


def prepare_input(user_input, df, target_col):
    full_cols = df.drop(columns=[target_col]).columns
    input_df = pd.DataFrame([user_input])

    for col in full_cols:
        if col not in input_df.columns:
            if df[col].dtype == "object":
                input_df[col] = df[col].mode()[0]
            else:
                input_df[col] = df[col].mean()
    return input_df[full_cols]

def collect_user_inputs(df, exclude_cols=[], dataset_name="dataset"):
    sample = {}
    for col in df.columns:
        if col in exclude_cols: 
            continue
        if df[col].dtype == "object":
            sample[col] = st.selectbox(
                f"{col}", df[col].unique(), key=f"{dataset_name}_{col}"
            )
        else:
            sample[col] = st.number_input(
                f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()),
                key=f"{dataset_name}_{col}"
            )
    return pd.DataFrame([sample])

def show_model_metrics(recomm):
    metrics = recomm["metrics"]
    st.markdown("### 📈 Model Performance")
    st.table(pd.DataFrame([metrics]))

    # ROC Curve
    fpr, tpr, _ = roc_curve(recomm["y_test"], recomm["y_probs"][:,1], pos_label=1)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
    fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)

    # Feature Importance
    clf = recomm["pipeline"].named_steps["clf"]
    feat_names = recomm["pipeline"].named_steps["preprocessor"].get_feature_names_out()
    importances = clf.feature_importances_
    feat_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(10)
    st.bar_chart(feat_df.set_index("feature"))


df_life = pd.read_csv("life_health_insurance_dataset - life_health_insurance_d.csv")
df_motor = pd.read_csv("LINDA_ Motors - motor_insurance_dataset.csv")


def generate_synthetic_health(n=200):
    genders = ["Male", "Female"]
    marital_statuses = ["Single", "Married", "Divorced", "Widowed"]
    occupations = ["Salaried", "Self-Employed", "Business", "Student", "Retired"]
    income_levels = ["Low", "Medium", "High"]
    education_levels = ["High School", "Graduate", "Postgraduate", "Doctorate"]
    locations = ["Urban", "Semi-Urban", "Rural"]
    health_conditions = ["None", "Diabetes", "Hypertension", "Heart Disease", "Asthma", "Cancer"]
    smoking_status = ["Yes", "No"]
    alcohol_status = ["Yes", "No"]
    claim_history = ["Yes", "No"]
    lifestyles = ["Active", "Sedentary", "Moderate"]
    risk_appetite = ["Low", "Medium", "High"]
    travel_frequency = ["Rarely", "Often", "Frequent"]
    addons = ["Maternity", "Accidental Cover", "Critical Illness", "Hospital Cash", "None"]
    recommended_products = ["Individual", "Family Floater", "Senior Citizen", "Critical Illness"]

    def generate_record(i):
        return {
            "Customer_ID": f"CUST{i:04d}",
            "Age": random.randint(18, 70),
            "Gender": random.choice(genders),
            "Marital_Status": random.choice(marital_statuses),
            "Occupation": random.choice(occupations),
            "Income_Level": random.choice(income_levels),
            "Education": random.choice(education_levels),
            "Location": random.choice(locations),
            "Family_Size": random.randint(1, 6),
            "Health_Condition": random.choice(health_conditions),
            "Smoking": random.choice(smoking_status),
            "Alcohol": random.choice(alcohol_status),
            "Claim_History": random.choice(claim_history),
            "Lifestyle": random.choice(lifestyles),
            "Risk_Appetite": random.choice(risk_appetite),
            "Travel_Frequency": random.choice(travel_frequency),
            "Coverage_Amount": random.randint(200000, 2000000),
            "Premium_Budget": random.randint(5000, 50000),
            "Tenure_Preference": random.choice([5, 10, 15, 20, 25]),
            "Addons_Interested": random.choice(addons),
            "Recommended_Product": random.choice(recommended_products)
        }

    return pd.DataFrame([generate_record(i) for i in range(1, n+1)])

df_synth = generate_synthetic_health()


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Home",
    "📊 Data Explorer", 
    "🏥 Life/Health Recommendation", 
    "🚗 Motor Recommendation",
    "🧪  Health Recommendation"
])


with tab1:
    st.markdown("<div class='main-title'>Linda – Voice-Ready AI Assistant</div>", unsafe_allow_html=True)
    st.markdown("""
    **Linda** is a voice- and text-based AI chatbot designed for fast, intelligent support across **Web, Mobile, and WhatsApp**.<br>
    Whether your customers prefer speaking or typing, Linda delivers accurate and insightful help—24/7, in any context.
    
    ---
    ### What is Linda?
    Linda is a next-generation voice-first virtual assistant that enables conversational, human-like interactions in multiple languages.  
    It uses advanced conversational AI to understand intent, emotion, and context, turning customer engagement into smart dialogue and actionable outcomes.
    - **Flexible, natural conversations**: Voice or text, on Web, Mobile, WhatsApp.
    - **Always-on support**: AI-driven insights around the clock.
    - **Brand & domain customization**: Linda learns your unique voice and business needs.
    
    ---
    ### Key Capabilities
    - Voice + Text conversations (Web & WhatsApp)
    - NLP powered by Rasa, Dialogflow, or custom models
    - Voice-to-text via Whisper; smart generative AI responses
    - Deployed anywhere: Web, Mobile, WhatsApp
    - Multilingual & regional language adaptation
    - Brand voice and tone customization
    
    ---
    ### What Makes Linda Different?
    Linda isn’t just a chatbot—it’s a fully AI-enabled virtual assistant:
    - Advanced AI for intent, sentiment, and context understanding
    - Multi-turn conversation support with contextual memory
    - Dynamic, unscripted answers using generative AI and LLM integration
    - Adaptive learning from ongoing user interactions
    - Multimodal NLP (voice, text, user behavior)
    
    ---
    ### Technology Stack
    - Speech Recognition: **OpenAI Whisper**
    - NLP & Dialog Management: **Rasa, Dialogflow, custom transformers**
    - Smart AI Response: **OpenAI/GPT-based models**
    - TTS: **ElevenLabs**, Azure TTS
    - Platform: Web, Mobile, WhatsApp
    - Hosting: On-premise | Private Cloud | Public Cloud
    - Security: End-to-end encryption, role-based access
    
    ---
    ### Real Business Value
    - Reduce support costs by up to **50%**
    - Resolve queries **70% faster** with conversational voice automation
    - Increase conversion rates & customer satisfaction
    - 24/7, multilingual, always-on engagement
    - Seamless integration with CRMs, ERPs, and key business tools
    
    ---
    ### Industries & Use Cases
    | Industry             | Use Cases                                                      |
    |----------------------|---------------------------------------------------------------|
    | Insurance            | Claims, renewals, IVR automation                              |
    | Banking & Finance    | Balance checks, fraud alerts, loan help                       |
    | Healthcare           | Appointments, triage, patient follow-ups                      |
    | Retail & eCommerce   | Order tracking, returns, recommendations                      |
    | Education            | Student Q&A, voice tutors, admissions info                    |
    | Energy & Utilities   | Billing, outages, new service requests                        |
    | Real Estate          | Property info, site visits, lead engagement                   |
    | Travel & Hospitality | Bookings, check-ins, itinerary updates                        |
    
    ---
    ### Customization
    - Multilingual and regional adaptation for diverse audiences
    - Brand voice and tone configuration
    - Domain-specific knowledge and training
    - API and legacy system integration
    
    ---
    ### Linda = Smart Conversations + Tangible Outcomes
    Built by MS Risktec. Powered by AI. Tailored for Your Industry.<br>
    📩 info@msrisktec.com &nbsp;&nbsp; 🌐 www.msrisktec.com
    """, unsafe_allow_html=True)

with tab2:
    st.subheader("Explore Insurance Datasets")
    # Add Custom Upload Option
    datasetchoice = st.selectbox("Select Dataset", ["LifeHealth", "Motor", "Health", "Custom Upload"])
    dfcustom = None
    custom_cols = []
    custom_target = None

    if datasetchoice == "Custom Upload":
        uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")
        if uploaded_file is not None:
            try:
                
                dfcustom = pd.read_csv(uploaded_file)
                dfcustom.columns = dfcustom.columns.str.strip()              # STRIP SPACES
                dfcustom = dfcustom.loc[:, ~dfcustom.columns.duplicated()]   # REMOVE DUPLICATES
                st.success(f"Loaded {uploaded_file.name} ({dfcustom.shape[0]} rows, {dfcustom.shape[1]} columns)")
                st.dataframe(dfcustom.head())
                custom_cols = list(dfcustom.columns)
                custom_target = st.selectbox("Select target (for ML)", custom_cols)
            except Exception as e:
                st.error(f"Could not load file: {e}")

    # Select active DataFrame
    if datasetchoice == "LifeHealth":
        df = df_life
        targetcol = "Recommended_Product"
    elif datasetchoice == "Motor":
        df = df_motor
        targetcol = "RecommendedProduct"
    elif datasetchoice == "Health":
        df = df_synth
        targetcol = "Recommended_Product"
    elif datasetchoice == "Custom Upload" and dfcustom is not None and custom_target:
        df = dfcustom
        targetcol = custom_target.strip()
    else:
        df = df_synth
        targetcol = "Recommended_Product"  # fallback

    st.dataframe(df.head())

    mode = st.radio("Select mode", ["Query Dataset", "Run ML Recommender"], key="tab2mode")
    inputmethod = st.radio("Choose input method", ["Text", "Voice"], key="queryinputmethod")

    userquery = None
    if inputmethod == "Text":
        userquery = st.text_area("Enter your question or customer description here", key="querytext")
    else:
        if st.button("Speak", key="queryspeak"):
            userquery = get_voice_input()
        else:
            userquery = None

    if userquery:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        if mode == "Query Dataset":
            st.write(f"Querying dataset with: {userquery}")
            pandascode = ask_ai(userquery, df)
            pandascode = fix_column_names(df, pandascode)
            st.code(pandascode, language="python")
            try:
                result = eval(pandascode, {"df": df, "pd": pd, "np": np})
                st.write("Result:")
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    st.dataframe(result)
                if isinstance(result, pd.DataFrame) and result.select_dtypes(include=np.number).shape[1] > 0:
                    fig = px.histogram(result, x=result.columns[0])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write(result)
            except Exception as e:
                st.warning(f"Could not process as a data query. Switching to AI chat mode ...")
                genai.configure(api_key="AIzaSyADkL60GAaUgT9MzWoEGbTJkZfLgWtSE0")
                model = genai.GenerativeModel(model_name="gemini-2.0-flash")
                chatresponse = model.generate_content(f"You are Linda, a helpful AI assistant. Answer the following as clearly as possible for the user: {userquery}")
                st.markdown(f"AI: {chatresponse.text.strip()}")
        elif mode == "Run ML Recommender":
            # Extract features from text, fall back to chat
            prompt = f"""
You are an assistant that extracts insurance customer features from plain English descriptions when provided.
If the input contains identifiable insurance customer data (e.g. Age, Gender, Vehicle type), output ONLY a Python dictionary with keys and values matching these features.
If the input is a general question or unclear, respond naturally and helpfully in text.
Do not mix dictionary output and chat responses.

Examples:
Input: "Male, Age 30"
Output: {{'Gender': 'Male', 'Age': 30}}

Input: "What insurance coverage do I need for a 30-year-old male?"
Output: "For a 30-year-old male, recommended coverage..."

Input: "{userquery}"
"""


            genai.configure(api_key="AIzaSyA0L73IYOdcRIB27Lm3MHsfEEeWE-acyGs")
            model = genai.GenerativeModel(model_name="gemini-2.0-flash")
            response = model.generate_content(prompt)
            featuresdictstr = response.text.strip()
            try:
                featuresdict = ast.literal_eval(featuresdictstr)
                st.write("Extracted features:", featuresdict)
            except Exception:
                featuresdict = None
            if not featuresdict or not isinstance(featuresdict, dict):
                chatprompt = f"You are Linda, a helpful AI assistant. Answer the following question clearly: {userquery}"
                chatresponse = model.generate_content(chatprompt)
                st.markdown(f'<div style="font-size: 1.6em; font-weight: bold; color: #0084ff; padding: 15px; border-radius: 10px; background-color: #e6f0ff;">{chatresponse.text.strip()}</div>', unsafe_allow_html=True)
            else:
                inputdf = prepare_input(featuresdict, df, targetcol)
                try:
                    recomm = train_recommender(df, targetcol)
                    probs = recomm["pipeline"].predict_proba(inputdf)
                    predidx = np.argmax(probs, axis=1)[0]
                    predlabel = recomm["le"].inverse_transform([predidx])[0]
                    st.success(f"Recommended Product: {predlabel}")
                    probseries = pd.Series(probs[0], index=recomm["le"].classes_).sort_values(ascending=False)
                    st.bar_chart(probseries)
                    top3idx = np.argsort(probs[0])[-3:][::-1]
                    top3labels = recomm["le"].inverse_transform(top3idx)
                    top3probs = probs[0][top3idx]
                    st.markdown("Top 3 Recommendations:")
                    for label, prob in zip(top3labels, top3probs):
                        st.write(f"- {label}: {prob*100:.1f}%")
                except Exception as e:
                    st.error(f"Could not run prediction: {e}")

def show_model_metrics(recomm):
    metrics = recomm["metrics"]
    st.markdown("### 📈 Model Performance")
    st.table(pd.DataFrame([metrics]))

    # ROC Curve
    fpr, tpr, _ = roc_curve(recomm["y_test"], recomm["y_probs"][:,1], pos_label=1)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
    fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)

    # Feature Importance
    clf = recomm["pipeline"].named_steps["clf"]
    feat_names = recomm["pipeline"].named_steps["preprocessor"].get_feature_names_out()
    importances = clf.feature_importances_
    feat_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(10)
    st.bar_chart(feat_df.set_index("feature"))

# ---------------------------
# 🏥 Tab 2: Life/Health Recommender
# ---------------------------
with tab3:
    st.header("🏥 Life/Health Insurance Product Recommendation")
    st.dataframe(df_life.head())

    recomm = train_recommender(df_life, target_col="Recommended_Product")

    sample = collect_user_inputs(df_life, exclude_cols=["Recommended_Product"], dataset_name="life")
    st.dataframe(sample)

    try:
        user_df = prepare_input(sample.iloc[0].to_dict(), df_life, target_col="Recommended_Product")
        probs = recomm["pipeline"].predict_proba(user_df)
        pred_idx = np.argmax(probs, axis=1)[0]
        pred_label = recomm["le"].inverse_transform([pred_idx])[0]

        st.success(f"✅ Recommended Product: {pred_label}")

# Bar chart for all class probabilities
        prob_series = pd.Series(probs[0], index=recomm["le"].classes_).sort_values(ascending=False)
        st.bar_chart(prob_series)

        # Show Top 3 recommendations with % values
        top3_idx = np.argsort(probs[0])[::-1][:3]
        top3_labels = recomm["le"].inverse_transform(top3_idx)
        top3_probs = probs[0][top3_idx]

        st.markdown("### 🔝 Top 3 Recommendations:")
        for label, prob in zip(top3_labels, top3_probs):
            st.write(f"- **{label}** : {prob*100:.1f}%")

    except Exception as e:
        st.error(f"Could not predict: {e}")

    # show_model_metrics(recomm)

# ---------------------------
# 🚗 Tab 3: Motor Recommender
# ---------------------------
with tab4:
    st.header("🚗 Motor Insurance Product Recommendation")
    st.dataframe(df_motor.head())

    recomm_motor = train_recommender(df_motor, target_col="RecommendedProduct")

    sample_motor = collect_user_inputs(df_motor, exclude_cols=["RecommendedProduct"], dataset_name="motor")
    st.dataframe(sample_motor)

    try:
        user_df_motor = prepare_input(sample_motor.iloc[0].to_dict(), df_motor, target_col="RecommendedProduct")
        probs_motor = recomm_motor["pipeline"].predict_proba(user_df_motor)
        pred_idx_motor = np.argmax(probs_motor, axis=1)[0]
        pred_label_motor = recomm_motor["le"].inverse_transform([pred_idx_motor])[0]

        st.success(f"✅ Recommended Product: {pred_label}")

        # Bar chart for all class probabilities
        prob_series = pd.Series(probs[0], index=recomm["le"].classes_).sort_values(ascending=False)
        st.bar_chart(prob_series)

        # Show Top 3 recommendations with % values
        top3_idx = np.argsort(probs[0])[::-1][:3]
        top3_labels = recomm["le"].inverse_transform(top3_idx)
        top3_probs = probs[0][top3_idx]

        st.markdown("### 🔝 Top 3 Recommendations:")
        for label, prob in zip(top3_labels, top3_probs):
            st.write(f"- **{label}** : {prob*100:.1f}%")

    except Exception as e:
        st.error(f"Could not predict: {e}")

    # show_model_metrics(recomm_motor)

# ---------------------------
# 🧪 Tab 4: Synthetic Health Recommender
# ---------------------------
with tab5:
    st.header("🧪 Health Insurance Product Recommendation")
    st.dataframe(df_synth.head())

    recomm_synth = train_recommender(df_synth, target_col="Recommended_Product")

    sample_synth = collect_user_inputs(df_synth, exclude_cols=["Recommended_Product"], dataset_name="synth")
    st.dataframe(sample_synth)

    try:
        user_df_synth = prepare_input(sample_synth.iloc[0].to_dict(), df_synth, target_col="Recommended_Product")
        probs_synth = recomm_synth["pipeline"].predict_proba(user_df_synth)
        pred_idx_synth = np.argmax(probs_synth, axis=1)[0]
        pred_label_synth = recomm_synth["le"].inverse_transform([pred_idx_synth])[0]

        st.success(f"✅ Recommended Product: {pred_label}")

# Bar chart for all class probabilities
        prob_series = pd.Series(probs[0], index=recomm["le"].classes_).sort_values(ascending=False)
        st.bar_chart(prob_series)

        # Show Top 3 recommendations with % values
        top3_idx = np.argsort(probs[0])[::-1][:3]
        top3_labels = recomm["le"].inverse_transform(top3_idx)
        top3_probs = probs[0][top3_idx]

        st.markdown("### 🔝 Top 3 Recommendations:")
        for label, prob in zip(top3_labels, top3_probs):
            st.write(f"- **{label}** : {prob*100:.1f}%")

    except Exception as e:
        st.error(f"Could not predict: {e}")

    # show_model_metrics(recomm_synth)
