import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------
# Load Dataset
# ---------------------------------
df = pd.read_csv("/workspaces/project-131/notebook/Gift_Recommendations_Min5.csv")

# ---------------------------------
# Prepare Data
# ---------------------------------
X = df[['Gender', 'Age', 'Interest/Hobby']].copy()
y = df['Gift Recommendation']

gender_encoder = LabelEncoder()
hobby_encoder = LabelEncoder()

X['Gender'] = gender_encoder.fit_transform(X['Gender'])
X['Interest/Hobby'] = hobby_encoder.fit_transform(X['Interest/Hobby'])

scaler = StandardScaler()
X['Age'] = scaler.fit_transform(X[['Age']])

# ---------------------------------
# Train Model
# ---------------------------------
xtrain, xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)

model.fit(xtrain, ytrain)

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="AI Gift Recommendation System",
    page_icon="🎁",
    layout="centered"
)

st.title("🎁 AI-Based Gift Preference Predictor")
st.write("Get personalized gift recommendations based on age, gender, and interests.")

# ---------------------------------
# Reset Button Logic
# ---------------------------------
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

# ---------------------------------
# Form
# ---------------------------------
with st.form(key=f"gift_form_{st.session_state.reset_counter}"):

    gender = st.selectbox(
        "Gender",
        ["Select Gender"] + sorted(df["Gender"].unique().tolist())
    )

    hobby = st.selectbox(
        "Interest/Hobby",
        ["Select Interest/Hobby"] + sorted(df["Interest/Hobby"].unique().tolist())
    )
    
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=70,
        value=None
    
    )

    col1, col2 = st.columns(2)

    with col1:
        predict_btn = st.form_submit_button("🎁 Recommend Gift")

    with col2:
        reset_btn = st.form_submit_button("🔄 Reset")

# ---------------------------------
# Prediction
# ---------------------------------
if gender == "Select Gender":
    st.warning("Please select a gender.")

elif hobby == "Select Interest/Hobby":
    st.warning("Please select an interest/hobby.")
elif hobby not in df["Interest/Hobby"].unique():
    st.warning("Selected hobby is not available in the dataset.")

elif age is None:
    st.warning("Please enter a valid age.")

else:
    gender_encoded = gender_encoder.transform([gender])[0]
    hobby_encoded = hobby_encoder.transform([hobby])[0]
    age_scaled = scaler.transform([[age]])[0][0]

    input_data = pd.DataFrame(
        [[gender_encoded, age_scaled, hobby_encoded]],
        columns=['Gender', 'Age', 'Interest/Hobby']
    )

    prediction = model.predict(input_data)

    st.success(f"🎁 Recommended Gift: {prediction[0]}")
# ---------------------------------
# Reset Form
# ---------------------------------
if reset_btn:
    st.session_state.reset_counter += 1
    st.rerun()