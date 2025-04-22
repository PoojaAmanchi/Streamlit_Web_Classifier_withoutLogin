import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Web Service Classifier & Recommender", layout="wide")

st.title("🌐 Web Service Classifier & Recommender")

tabs = st.tabs(["📊 Classification", "🤖 Recommendation"])

# Convert dataframe to CSV function
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Load or upload data
with tabs[0]:
    st.subheader("Step 1: Upload Web Service Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

        # Simulate classification model
        feature_cols = ['Response Time', 'Availability', 'Throughput', 'Reliability']
        X = df[feature_cols]
        y = np.random.choice(["Bronze", "Silver", "Gold", "Platinum"], size=len(X))  # Simulated labels

        model = RandomForestClassifier().fit(X, y)  # Fit for LIME use

        df['Classification'] = y
        st.success("✅ Classification Done")
        st.dataframe(df[['Service Name', 'Classification']])

        # Explain one prediction using LIME
        st.subheader("Explain a Prediction (LIME)")
        explainer = LimeTabularExplainer(X.values, feature_names=feature_cols, class_names=model.classes_, mode="classification")
        index = st.slider("Select a service to explain", 0, len(X) - 1, 0)
        exp = explainer.explain_instance(X.iloc[index].values, model.predict_proba)
        st.text(f"Explanation for Service: {df.iloc[index]['Service Name']}")
        st.markdown(exp.as_list())

        # Visualize the model's performance
        st.subheader("Model Performance (Confusion Matrix)")
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(4, 3))  # Create a new figure and axis
        disp.plot(ax=ax)  # Plot the confusion matrix on the specified axis
        st.pyplot(fig)

        # Feature importances visualization
        st.subheader("Feature Importances")
        feature_importances = model.feature_importances_
        fig, ax = plt.subplots()
        ax.barh(feature_cols, feature_importances)
        ax.set_xlabel('Importance')
        st.pyplot(fig)

# Recommendation Tab
with tabs[1]:
    st.subheader("Step 2: Enter QoS Values for Recommendations")
    rt = st.number_input("Response Time", min_value=0)
    av = st.number_input("Availability", min_value=0.0, max_value=1.0)
    tp = st.number_input("Throughput", min_value=0)
    rel = st.number_input("Reliability", min_value=0.0, max_value=1.0)

    if uploaded_file:
        feature_cols = ['Response Time', 'Availability', 'Throughput', 'Reliability']

        if st.button("Get Top 10 Recommendations"):
            features = df[feature_cols]
            query = [rt, av, tp, rel]

            def content_based_recommendation(query, df, feature_cols):
                cosine_sim = cosine_similarity(df[feature_cols], [query])
                return np.argsort(cosine_sim.flatten())[:10]

            recommended_indices = content_based_recommendation(query, df, feature_cols)
            st.session_state["recommendations"] = recommended_indices

        if "recommendations" in st.session_state:
            st.success("🎯 Top 10 Similar Services")
            for rank, idx in enumerate(st.session_state["recommendations"], 1):
                st.markdown(f"**{rank}. {df.iloc[idx]['Service Name']}** - [Visit]({df.iloc[idx]['URL']})")

            # Export recommendations as CSV
            recommended_df = df.iloc[st.session_state["recommendations"]][['Service Name', 'URL']]
            csv = convert_df(recommended_df)
            st.download_button(
                label="Download Recommendations",
                data=csv,
                file_name='recommendations.csv',
                mime='text/csv',
            )

        # Visualize QoS features on a scatter plot
        st.subheader("QoS Feature Visualization")
        fig = px.scatter(df, x="Response Time", y="Availability", color="Classification", hover_data=["Service Name"])
        st.plotly_chart(fig)

        # Export classification results as CSV
        classification_df = df[['Service Name', 'Classification']]
        csv = convert_df(classification_df)
        st.download_button(
            label="Download Classification Results",
            data=csv,
            file_name='classification_results.csv',
            mime='text/csv',
        )
