# app
This application uses **Scikit-Learn's Diabetes Dataset** to train a **Random Forest Classifier**. By analyzing 10 key biological indicators—including blood pressure, body mass index, and serum lipid levels—the model classifies a patient's risk level into **Low Risk** or **High Risk** alongside exact probability scores.

# Diabetes Risk Predictor

An interactive machine learning web dashboard built with **Streamlit**, **Scikit-Learn**, and **Plotly** that assesses a patient's risk of high diabetes progression based on standardized baseline clinical features.

## Features

- **Interactive Sidebar Controls:** Easily adjust clinical values (Age, BMI, Blood Pressure, Cholesterol levels, etc.) via sliders and select boxes.
- **Instant Risk Scoring:** Real-time risk classification backed by probability percentages and progress bars.
- **Dynamic Feature Importance Visualizations:** Displays an interactive **Plotly horizontal bar chart** showing which physiological metrics drive the model's decisions.
- **Streamlit Caching (`st.cache_resource`):** Optimized performance ensures the machine learning model loads once without slowing down user interactions.


## Tech Stack

- **Frontend / Web App:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualizations:** [Plotly Express](https://plotly.com/python/)


## Deployment Instructions (Streamlit Community Cloud)

1. **Fork or Push to GitHub:**
   Upload `app.py` and `requirements.txt` to a **public** GitHub repository.

2. **Deploy to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
   - Click **New app**.
   - Select your **Repository**, **Branch** (`main`), and set the main file path to `app.py`.
   - Click **Deploy!**


## Local Setup

```bash
# Clone repository
git clone [https://github.com/your-username/diabetes-risk-predictor.git](https://github.com/your-username/diabetes-risk-predictor.git)
cd diabetes-risk-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
