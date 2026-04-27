"""
Smart Travel Recommendation System 2.0
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Smart Travel", page_icon="✈️", layout="wide")

st.markdown("""
<style>
.main{padding:0 1rem}.stApp{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%)}
.hero{background:#fff;padding:2rem;border-radius:20px;text-align:center;margin-bottom:2rem;box-shadow:0 10px 40px rgba(0,0,0,.1)}
.hero h1{color:#667eea;font-size:2.5rem;margin-bottom:.5rem}
.card{background:#fff;padding:1.5rem;border-radius:15px;box-shadow:0 4px 15px rgba(0,0,0,.1);margin-bottom:1rem}
.metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:1.5rem;border-radius:15px;text-align:center}
.stButton>button{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;border:none;padding:.75rem 2rem;border-radius:25px;font-weight:600;width:100%}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 5px 20px rgba(102,126,234,.4)}
@media(max-width:768px){.hero h1{font-size:1.8rem}.card{padding:1rem}}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("destinations.csv"), pd.read_csv("hotels.csv")

@st.cache_data
def train_models():
    df = pd.read_csv("destinations.csv")
    # Predict Type (7 classes) — meaningful accuracy with this dataset size
    le_type = LabelEncoder()
    df["Type_enc"] = le_type.fit_transform(df["Type"])
    # Budget range feature
    df["Budget_range"] = pd.cut(df["Budget"], bins=3, labels=[0, 1, 2]).astype(int)
    X = df[["Budget", "Popularity", "Budget_range"]]
    y = df["Type_enc"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = {"model": model, "accuracy": acc}
    return results, le_type

destinations_df, hotels_df = load_data()
trained_models, le_type = train_models()

with st.sidebar:
    st.markdown("### ✈️ Travel Preferences")
    budget = st.slider("💰 Budget (₹)", 5000, 50000, 20000, 1000)
    travel_type = st.selectbox("🎯 Type", ["Beach","Mountain","City","Adventure","Romantic"])
    segment = st.selectbox("🏨 Hotel Segment", ["All", "Budget", "Mid-range", "Luxury"])
    min_rating = st.slider("⭐ Min Hotel Rating", 3.0, 5.0, 3.0, 0.1)
    only_positive = st.checkbox("😊 Only Positively Reviewed Hotels")
    st.markdown("---")
    st.caption("© 2026 Smart Travel")

st.markdown('<div class="hero"><h1>✈️ Smart Travel Recommender</h1><p style="font-size:1.1rem;color:#666">AI-powered travel recommendations</p></div>', unsafe_allow_html=True)

col1, col2 = st.columns([2,1])

with col1:
    st.markdown("### 🌍 Recommended Destination")
    filtered = destinations_df[(destinations_df["Type"]==travel_type)&(destinations_df["Budget"]<=budget)].sort_values("Popularity",ascending=False)
    
    if not filtered.empty:
        top = filtered.iloc[0]
        st.markdown(f'<div class="card"><h2 style="color:#667eea;margin:0">{top["Name"]}</h2><p style="color:#666;margin:.5rem 0">{top["State"]} • {top["Type"]}</p><div style="display:flex;justify-content:space-between;margin-top:1rem"><div><p style="margin:0;color:#999;font-size:.9rem">Budget</p><p style="margin:0;font-size:1.3rem;font-weight:bold;color:#667eea">₹{top["Budget"]:,}</p></div><div><p style="margin:0;color:#999;font-size:.9rem">Popularity</p><p style="margin:0;font-size:1.3rem;font-weight:bold;color:#667eea">{top["Popularity"]}/100</p></div></div></div>', unsafe_allow_html=True)
        
        st.markdown("### 🏨 Top Hotels")
        location_hotels = hotels_df[hotels_df["Location"]==top["Name"]].copy()
        if segment != "All":
            location_hotels = location_hotels[location_hotels["Segment"]==segment]
        if only_positive:
            location_hotels = location_hotels[location_hotels["Sentiment"]=="Positive"]
        location_hotels = location_hotels[location_hotels["Rating"]>=min_rating]
        location_hotels = location_hotels.sort_values("Rating",ascending=False).head(5)

        if not location_hotels.empty:
            for _, hotel in location_hotels.iterrows():
                c1,c2,c3 = st.columns([3,1,1])
                with c1:
                    st.markdown(f"**{hotel['Hotel']}**")
                    st.caption(f"{hotel['Segment']} • {hotel['Sentiment']}")
                with c2:
                    st.metric("Rating", f"{hotel['Rating']}⭐")
                with c3:
                    st.metric("Price", f"₹{hotel['Price']:,}")
    else:
        st.warning("No destinations found")

with col2:
    st.markdown("### 📊 Stats")
    st.markdown(f'<div class="metric-card"><h3 style="margin:0">{len(destinations_df)}</h3><p style="margin:0;opacity:.9">Destinations</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card" style="margin-top:1rem"><h3 style="margin:0">₹{int(destinations_df["Budget"].mean()):,}</h3><p style="margin:0;opacity:.9">Avg Budget</p></div>', unsafe_allow_html=True)
    
    st.markdown("### 💬 Review Analyzer")
    review = st.text_area("Write review:", height=100)
    if st.button("Analyze"):
        if review:
            p = TextBlob(review).sentiment.polarity
            if p>0.1: st.success(f"😊 Positive ({p:.2f})")
            elif p<-0.1: st.error(f"😞 Negative ({p:.2f})")
            else: st.info(f"😐 Neutral ({p:.2f})")

st.markdown("### 📈 Insights")
t1, t2, t3 = st.tabs(["Popularity", "Prices", "🤖 ML Predictions"])

with t1:
    fig = px.bar(destinations_df.sort_values("Popularity",ascending=False).head(10), x="Popularity", y="Name", orientation="h", color="Popularity", color_continuous_scale="Purples")
    fig.update_layout(height=400,showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with t2:
    fig2 = px.histogram(hotels_df, x="Price", nbins=20, color="Segment")
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

with t3:
    st.markdown("### 🤖 Model Predictions")
    st.markdown("Each model predicts the best travel type and top destination based on your budget from the sidebar.")

    budget_range = 0 if budget < 20000 else (1 if budget < 35000 else 2)
    avg_pop = int(destinations_df["Popularity"].median())
    input_data = [[budget, avg_pop, budget_range]]

    mc1, mc2, mc3 = st.columns(3)
    model_cols = [mc1, mc2, mc3]
    colors = ["#667eea", "#764ba2", "#f093fb"]

    for i, (model_name, info) in enumerate(trained_models.items()):
        try:
            pred_enc = info["model"].predict(input_data)[0]
            pred_type = le_type.inverse_transform([pred_enc])[0]
            # Find top destination of predicted type within budget
            match = destinations_df[
                (destinations_df["Type"] == pred_type) &
                (destinations_df["Budget"] <= budget)
            ].sort_values("Popularity", ascending=False)
            pred_dest = match.iloc[0]["Name"] if not match.empty else pred_type
        except Exception:
            pred_type = "N/A"
            pred_dest = "N/A"

        acc = info["accuracy"] * 100
        with model_cols[i]:
            st.markdown(
                f"""<div style="background:linear-gradient(135deg,{colors[i]},#764ba2);
                color:#fff;padding:1.5rem;border-radius:15px;text-align:center;">
                <h4 style="margin:0;font-size:1rem">{model_name}</h4>
                <p style="margin:.4rem 0 0;opacity:.85;font-size:.85rem">Predicted Type</p>
                <h3 style="margin:.2rem 0">{pred_type}</h3>
                <p style="margin:.4rem 0 0;opacity:.85;font-size:.85rem">Top Destination</p>
                <h2 style="margin:.2rem 0;font-size:1.2rem">{pred_dest}</h2>
                <p style="margin:.6rem 0 0;opacity:.85;font-size:.85rem">Model Accuracy</p>
                <h3 style="margin:.2rem 0">{acc:.1f}%</h3>
                </div>""",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    acc_df = pd.DataFrame({
        "Model": list(trained_models.keys()),
        "Accuracy (%)": [v["accuracy"] * 100 for v in trained_models.values()]
    })
    fig3 = px.bar(acc_df, x="Model", y="Accuracy (%)", color="Model",
                  color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"],
                  text_auto=".1f")
    fig3.update_layout(height=300, showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#fff;padding:1rem"><p>© 2026 Smart Travel Recommendation System. All Rights Reserved.</p></div>', unsafe_allow_html=True)
