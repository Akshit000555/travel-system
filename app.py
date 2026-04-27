"""
Smart Travel Recommendation System 2.0
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob

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

destinations_df, hotels_df = load_data()

with st.sidebar:
    st.markdown("### ✈️ Travel Preferences")
    budget = st.slider("💰 Budget (₹)", 5000, 50000, 20000, 1000)
    travel_type = st.selectbox("🎯 Type", ["Beach","Mountain","City","Adventure","Romantic"])
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
        location_hotels = hotels_df[hotels_df["Location"]==top["Name"]].sort_values("Rating",ascending=False).head(5)
        
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
t1,t2 = st.tabs(["Popularity","Prices"])

with t1:
    fig = px.bar(destinations_df.sort_values("Popularity",ascending=False).head(10), x="Popularity", y="Name", orientation="h", color="Popularity", color_continuous_scale="Purples")
    fig.update_layout(height=400,showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with t2:
    fig2 = px.histogram(hotels_df, x="Price", nbins=20, color="Segment")
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#fff;padding:1rem"><p>© 2026 Smart Travel Recommendation System. All Rights Reserved.</p></div>', unsafe_allow_html=True)
