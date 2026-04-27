# ✈️ Smart Travel Recommendation System 2.0

An AI-powered travel recommendation web application built with Streamlit. Get personalized destination and hotel suggestions based on your budget and travel preferences, with real-time sentiment analysis.

## 🌟 Features

- **Smart Recommendations** - AI-powered destination suggestions based on budget and travel type
- **Hotel Listings** - Top-rated hotels with pricing and sentiment analysis
- **Review Analyzer** - Real-time sentiment analysis using TextBlob
- **Interactive Visualizations** - Popularity charts and price distribution graphs
- **Responsive Design** - Works seamlessly on desktop, tablet, and mobile
- **Modern UI** - Beautiful gradient design with card-based layout

## 🚀 Tech Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas
- **Visualizations:** Plotly
- **NLP:** TextBlob
- **Deployment:** Render / Streamlit Cloud

## 📦 Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Akshit000555/travel-system.git
cd travel-system

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🌐 Deploy on Render

1. Go to [render.com](https://render.com) and sign in
2. Click **New** → **Web Service**
3. Connect your GitHub repository: `travel-system`
4. Configure settings:
   - **Name:** travel-recommender (or any name)
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python -m streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
5. Click **Create Web Service**
6. Wait 2-3 minutes for deployment

Your app will be live at `https://your-app-name.onrender.com`

## 🎯 Deploy on Streamlit Cloud (Alternative)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select repository: `Akshit000555/travel-system`
5. Set main file: `app.py`
6. Click **Deploy**

## 📊 Dataset

The app uses three CSV files:
- `destinations.csv` - Travel destinations with budget and popularity
- `hotels.csv` - Hotel listings with ratings, prices, and segments
- `reviews.csv` - Sample reviews for sentiment analysis

## 🎨 Features Breakdown

### 1. Destination Recommendations
- Filter by travel type (Beach, Mountain, City, Adventure, Romantic)
- Budget-based filtering
- Popularity ranking

### 2. Hotel Recommendations
- Location-based hotel suggestions
- Rating and price comparison
- Segment classification (Budget, Mid-range, Luxury)
- Sentiment indicators (Positive, Neutral, Negative)

### 3. Sentiment Analysis
- Real-time review analysis
- Polarity scoring
- Emoji-based feedback

### 4. Data Visualizations
- Top destinations by popularity (Bar chart)
- Hotel price distribution (Histogram)
- Interactive Plotly charts

## 🛠️ Project Structure

```
travel-system/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── destinations.csv       # Destinations data
├── hotels.csv            # Hotels data
├── reviews.csv           # Reviews data
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md             # Documentation
```

## 📱 Responsive Design

The app is fully responsive with:
- Mobile-first approach
- Flexible grid layouts
- Adaptive font sizes
- Touch-friendly controls

## 🔧 Configuration

Edit `.streamlit/config.toml` to customize:
```toml
[server]
headless = true
enableCORS = false
port = 8501
```

## 📝 License

© 2026 Smart Travel Recommendation System. All Rights Reserved.

## 👥 Contributing

This is an academic project. For suggestions or issues, please open an issue on GitHub.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/)
- NLP using [TextBlob](https://textblob.readthedocs.io/)

---

**Made with ❤️ for travelers**
