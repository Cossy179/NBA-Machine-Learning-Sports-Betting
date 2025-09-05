# 🔑 NBA Machine Learning Sports Betting - API Setup Guide

This guide will help you set up the necessary API keys for the enhanced Phase 2 system. The system now uses proper environment variable management and real API integrations.

## 📋 Quick Setup

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** with your API keys (see sections below)

3. **Test the configuration:**
   ```bash
   python src/Utils/ConfigManager.py
   ```

## 🔧 Required APIs

### 1. The Odds API (CRITICAL - For Betting Odds)
- **Website:** https://the-odds-api.com/
- **Free Tier:** 500 requests/month
- **Cost:** $10/month for 10,000 requests
- **What it provides:** Real betting odds, line movements, sportsbook data

**Setup Steps:**
1. Go to https://the-odds-api.com/
2. Sign up for a free account
3. Get your API key from the dashboard
4. Add to `.env`: `THE_ODDS_API_KEY=your_api_key_here`

**Why it's critical:** This is the only reliable source for real-time betting odds and line movements, which are essential for the system's market intelligence features.

### 2. SportsRadar NBA API (RECOMMENDED - For Injury Data)
- **Website:** https://developer.sportradar.com/
- **Free Trial:** Available
- **Cost:** $69/month for basic NBA package
- **What it provides:** Comprehensive NBA data, injury reports, player stats

**Setup Steps:**
1. Go to https://developer.sportradar.com/
2. Create an account and request NBA API access
3. Choose the "NBA v8 Trial" or paid plan
4. Get your API key
5. Add to `.env`: `SPORTSRADAR_API_KEY=your_api_key_here`

**Alternative:** If too expensive, the system will fall back to free APIs.

## 🆓 Free/Optional APIs

### 3. NBA Stats API (FREE - Basic NBA Data)
- **Website:** https://stats.nba.com/ (public API)
- **Free Tier:** Rate limited but free
- **What it provides:** Official NBA statistics, team rosters

**Setup:**
- No API key required
- Set in `.env`: `NBA_STATS_API_KEY=not_required`
- The system uses proper headers to avoid rate limiting

### 4. RapidAPI Sports (FREEMIUM - Alternative NBA Data)
- **Website:** https://rapidapi.com/api-sports/api/api-nba/
- **Free Tier:** 100 requests/day
- **Cost:** $10/month for 10,000 requests
- **What it provides:** Alternative NBA data source

**Setup Steps:**
1. Go to https://rapidapi.com/api-sports/api/api-nba/
2. Subscribe to the free plan
3. Get your RapidAPI key
4. Add to `.env`: `RAPIDAPI_KEY=your_rapidapi_key_here`

### 5. News API (FREE - Media Sentiment)
- **Website:** https://newsapi.org/
- **Free Tier:** 1,000 requests/day
- **What it provides:** News articles for sentiment analysis

**Setup Steps:**
1. Go to https://newsapi.org/
2. Register for free account
3. Get your API key
4. Add to `.env`: `NEWS_API_KEY=your_news_api_key_here`

## 💰 Cost Breakdown

### Minimum Setup (Basic Functionality)
- **The Odds API:** $10/month
- **Total:** $10/month

### Recommended Setup (Full Features)
- **The Odds API:** $10/month  
- **SportsRadar:** $69/month
- **Total:** $79/month

### Free Alternative Setup
- **The Odds API:** $10/month (essential)
- **All other APIs:** Free
- **Total:** $10/month

## 🔒 Security Notes

- The `.env` file is automatically ignored by git
- Never commit API keys to version control
- Keep your API keys secure and rotate them periodically
- Monitor your API usage to avoid unexpected charges

## 🧪 Testing Your Setup

After setting up your APIs, test the configuration:

```bash
# Test configuration
python src/Utils/ConfigManager.py

# Test real-time data provider
python -c "from src.DataProviders.RealTimeDataProvider import RealTimeDataProvider; provider = RealTimeDataProvider(); print('✅ Setup successful!')"
```

## 📊 API Usage Monitoring

The system includes built-in rate limiting and usage tracking:

- **Automatic retries** with exponential backoff
- **Request caching** to minimize API calls
- **Fallback systems** when primary APIs are unavailable
- **Usage logging** to monitor API consumption

## 🚫 APIs We Removed

- **Weather API:** Removed as weather doesn't affect indoor basketball games
- **Social Media APIs:** Replaced with news-based sentiment analysis

## 🆘 Troubleshooting

### Common Issues:

1. **"No API key" errors:**
   - Check your `.env` file exists and has correct keys
   - Ensure no extra spaces around the `=` sign

2. **Rate limiting errors:**
   - The system will automatically retry with backoff
   - Consider upgrading to higher API tiers

3. **Import errors:**
   - Install python-dotenv: `pip install python-dotenv`
   - Check file paths in your configuration

### Getting Help:

- Check the configuration status: `python src/Utils/ConfigManager.py`
- Review API documentation for specific endpoints
- Monitor the console output for detailed error messages

## 🎯 What Each API Provides

| API | Injuries | Lineups | Odds | News | Player Stats |
|-----|----------|---------|------|------|--------------|
| The Odds API | ❌ | ❌ | ✅ | ❌ | ❌ |
| SportsRadar | ✅ | ✅ | ❌ | ❌ | ✅ |
| NBA Stats | ⚠️ | ⚠️ | ❌ | ❌ | ✅ |
| RapidAPI | ✅ | ✅ | ❌ | ❌ | ✅ |
| News API | ❌ | ❌ | ❌ | ✅ | ❌ |

✅ = Full support, ⚠️ = Limited support, ❌ = Not available

## 🚀 Next Steps

Once your APIs are set up:

1. **Test the system:** Run predictions with real data
2. **Monitor usage:** Keep track of API consumption  
3. **Optimize calls:** The system caches data to minimize requests
4. **Scale up:** Upgrade API tiers as needed for production use

---

**💡 Pro Tip:** Start with the free/cheap APIs to test the system, then upgrade to premium APIs once you're satisfied with the performance!
