# Advanced Sentiment Analysis Improvements

## Overview

The sentiment analysis module has been significantly upgraded from a simple keyword-based approach to a sophisticated BERT-based system that captures context, sarcasm, and nuanced sentiment in NBA news and social media.

## Key Improvements

### 1. BERT-Based Sentiment Classification

**Before:** Simple keyword counting (positive/negative word matching)  
**After:** Transformer models (BERT/RoBERTa) that understand context and subtle language nuances

- Uses pre-trained models optimized for social media and news sentiment
- Falls back gracefully to keyword-based method if transformers unavailable
- Models supported:
  - `cardiffnlp/twitter-roberta-base-sentiment-latest` (default, fast, good for social media)
  - `nlptown/bert-base-multilingual-uncased-sentiment` (multilingual)
  - `finiteautomata/bertweet-base-sentiment-analysis` (Twitter-optimized)

### 2. Expanded Data Sources

**Before:** ESPN articles and Reddit posts only  
**After:** Multiple sources with comprehensive coverage

- **ESPN** - Team news and headlines
- **The Athletic** - Premium sports journalism (via RSS feeds)
- **Reddit** - r/NBA discussions and posts
- **Team Press Releases** - Official team announcements
- **Injury Reports** - ESPN injury pages

### 3. Time-Decay Weighting

Recent news is weighted more heavily than older articles:

- Articles weighted using exponential decay: `weight = exp(-log(2) * age_days / half_life)`
- Default half-life: 3 days (weight drops to 50% after 3 days)
- Minimum weight: 0.1 (even old articles contribute)
- Recent breaking news has maximum influence

### 4. High-Impact News Detection

Automatically classifies news by impact level:

- **High Impact:** Season-ending injuries, trades, signings, suspensions
- **Medium Impact:** Day-to-day injuries, contract extensions, minor trades
- **Routine:** Game summaries, previews, general news

High-impact news:
- Receives 2x weight in sentiment calculation
- Allows larger probability adjustments (±10% vs ±5%)
- Triggers additional confidence boosts

### 5. Integrated Feature Engineering

Injury-flagging and public-confidence are now integrated as features:

- **Injury Concerns:** Reduces overall sentiment (0-1 scale)
- **Public Confidence:** Slight boost when public sentiment aligns
- Both feed into the final sentiment calculation rather than simple adjustments

### 6. Calibrated Probability Adjustments

**Before:** Fixed ±5% adjustment  
**After:** Dynamic ±5-10% range based on sentiment strength and news impact

- **Regular news:** ±5% maximum adjustment
- **High-impact news:** ±10% maximum adjustment
- **Scaling:** Adjustments scaled by sentiment strength (weak/moderate/strong)
- Prevents over-correction while allowing significant adjustments for major news

## Usage

### Basic Usage

The improved sentiment analyzer is backward-compatible with existing code:

```python
from src.Utils.SentimentAnalysis import NBASentimentAnalyzer

analyzer = NBASentimentAnalyzer(use_bert=True)

# Get sentiment for a game
sentiment = analyzer.get_game_sentiment("Lakers", "Celtics")

# Adjust predictions
adjusted_prediction = analyzer.adjust_prediction_with_sentiment(
    base_prediction, sentiment
)
```

### Fine-Tuning on NBA-Specific Data

To improve accuracy further, fine-tune the model on NBA-specific data:

1. **Collect labeled data:**
```bash
py src/Utils/finetune_sentiment_model.py --collect_data --data_path data/nba_sentiment_corpus.csv
```

2. **Label your data:**
   - Create CSV with columns: `text`, `label` (0=negative, 1=neutral, 2=positive), `source`, `date`
   - Include examples from ESPN, Reddit, injury reports, etc.

3. **Fine-tune model:**
```bash
py src/Utils/finetune_sentiment_model.py \
    --data_path data/nba_sentiment_corpus.csv \
    --base_model cardiffnlp/twitter-roberta-base-sentiment-latest \
    --output_dir Models/SentimentModels/nba_bert_sentiment \
    --epochs 3 \
    --batch_size 16
```

4. **Use fine-tuned model:**
   - Update `SentimentAnalysis.py` to load your custom model
   - Or specify model path in `BERTSentimentClassifier(model_name="path/to/model")`

## Technical Details

### Sentiment Calculation

The overall sentiment combines multiple factors:

```python
base_sentiment = weighted_sentiment  # Time-decay weighted articles
injury_factor = 1.0 - (injury_concerns * 0.3)
public_factor = public_confidence * 0.15 + 0.85

overall_sentiment = (
    base_sentiment * 0.40 +
    social_buzz * 0.20 +
    base_sentiment * injury_factor * 0.20 +
    momentum_narrative * 0.10 +
    public_confidence * 0.05 +
    media_attention * 0.05
)
```

### Probability Adjustment

```python
# Determine adjustment range
adjustment_range = 0.10 if high_impact_news else 0.05

# Scale by sentiment strength
if abs(sentiment_diff) > 0.3:
    scale_factor = 1.0  # Strong sentiment
elif abs(sentiment_diff) > 0.15:
    scale_factor = 0.7  # Moderate sentiment
else:
    scale_factor = 0.4  # Weak sentiment

probability_adjustment = sentiment_diff * adjustment_range * scale_factor
```

### Time-Decay Weighting

```python
def get_time_weight(age_days, decay_half_life_days=3.0):
    weight = exp(-log(2) * age_days / decay_half_life_days)
    return max(0.1, weight)  # Minimum 0.1
```

## Installation

Install required dependencies:

```bash
pip install transformers torch sentencepiece feedparser numpy
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Performance Considerations

- **BERT models:** First inference may take 5-10 seconds to load model
- **Caching:** Results cached for 1 hour to avoid redundant API calls
- **Rate limiting:** Built-in delays between requests to respect rate limits
- **Fallback:** Automatically falls back to keyword-based method if BERT unavailable

## Limitations & Future Improvements

1. **Data Collection:** The Athletic and team press releases may require authentication/subscription
2. **Fine-tuning:** Requires manually labeled NBA sentiment corpus for best results
3. **Real-time Updates:** Consider implementing webhook/streaming updates for breaking news
4. **Multi-language:** Current models optimized for English; could add multilingual support

## Example Output

```
📊 Analyzing sentiment (BERT-enhanced): Lakers vs Celtics
    Home sentiment: 0.68
    Away sentiment: 0.52
    Narrative: ⬆️ Lakers surging vs struggling Celtics
    ⚠️  High-impact news detected

Sentiment Differential: 0.160
Combined Buzz: 0.620
High-Impact News: True
```

## Integration Points

The improved sentiment analysis integrates with:

- `predict.py` - Main prediction pipeline
- `src/Predict/SuperAdvanced_Prediction_Engine.py` - Advanced prediction engine
- `src/DataProviders/RealTimeDataProvider.py` - Real-time data provider

All existing code continues to work with backward-compatible improvements.

