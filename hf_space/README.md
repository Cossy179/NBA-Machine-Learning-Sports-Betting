---
title: NBA Predictions API
emoji: 🏀
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 🏀 NBA Predictions API

AI-powered NBA game predictions using LightGBM model with 839% ROI in backtests.

## Features

- **Daily Game Predictions**: Automated predictions for all NBA games
- **Confidence Scores**: Each prediction includes a confidence percentage
- **Parlay Suggestions**: AI-generated parlay combinations
- **Kelly Criterion**: Optimal bet sizing recommendations
- **Real-time Odds**: Integration with FanDuel sportsbook odds
- **Caching**: Predictions cached daily (expires at midnight UTC)

## API Endpoints

### `GET /`
Health check and API information

### `GET /api/predictions`
Get today's NBA predictions including:
- Game predictions with winner, confidence, and scores
- Parlay suggestions
- Betting recommendations

**Query Parameters:**
- `force_refresh` (bool): Force regenerate predictions (ignore cache)

**Response:**
```json
{
  "date": "2025-10-13",
  "generated_at": "2025-10-13T12:00:00Z",
  "games": [
    {
      "id": "unique_id",
      "home_team": "Lakers",
      "away_team": "Warriors",
      "game_time": "19:00",
      "home_odds": -110,
      "away_odds": +105,
      "prediction": {
        "winner": "Lakers",
        "confidence": 68.5,
        "home_score": 115,
        "away_score": 108,
        "spread_prediction": -7,
        "total_prediction": 223
      }
    }
  ],
  "parlays": [
    {
      "legs": ["Lakers ML", "Celtics ML"],
      "combined_odds": 2.65,
      "confidence": 62.3
    }
  ]
}
```

### `GET /api/games`
Get only game predictions (no parlays)

### `GET /api/parlays`
Get only parlay suggestions (no individual games)

### `GET /api/health`
Detailed health check including model status

### `GET /api/cache-status`
Check prediction cache status

### `POST /api/refresh`
Manually refresh predictions

## Model Details

- **Model Type**: LightGBM Gradient Boosting
- **Backtest Performance**: 839% ROI
- **Accuracy**: ~68.9% win rate on moneyline bets
- **Features**: 106 advanced statistical features
- **Training Data**: 2012-2024 NBA seasons

## Usage Example

```python
import requests

# Get today's predictions
response = requests.get('https://huggingface.co/spaces/Cossy179/Goon-Steen/api/predictions')
data = response.json()

for game in data['games']:
    pred = game['prediction']
    print(f"{game['away_team']} @ {game['home_team']}")
    print(f"Winner: {pred['winner']} ({pred['confidence']:.1f}% confidence)")
    print(f"Score: {pred['home_score']}-{pred['away_score']}")
    print()
```

## PHP Integration

```php
// Cache predictions for 1 hour
function fetchPredictions() {
    $cacheFile = __DIR__ . '/cache/predictions.json';
    if (file_exists($cacheFile) && time() - filemtime($cacheFile) < 3600) {
        return json_decode(file_get_contents($cacheFile), true);
    }
    
    $url = 'https://huggingface.co/spaces/Cossy179/Goon-Steen/api/predictions';
    $response = file_get_contents($url);
    file_put_contents($cacheFile, $response);
    return json_decode($response, true);
}

$predictions = fetchPredictions();
```

## Caching Strategy

- Predictions are generated once daily
- Cache expires at midnight UTC
- Use `force_refresh=true` to regenerate
- PHP backend should cache API responses for 1 hour

## License

MIT License - Free for personal and commercial use

## Credits

Built by Alex Halliday
Model trained on 2012-2024 NBA data
Powered by LightGBM and FastAPI

