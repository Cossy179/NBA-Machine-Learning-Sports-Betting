# 🏀 NBA Predictions - HuggingFace Space Deployment Guide

Complete step-by-step guide to deploy your NBA prediction models to HuggingFace Space.

## Prerequisites

- HuggingFace account (free): https://huggingface.co/join
- Git installed on your computer
- Git LFS installed: https://git-lfs.github.com/

## Step 1: Create HuggingFace Space

1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - **Owner**: Your username (Cossy179)
   - **Space name**: `Goon-Steen` (or your preferred name)
   - **License**: MIT
   - **Select SDK**: Docker
   - **Space hardware**: CPU basic (free tier)
   - **Visibility**: Public or Private

3. Click **Create Space**

## Step 2: Clone the Space Repository

Open terminal/PowerShell and run:

```bash
# Clone your new space
git clone https://huggingface.co/spaces/Cossy179/Goon-Steen
cd Goon-Steen

# Install Git LFS (if not already installed)
git lfs install
```

## Step 3: Copy Files to Space

Copy all files from `hf_space/` directory to your cloned space:

**Windows (PowerShell):**
```powershell
# From your NBA-Machine-Learning-Sports-Betting directory
Copy-Item -Path "hf_space\*" -Destination "path\to\Goon-Steen\" -Recurse -Force
```

**Mac/Linux:**
```bash
# From your NBA-Machine-Learning-Sports-Betting directory
cp -r hf_space/* /path/to/Goon-Steen/
```

Your space directory should now contain:
```
Goon-Steen/
├── app.py
├── model_runner.py
├── data_provider.py
├── requirements.txt
├── Dockerfile
├── README.md
├── .gitattributes
└── models/
    ├── SuperAdvanced_XGB_v1_lightgbm.txt
    ├── SuperAdvanced_XGB_v1_features.pkl
    └── player_correlations.csv
```

## Step 4: Configure Git LFS for Large Files

```bash
cd Goon-Steen

# Track large model files with Git LFS
git lfs track "models/*.txt"
git lfs track "models/*.pkl"
git lfs track "models/*.csv"

# Verify LFS is tracking
git lfs ls-files
```

## Step 5: Commit and Push to HuggingFace

```bash
# Add all files
git add .

# Commit
git commit -m "Initial deployment: NBA prediction API with LightGBM model"

# Push to HuggingFace
git push origin main
```

**Note**: This may take a few minutes as the model files are uploaded to Git LFS.

## Step 6: Wait for Build

1. Go to your space: https://huggingface.co/spaces/Cossy179/Goon-Steen
2. You'll see "Building..." status
3. Wait 2-5 minutes for Docker image to build
4. Once complete, you'll see "Running" status

## Step 7: Test the API

Once deployed, test your API endpoints:

### Health Check
```bash
curl https://cossy179-goon-steen.hf.space/api/health
```

### Get Predictions
```bash
curl https://cossy179-goon-steen.hf.space/api/predictions
```

You should see JSON response with game predictions!

## Step 8: Update PHP Backend

Your PHP backend is already configured to call:
```
https://cossy179-goon-steen.hf.space/api/predictions
```

**Important**: Create the cache directory:
```bash
cd web
mkdir cache
chmod 755 cache
```

## Step 9: Test Website Integration

1. Open your website dashboard
2. Check if games are loading
3. Verify predictions are showing
4. Test Kelly criterion calculator
5. Check parlay suggestions

## Troubleshooting

### Space Not Building

**Error**: "Dockerfile not found"
- Make sure `Dockerfile` is in the root of your space

**Error**: "Requirements installation failed"
- Check `requirements.txt` for typos
- Verify Python package versions are compatible

### API Returns Errors

**Error**: "Model file not found"
- Verify model files are in `models/` directory
- Check Git LFS tracked the files: `git lfs ls-files`

**Error**: "No games found"
- This is normal during NBA off-season
- The API will return empty games array

### Website Not Showing Predictions

**Error**: "Failed to fetch from HuggingFace"
- Verify Space is running (green "Running" status)
- Check the URL is correct in `web/api.php` (line 265)
- Ensure cache directory exists and is writable

**Error**: "CORS error"
- HuggingFace Spaces automatically handle CORS
- If still seeing errors, check browser console

## Updating the Model

To update your model in the future:

```bash
cd Goon-Steen

# Copy new model file
cp path/to/new_model.txt models/

# Commit and push
git add models/
git commit -m "Update model to version X"
git push origin main
```

The space will automatically rebuild.

## Monitoring

### View Logs
1. Go to your space page
2. Click "Logs" tab
3. See real-time application logs

### Check Performance
- Monitor response times in logs
- Check cache hit rates
- Verify predictions are updating daily

## Cost Optimization

HuggingFace Spaces are **FREE** for:
- Public spaces
- CPU basic tier
- Up to 2 CPU cores

Your setup should run completely free!

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/health` | GET | Detailed health status |
| `/api/predictions` | GET | All predictions (games + parlays) |
| `/api/games` | GET | Game predictions only |
| `/api/parlays` | GET | Parlay suggestions only |
| `/api/cache-status` | GET | Cache status |
| `/api/refresh` | POST | Force refresh predictions |

## Security Notes

- API is public (no authentication needed)
- Predictions are cached to prevent abuse
- Rate limiting handled by HuggingFace
- No sensitive data exposed

## Next Steps

1. ✅ Deploy to HuggingFace (you're here!)
2. ✅ Test API endpoints
3. ✅ Verify website integration
4. 📊 Monitor performance
5. 🎯 Collect user feedback
6. 🔄 Update models as needed

## Support

If you encounter issues:
1. Check HuggingFace Space logs
2. Review browser console errors
3. Verify all files are uploaded
4. Check Git LFS is working

## Success Checklist

- [ ] Space created on HuggingFace
- [ ] Files copied and pushed
- [ ] Space shows "Running" status
- [ ] API health check returns 200
- [ ] Predictions endpoint returns data
- [ ] Website displays games
- [ ] Kelly criterion works
- [ ] Parlays are showing

Congratulations! Your NBA prediction system is now live! 🎉

