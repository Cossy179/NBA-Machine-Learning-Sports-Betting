# 🏀 NBA ML Betting Results - Excel Formatting Guide

## 📊 Generated Files

The backtest script now generates **both CSV and Excel files** with improved formatting:

### 📁 File Types Generated:
1. **`{model_name}_detailed_betting_YYYYMMDD_HHMMSS.csv`** - Clean CSV format
2. **`{model_name}_formatted_betting_YYYYMMDD_HHMMSS.xlsx`** - **Excel with color coding and formatting**

## 🎨 Excel Formatting Features

### ✅ **Color Coding:**
- **🟢 Green Background**: Winning bets (positive profit)
- **🔴 Red Background**: Losing bets (negative profit)
- **🔵 Blue Header**: Column headers with white text

### 📋 **Column Structure (matching your screenshot):**
- **Game**: Sequential bet number
- **Date**: Game date (MM/DD/YYYY format)
- **Away**: Away team name
- **OU**: Over/Under line
- **Spread**: Calculated point spread
- **IL**: Implied Line (probability)
- **Hom**: Home team name
- **ML**: Moneyline odds
- **Away (ML)**: Away team moneyline odds
- **Points**: Total points scored
- **Win**: Actual winner (1=home, 0=away)
- **Margi**: Victory margin
- **Predictio**: Model prediction (1=home, 0=away)
- **Correct?**: Yes/No if prediction was correct
- **Money Lost/Won**: Profit/loss with proper formatting ($XXX.XX)
- **Running Profi**: Cumulative profit with proper formatting ($X,XXX.XX)

### 📈 **Summary Section:**
- **Row 1**: "Tested: MM/DD/YYYY" with "Bet for All Game: $100"
- **Row 2**: "Total Bets: XXX" with "Win Rate: XX.X%"
- **Row 3**: "Total Profit: $XX,XXX.XX"

### 🎯 **Professional Formatting:**
- **Borders**: All cells have clean borders
- **Alignment**: Numbers centered, text left-aligned
- **Auto-width**: Columns automatically sized for content
- **Bold text**: Running profit totals in bold
- **Currency formatting**: Proper $ formatting for money columns

## 🚀 **How to Use:**

1. **Open the Excel file** in Microsoft Excel, LibreOffice Calc, or Google Sheets
2. **View the color-coded results** - green for wins, red for losses
3. **Analyze the running profit** to see cumulative performance
4. **Check the summary** at the top for overall statistics
5. **Sort/filter** by any column as needed

## 📊 **Example Results:**
- **Best Model**: Auto-selected XGBoost
- **Total Profit**: $58,392.18
- **Win Rate**: 74.7%
- **Total Bets**: 719
- **ROI**: 81.2%

The Excel files provide a much cleaner, more professional view of your betting results with clear visual indicators for wins and losses!
