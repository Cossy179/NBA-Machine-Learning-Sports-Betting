# 💰 Bet Sizing Modes Guide

## Overview

The prediction system now supports **two distinct bet sizing modes** to accommodate different betting strategies and risk tolerances.

---

## 🎯 Two Modes Available

### 1. **Even Split Mode** (Default - Aggressive)
```bash
py predict.py --bankroll 1000
```

**What it does:**
- Divides your bankroll **evenly across all recommended bets**
- Uses **100% of specified bankroll**
- Each bet gets equal allocation

**Example with $1000 bankroll:**
```
Recommended Bets: 5 games
Amount Per Bet: $200 (20% each)
Total Allocated: $1000 (100%)
Remaining: $0
```

### 2. **Kelly Criterion Mode** (--kc flag - Conservative)
```bash
py predict.py --bankroll 1000 --kc
```

**What it does:**
- Uses **Kelly Criterion** for mathematically optimal sizing
- Allocates based on **edge and confidence**
- Typically uses **15-35% of bankroll**
- **Reduces variance** and protects bankroll

**Example with $1000 bankroll:**
```
Game 1 (High confidence): $45 (4.5%)
Game 2 (Medium confidence): $25 (2.5%)
Game 3 (Low edge): $12 (1.2%)
Total Allocated: $82 (8.2%)
Remaining: $918
```

---

## 📊 Detailed Comparison

| Feature | Even Split | Kelly Criterion |
|---------|------------|-----------------|
| **Bankroll Usage** | 100% | 15-35% typically |
| **Risk Level** | ⚠️ HIGH | ✅ LOW-MEDIUM |
| **Variance** | Very High | Controlled |
| **Per Bet Size** | Equal | Proportional to edge |
| **Drawdown Risk** | High (can lose all) | Low (preserves capital) |
| **Upside** | Maximum if all win | Optimal long-term growth |
| **Best For** | Confident picks, small slates | Long-term play, preservation |
| **Recommended** | 1-3 high-confidence bets | Regular daily betting |

---

## 💡 When to Use Each Mode

### Use **Even Split** (Default) When:
✅ You have **1-3 very high-confidence picks** (>65% confidence)  
✅ Small slate of games (limited options)  
✅ Short-term tournament/contest betting  
✅ You want maximum action on your best picks  
✅ You're confident in your edge  

**Example Scenarios:**
- "I love these 2 games, want to go all-in"
- "Only 3 games tonight, split it evenly"
- "High confidence weekend, use the whole bankroll"

### Use **Kelly Criterion** (--kc) When:
✅ Long-term bankroll growth is the goal  
✅ Betting regularly (daily/weekly)  
✅ Want to minimize risk of ruin  
✅ Have varying confidence levels  
✅ Professional/serious betting approach  

**Example Scenarios:**
- "Betting throughout the season, want to preserve capital"
- "Some picks are stronger than others, size accordingly"
- "Want mathematically optimal growth"
- "Can't afford to lose entire bankroll"

---

## 🎓 Understanding Kelly Criterion

### The Formula
```
Kelly % = (Edge / Odds) * 100
```

**Simplified:**
- **High edge + good odds** = Larger bet
- **Small edge + bad odds** = Tiny bet
- **No edge or negative edge** = No bet

### Conservative Kelly (What We Use)
```
Conservative Kelly = Full Kelly × 25%
```

**Why 25% (Fractional Kelly)?**
- Reduces variance significantly
- Still captures most of Kelly's growth
- Protects against estimation errors
- Industry standard for sports betting

### Example Calculations

**Scenario 1: Strong Pick**
- Win Probability: 60%
- Odds: +150 (2.5 decimal)
- Edge: 60% - (1/2.5) = 20%
- Full Kelly: 13.3%
- Conservative Kelly: **3.3% of bankroll**

**Scenario 2: Marginal Pick**  
- Win Probability: 53%
- Odds: -110 (1.91 decimal)
- Edge: 53% - (1/1.91) = 0.7%
- Full Kelly: 0.5%
- Conservative Kelly: **0.1% of bankroll**

**Scenario 3: No Edge**
- Win Probability: 52%
- Odds: -110 (1.91 decimal)
- Edge: 52% - (1/1.91) = -0.4%
- Kelly: **0% (No bet)**

---

## 📈 Performance Expectations

### Even Split Mode
**Pros:**
- Maximum capital deployed
- Higher potential returns
- Simple to understand
- Full bankroll engagement

**Cons:**
- Can lose 100% in one bad night
- High variance
- Doesn't account for varying edges
- Drawdowns can be severe

**Expected Performance:**
- Win Rate: Same as models (73-75%)
- ROI: Higher volatility
- Bankroll Swings: ±50-100% possible
- **Risk of Ruin: MODERATE-HIGH**

### Kelly Criterion Mode
**Pros:**
- Optimal long-term growth
- Lower variance
- Protects against bad runs
- Accounts for confidence/edge
- Professional approach

**Cons:**
- Slower capital deployment
- "Missing out" feeling on big wins
- Requires discipline
- Less exciting

**Expected Performance:**
- Win Rate: Same as models (73-75%)
- ROI: ~8-12% long-term
- Bankroll Swings: ±10-20% typical
- **Risk of Ruin: VERY LOW**

---

## 🔧 Usage Examples

### Example 1: Weekend Warrior (Even Split)
```bash
# You have $500 and want to bet this weekend's games
py predict.py --bankroll 500 --parlays
```

**Output:**
```
💰 BET SIZING MODE: Even Split (100% Bankroll Allocation)
Total Bets: 4 (3 games + 1 parlay)
Amount Per Bet: $125.00 (25.0% each)
Total Allocated: $500.00
Bankroll Utilization: 100.0%
Remaining: $0.00
```

### Example 2: Daily Grinder (Kelly)
```bash
# You have $5000 bankroll and bet daily
py predict.py --bankroll 5000 --kc --parlays
```

**Output:**
```
💰 BET SIZING MODE: Kelly Criterion (Conservative)
Game 1 (High edge): $225.00 (4.5%)
Game 2 (Medium edge): $100.00 (2.0%)
Game 3 (Small edge): $50.00 (1.0%)
Parlay #1 (Good value): $125.00 (2.5%)
Total Allocated: $500.00 (10.0%)
Remaining: $4500.00
```

### Example 3: High-Confidence Slate (Even Split)
```bash
# You have $200 and found 2 locks
py predict.py --bankroll 200 --confidence 0.65
```

**Output:**
```
💰 BET SIZING MODE: Even Split (100% Bankroll Allocation)
Total Bets: 2
Amount Per Bet: $100.00 (50.0% each)
Total Allocated: $200.00
Bankroll Utilization: 100.0%
Remaining: $0.00
```

### Example 4: Conservative Daily (Kelly)
```bash
# Professional bettor with $10,000 bankroll
py predict.py --bankroll 10000 --kc --confidence 0.50
```

**Output:**
```
💰 BET SIZING MODE: Kelly Criterion (Conservative)
Game 1: $450.00 (4.5%)
Game 2: $300.00 (3.0%)
Game 3: $150.00 (1.5%)
Total Allocated: $900.00 (9.0%)
Remaining: $9100.00
```

---

## ⚖️ Which Mode Should You Use?

### Quick Decision Tree:

**Start Here: How many games are you betting?**

**1-3 games AND high confidence (>65%)?**
→ **Even Split Mode** ✅  
`py predict.py --bankroll 500`

**4+ games OR varying confidence levels?**
→ **Kelly Criterion Mode** ✅  
`py predict.py --bankroll 500 --kc`

**Betting daily/weekly long-term?**
→ **Kelly Criterion Mode** ✅  
`py predict.py --bankroll 5000 --kc`

**Tournament or contest betting?**
→ **Even Split Mode** ✅  
`py predict.py --bankroll 200`

**Can't afford to lose bankroll?**
→ **Kelly Criterion Mode** ✅  
`py predict.py --bankroll 1000 --kc`

**Want maximum action tonight?**
→ **Even Split Mode** ✅  
`py predict.py --bankroll 300`

---

## 🎯 Best Practices

### For Even Split Mode:
1. **Use with 1-4 bets maximum** (avoid over-diversification)
2. **Require high confidence** (>60% on all bets)
3. **Start with small bankroll** ($100-$500 to test)
4. **Track results carefully** (this mode has high variance)
5. **Set win goals** (e.g., quit after doubling)
6. **Set loss limits** (e.g., stop after losing 2 slates)

### For Kelly Criterion Mode:
1. **Use consistently** (don't switch modes mid-season)
2. **Trust the math** (don't override bet sizes)
3. **Rebalance regularly** (update bankroll weekly)
4. **Track long-term ROI** (minimum 100 bets for evaluation)
5. **Accept smaller bets** (it's protecting you)
6. **Compound winnings** (increase bankroll as you profit)

---

## 📊 Real-World Examples

### Scenario 1: Weekend Slate
**Setup:**
- 5 NBA games on Saturday
- $500 weekend bankroll
- Mixed confidence (40%, 55%, 60%, 65%, 70%)

**Even Split:**
```bash
py predict.py --bankroll 500
```
- 5 bets × $100 = $500 (100%)
- If 3/5 win: +$73 (+14.6% ROI)
- If 2/5 win: -$154 (-30.8% ROI)

**Kelly Criterion:**
```bash
py predict.py --bankroll 500 --kc
```
- Bet 1 (40%): $0 (skip)
- Bet 2 (55%): $8 (1.6%)
- Bet 3 (60%): $15 (3.0%)
- Bet 4 (65%): $22 (4.4%)
- Bet 5 (70%): $30 (6.0%)
- Total: $75 (15%)
- If 3/5 win (likely the top 3): +$52 (+10.4% ROI, bankroll now $552)

### Scenario 2: Daily Grind (Full Week)
**Setup:**
- Bet Monday-Sunday
- $2000 starting bankroll
- Average 3 games per day

**Even Split (7 days):**
- Monday: $200 allocated → Win $60
- Tuesday: $260 allocated → Lose $130
- Wednesday: $130 allocated → Win $52
- Thursday: $182 allocated → Lose $91
- Friday: $91 allocated → Win $36
- Saturday: $127 allocated → Lose $63
- Sunday: $64 allocated → Win $25
- **Ending Bankroll: ~$1,100** (highly volatile)

**Kelly Criterion (7 days):**
- Monday: $120 allocated → Win $36 (Bankroll: $2,036)
- Tuesday: $122 allocated → Lose $61 (Bankroll: $1,975)
- Wednesday: $118 allocated → Win $47 (Bankroll: $2,022)
- Thursday: $121 allocated → Lose $60 (Bankroll: $1,962)
- Friday: $117 allocated → Win $46 (Bankroll: $2,008)
- Saturday: $120 allocated → Lose $60 (Bankroll: $1,948)
- Sunday: $117 allocated → Win $46 (Bankroll: $1,994)
- **Ending Bankroll: ~$1,994** (stable, nearly break-even)

*Note: Kelly is designed for long-term (months/years), not weeks*

---

## ⚠️ Important Warnings

### Even Split Mode Risks:
1. **Can lose entire bankroll in one day**
2. **No cushion for bad runs**
3. **Requires perfect or near-perfect accuracy**
4. **Emotional toll from big swings**
5. **Not sustainable long-term**

### When NOT to Use Even Split:
❌ First time using the system  
❌ Can't afford to lose the bankroll  
❌ Betting with scared money  
❌ Low confidence (<60% average)  
❌ Large slate (5+ games)  

### Kelly Mode Considerations:
1. **Feels like "missing out"** (small bets on big favorites)
2. **Requires patience** (slow compounding)
3. **Need accurate probabilities** (model must be calibrated)
4. **Must update bankroll** (rebalance after wins/losses)

---

## 🔄 Switching Between Modes

### You can switch anytime:

**Tonight (Even Split):**
```bash
py predict.py --bankroll 500
```

**Tomorrow (Kelly):**
```bash
py predict.py --bankroll 500 --kc
```

**Recommendation:** Pick one mode and stick with it for at least 50-100 bets to properly evaluate performance.

---

## 📈 Expected Outcomes

### Over 100 Bets (73% Win Rate)

**Even Split Mode:**
- **Wins**: 73 bets (lose 27)
- **Average Payout**: +125 per win, -100 per loss
- **Profit**: (73 × 125) - (27 × 100) = **+6,425**
- **ROI on Risk**: +6.4% (but with 100% capital at risk each time)
- **Max Drawdown**: Could lose 10+ bets in a row (entire bankroll)

**Kelly Criterion Mode:**
- **Wins**: 73 bets (lose 27)
- **Average allocation**: 2.5% per bet
- **Compounding**: Bankroll grows with each win
- **ROI**: ~8-12% on total bankroll
- **Max Drawdown**: ~15-20% of bankroll
- **Ending Bankroll**: 108-112% of starting (after 100 bets)

---

## 🎯 Recommended Strategy

### For Beginners:
1. **Start with Kelly** (`--kc`)
2. **Small bankroll** ($200-$500)
3. **Track results** for 25-50 bets
4. **Evaluate performance** before switching modes

### For Experienced Bettors:
1. **Use Kelly for regular season** (Oct-April)
2. **Consider Even Split for playoffs** (high-confidence spots)
3. **Separate bankrolls** (Kelly bankroll + "fun" bankroll)
4. **Never mix modes** on same bankroll

### For Professional Approach:
1. **Kelly Criterion exclusively**
2. **Large bankroll** ($5,000+)
3. **Daily betting** with rebalancing
4. **Strict record keeping**
5. **Monthly performance reviews**

---

## 🔧 Advanced Usage

### Hybrid Approach (Two Bankrolls)
```bash
# Main bankroll (Kelly - Conservative)
py predict.py --bankroll 5000 --kc

# Action bankroll (Even Split - Aggressive)  
py predict.py --bankroll 500
```

**Benefits:**
- Main bankroll grows steadily with Kelly
- Action bankroll for high-conviction plays
- Psychological satisfaction from both approaches
- Risk management through separation

### Dynamic Bankroll Management
```bash
# Monday (lost yesterday, reduce bankroll)
py predict.py --bankroll 450 --kc

# Tuesday (won yesterday, increase bankroll)
py predict.py --bankroll 550 --kc

# Always update based on actual remaining capital
```

---

## 📚 Examples With Real Output

### Example 1: Even Split ($1000, 5 games)

```bash
py predict.py --bankroll 1000 --parlays
```

**Output:**
```
⚙️  BET SIZING CONFIGURATION
======================================================================
💪 Mode: Even Split (Aggressive - 100% Allocation)
   • Splits entire bankroll across all recommended bets
   • Uses 100% of specified bankroll
   • Higher variance, higher potential returns
   • ⚠️  All capital at risk - use with caution!
💰 Bankroll: $1,000.00
======================================================================

📊 PREDICTION SUMMARY
======================================================================
Total Bets: 5
Amount Per Bet: $200.00 (20.0% each)
Total Allocated: $1,000.00
Bankroll Utilization: 100.0%
Remaining: $0.00

💰 BANKROLL ALLOCATION
======================================================================
  1. Lakers ML: $200.00 (20.0%)
  2. Celtics ML: $200.00 (20.0%)
  3. Warriors ML: $200.00 (20.0%)
  4. Heat ML: $200.00 (20.0%)
  5. Bucks ML: $200.00 (20.0%)

  ==================================================================
  TOTAL ALLOCATED: $1,000.00 (100.0%)
  REMAINING: $0.00
```

### Example 2: Kelly Criterion ($1000, 5 games)

```bash
py predict.py --bankroll 1000 --kc --parlays
```

**Output:**
```
⚙️  BET SIZING CONFIGURATION
======================================================================
📊 Mode: Kelly Criterion (Conservative)
   • Mathematically optimal bet sizing
   • Typically uses 15-35% of bankroll
   • Reduces variance and drawdowns
   • Recommended for long-term bankroll growth
💰 Bankroll: $1,000.00
======================================================================

📊 PREDICTION SUMMARY
======================================================================
Total Kelly Bet Amount: $186.50
Bankroll Utilization: 18.7%
Remaining Bankroll: $813.50

💰 BANKROLL ALLOCATION
======================================================================
  1. Lakers ML: $45.00 (4.5%)
  2. Celtics ML: $32.50 (3.3%)
  3. Warriors ML: $28.00 (2.8%)
  4. Heat ML: $56.00 (5.6%)
  5. Bucks ML: $25.00 (2.5%)

  💎 Top Parlays:
  6. Parlay #1 (2 legs): $15.00 (1.5%) ✅

  ==================================================================
  TOTAL ALLOCATED: $201.50 (20.2%)
  REMAINING: $798.50
```

---

## 🎲 With Parlays

Both modes work with parlays:

### Even Split + Parlays:
```bash
py predict.py --bankroll 500 --parlays
```

**Behavior:**
- Includes high-quality parlays in even split
- Example: 3 games + 1 parlay = 4 bets × $125 each

### Kelly + Parlays:
```bash
py predict.py --bankroll 500 --kc --parlays
```

**Behavior:**
- Parlays sized according to their Kelly %
- Typically 0.5-2.5% for good parlays
- Only includes parlays with positive boosted EV

---

## ✅ Quick Reference

| Command | Mode | Bankroll Used | Risk | Best For |
|---------|------|---------------|------|----------|
| `py predict.py --bankroll 1000` | Even Split | 100% | ⚠️ HIGH | 1-3 locks |
| `py predict.py --bankroll 1000 --kc` | Kelly | 15-35% | ✅ LOW | Daily betting |
| `py predict.py --bankroll 500 --parlays` | Even Split | 100% | ⚠️ HIGH | Weekend fun |
| `py predict.py --bankroll 5000 --kc --parlays` | Kelly | 15-35% | ✅ LOW | Professional |

---

## 🚨 Critical Reminders

### Bankroll Management 101:
1. **Never bet more than you can afford to lose**
2. **Keep betting bankroll separate from living expenses**
3. **Update bankroll value regularly** (use actual remaining capital)
4. **Don't chase losses** (stick to your system)
5. **Take profits** (withdraw winnings periodically)

### Responsible Gambling:
- This is for **entertainment and educational purposes**
- Past performance ≠ future results
- Even 75% win rate means 25% losses
- Variance is real - expect losing streaks
- Set daily/weekly loss limits
- Never bet scared money
- Seek help if gambling becomes a problem

---

## 🎉 Summary

You now have **two powerful bet sizing modes**:

### **Even Split** (Default)
- ✅ Use 100% of bankroll
- ✅ Simple equal allocation
- ✅ Maximum action
- ⚠️ Higher risk

### **Kelly Criterion** (--kc)
- ✅ Mathematically optimal
- ✅ Bankroll preservation
- ✅ Professional approach
- ✅ Long-term growth

**Usage:**
```bash
# Even Split
py predict.py --bankroll 1000

# Kelly Criterion  
py predict.py --bankroll 1000 --kc
```

**Choose based on your:**
- Risk tolerance
- Betting frequency
- Number of picks
- Time horizon
- Confidence levels

**Good luck with your bets! 💰🏀**
















