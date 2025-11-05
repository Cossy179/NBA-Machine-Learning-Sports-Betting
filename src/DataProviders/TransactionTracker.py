"""
NBA Transaction and Roster Tracker for 2025-26 season.
Tracks trades, signings, retirements, and roster changes to maintain data accuracy.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
import re
import time
import warnings
warnings.filterwarnings('ignore')


class TransactionTracker:
    """Tracks NBA transactions (trades, signings, retirements)"""
    
    def __init__(self, db_path: str = "Data/Transactions.sqlite"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize transaction tracking database"""
        con = sqlite3.connect(self.db_path)
        cursor = con.cursor()
        
        # Transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                player_name TEXT NOT NULL,
                player_id TEXT,
                transaction_type TEXT NOT NULL,
                from_team TEXT,
                to_team TEXT,
                details TEXT,
                season TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, player_name, transaction_type, from_team, to_team)
            )
        """)
        
        # Player team history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_team_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT,
                season TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        con.commit()
        con.close()
    
    def scrape_basketball_reference_transactions(
        self,
        season: str = "2025-26",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Scrape transactions from Basketball Reference.
        
        Args:
            season: NBA season (e.g., "2025-26")
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with transaction data
        """
        year = season.split('-')[0]
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_transactions.html"
        
        print(f"Scraping transactions from Basketball Reference...")
        print(f"  URL: {url}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            transactions = []
            
            # Find transaction sections
            transaction_sections = soup.find_all(['h2', 'h3'])
            
            for section in transaction_sections:
                date_str = section.get_text().strip()
                
                # Try to parse date
                try:
                    trans_date = self._parse_date(date_str)
                    if trans_date:
                        # Check date range
                        if date_from and trans_date < date_from:
                            continue
                        if date_to and trans_date > date_to:
                            continue
                        
                        # Find transaction list for this date
                        next_sibling = section.find_next_sibling('ul')
                        if next_sibling:
                            items = next_sibling.find_all('li')
                            for item in items:
                                trans_text = item.get_text()
                                trans = self._parse_transaction(trans_text, trans_date, season)
                                if trans:
                                    transactions.append(trans)
                except:
                    continue
            
            if transactions:
                df = pd.DataFrame(transactions)
                print(f"✅ Found {len(df)} transactions")
                return df
            else:
                print("⚠️  No transactions found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error scraping transactions: {e}")
            return pd.DataFrame()
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse date string to YYYY-MM-DD format"""
        try:
            # Try various date formats
            for fmt in ['%B %d, %Y', '%b %d, %Y', '%Y-%m-%d', '%m/%d/%Y']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%Y-%m-%d')
                except:
                    continue
        except:
            pass
        return None
    
    def _parse_transaction(
        self,
        trans_text: str,
        date: str,
        season: str
    ) -> Optional[Dict]:
        """Parse individual transaction text"""
        trans_text = trans_text.strip()
        
        # Determine transaction type
        trans_type = None
        if 'traded' in trans_text.lower() or 'trade' in trans_text.lower():
            trans_type = 'trade'
        elif 'signed' in trans_text.lower() or 'signing' in trans_text.lower():
            trans_type = 'signing'
        elif 'waived' in trans_text.lower() or 'waiver' in trans_text.lower():
            trans_type = 'waiver'
        elif 'retired' in trans_text.lower():
            trans_type = 'retirement'
        elif 'drafted' in trans_text.lower():
            trans_type = 'draft'
        else:
            trans_type = 'other'
        
        # Extract player name (usually at the start)
        player_match = re.match(r'^([A-Z][a-z]+ [A-Z][a-z]+)', trans_text)
        player_name = player_match.group(1) if player_match else None
        
        # Extract teams
        teams = re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+)?)', trans_text)
        from_team = None
        to_team = None
        
        if trans_type == 'trade' and len(teams) >= 2:
            from_team = teams[0] if teams else None
            to_team = teams[-1] if len(teams) > 1 else None
        
        return {
            'date': date,
            'player_name': player_name,
            'transaction_type': trans_type,
            'from_team': from_team,
            'to_team': to_team,
            'details': trans_text,
            'season': season
        }
    
    def get_nba_api_transactions(
        self,
        season: str = "2025-26",
        date_from: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get transactions from NBA Stats API (if available).
        This is a placeholder for API-based transaction fetching.
        """
        # NBA Stats API doesn't have a direct transactions endpoint
        # This would need to be implemented using a third-party API or scraping
        print("⚠️  NBA Stats API transactions not directly available")
        print("   Using Basketball Reference scraping instead")
        return pd.DataFrame()
    
    def save_transactions(self, transactions_df: pd.DataFrame):
        """Save transactions to database"""
        if transactions_df.empty:
            return
        
        try:
            con = sqlite3.connect(self.db_path)
            transactions_df.to_sql(
                'transactions',
                con,
                if_exists='append',
                index=False,
                method='multi'
            )
            con.close()
            print(f"✅ Saved {len(transactions_df)} transactions to database")
        except Exception as e:
            print(f"❌ Error saving transactions: {e}")
    
    def update_player_team_history(
        self,
        player_id: str,
        player_name: str,
        new_team: str,
        transaction_date: str,
        season: str
    ):
        """Update player team history when a transaction occurs"""
        con = sqlite3.connect(self.db_path)
        cursor = con.cursor()
        
        # End previous team assignment
        cursor.execute("""
            UPDATE player_team_history
            SET end_date = ?
            WHERE player_id = ? AND end_date IS NULL
        """, (transaction_date, player_id))
        
        # Add new team assignment
        cursor.execute("""
            INSERT INTO player_team_history (player_id, player_name, team, start_date, season)
            VALUES (?, ?, ?, ?, ?)
        """, (player_id, player_name, new_team, transaction_date, season))
        
        con.commit()
        con.close()
    
    def get_player_current_team(
        self,
        player_id: str,
        date: Optional[str] = None
    ) -> Optional[str]:
        """Get player's current team as of a specific date"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        con = sqlite3.connect(self.db_path)
        cursor = con.cursor()
        
        cursor.execute("""
            SELECT team FROM player_team_history
            WHERE player_id = ?
            AND start_date <= ?
            AND (end_date IS NULL OR end_date >= ?)
            ORDER BY start_date DESC
            LIMIT 1
        """, (player_id, date, date))
        
        result = cursor.fetchone()
        con.close()
        
        return result[0] if result else None
    
    def reset_team_dependent_features(
        self,
        player_id: str,
        transaction_date: str
    ):
        """
        Mark that team-dependent features should be reset for a player.
        This is called when a player changes teams mid-season.
        """
        # This would be used by the feature engineering pipeline
        # to reset team-specific features like team synergy, lineup combinations, etc.
        print(f"⚠️  Player {player_id} changed teams on {transaction_date}")
        print(f"   Team-dependent features should be reset")


def track_2025_26_transactions(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to track 2025-26 season transactions.
    
    Args:
        date_from: Start date (defaults to season start)
        date_to: End date (defaults to today)
    
    Returns:
        DataFrame with transactions
    """
    tracker = TransactionTracker()
    
    if date_from is None:
        date_from = "2025-10-22"  # 2025-26 opening night
    
    if date_to is None:
        date_to = datetime.now().strftime("%Y-%m-%d")
    
    transactions = tracker.scrape_basketball_reference_transactions(
        season="2025-26",
        date_from=date_from,
        date_to=date_to
    )
    
    if not transactions.empty:
        tracker.save_transactions(transactions)
    
    return transactions


if __name__ == "__main__":
    print("NBA Transaction Tracker - 2025-26 Season")
    print("=" * 60)
    
    transactions = track_2025_26_transactions(
        date_from="2025-10-22",
        date_to="2025-11-03"
    )
    
    if not transactions.empty:
        print(f"\n✅ Found {len(transactions)} transactions")
        print("\nSample transactions:")
        print(transactions.head())
    else:
        print("\n⚠️  No transactions found in date range")

