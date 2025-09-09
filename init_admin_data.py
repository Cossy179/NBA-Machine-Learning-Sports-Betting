#!/usr/bin/env python3
"""
Initialize Admin Data
Script to create sample data for the admin dashboard
"""

import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
import random

def hash_password(password: str, salt: str = None) -> tuple:
    """Hash password with salt"""
    if salt is None:
        salt = secrets.token_hex(16)
    
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    return password_hash.hex(), salt

def init_sample_data():
    """Initialize the database with sample data"""
    conn = sqlite3.connect('web_database.db')
    cursor = conn.cursor()
    
    try:
        # Create admin user if not exists
        admin_password_hash, admin_salt = hash_password('admin123')
        
        cursor.execute('''
            INSERT OR IGNORE INTO users (
                username, email, password_hash, salt, first_name, last_name, 
                date_of_birth, is_admin, email_verified, terms_accepted, responsible_gambling, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [
            'admin', 'admin@goonsteen.com', admin_password_hash, admin_salt,
            'Admin', 'User', '1990-01-01', True, True, True, True, 'active'
        ])
        
        # Create sample users
        sample_users = [
            ('john_doe', 'john.doe@email.com', 'John', 'Doe'),
            ('jane_smith', 'jane.smith@email.com', 'Jane', 'Smith'),
            ('mike_brown', 'mike.brown@email.com', 'Mike', 'Brown'),
            ('sarah_wilson', 'sarah.wilson@email.com', 'Sarah', 'Wilson'),
            ('david_jones', 'david.jones@email.com', 'David', 'Jones'),
            ('lisa_garcia', 'lisa.garcia@email.com', 'Lisa', 'Garcia'),
            ('tom_miller', 'tom.miller@email.com', 'Tom', 'Miller'),
            ('amy_davis', 'amy.davis@email.com', 'Amy', 'Davis'),
        ]
        
        user_ids = []
        for username, email, first_name, last_name in sample_users:
            password_hash, salt = hash_password('password123')
            
            # Create user with random creation time in last 30 days
            created_at = datetime.now() - timedelta(days=random.randint(1, 30))
            last_login = created_at + timedelta(hours=random.randint(1, 48))
            status = random.choice(['active', 'active', 'active', 'premium', 'suspended'])
            
            cursor.execute('''
                INSERT OR IGNORE INTO users (
                    username, email, password_hash, salt, first_name, last_name,
                    date_of_birth, email_verified, terms_accepted, responsible_gambling,
                    status, subscription_type, created_at, last_login
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [
                username, email, password_hash, salt, first_name, last_name,
                '1995-01-01', True, True, True, status,
                'premium' if status == 'premium' else 'free',
                created_at.isoformat(), last_login.isoformat()
            ])
            
            user_id = cursor.lastrowid
            if user_id:
                user_ids.append(user_id)
                
                # Create bankroll for user
                cursor.execute('''
                    INSERT OR IGNORE INTO bankrolls (
                        user_id, total_balance, available_balance, total_deposited, total_profit_loss
                    ) VALUES (?, ?, ?, ?, ?)
                ''', [
                    user_id, 
                    random.uniform(500, 5000),
                    random.uniform(100, 1000),
                    random.uniform(1000, 10000),
                    random.uniform(-500, 2000)
                ])
        
        # Create sample bets for users
        for user_id in user_ids:
            num_bets = random.randint(5, 50)
            for _ in range(num_bets):
                placed_at = datetime.now() - timedelta(days=random.randint(1, 90))
                status = random.choice(['won', 'lost', 'pending', 'won', 'lost'])  # Bias toward won/lost
                stake = random.uniform(10, 500)
                
                if status == 'won':
                    odds = random.uniform(1.5, 3.0)
                    payout = stake * odds
                elif status == 'lost':
                    payout = 0
                else:  # pending
                    odds = random.uniform(1.5, 3.0)
                    payout = stake * odds
                
                cursor.execute('''
                    INSERT OR IGNORE INTO bets (
                        user_id, bet_type, status, stake, potential_payout, actual_payout,
                        odds, bet_details, placed_at, settled_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', [
                    user_id, 'single', status, stake, stake * random.uniform(1.5, 3.0),
                    payout if status != 'pending' else 0,
                    random.uniform(1.5, 3.0),
                    '{"team": "Lakers", "bet_type": "moneyline"}',
                    placed_at.isoformat(),
                    placed_at.isoformat() if status != 'pending' else None
                ])
        
        # Create sample user activity
        for user_id in user_ids:
            activities = [
                ('login_success', 'User logged in successfully'),
                ('bet_placed', 'User placed a bet'),
                ('profile_updated', 'User updated profile'),
                ('password_changed', 'User changed password'),
                ('logout', 'User logged out')
            ]
            
            for activity_type, description in activities:
                activity_time = datetime.now() - timedelta(hours=random.randint(1, 72))
                cursor.execute('''
                    INSERT OR IGNORE INTO user_activity (
                        user_id, activity_type, description, created_at
                    ) VALUES (?, ?, ?, ?)
                ''', [user_id, activity_type, description, activity_time.isoformat()])
        
        # Create sample games for today
        today = datetime.now().date()
        sample_games = [
            ('Lakers', 'Warriors', '19:30:00'),
            ('Celtics', 'Heat', '20:00:00'),
            ('Bulls', 'Nets', '20:30:00'),
        ]
        
        for home_team, away_team, game_time in sample_games:
            # Get team IDs
            home_team_id = cursor.execute('SELECT id FROM teams WHERE name LIKE ?', [f'%{home_team}%']).fetchone()
            away_team_id = cursor.execute('SELECT id FROM teams WHERE name LIKE ?', [f'%{away_team}%']).fetchone()
            
            if home_team_id and away_team_id:
                cursor.execute('''
                    INSERT OR IGNORE INTO games (
                        home_team_id, away_team_id, game_date, game_time, season, status
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', [
                    home_team_id[0], away_team_id[0], today.isoformat(), game_time,
                    '2024-25', 'scheduled'
                ])
        
        # Create sample predictions
        games = cursor.execute('SELECT id FROM games WHERE game_date = ?', [today.isoformat()]).fetchall()
        for game in games:
            game_id = game[0]
            
            # Create prediction with random but realistic data
            confidence = random.uniform(60, 95)
            home_score = random.randint(95, 125)
            away_score = random.randint(95, 125)
            
            cursor.execute('''
                INSERT OR IGNORE INTO predictions (
                    game_id, model_name, model_version, prediction_type,
                    predicted_home_score, predicted_away_score, predicted_total,
                    confidence, probability, expected_value
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [
                game_id, 'Ensemble_NBA_v1', '1.0', 'moneyline',
                home_score, away_score, home_score + away_score,
                confidence, confidence / 100, random.uniform(0.05, 0.15)
            ])
        
        # Create sample model performance data
        models = ['XGBoost_ML', 'Neural_Network', 'Ensemble_NBA_v1']
        for model in models:
            for days_back in range(30):
                date_str = (datetime.now() - timedelta(days=days_back)).date().isoformat()
                
                cursor.execute('''
                    INSERT OR IGNORE INTO model_performance (
                        model_name, model_version, prediction_type,
                        total_predictions, correct_predictions, accuracy,
                        date_from, date_to
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', [
                    model, '1.0', 'moneyline',
                    random.randint(5, 15), random.randint(3, 12),
                    random.uniform(60, 75), date_str, date_str
                ])
        
        conn.commit()
        print("Sample data created successfully!")
        
        # Print admin credentials
        print("\n" + "="*50)
        print("ADMIN LOGIN CREDENTIALS:")
        print("Username: admin")
        print("Password: admin123")
        print("="*50)
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == '__main__':
    init_sample_data()
