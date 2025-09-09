#!/usr/bin/env python3
"""
Script to undo the effects of init_admin_data.py
This removes all test data and sample users created by the initialization script
"""

import sqlite3
import sys

def undo_init_data():
    """Remove all data created by init_admin_data.py"""
    conn = sqlite3.connect('web_database.db')
    cursor = conn.cursor()
    
    try:
        print("🔄 Undoing init_admin_data.py changes...")
        print("=" * 50)
        
        # Remove all test users (keep only your real admin account)
        cursor.execute('DELETE FROM users WHERE username != ?', ('alexhalliday',))
        users_deleted = cursor.rowcount
        print(f"✅ Removed {users_deleted} test users")
        
        # Remove all sample data from other tables
        tables_to_clean = [
            'user_sessions', 'user_activity', 'bankrolls', 'bankroll_transactions',
            'teams', 'games', 'betting_odds', 'predictions', 'bets', 'bet_legs',
            'model_performance', 'notifications', 'system_logs', 'api_usage',
            'user_subscriptions'
        ]
        
        total_records_deleted = 0
        for table in tables_to_clean:
            try:
                cursor.execute(f'DELETE FROM {table}')
                records_deleted = cursor.rowcount
                if records_deleted > 0:
                    print(f"✅ Cleaned {records_deleted} records from {table}")
                    total_records_deleted += records_deleted
            except sqlite3.Error as e:
                print(f"⚠️  Could not clean {table}: {e}")
        
        # Reset auto-increment counters
        cursor.execute('DELETE FROM sqlite_sequence WHERE name != "users"')
        print("✅ Reset auto-increment counters")
        
        # Commit all changes
        conn.commit()
        
        print("\n" + "=" * 50)
        print("✅ UNDO COMPLETE!")
        print(f"   Removed {users_deleted} test users")
        print(f"   Cleaned {total_records_deleted} sample records")
        
        # Show remaining users
        cursor.execute('SELECT username, email, is_admin FROM users')
        remaining_users = cursor.fetchall()
        print(f"\nRemaining users ({len(remaining_users)}):")
        for user in remaining_users:
            admin_status = "🔑 ADMIN" if user[2] else "👤 User"
            print(f"  {admin_status}: {user[0]} ({user[1]})")
        
    except Exception as e:
        print(f"❌ Error during undo: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    print("🔄 GoonSteen Data Undo Tool")
    print("This will remove all test data created by init_admin_data.py")
    print("=" * 60)
    
    # Ask for confirmation
    response = input("This will remove all test data. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Undo cancelled.")
        sys.exit(0)
    
    undo_init_data()
    
    print("\n💡 Your real admin account (alexhalliday) has been preserved.")
    print("   All test data has been removed.")
