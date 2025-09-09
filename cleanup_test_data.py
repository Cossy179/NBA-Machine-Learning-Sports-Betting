#!/usr/bin/env python3
"""
Cleanup script to remove test users and weak admin accounts
Run this script to clean up test data from the database
"""

import sqlite3
import sys

def cleanup_test_data():
    """Remove test users and weak admin accounts"""
    conn = sqlite3.connect('web_database.db')
    cursor = conn.cursor()
    
    try:
        # Get current users before cleanup
        cursor.execute('SELECT username, email, is_admin FROM users')
        users_before = cursor.fetchall()
        print("Users before cleanup:")
        for user in users_before:
            print(f"  {user[0]} ({user[1]}) - Admin: {bool(user[2])}")
        
        print("\n" + "="*50)
        
        # Remove weak admin user (admin/admin123)
        cursor.execute('DELETE FROM users WHERE username = ? AND email = ?', ('admin', 'admin@goonsteen.com'))
        admin_deleted = cursor.rowcount
        if admin_deleted > 0:
            print(f"✅ Removed weak admin user: admin@goonsteen.com")
        
        # Remove all test users (common test patterns)
        test_patterns = [
            'john_doe', 'jane_smith', 'mike_brown', 'sarah_wilson', 
            'david_jones', 'lisa_garcia', 'tom_miller', 'amy_davis'
        ]
        
        test_users_deleted = 0
        for pattern in test_patterns:
            cursor.execute('DELETE FROM users WHERE username = ?', (pattern,))
            test_users_deleted += cursor.rowcount
        
        if test_users_deleted > 0:
            print(f"✅ Removed {test_users_deleted} test users")
        
        # Remove test users by email domain patterns
        test_email_domains = ['@email.com', '@test.com', '@example.com']
        for domain in test_email_domains:
            cursor.execute('DELETE FROM users WHERE email LIKE ?', (f'%{domain}',))
            domain_deleted = cursor.rowcount
            if domain_deleted > 0:
                print(f"✅ Removed {domain_deleted} users with {domain} emails")
        
        # Clean up related data
        # Remove sessions for deleted users
        cursor.execute('''
            DELETE FROM user_sessions 
            WHERE user_id NOT IN (SELECT id FROM users)
        ''')
        sessions_deleted = cursor.rowcount
        if sessions_deleted > 0:
            print(f"✅ Cleaned up {sessions_deleted} orphaned sessions")
        
        # Remove activity logs for deleted users
        cursor.execute('''
            DELETE FROM user_activity 
            WHERE user_id NOT IN (SELECT id FROM users)
        ''')
        activity_deleted = cursor.rowcount
        if activity_deleted > 0:
            print(f"✅ Cleaned up {activity_deleted} orphaned activity logs")
        
        # Commit all changes
        conn.commit()
        
        print("\n" + "="*50)
        print("CLEANUP COMPLETE!")
        
        # Show remaining users
        cursor.execute('SELECT username, email, is_admin FROM users')
        users_after = cursor.fetchall()
        print("\nRemaining users:")
        for user in users_after:
            admin_status = "🔑 ADMIN" if user[2] else "👤 User"
            print(f"  {admin_status}: {user[0]} ({user[1]})")
        
        if not users_after:
            print("  ⚠️  No users remaining! You may need to create an admin account.")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()

def validate_admin_password(password):
    """Validate admin password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if password.lower() in ['admin', 'admin123', 'password', '123456', 'password123']:
        return False, "Password is too weak/common"
    
    # Check for at least one uppercase, lowercase, digit, and special character
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    
    if not (has_upper and has_lower and has_digit and has_special):
        return False, "Password must contain uppercase, lowercase, digit, and special character"
    
    return True, "Password is strong"

if __name__ == "__main__":
    print("🧹 GoonSteen Database Cleanup Tool")
    print("="*50)
    
    # Ask for confirmation
    response = input("This will remove test users and weak admin accounts. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Cleanup cancelled.")
        sys.exit(0)
    
    cleanup_test_data()
    
    print("\n💡 Tip: Use 'py -m flask --app web_backend create-admin' to create a new admin account")
    print("   Make sure to use a strong password!")
