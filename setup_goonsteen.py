#!/usr/bin/env python3
"""
GoonSteen Setup Script
Complete setup for the NBA sports betting platform
"""

import os
import subprocess
import sys
import sqlite3
import hashlib
import secrets
from datetime import datetime

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print("✅ Python version check passed")

def install_requirements():
    """Install required Python packages"""
    print("📦 Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'web_requirements.txt'])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        sys.exit(1)

def initialize_database():
    """Initialize the database with schema"""
    print("🗄️ Initializing database...")
    
    if os.path.exists('web_database.db'):
        backup_name = f'web_database_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
        os.rename('web_database.db', backup_name)
        print(f"📋 Existing database backed up as {backup_name}")
    
    try:
        conn = sqlite3.connect('web_database.db')
        with open('database_schema.sql', 'r') as f:
            conn.executescript(f.read())
        conn.commit()
        conn.close()
        print("✅ Database schema created successfully")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        sys.exit(1)

def create_admin_user():
    """Create the admin user"""
    print("👤 Creating admin user...")
    
    def hash_password(password: str, salt: str = None) -> tuple:
        if salt is None:
            salt = secrets.token_hex(16)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return password_hash.hex(), salt
    
    try:
        conn = sqlite3.connect('web_database.db')
        cursor = conn.cursor()
        
        # Check if admin already exists
        admin_exists = cursor.execute('SELECT id FROM users WHERE username = ?', ['admin']).fetchone()
        
        if admin_exists:
            print("⚠️ Admin user already exists")
        else:
            password_hash, salt = hash_password('admin123')
            
            cursor.execute('''
                INSERT INTO users (
                    username, email, password_hash, salt, first_name, last_name,
                    date_of_birth, is_admin, email_verified, terms_accepted, responsible_gambling, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [
                'admin', 'admin@goonsteen.com', password_hash, salt,
                'Admin', 'User', '1990-01-01', True, True, True, True, 'active'
            ])
            
            conn.commit()
            print("✅ Admin user created successfully")
        
        conn.close()
    except Exception as e:
        print(f"❌ Failed to create admin user: {e}")
        sys.exit(1)

def create_sample_data():
    """Create sample data for testing"""
    print("🎲 Creating sample data...")
    try:
        subprocess.check_call([sys.executable, 'init_admin_data.py'])
        print("✅ Sample data created successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create sample data: {e}")
        print("⚠️ Continuing without sample data...")

def create_env_file():
    """Create environment file with default settings"""
    print("⚙️ Creating environment configuration...")
    
    env_content = f"""# GoonSteen Environment Configuration
# Generated on {datetime.now().isoformat()}

# Server Configuration
HOST=127.0.0.1
PORT=5000
DEBUG=True

# Database
DATABASE_PATH=web_database.db

# Security (CHANGE THESE IN PRODUCTION!)
SECRET_KEY={secrets.token_hex(32)}
JWT_SECRET={secrets.token_hex(32)}
JWT_EXPIRATION_HOURS=24

# Email Configuration (Optional)
# SMTP_SERVER=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USERNAME=your-email@gmail.com
# SMTP_PASSWORD=your-app-password

# API Configuration
API_RATE_LIMIT=100
MAX_LOGIN_ATTEMPTS=5
SESSION_TIMEOUT=1440
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Environment file created (.env)")

def print_success_message():
    """Print setup completion message"""
    print("\n" + "="*60)
    print("🎉 GOONSTEEN SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 NEXT STEPS:")
    print("1. Start the server:")
    print("   python web_backend.py")
    print("\n2. Open your browser to:")
    print("   http://localhost:5000")
    print("\n🔑 ADMIN LOGIN CREDENTIALS:")
    print("   Username: admin")
    print("   Password: admin123")
    print("\n📁 PROJECT STRUCTURE:")
    print("   web/           - Frontend files (HTML, CSS, JS)")
    print("   web_backend.py - Flask backend server")
    print("   web_database.db - SQLite database")
    print("   .env           - Environment configuration")
    print("\n⚠️ IMPORTANT SECURITY NOTES:")
    print("   - Change admin password after first login")
    print("   - Update SECRET_KEY and JWT_SECRET in .env for production")
    print("   - Configure proper email settings for production")
    print("\n🏀 Ready to start your NBA betting platform!")
    print("="*60)

def main():
    """Main setup function"""
    print("🏀 Welcome to GoonSteen Setup!")
    print("Setting up your NBA Sports Betting AI Platform...\n")
    
    # Run setup steps
    check_python_version()
    install_requirements()
    initialize_database()
    create_admin_user()
    create_sample_data()
    create_env_file()
    
    print_success_message()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
