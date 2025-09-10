-- GoonSteen Database Schema
-- SQLite Database for NBA Sports Betting Platform

-- Users Table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    salt VARCHAR(32) NOT NULL,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    date_of_birth DATE NOT NULL,
    phone VARCHAR(20),
    avatar_url VARCHAR(255),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'inactive', 'banned')),
    subscription_type VARCHAR(20) DEFAULT 'free' CHECK (subscription_type IN ('free', 'premium', 'pro')),
    is_admin BOOLEAN DEFAULT FALSE,
    email_verified BOOLEAN DEFAULT FALSE,
    email_verification_token VARCHAR(100),
    password_reset_token VARCHAR(100),
    password_reset_expires DATETIME,
    last_login DATETIME,
    login_attempts INTEGER DEFAULT 0,
    locked_until DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    terms_accepted BOOLEAN DEFAULT FALSE,
    marketing_emails BOOLEAN DEFAULT FALSE,
    responsible_gambling BOOLEAN DEFAULT FALSE
);

-- User Sessions Table
CREATE TABLE user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    expires_at DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- User Activity Log
CREATE TABLE user_activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    activity_type VARCHAR(50) NOT NULL,
    description TEXT,
    ip_address VARCHAR(45),
    user_agent TEXT,
    metadata JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Bankroll Management
CREATE TABLE bankrolls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    total_balance DECIMAL(10,2) DEFAULT 0.00,
    available_balance DECIMAL(10,2) DEFAULT 0.00,
    reserved_balance DECIMAL(10,2) DEFAULT 0.00,
    total_deposited DECIMAL(10,2) DEFAULT 0.00,
    total_withdrawn DECIMAL(10,2) DEFAULT 0.00,
    total_profit_loss DECIMAL(10,2) DEFAULT 0.00,
    daily_limit DECIMAL(10,2) DEFAULT 100.00,
    weekly_limit DECIMAL(10,2) DEFAULT 500.00,
    monthly_limit DECIMAL(10,2) DEFAULT 2000.00,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Teams Table (NBA Teams)
CREATE TABLE teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    abbreviation VARCHAR(5) NOT NULL UNIQUE,
    city VARCHAR(100) NOT NULL,
    conference VARCHAR(10) CHECK (conference IN ('Eastern', 'Western')),
    division VARCHAR(20),
    logo_url VARCHAR(255),
    primary_color VARCHAR(7),
    secondary_color VARCHAR(7),
    active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Games Table
CREATE TABLE games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_game_id VARCHAR(50) UNIQUE,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    game_time TIME NOT NULL,
    season VARCHAR(10) NOT NULL,
    week INTEGER,
    status VARCHAR(20) DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'live', 'completed', 'postponed', 'cancelled')),
    home_score INTEGER DEFAULT 0,
    away_score INTEGER DEFAULT 0,
    quarter INTEGER DEFAULT 1,
    time_remaining VARCHAR(10),
    venue VARCHAR(100),
    weather_conditions TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (home_team_id) REFERENCES teams(id),
    FOREIGN KEY (away_team_id) REFERENCES teams(id)
);

-- AI Predictions
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20),
    prediction_type VARCHAR(50) NOT NULL CHECK (prediction_type IN ('moneyline', 'spread', 'total', 'score')),
    predicted_winner INTEGER,
    predicted_home_score DECIMAL(4,1),
    predicted_away_score DECIMAL(4,1),
    predicted_total DECIMAL(4,1),
    predicted_spread DECIMAL(4,1),
    confidence DECIMAL(5,2) NOT NULL,
    probability DECIMAL(5,4),
    expected_value DECIMAL(8,4),
    kelly_criterion DECIMAL(6,4),
    features JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE,
    FOREIGN KEY (predicted_winner) REFERENCES teams(id)
);

-- User Bets
CREATE TABLE bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    game_id INTEGER,
    bet_type VARCHAR(50) NOT NULL CHECK (bet_type IN ('single', 'parlay', 'system')),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'won', 'lost', 'pushed', 'cancelled', 'void')),
    stake DECIMAL(10,2) NOT NULL,
    potential_payout DECIMAL(10,2) NOT NULL,
    actual_payout DECIMAL(10,2) DEFAULT 0.00,
    odds DECIMAL(8,2) NOT NULL,
    bet_details JSON NOT NULL,
    placed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    settled_at DATETIME,
    notes TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE SET NULL
);

-- Notifications
CREATE TABLE notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    notification_type VARCHAR(50) NOT NULL,
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    action_url VARCHAR(255),
    is_read BOOLEAN DEFAULT FALSE,
    priority VARCHAR(10) DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    expires_at DATETIME,
    metadata JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- System Settings
CREATE TABLE system_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value TEXT,
    setting_type VARCHAR(20) DEFAULT 'string' CHECK (setting_type IN ('string', 'integer', 'decimal', 'boolean', 'json')),
    description TEXT,
    category VARCHAR(50),
    is_public BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Model Performance Tracking
CREATE TABLE model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20),
    prediction_type VARCHAR(50) NOT NULL,
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy DECIMAL(5,2) DEFAULT 0.00,
    profit_loss DECIMAL(10,2) DEFAULT 0.00,
    roi DECIMAL(6,2) DEFAULT 0.00,
    sharpe_ratio DECIMAL(6,4),
    max_drawdown DECIMAL(6,2),
    win_rate DECIMAL(5,2) DEFAULT 0.00,
    avg_odds DECIMAL(6,2),
    kelly_performance DECIMAL(6,4),
    date_from DATE NOT NULL,
    date_to DATE NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insert default admin user (password: admin123)
INSERT OR IGNORE INTO users (
    username, 
    email, 
    password_hash, 
    salt, 
    first_name, 
    last_name, 
    date_of_birth, 
    is_admin, 
    email_verified, 
    terms_accepted, 
    responsible_gambling
) VALUES (
    'admin',
    'admin@goonsteen.com',
    'pbkdf2:sha256:260000$salt123$hash123', -- This should be properly hashed
    'salt123',
    'Admin',
    'User',
    '1990-01-01',
    TRUE,
    TRUE,
    TRUE,
    TRUE
);

-- Insert default system settings
INSERT OR IGNORE INTO system_settings (setting_key, setting_value, setting_type, description, category, is_public) VALUES
('site_name', 'GoonSteen', 'string', 'Website name', 'general', TRUE),
('site_description', 'NBA Sports Betting AI Platform', 'string', 'Website description', 'general', TRUE),
('max_login_attempts', '5', 'integer', 'Maximum login attempts before lockout', 'security', FALSE),
('lockout_duration', '30', 'integer', 'Account lockout duration in minutes', 'security', FALSE),
('session_timeout', '1440', 'integer', 'Session timeout in minutes', 'security', FALSE),
('default_bankroll', '1000.00', 'decimal', 'Default bankroll for new users', 'betting', FALSE),
('min_bet_amount', '1.00', 'decimal', 'Minimum bet amount', 'betting', TRUE),
('max_bet_amount', '1000.00', 'decimal', 'Maximum bet amount for free users', 'betting', TRUE),
('kelly_criterion_enabled', 'true', 'boolean', 'Enable Kelly Criterion calculations', 'betting', TRUE),
('model_update_frequency', '24', 'integer', 'Model update frequency in hours', 'ml', FALSE);

-- Insert NBA teams
INSERT OR IGNORE INTO teams (name, abbreviation, city, conference, division) VALUES
('Atlanta Hawks', 'ATL', 'Atlanta', 'Eastern', 'Southeast'),
('Boston Celtics', 'BOS', 'Boston', 'Eastern', 'Atlantic'),
('Brooklyn Nets', 'BKN', 'Brooklyn', 'Eastern', 'Atlantic'),
('Charlotte Hornets', 'CHA', 'Charlotte', 'Eastern', 'Southeast'),
('Chicago Bulls', 'CHI', 'Chicago', 'Eastern', 'Central'),
('Cleveland Cavaliers', 'CLE', 'Cleveland', 'Eastern', 'Central'),
('Dallas Mavericks', 'DAL', 'Dallas', 'Western', 'Southwest'),
('Denver Nuggets', 'DEN', 'Denver', 'Western', 'Northwest'),
('Detroit Pistons', 'DET', 'Detroit', 'Eastern', 'Central'),
('Golden State Warriors', 'GSW', 'Golden State', 'Western', 'Pacific'),
('Houston Rockets', 'HOU', 'Houston', 'Western', 'Southwest'),
('Indiana Pacers', 'IND', 'Indiana', 'Eastern', 'Central'),
('LA Clippers', 'LAC', 'Los Angeles', 'Western', 'Pacific'),
('Los Angeles Lakers', 'LAL', 'Los Angeles', 'Western', 'Pacific'),
('Memphis Grizzlies', 'MEM', 'Memphis', 'Western', 'Southwest'),
('Miami Heat', 'MIA', 'Miami', 'Eastern', 'Southeast'),
('Milwaukee Bucks', 'MIL', 'Milwaukee', 'Eastern', 'Central'),
('Minnesota Timberwolves', 'MIN', 'Minnesota', 'Western', 'Northwest'),
('New Orleans Pelicans', 'NOP', 'New Orleans', 'Western', 'Southwest'),
('New York Knicks', 'NYK', 'New York', 'Eastern', 'Atlantic'),
('Oklahoma City Thunder', 'OKC', 'Oklahoma City', 'Western', 'Northwest'),
('Orlando Magic', 'ORL', 'Orlando', 'Eastern', 'Southeast'),
('Philadelphia 76ers', 'PHI', 'Philadelphia', 'Eastern', 'Atlantic'),
('Phoenix Suns', 'PHX', 'Phoenix', 'Western', 'Pacific'),
('Portland Trail Blazers', 'POR', 'Portland', 'Western', 'Northwest'),
('Sacramento Kings', 'SAC', 'Sacramento', 'Western', 'Pacific'),
('San Antonio Spurs', 'SAS', 'San Antonio', 'Western', 'Southwest'),
('Toronto Raptors', 'TOR', 'Toronto', 'Eastern', 'Atlantic'),
('Utah Jazz', 'UTA', 'Utah', 'Western', 'Northwest'),
('Washington Wizards', 'WAS', 'Washington', 'Eastern', 'Southeast');
