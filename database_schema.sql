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

-- Bankroll Transactions
CREATE TABLE bankroll_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    transaction_type VARCHAR(20) NOT NULL CHECK (transaction_type IN ('deposit', 'withdrawal', 'bet_placed', 'bet_won', 'bet_lost', 'bet_refund', 'adjustment')),
    amount DECIMAL(10,2) NOT NULL,
    balance_before DECIMAL(10,2) NOT NULL,
    balance_after DECIMAL(10,2) NOT NULL,
    description TEXT,
    reference_id VARCHAR(100),
    reference_type VARCHAR(50),
    metadata JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
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

-- Betting Odds
CREATE TABLE betting_odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    sportsbook VARCHAR(50) NOT NULL,
    bet_type VARCHAR(50) NOT NULL CHECK (bet_type IN ('moneyline', 'spread', 'total', 'prop')),
    home_odds DECIMAL(8,2),
    away_odds DECIMAL(8,2),
    spread DECIMAL(4,1),
    total DECIMAL(4,1),
    over_odds DECIMAL(8,2),
    under_odds DECIMAL(8,2),
    prop_description TEXT,
    prop_odds DECIMAL(8,2),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE
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

-- Bet Legs (for parlays and system bets)
CREATE TABLE bet_legs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bet_id INTEGER NOT NULL,
    game_id INTEGER NOT NULL,
    bet_type VARCHAR(50) NOT NULL CHECK (bet_type IN ('moneyline', 'spread', 'total', 'prop')),
    selection VARCHAR(100) NOT NULL,
    odds DECIMAL(8,2) NOT NULL,
    line_value DECIMAL(4,1),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'won', 'lost', 'pushed', 'void')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (bet_id) REFERENCES bets(id) ON DELETE CASCADE,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE
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

-- System Logs
CREATE TABLE system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_level VARCHAR(10) NOT NULL CHECK (log_level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    logger_name VARCHAR(100),
    message TEXT NOT NULL,
    module VARCHAR(100),
    function_name VARCHAR(100),
    line_number INTEGER,
    user_id INTEGER,
    session_id VARCHAR(255),
    ip_address VARCHAR(45),
    request_id VARCHAR(100),
    metadata JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- API Usage Tracking
CREATE TABLE api_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    request_size INTEGER,
    response_size INTEGER,
    ip_address VARCHAR(45),
    user_agent TEXT,
    rate_limit_remaining INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Subscription Plans
CREATE TABLE subscription_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    description TEXT,
    price DECIMAL(8,2) NOT NULL,
    billing_period VARCHAR(20) NOT NULL CHECK (billing_period IN ('monthly', 'quarterly', 'yearly')),
    features JSON,
    max_bets_per_day INTEGER,
    max_bets_per_month INTEGER,
    api_rate_limit INTEGER,
    priority_support BOOLEAN DEFAULT FALSE,
    active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- User Subscriptions
CREATE TABLE user_subscriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    plan_id INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'cancelled', 'expired', 'suspended')),
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,
    cancelled_at DATETIME,
    payment_method VARCHAR(50),
    payment_reference VARCHAR(100),
    auto_renew BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (plan_id) REFERENCES subscription_plans(id)
);

-- Indexes for Performance
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_created_at ON users(created_at);

CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_expires ON user_sessions(expires_at);

CREATE INDEX idx_user_activity_user_id ON user_activity(user_id);
CREATE INDEX idx_user_activity_type ON user_activity(activity_type);
CREATE INDEX idx_user_activity_created_at ON user_activity(created_at);

CREATE INDEX idx_bankrolls_user_id ON bankrolls(user_id);

CREATE INDEX idx_bankroll_transactions_user_id ON bankroll_transactions(user_id);
CREATE INDEX idx_bankroll_transactions_type ON bankroll_transactions(transaction_type);
CREATE INDEX idx_bankroll_transactions_created_at ON bankroll_transactions(created_at);

CREATE INDEX idx_games_date ON games(game_date);
CREATE INDEX idx_games_status ON games(status);
CREATE INDEX idx_games_home_team ON games(home_team_id);
CREATE INDEX idx_games_away_team ON games(away_team_id);
CREATE INDEX idx_games_season ON games(season);

CREATE INDEX idx_betting_odds_game_id ON betting_odds(game_id);
CREATE INDEX idx_betting_odds_type ON betting_odds(bet_type);
CREATE INDEX idx_betting_odds_sportsbook ON betting_odds(sportsbook);

CREATE INDEX idx_predictions_game_id ON predictions(game_id);
CREATE INDEX idx_predictions_model ON predictions(model_name);
CREATE INDEX idx_predictions_type ON predictions(prediction_type);
CREATE INDEX idx_predictions_confidence ON predictions(confidence);

CREATE INDEX idx_bets_user_id ON bets(user_id);
CREATE INDEX idx_bets_game_id ON bets(game_id);
CREATE INDEX idx_bets_status ON bets(status);
CREATE INDEX idx_bets_placed_at ON bets(placed_at);

CREATE INDEX idx_bet_legs_bet_id ON bet_legs(bet_id);
CREATE INDEX idx_bet_legs_game_id ON bet_legs(game_id);
CREATE INDEX idx_bet_legs_status ON bet_legs(status);

CREATE INDEX idx_model_performance_model ON model_performance(model_name);
CREATE INDEX idx_model_performance_type ON model_performance(prediction_type);
CREATE INDEX idx_model_performance_date ON model_performance(date_from, date_to);

CREATE INDEX idx_notifications_user_id ON notifications(user_id);
CREATE INDEX idx_notifications_read ON notifications(is_read);
CREATE INDEX idx_notifications_created_at ON notifications(created_at);

CREATE INDEX idx_system_logs_level ON system_logs(log_level);
CREATE INDEX idx_system_logs_created_at ON system_logs(created_at);
CREATE INDEX idx_system_logs_user_id ON system_logs(user_id);

CREATE INDEX idx_api_usage_user_id ON api_usage(user_id);
CREATE INDEX idx_api_usage_endpoint ON api_usage(endpoint);
CREATE INDEX idx_api_usage_created_at ON api_usage(created_at);

CREATE INDEX idx_user_subscriptions_user_id ON user_subscriptions(user_id);
CREATE INDEX idx_user_subscriptions_status ON user_subscriptions(status);

-- Triggers for Updated_at timestamps
CREATE TRIGGER update_users_timestamp 
    AFTER UPDATE ON users 
    BEGIN 
        UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id; 
    END;

CREATE TRIGGER update_bankrolls_timestamp 
    AFTER UPDATE ON bankrolls 
    BEGIN 
        UPDATE bankrolls SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id; 
    END;

CREATE TRIGGER update_teams_timestamp 
    AFTER UPDATE ON teams 
    BEGIN 
        UPDATE teams SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id; 
    END;

CREATE TRIGGER update_games_timestamp 
    AFTER UPDATE ON games 
    BEGIN 
        UPDATE games SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id; 
    END;

CREATE TRIGGER update_betting_odds_timestamp 
    AFTER UPDATE ON betting_odds 
    BEGIN 
        UPDATE betting_odds SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id; 
    END;

CREATE TRIGGER update_model_performance_timestamp 
    AFTER UPDATE ON model_performance 
    BEGIN 
        UPDATE model_performance SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id; 
    END;

CREATE TRIGGER update_system_settings_timestamp 
    AFTER UPDATE ON system_settings 
    BEGIN 
        UPDATE system_settings SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id; 
    END;

CREATE TRIGGER update_subscription_plans_timestamp 
    AFTER UPDATE ON subscription_plans 
    BEGIN 
        UPDATE subscription_plans SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id; 
    END;

CREATE TRIGGER update_user_subscriptions_timestamp 
    AFTER UPDATE ON user_subscriptions 
    BEGIN 
        UPDATE user_subscriptions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id; 
    END;

-- Insert default admin user (password: admin123)
INSERT INTO users (
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
INSERT INTO system_settings (setting_key, setting_value, setting_type, description, category, is_public) VALUES
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
INSERT INTO teams (name, abbreviation, city, conference, division) VALUES
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

-- Insert subscription plans
INSERT INTO subscription_plans (name, display_name, description, price, billing_period, features, max_bets_per_day, max_bets_per_month, api_rate_limit) VALUES
('free', 'Free', 'Basic access to predictions', 0.00, 'monthly', '{"predictions": true, "basic_analytics": true}', 5, 50, 100),
('premium', 'Premium', 'Advanced features and higher limits', 19.99, 'monthly', '{"predictions": true, "advanced_analytics": true, "parlay_builder": true, "kelly_criterion": true}', 25, 500, 500),
('pro', 'Professional', 'All features with highest limits', 49.99, 'monthly', '{"predictions": true, "advanced_analytics": true, "parlay_builder": true, "kelly_criterion": true, "api_access": true, "priority_support": true}', 100, 2000, 2000);

-- Create Views for common queries
CREATE VIEW user_stats AS
SELECT 
    u.id,
    u.username,
    u.email,
    u.first_name,
    u.last_name,
    u.status,
    u.subscription_type,
    u.created_at,
    b.total_balance,
    b.total_profit_loss,
    COUNT(bets.id) as total_bets,
    COALESCE(AVG(CASE WHEN bets.status = 'won' THEN 1.0 ELSE 0.0 END) * 100, 0) as win_rate,
    COALESCE(SUM(CASE WHEN bets.status = 'won' THEN bets.actual_payout - bets.stake ELSE 0 END), 0) as total_profit
FROM users u
LEFT JOIN bankrolls b ON u.id = b.user_id
LEFT JOIN bets ON u.id = bets.user_id
GROUP BY u.id, u.username, u.email, u.first_name, u.last_name, u.status, u.subscription_type, u.created_at, b.total_balance, b.total_profit_loss;

CREATE VIEW game_predictions_view AS
SELECT 
    g.id as game_id,
    g.game_date,
    g.game_time,
    g.status,
    ht.name as home_team,
    ht.abbreviation as home_abbr,
    at.name as away_team,
    at.abbreviation as away_abbr,
    g.home_score,
    g.away_score,
    p.model_name,
    p.prediction_type,
    p.predicted_winner,
    p.predicted_home_score,
    p.predicted_away_score,
    p.predicted_total,
    p.confidence,
    p.expected_value
FROM games g
JOIN teams ht ON g.home_team_id = ht.id
JOIN teams at ON g.away_team_id = at.id
LEFT JOIN predictions p ON g.id = p.game_id
WHERE g.game_date >= date('now');

-- Data cleanup procedures (to be run periodically)
-- Clean up old sessions
DELETE FROM user_sessions WHERE expires_at < datetime('now', '-7 days');

-- Clean up old logs
DELETE FROM system_logs WHERE created_at < datetime('now', '-30 days');

-- Clean up old API usage data
DELETE FROM api_usage WHERE created_at < datetime('now', '-90 days');

-- Clean up old activity logs
DELETE FROM user_activity WHERE created_at < datetime('now', '-180 days');
