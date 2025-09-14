<?php
/**
 * Database connection and management class
 */

class Database {
    private $connection = null;
    
    public function getConnection() {
        if ($this->connection === null) {
            try {
                // Use SQLite for compatibility with existing data
                $dbPath = ROOT_PATH . '/web_database.db';
                
                $this->connection = new PDO('sqlite:' . $dbPath);
                $this->connection->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
                $this->connection->setAttribute(PDO::ATTR_DEFAULT_FETCH_MODE, PDO::FETCH_ASSOC);
                
                // Enable foreign keys for SQLite
                $this->connection->exec('PRAGMA foreign_keys = ON');
                
                // Initialize database if it doesn't exist
                if (!file_exists($dbPath)) {
                    $this->initializeDatabase();
                }
                
            } catch (PDOException $e) {
                logMessage('ERROR', 'Database connection failed: ' . $e->getMessage());
                throw new Exception('Database connection failed');
            }
        }
        
        return $this->connection;
    }
    
    private function initializeDatabase() {
        try {
            if (file_exists(DATABASE_SCHEMA_PATH)) {
                $schema = file_get_contents(DATABASE_SCHEMA_PATH);
                $this->connection->exec($schema);
                logMessage('INFO', 'Database initialized successfully');
            } else {
                // Create basic schema if schema file doesn't exist
                $this->createBasicSchema();
            }
        } catch (Exception $e) {
            logMessage('ERROR', 'Database initialization failed: ' . $e->getMessage());
            throw $e;
        }
    }
    
    private function createBasicSchema() {
        $schema = "
        -- Basic schema for GoonSteen
        CREATE TABLE IF NOT EXISTS users (
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
        
        CREATE TABLE IF NOT EXISTS user_sessions (
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
        
        CREATE TABLE IF NOT EXISTS user_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            activity_type VARCHAR(50) NOT NULL,
            description TEXT,
            ip_address VARCHAR(45),
            user_agent TEXT,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        );
        
        CREATE TABLE IF NOT EXISTS bankrolls (
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
        
        CREATE TABLE IF NOT EXISTS teams (
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
        
        CREATE TABLE IF NOT EXISTS games (
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
        
        CREATE TABLE IF NOT EXISTS predictions (
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
            features TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE,
            FOREIGN KEY (predicted_winner) REFERENCES teams(id)
        );
        
        CREATE TABLE IF NOT EXISTS bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            game_id INTEGER,
            bet_type VARCHAR(50) NOT NULL CHECK (bet_type IN ('single', 'parlay', 'system')),
            status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'won', 'lost', 'pushed', 'cancelled', 'void')),
            stake DECIMAL(10,2) NOT NULL,
            potential_payout DECIMAL(10,2) NOT NULL,
            actual_payout DECIMAL(10,2) DEFAULT 0.00,
            odds DECIMAL(8,2) NOT NULL,
            bet_details TEXT NOT NULL,
            placed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            settled_at DATETIME,
            notes TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE SET NULL
        );
        
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            notification_type VARCHAR(50) NOT NULL,
            title VARCHAR(200) NOT NULL,
            message TEXT NOT NULL,
            action_url VARCHAR(255),
            is_read BOOLEAN DEFAULT FALSE,
            priority VARCHAR(10) DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
            expires_at DATETIME,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        
        CREATE TABLE IF NOT EXISTS system_settings (
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
        
        CREATE TABLE IF NOT EXISTS model_performance (
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
        
        ";
        
        $this->connection->exec($schema);
        
        // Insert default admin user with proper password hash
        $salt = bin2hex(random_bytes(16));
        $passwordHash = hash_pbkdf2('sha256', 'admin123', $salt, 100000, 0, true);
        
        $stmt = $this->connection->prepare("
            INSERT OR IGNORE INTO users 
            (username, email, password_hash, salt, first_name, last_name, date_of_birth, is_admin, email_verified, terms_accepted, responsible_gambling) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ");
        $stmt->execute([
            'admin', 'admin@goonsteen.com', base64_encode($passwordHash), $salt,
            'Admin', 'User', '1990-01-01', 1, 1, 1, 1
        ]);
        
        // Insert NBA teams
        $teamsData = [
            ['Atlanta Hawks', 'ATL', 'Atlanta', 'Eastern', 'Southeast'],
            ['Boston Celtics', 'BOS', 'Boston', 'Eastern', 'Atlantic'],
            ['Brooklyn Nets', 'BKN', 'Brooklyn', 'Eastern', 'Atlantic'],
            ['Charlotte Hornets', 'CHA', 'Charlotte', 'Eastern', 'Southeast'],
            ['Chicago Bulls', 'CHI', 'Chicago', 'Eastern', 'Central'],
            ['Cleveland Cavaliers', 'CLE', 'Cleveland', 'Eastern', 'Central'],
            ['Dallas Mavericks', 'DAL', 'Dallas', 'Western', 'Southwest'],
            ['Denver Nuggets', 'DEN', 'Denver', 'Western', 'Northwest'],
            ['Detroit Pistons', 'DET', 'Detroit', 'Eastern', 'Central'],
            ['Golden State Warriors', 'GSW', 'Golden State', 'Western', 'Pacific']
        ];
        
        $teamStmt = $this->connection->prepare("INSERT OR IGNORE INTO teams (name, abbreviation, city, conference, division) VALUES (?, ?, ?, ?, ?)");
        foreach ($teamsData as $team) {
            $teamStmt->execute($team);
        }
        
        // Insert default system settings
        $settingsData = [
            ['site_name', 'GoonSteen', 'string', 'Website name', 'general', 1],
            ['max_login_attempts', '5', 'integer', 'Maximum login attempts', 'security', 0],
            ['default_bankroll', '1000.00', 'decimal', 'Default bankroll', 'betting', 0],
            ['kelly_criterion_enabled', 'true', 'boolean', 'Enable Kelly Criterion', 'betting', 1]
        ];
        
        $settingStmt = $this->connection->prepare("INSERT OR IGNORE INTO system_settings (setting_key, setting_value, setting_type, description, category, is_public) VALUES (?, ?, ?, ?, ?, ?)");
        foreach ($settingsData as $setting) {
            $settingStmt->execute($setting);
        }
        
        logMessage('INFO', 'Basic database schema created');
    }
    
    public function query($sql, $params = []) {
        try {
            $stmt = $this->connection->prepare($sql);
            $stmt->execute($params);
            return $stmt;
        } catch (PDOException $e) {
            logMessage('ERROR', 'Database query failed: ' . $e->getMessage(), ['sql' => $sql, 'params' => $params]);
            throw new Exception('Database query failed');
        }
    }
    
    public function fetch($sql, $params = []) {
        $stmt = $this->query($sql, $params);
        return $stmt->fetch();
    }
    
    public function prepare($sql) {
        return $this->connection->prepare($sql);
    }
    
    public function fetchAll($sql, $params = []) {
        return $this->query($sql, $params)->fetchAll();
    }
    
    public function execute($sql, $params = []) {
        $stmt = $this->query($sql, $params);
        return $this->connection->lastInsertId();
    }
    
    public function beginTransaction() {
        return $this->connection->beginTransaction();
    }
    
    public function commit() {
        return $this->connection->commit();
    }
    
    public function rollback() {
        return $this->connection->rollback();
    }
}
