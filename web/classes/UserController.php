<?php
/**
 * User controller for handling user-related API endpoints
 */

class UserController {
    private $db;
    private $auth;
    
    public function __construct($database, $auth) {
        $this->db = $database;
        $this->auth = $auth;
    }
    
    public function signup($params = []) {
        $router = new Router();
        $data = $router->getJsonInput();
        
        // Validate required fields
        $required = ['first_name', 'last_name', 'username', 'email', 'password', 'age_verification'];
        foreach ($required as $field) {
            if (empty($data[$field])) {
                respondError("$field is required");
            }
        }
        
        // Validate email format
        if (!$this->auth->validateEmail($data['email'])) {
            respondError('Invalid email format');
        }
        
        // Validate username format
        if (!$this->auth->validateUsername($data['username'])) {
            respondError('Username must be 3-20 characters, letters and numbers only');
        }
        
        // Check age verification
        $birthDate = $this->auth->parseBirthDate($data['age_verification']);
        if (!$birthDate) {
            respondError('Invalid birth date format');
        }
        
        $age = $this->auth->calculateAge($birthDate);
        if ($age < 18) {
            respondError('You must be 18 or older to register');
        }
        
        // Check if user already exists
        $existing = $this->db->fetch(
            'SELECT id FROM users WHERE username = ? OR email = ?',
            [$data['username'], $data['email']]
        );
        
        if ($existing) {
            respondError('Username or email already exists');
        }
        
        // Hash password
        list($passwordHash, $salt) = $this->auth->hashPassword($data['password']);
        
        try {
            $this->db->beginTransaction();
            
            // Create user
            $userId = $this->db->execute(
                'INSERT INTO users (username, email, password_hash, salt, first_name, last_name, date_of_birth, terms_accepted, marketing_emails, responsible_gambling) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                [
                    $data['username'],
                    $data['email'],
                    base64_encode($passwordHash),
                    $salt,
                    $data['first_name'],
                    $data['last_name'],
                    $birthDate,
                    !empty($data['terms_accepted']),
                    !empty($data['marketing_emails']),
                    !empty($data['responsible_gambling'])
                ]
            );
            
            // Create initial bankroll
            $this->db->execute(
                'INSERT INTO bankrolls (user_id, total_balance, available_balance) VALUES (?, 1000.00, 1000.00)',
                [$userId]
            );
            
            $this->db->commit();
            
            // Log activity
            $this->auth->logActivity($userId, 'user_registration', 'User account created');
            
            logMessage('INFO', "New user registered: {$data['username']} (ID: $userId)");
            
            return [
                'message' => 'Account created successfully',
                'user_id' => $userId
            ];
            
        } catch (Exception $e) {
            $this->db->rollback();
            logMessage('ERROR', 'Signup error: ' . $e->getMessage());
            respondError('Registration failed', 500);
        }
    }
    
    public function login($params = []) {
        $router = new Router();
        $data = $router->getJsonInput();
        
        $username = $data['username'] ?? '';
        $password = $data['password'] ?? '';
        $rememberMe = !empty($data['remember_me']);
        
        if (!$username || !$password) {
            respondError('Username and password are required');
        }
        
        // Find user by username or email
        $user = $this->db->fetch(
            'SELECT * FROM users WHERE username = ? OR email = ?',
            [$username, $username]
        );
        
        if (!$user) {
            $this->auth->logActivity(null, 'login_failed', "Login attempt with unknown username: $username");
            respondError('Invalid credentials', 401);
        }
        
        // Check if account is locked
        if ($this->auth->isAccountLocked($user)) {
            respondError('Account is temporarily locked', 423);
        }
        
        // Verify password
        $storedHash = base64_decode($user['password_hash']);
        if (!$this->auth->verifyPassword($password, $storedHash, $user['salt'])) {
            $attempts = $this->auth->incrementLoginAttempts($user['id']);
            $this->auth->logActivity($user['id'], 'login_failed', 'Invalid password');
            respondError('Invalid credentials', 401);
        }
        
        // Check account status
        if ($user['status'] !== 'active') {
            $this->auth->logActivity($user['id'], 'login_failed', "Login attempt with {$user['status']} account");
            respondError("Account is {$user['status']}", 403);
        }
        
        // Reset login attempts on successful login
        $this->auth->resetLoginAttempts($user['id']);
        
        // Create JWT token
        $token = $this->auth->createJwtToken($user['id']);
        
        // Create session record
        $sessionToken = $this->auth->createSession($user['id'], $rememberMe);
        
        // Log successful login
        $this->auth->logActivity($user['id'], 'login_success', 'User logged in successfully');
        
        logMessage('INFO', "User logged in: {$user['username']} (ID: {$user['id']})");
        
        return [
            'message' => 'Login successful',
            'token' => $token,
            'user' => [
                'id' => (int)$user['id'],
                'username' => $user['username'],
                'email' => $user['email'],
                'first_name' => $user['first_name'],
                'last_name' => $user['last_name'],
                'is_admin' => (bool)$user['is_admin'],
                'subscription_type' => $user['subscription_type']
            ]
        ];
    }
    
    public function logout($params = []) {
        $this->auth->requireAuth();
        $user = $this->auth->getCurrentUser();
        
        // Log activity
        $this->auth->logActivity($user['id'], 'logout', 'User logged out');
        
        return ['message' => 'Logged out successfully'];
    }
    
    public function getSession($params = []) {
        $this->auth->requireAuth();
        $user = $this->auth->getCurrentUser();
        
        return [
            'authenticated' => true,
            'user' => [
                'id' => (int)$user['id'],
                'username' => $user['username'],
                'email' => $user['email'],
                'first_name' => $user['first_name'],
                'last_name' => $user['last_name'],
                'is_admin' => (bool)$user['is_admin'],
                'subscription_type' => $user['subscription_type']
            ]
        ];
    }
    
    public function checkUsername($params = []) {
        $username = $_GET['username'] ?? '';
        
        if (strlen($username) < 3) {
            return ['available' => false, 'error' => 'Username too short'];
        }
        
        $existing = $this->db->fetch('SELECT id FROM users WHERE username = ?', [$username]);
        
        return ['available' => !$existing];
    }
    
    public function getProfile($params = []) {
        $this->auth->requireAuth();
        $user = $this->auth->getCurrentUser();
        
        $userData = $this->db->fetch(
            'SELECT u.*, b.total_balance, b.available_balance, b.daily_limit, b.total_profit_loss FROM users u LEFT JOIN bankrolls b ON u.id = b.user_id WHERE u.id = ?',
            [$user['id']]
        );
        
        if ($userData) {
            // Convert numeric fields
            if (isset($userData['total_balance'])) {
                $userData['total_balance'] = (float)$userData['total_balance'];
            }
            if (isset($userData['available_balance'])) {
                $userData['available_balance'] = (float)$userData['available_balance'];
            }
            if (isset($userData['daily_limit'])) {
                $userData['daily_limit'] = (float)$userData['daily_limit'];
            }
            if (isset($userData['total_profit_loss'])) {
                $userData['total_profit_loss'] = (float)$userData['total_profit_loss'];
            }
            
            return $userData;
        }
        
        respondError('User not found', 404);
    }
    
    public function getBankroll($params = []) {
        $this->auth->requireAuth();
        $user = $this->auth->getCurrentUser();
        
        $bankroll = $this->db->fetch('SELECT * FROM bankrolls WHERE user_id = ?', [$user['id']]);
        
        if ($bankroll) {
            // Convert to float
            foreach (['total_balance', 'available_balance', 'reserved_balance', 'daily_limit', 'weekly_limit', 'monthly_limit', 'total_profit_loss'] as $field) {
                if (isset($bankroll[$field])) {
                    $bankroll[$field] = (float)$bankroll[$field];
                }
            }
            return $bankroll;
        }
        
        // Create initial bankroll
        $this->db->execute(
            'INSERT INTO bankrolls (user_id, total_balance, available_balance) VALUES (?, 0.00, 0.00)',
            [$user['id']]
        );
        
        return [
            'user_id' => $user['id'],
            'total_balance' => 0.00,
            'available_balance' => 0.00,
            'daily_limit' => 100.00
        ];
    }
    
    public function updateBankroll($params = []) {
        $this->auth->requireAuth();
        $user = $this->auth->getCurrentUser();
        
        $router = new Router();
        $data = $router->getJsonInput();
        
        $totalBalance = (float)($data['total_balance'] ?? 0);
        $dailyLimit = (float)($data['daily_limit'] ?? 100);
        
        try {
            // Check if bankroll exists
            $existing = $this->db->fetch('SELECT id FROM bankrolls WHERE user_id = ?', [$user['id']]);
            
            if ($existing) {
                $this->db->execute(
                    'UPDATE bankrolls SET total_balance = ?, available_balance = ?, daily_limit = ?, updated_at = ? WHERE user_id = ?',
                    [$totalBalance, $totalBalance, $dailyLimit, date('Y-m-d H:i:s'), $user['id']]
                );
            } else {
                $this->db->execute(
                    'INSERT INTO bankrolls (user_id, total_balance, available_balance, daily_limit) VALUES (?, ?, ?, ?)',
                    [$user['id'], $totalBalance, $totalBalance, $dailyLimit]
                );
            }
            
            // Log the bankroll update
            $this->auth->logActivity($user['id'], 'bankroll_updated', "Bankroll updated to \$$totalBalance");
            
            return ['message' => 'Bankroll updated successfully'];
            
        } catch (Exception $e) {
            logMessage('ERROR', 'Error updating bankroll: ' . $e->getMessage());
            respondError('Failed to update bankroll', 500);
        }
    }
    
    public function trackBet($params = []) {
        $this->auth->requireAuth();
        $user = $this->auth->getCurrentUser();
        
        $router = new Router();
        $data = $router->getJsonInput();
        
        $gameId = $data['game_id'] ?? null;
        $betAmount = (float)($data['bet_amount'] ?? 0);
        $betType = $data['bet_type'] ?? '';
        $odds = $data['odds'] ?? '';
        
        if (!$gameId || !$betAmount || !$betType || !$odds) {
            respondError('All fields are required');
        }
        
        try {
            // Parse odds to calculate potential payout
            $decimalOdds = 1.0;
            if (strpos($odds, '+') === 0) {
                $decimalOdds = (floatval(substr($odds, 1)) / 100) + 1;
            } elseif (strpos($odds, '-') === 0) {
                $decimalOdds = (100 / floatval(substr($odds, 1))) + 1;
            } else {
                $decimalOdds = floatval($odds);
            }
            
            $potentialPayout = $betAmount * $decimalOdds;
            
            // Create bet record
            $betId = $this->db->execute(
                'INSERT INTO bets (user_id, game_id, bet_type, status, stake, potential_payout, odds, bet_details, placed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                [
                    $user['id'],
                    $gameId,
                    'single',
                    'pending',
                    $betAmount,
                    $potentialPayout,
                    $decimalOdds,
                    json_encode(['bet_type' => $betType, 'odds_display' => $odds]),
                    date('Y-m-d H:i:s')
                ]
            );
            
            // Log the bet
            $this->auth->logActivity($user['id'], 'bet_tracked', "Tracked \$$betAmount bet on game $gameId");
            
            return [
                'message' => 'Bet tracked successfully',
                'bet_id' => $betId,
                'potential_payout' => $potentialPayout
            ];
            
        } catch (Exception $e) {
            logMessage('ERROR', 'Error tracking bet: ' . $e->getMessage());
            respondError('Failed to track bet', 500);
        }
    }
    
    public function calculateKelly($params = []) {
        $this->auth->requireAuth();
        $user = $this->auth->getCurrentUser();
        
        $router = new Router();
        $data = $router->getJsonInput();
        
        $gameId = $data['game_id'] ?? null;
        $betAmount = (float)($data['bet_amount'] ?? 0);
        $odds = $data['odds'] ?? '';
        
        try {
            // Get prediction confidence for the game
            $prediction = $this->db->fetch(
                'SELECT confidence, probability FROM predictions WHERE game_id = ? ORDER BY created_at DESC LIMIT 1',
                [$gameId]
            );
            
            if (!$prediction) {
                return ['kelly_amount' => 0, 'message' => 'No prediction available'];
            }
            
            // Get user bankroll
            $bankroll = $this->db->fetch('SELECT total_balance FROM bankrolls WHERE user_id = ?', [$user['id']]);
            $totalBankroll = $bankroll ? (float)$bankroll['total_balance'] : 1000;
            
            // Parse odds
            $decimalOdds = 1.0;
            if (strpos($odds, '+') === 0) {
                $decimalOdds = (floatval(substr($odds, 1)) / 100) + 1;
            } elseif (strpos($odds, '-') === 0) {
                $decimalOdds = (100 / floatval(substr($odds, 1))) + 1;
            } else {
                $decimalOdds = floatval($odds);
            }
            
            // Kelly Criterion calculation
            $winProbability = $prediction['confidence'] / 100;
            $loseProbability = 1 - $winProbability;
            $b = $decimalOdds - 1;
            
            $kellyPercentage = ($b * $winProbability - $loseProbability) / $b;
            $kellyAmount = max(0, $kellyPercentage * $totalBankroll);
            
            // Cap at 5% of bankroll for safety
            $kellyAmount = min($kellyAmount, $totalBankroll * 0.05);
            
            return [
                'kelly_amount' => round($kellyAmount, 2),
                'kelly_percentage' => round($kellyPercentage * 100, 2),
                'confidence' => $prediction['confidence']
            ];
            
        } catch (Exception $e) {
            logMessage('ERROR', 'Error calculating Kelly: ' . $e->getMessage());
            return ['kelly_amount' => 0, 'error' => 'Calculation failed'];
        }
    }
    
    public function getAnalytics($params = []) {
        $this->auth->requireAuth();
        $user = $this->auth->getCurrentUser();
        
        $analytics = $this->db->fetch(
            'SELECT COUNT(*) as total_bets, AVG(stake) as avg_bet, AVG(CASE WHEN status IN ("won", "lost") THEN CASE WHEN status = "won" THEN 1.0 ELSE 0.0 END END) * 100 as win_rate, MAX(CASE WHEN status = "won" THEN 1 ELSE 0 END) as best_streak FROM bets WHERE user_id = ?',
            [$user['id']]
        );
        
        return [
            'total_bets' => (int)($analytics['total_bets'] ?? 0),
            'avg_bet' => (float)($analytics['avg_bet'] ?? 0),
            'win_rate' => round((float)($analytics['win_rate'] ?? 0), 1),
            'best_streak' => (int)($analytics['best_streak'] ?? 0)
        ];
    }
    
    public function getBettingHistory($params = []) {
        $this->auth->requireAuth();
        $user = $this->auth->getCurrentUser();
        
        $limit = (int)($_GET['limit'] ?? 50);
        $statusFilter = $_GET['status'] ?? '';
        $period = $_GET['period'] ?? 'all';
        
        $whereClauses = ['user_id = ?'];
        $queryParams = [$user['id']];
        
        if ($statusFilter) {
            $whereClauses[] = 'status = ?';
            $queryParams[] = $statusFilter;
        }
        
        if ($period !== 'all') {
            switch ($period) {
                case '7d':
                    $whereClauses[] = 'placed_at >= date("now", "-7 days")';
                    break;
                case '30d':
                    $whereClauses[] = 'placed_at >= date("now", "-30 days")';
                    break;
                case '90d':
                    $whereClauses[] = 'placed_at >= date("now", "-90 days")';
                    break;
            }
        }
        
        $whereClause = implode(' AND ', $whereClauses);
        $queryParams[] = $limit;
        
        $bets = $this->db->fetchAll(
            "SELECT *, json_extract(bet_details, '\$.bet_type') as bet_type_detail, json_extract(bet_details, '\$.odds_display') as odds_display FROM bets WHERE $whereClause ORDER BY placed_at DESC LIMIT ?",
            $queryParams
        );
        
        // Convert numeric fields
        foreach ($bets as &$bet) {
            $bet['stake'] = (float)$bet['stake'];
            $bet['potential_payout'] = (float)$bet['potential_payout'];
            $bet['actual_payout'] = (float)$bet['actual_payout'];
            $bet['odds'] = (float)$bet['odds'];
        }
        
        return $bets;
    }
    
    public function getNotifications($params = []) {
        $this->auth->requireAuth();
        $user = $this->auth->getCurrentUser();
        
        $notifications = $this->db->fetchAll(
            'SELECT * FROM notifications WHERE user_id = ? OR user_id IS NULL ORDER BY created_at DESC LIMIT 20',
            [$user['id']]
        );
        
        return $notifications;
    }
    
    public function markNotificationRead($params = []) {
        $this->auth->requireAuth();
        $user = $this->auth->getCurrentUser();
        
        $notificationId = $params['id'] ?? null;
        
        if (!$notificationId) {
            respondError('Notification ID required');
        }
        
        try {
            $this->db->execute(
                'UPDATE notifications SET is_read = 1 WHERE id = ? AND user_id = ?',
                [$notificationId, $user['id']]
            );
            
            return ['message' => 'Notification marked as read'];
            
        } catch (Exception $e) {
            logMessage('ERROR', 'Error marking notification as read: ' . $e->getMessage());
            respondError('Failed to mark notification as read', 500);
        }
    }
}
