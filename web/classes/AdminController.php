<?php
/**
 * Admin controller for handling admin-related API endpoints
 */

class AdminController {
    private $db;
    private $auth;
    
    public function __construct($database, $auth) {
        $this->db = $database;
        $this->auth = $auth;
    }
    
    public function getOverview($params = []) {
        $this->auth->requireAdmin();
        
        // Get user statistics
        $userStats = $this->db->fetch(
            'SELECT 
                COUNT(*) as total_users,
                COUNT(CASE WHEN created_at >= date("now", "-7 days") THEN 1 END) as new_users_week,
                COUNT(CASE WHEN status = "active" THEN 1 END) as active_users,
                COUNT(CASE WHEN last_login >= date("now", "-30 days") THEN 1 END) as active_monthly
            FROM users'
        );
        
        // Get betting statistics
        $betStats = $this->db->fetch(
            'SELECT 
                COUNT(*) as total_bets,
                COUNT(CASE WHEN placed_at >= date("now") THEN 1 END) as bets_today,
                AVG(CASE WHEN status IN ("won", "lost") THEN 
                    CASE WHEN status = "won" THEN 1.0 ELSE 0.0 END 
                END) * 100 as win_rate
            FROM bets'
        );
        
        $totalUsers = (int)($userStats['total_users'] ?? 0);
        $activeMonthly = (int)($userStats['active_monthly'] ?? 0);
        
        return [
            'total_users' => $totalUsers,
            'new_users_week' => (int)($userStats['new_users_week'] ?? 0),
            'active_users_percentage' => $totalUsers > 0 ? intval(($activeMonthly / $totalUsers) * 100) : 0,
            'total_bets' => (int)($betStats['total_bets'] ?? 0),
            'bets_today' => (int)($betStats['bets_today'] ?? 0),
            'win_rate' => intval($betStats['win_rate'] ?? 0),
            'revenue' => 24650.00, // Calculate from actual data
            'revenue_growth' => 8.5,
            'revenue_target_percentage' => 92,
            'model_accuracy' => 68.9,
            'accuracy_improvement' => 2.1
        ];
    }
    
    public function getRecentUsers($params = []) {
        $this->auth->requireAdmin();
        
        $users = $this->db->fetchAll(
            'SELECT id, first_name, last_name, username, email, status, created_at FROM users ORDER BY created_at DESC LIMIT 10'
        );
        
        return $users;
    }
    
    public function getActivity($params = []) {
        $this->auth->requireAdmin();
        
        $activities = $this->db->fetchAll(
            'SELECT activity_type as type, description as title, description, created_at as timestamp FROM user_activity ORDER BY created_at DESC LIMIT 20'
        );
        
        return $activities;
    }
    
    public function getUser($params = []) {
        $this->auth->requireAdmin();
        
        $userId = $params['id'] ?? null;
        if (!$userId) {
            respondError('User ID required');
        }
        
        $user = $this->db->fetch('SELECT * FROM users WHERE id = ?', [$userId]);
        
        if (!$user) {
            respondError('User not found', 404);
        }
        
        // Get user stats
        $stats = $this->db->fetch(
            'SELECT 
                COUNT(*) as total_bets,
                AVG(CASE WHEN status IN ("won", "lost") THEN 
                    CASE WHEN status = "won" THEN 1.0 ELSE 0.0 END 
                END) * 100 as win_rate,
                SUM(CASE WHEN status = "won" THEN actual_payout - stake ELSE 0 END) as total_profit
            FROM bets 
            WHERE user_id = ?',
            [$userId]
        );
        
        if ($stats) {
            $user['total_bets'] = (int)($stats['total_bets'] ?? 0);
            $user['win_rate'] = (float)($stats['win_rate'] ?? 0);
            $user['total_profit'] = (float)($stats['total_profit'] ?? 0);
        }
        
        return $user;
    }
    
    public function updateUser($params = []) {
        $this->auth->requireAdmin();
        
        $userId = $params['id'] ?? null;
        if (!$userId) {
            respondError('User ID required');
        }
        
        $input = file_get_contents('php://input');
        $data = json_decode($input, true) ?: [];
        
        $allowedFields = ['first_name', 'last_name', 'email', 'status', 'subscription_type'];
        $fields = [];
        $values = [];
        
        foreach ($allowedFields as $field) {
            if (isset($data[$field])) {
                $fields[] = "$field = ?";
                $values[] = $data[$field];
            }
        }
        
        if (empty($fields)) {
            respondError('No valid fields provided');
        }
        
        $values[] = date('Y-m-d H:i:s'); // updated_at
        $values[] = $userId;
        
        try {
            $this->db->execute(
                'UPDATE users SET ' . implode(', ', $fields) . ', updated_at = ? WHERE id = ?',
                $values
            );
            
            $currentUser = $this->auth->getCurrentUser();
            $this->auth->logActivity($userId, 'user_updated', "User updated by admin {$currentUser['username']}");
            
            return ['message' => 'User updated successfully'];
            
        } catch (Exception $e) {
            logMessage('ERROR', "Error updating user $userId: " . $e->getMessage());
            respondError('Failed to update user', 500);
        }
    }
    
    public function suspendUser($params = []) {
        $this->auth->requireAdmin();
        
        $userId = $params['id'] ?? null;
        if (!$userId) {
            respondError('User ID required');
        }
        
        try {
            $this->db->execute('UPDATE users SET status = ? WHERE id = ?', ['suspended', $userId]);
            
            $currentUser = $this->auth->getCurrentUser();
            $this->auth->logActivity($userId, 'account_suspended', "Account suspended by admin {$currentUser['username']}");
            
            logMessage('INFO', "User $userId suspended by admin {$currentUser['id']}");
            
            return ['message' => 'User suspended successfully'];
            
        } catch (Exception $e) {
            logMessage('ERROR', "Error suspending user $userId: " . $e->getMessage());
            respondError('Failed to suspend user', 500);
        }
    }
    
    public function unsuspendUser($params = []) {
        $this->auth->requireAdmin();
        
        $userId = $params['id'] ?? null;
        if (!$userId) {
            respondError('User ID required');
        }
        
        try {
            $this->db->execute('UPDATE users SET status = ? WHERE id = ?', ['active', $userId]);
            
            $currentUser = $this->auth->getCurrentUser();
            $this->auth->logActivity($userId, 'account_unsuspended', "Account unsuspended by admin {$currentUser['username']}");
            
            logMessage('INFO', "User $userId unsuspended by admin {$currentUser['id']}");
            
            return ['message' => 'User unsuspended successfully'];
            
        } catch (Exception $e) {
            logMessage('ERROR', "Error unsuspending user $userId: " . $e->getMessage());
            respondError('Failed to unsuspend user', 500);
        }
    }
    
    public function getSystemHealth($params = []) {
        $this->auth->requireAdmin();
        
        try {
            // Get system metrics (simplified for PHP)
            $cpuPercent = 23.0; // Placeholder - would need system monitoring
            $memoryPercent = 67.0; // Placeholder
            $diskPercent = 12.0; // Placeholder
            
            // Database health check
            $startTime = microtime(true);
            $this->db->fetch('SELECT 1');
            $dbResponseTime = (microtime(true) - $startTime) * 1000; // Convert to milliseconds
            
            // Calculate database size
            $dbPath = ROOT_PATH . '/web_database.db';
            $dbSize = file_exists($dbPath) ? filesize($dbPath) / (1024 * 1024) : 0; // MB
            
            // Determine overall system status
            $status = 'healthy';
            if ($cpuPercent > 85 || $memoryPercent > 90 || $dbResponseTime > 1000) {
                $status = 'error';
            } elseif ($cpuPercent > 70 || $memoryPercent > 80 || $dbResponseTime > 500) {
                $status = 'warning';
            }
            
            return [
                'cpu' => round($cpuPercent, 1),
                'memory' => round($memoryPercent, 1),
                'disk' => round($diskPercent, 1),
                'database_size_mb' => round($dbSize, 2),
                'database_response_ms' => round($dbResponseTime, 1),
                'api_response' => round($dbResponseTime, 1),
                'status' => $status,
                'has_psutil' => false
            ];
            
        } catch (Exception $e) {
            logMessage('ERROR', 'Error getting system health: ' . $e->getMessage());
            return [
                'cpu' => 0,
                'memory' => 0,
                'disk' => 0,
                'database_size_mb' => 0,
                'database_response_ms' => 0,
                'api_response' => 0,
                'status' => 'error',
                'has_psutil' => false
            ];
        }
    }
    
    public function getChartData($params = []) {
        $this->auth->requireAdmin();
        
        $metric = $_GET['metric'] ?? 'users';
        
        switch ($metric) {
            case 'users':
                // Get user registration data for last 7 days
                $data = $this->db->fetchAll(
                    'SELECT 
                        date(created_at) as date,
                        COUNT(*) as count
                    FROM users 
                    WHERE created_at >= date("now", "-7 days")
                    GROUP BY date(created_at)
                    ORDER BY date'
                );
                break;
                
            case 'bets':
                // Get betting data for last 7 days
                $data = $this->db->fetchAll(
                    'SELECT 
                        date(placed_at) as date,
                        COUNT(*) as count
                    FROM bets 
                    WHERE placed_at >= date("now", "-7 days")
                    GROUP BY date(placed_at)
                    ORDER BY date'
                );
                break;
                
            case 'revenue':
                // Placeholder revenue data
                $data = [];
                break;
                
            default:
                $data = [];
        }
        
        $labels = [];
        $values = [];
        
        foreach ($data as $row) {
            $labels[] = $row['date'];
            $values[] = (int)$row['count'];
        }
        
        return [
            'labels' => $labels,
            'values' => $values
        ];
    }
    
    public function getRealtime($params = []) {
        $this->auth->requireAdmin();
        
        // Get current active users (simplified)
        $activeUsers = $this->db->fetch(
            'SELECT COUNT(DISTINCT user_id) as count FROM user_sessions WHERE last_activity >= datetime("now", "-1 hour")'
        );
        
        // System health (simplified)
        $systemHealth = [
            'cpu' => 23,
            'memory' => 67,
            'disk' => 12,
            'api_response' => 156
        ];
        
        return [
            'active_users' => (int)($activeUsers['count'] ?? 0),
            'system_health' => $systemHealth
        ];
    }
    
    public function getAllUsers($params = []) {
        $this->auth->requireAdmin();
        
        $page = (int)($_GET['page'] ?? 1);
        $perPage = (int)($_GET['per_page'] ?? 50);
        $statusFilter = $_GET['status'] ?? '';
        $searchTerm = $_GET['search'] ?? '';
        
        $whereClauses = [];
        $queryParams = [];
        
        if ($statusFilter) {
            $whereClauses[] = 'status = ?';
            $queryParams[] = $statusFilter;
        }
        
        if ($searchTerm) {
            $whereClauses[] = '(username LIKE ? OR email LIKE ? OR first_name LIKE ? OR last_name LIKE ?)';
            $searchPattern = "%$searchTerm%";
            $queryParams = array_merge($queryParams, [$searchPattern, $searchPattern, $searchPattern, $searchPattern]);
        }
        
        $whereClause = !empty($whereClauses) ? 'WHERE ' . implode(' AND ', $whereClauses) : '';
        $offset = ($page - 1) * $perPage;
        $queryParams = array_merge($queryParams, [$perPage, $offset]);
        
        $users = $this->db->fetchAll(
            "SELECT id, first_name, last_name, username, email, status, subscription_type, created_at, last_login, email_verified FROM users $whereClause ORDER BY created_at DESC LIMIT ? OFFSET ?",
            $queryParams
        );
        
        return $users;
    }
    
    public function getModelPerformance($params = []) {
        $this->auth->requireAdmin();
        
        $models = $this->db->fetchAll(
            'SELECT 
                model_name as name,
                AVG(accuracy) as accuracy,
                SUM(total_predictions) as total_predictions,
                SUM(correct_predictions) as correct_predictions,
                AVG(roi) as roi
            FROM model_performance 
            WHERE date_from >= date("now", "-30 days")
            GROUP BY model_name
            ORDER BY accuracy DESC'
        );
        
        $modelData = [];
        foreach ($models as $model) {
            $modelData[] = [
                'name' => $model['name'],
                'accuracy' => round((float)($model['accuracy'] ?? 0), 1),
                'total_predictions' => (int)($model['total_predictions'] ?? 0),
                'roi' => round((float)($model['roi'] ?? 0), 2)
            ];
        }
        
        return $modelData;
    }
    
    public function getDetailedActivity($params = []) {
        $this->auth->requireAdmin();
        
        $activityType = $_GET['type'] ?? '';
        $limit = (int)($_GET['limit'] ?? 100);
        
        $whereClause = $activityType ? 'WHERE ua.activity_type = ?' : '';
        $queryParams = $activityType ? [$activityType] : [];
        $queryParams[] = $limit;
        
        $activities = $this->db->fetchAll(
            "SELECT 
                ua.activity_type, ua.description, ua.created_at, ua.ip_address,
                u.username, u.first_name, u.last_name
            FROM user_activity ua
            LEFT JOIN users u ON ua.user_id = u.id
            $whereClause
            ORDER BY ua.created_at DESC 
            LIMIT ?",
            $queryParams
        );
        
        return $activities;
    }
    
    public function getBettingAnalytics($params = []) {
        $this->auth->requireAdmin();
        
        $analytics = $this->db->fetch(
            'SELECT 
                SUM(stake) as total_volume,
                AVG(stake) as avg_stake,
                COUNT(*) as total_bets,
                AVG(CASE WHEN status IN ("won", "lost") THEN 
                    CASE WHEN status = "won" THEN 1.0 ELSE 0.0 END 
                END) * 100 as win_rate,
                SUM(CASE WHEN status = "won" THEN actual_payout - stake 
                         WHEN status = "lost" THEN -stake 
                         ELSE 0 END) as total_profit,
                SUM(stake) as total_stakes
            FROM bets'
        );
        
        $profitMargin = 0;
        $totalStakes = (float)($analytics['total_stakes'] ?? 0);
        if ($totalStakes > 0) {
            $profitMargin = ((float)($analytics['total_profit'] ?? 0) / $totalStakes) * 100;
        }
        
        return [
            'total_volume' => (float)($analytics['total_volume'] ?? 0),
            'avg_stake' => (float)($analytics['avg_stake'] ?? 0),
            'win_rate' => round((float)($analytics['win_rate'] ?? 0), 1),
            'profit_margin' => round($profitMargin, 1),
            'total_bets' => (int)($analytics['total_bets'] ?? 0)
        ];
    }
    
    public function getRecentBets($params = []) {
        $this->auth->requireAdmin();
        
        $limit = (int)($_GET['limit'] ?? 50);
        
        $bets = $this->db->fetchAll(
            'SELECT 
                b.id, b.bet_type, b.stake, b.odds, b.status, b.actual_payout, b.placed_at,
                u.username, u.first_name, u.last_name
            FROM bets b
            JOIN users u ON b.user_id = u.id
            ORDER BY b.placed_at DESC
            LIMIT ?',
            [$limit]
        );
        
        // Convert numeric fields
        foreach ($bets as &$bet) {
            $bet['stake'] = (float)$bet['stake'];
            $bet['odds'] = (float)$bet['odds'];
            $bet['actual_payout'] = (float)$bet['actual_payout'];
        }
        
        return $bets;
    }
    
    public function getSettings($params = []) {
        $this->auth->requireAdmin();
        
        // Get current settings
        $settings = $this->db->fetchAll('SELECT setting_key, setting_value FROM system_settings');
        $settingsDict = [];
        
        foreach ($settings as $setting) {
            $settingsDict[$setting['setting_key']] = $setting['setting_value'];
        }
        
        // Add default values for missing settings
        $defaults = [
            'maxLoginAttempts' => '5',
            'sessionTimeout' => '1440',
            'passwordMinLength' => '8',
            'defaultBankroll' => '1000.00',
            'minBetAmount' => '1.00',
            'maxBetAmount' => '1000.00',
            'modelUpdateFreq' => '24',
            'confidenceThreshold' => '60',
            'kellyEnabled' => 'true'
        ];
        
        foreach ($defaults as $key => $defaultValue) {
            if (!isset($settingsDict[$key])) {
                $settingsDict[$key] = $defaultValue;
            }
        }
        
        return $settingsDict;
    }
    
    public function updateSettings($params = []) {
        $this->auth->requireAdmin();
        
        $input = file_get_contents('php://input');
        $data = json_decode($input, true) ?: [];
        
        try {
            foreach ($data as $key => $value) {
                // Convert boolean to string for storage
                if (is_bool($value)) {
                    $value = $value ? 'true' : 'false';
                }
                
                // Update or insert setting
                $existing = $this->db->fetch('SELECT id FROM system_settings WHERE setting_key = ?', [$key]);
                
                if ($existing) {
                    $this->db->execute(
                        'UPDATE system_settings SET setting_value = ?, updated_at = ? WHERE setting_key = ?',
                        [strval($value), date('Y-m-d H:i:s'), $key]
                    );
                } else {
                    $this->db->execute(
                        'INSERT INTO system_settings (setting_key, setting_value, setting_type) VALUES (?, ?, ?)',
                        [$key, strval($value), 'string']
                    );
                }
            }
            
            $currentUser = $this->auth->getCurrentUser();
            $this->auth->logActivity($currentUser['id'], 'settings_updated', 'Admin updated system settings');
            
            return ['message' => 'Settings updated successfully'];
            
        } catch (Exception $e) {
            logMessage('ERROR', 'Error updating settings: ' . $e->getMessage());
            respondError('Failed to update settings', 500);
        }
    }
    
    public function cleanup($params = []) {
        $this->auth->requireAdmin();
        
        try {
            // Clean up old sessions (older than 7 days)
            $this->db->execute('DELETE FROM user_sessions WHERE expires_at < datetime("now", "-7 days")');
            
            // Clean up old activity logs (older than 180 days)
            $this->db->execute('DELETE FROM user_activity WHERE created_at < datetime("now", "-180 days")');
            
            $currentUser = $this->auth->getCurrentUser();
            $this->auth->logActivity($currentUser['id'], 'data_cleanup', 'Admin performed data cleanup');
            
            logMessage('INFO', "Data cleanup performed by admin {$currentUser['id']}");
            
            return ['message' => 'Old data cleaned up successfully'];
            
        } catch (Exception $e) {
            logMessage('ERROR', 'Error during cleanup: ' . $e->getMessage());
            respondError('Failed to cleanup old data', 500);
        }
    }
    
    public function optimizeDb($params = []) {
        $this->auth->requireAdmin();
        
        try {
            // Run VACUUM to optimize database
            $this->db->getConnection()->exec('VACUUM');
            
            // Analyze tables for better query planning
            $this->db->getConnection()->exec('ANALYZE');
            
            $currentUser = $this->auth->getCurrentUser();
            $this->auth->logActivity($currentUser['id'], 'db_optimized', 'Admin optimized database');
            
            logMessage('INFO', "Database optimized by admin {$currentUser['id']}");
            
            return ['message' => 'Database optimized successfully'];
            
        } catch (Exception $e) {
            logMessage('ERROR', 'Error optimizing database: ' . $e->getMessage());
            respondError('Failed to optimize database', 500);
        }
    }
    
    public function backup($params = []) {
        $this->auth->requireAdmin();
        
        try {
            $backupFilename = 'goonsteen_backup_' . date('Ymd_His') . '.db';
            $dbPath = ROOT_PATH . '/web_database.db';
            
            if (!file_exists($dbPath)) {
                respondError('Database file not found', 404);
            }
            
            $currentUser = $this->auth->getCurrentUser();
            $this->auth->logActivity($currentUser['id'], 'db_backup', 'Admin created database backup');
            
            logMessage('INFO', "Database backup created by admin {$currentUser['id']}");
            
            // Send file as download
            header('Content-Type: application/octet-stream');
            header('Content-Disposition: attachment; filename="' . $backupFilename . '"');
            header('Content-Length: ' . filesize($dbPath));
            readfile($dbPath);
            exit();
            
        } catch (Exception $e) {
            logMessage('ERROR', 'Error creating backup: ' . $e->getMessage());
            respondError('Failed to create backup', 500);
        }
    }
    
    public function broadcast($params = []) {
        $this->auth->requireAdmin();
        
        $input = file_get_contents('php://input');
        $data = json_decode($input, true) ?: [];
        
        $title = $data['title'] ?? '';
        $content = $data['content'] ?? '';
        $messageType = $data['type'] ?? 'info';
        
        if (!$title || !$content) {
            respondError('Title and content are required');
        }
        
        try {
            // Get all active users
            $users = $this->db->fetchAll('SELECT id FROM users WHERE status = "active"');
            
            // Create notification for each user
            foreach ($users as $user) {
                $this->db->execute(
                    'INSERT INTO notifications (user_id, notification_type, title, message, priority) VALUES (?, ?, ?, ?, ?)',
                    [$user['id'], "broadcast_$messageType", $title, $content, 'normal']
                );
            }
            
            $currentUser = $this->auth->getCurrentUser();
            $this->auth->logActivity($currentUser['id'], 'broadcast_sent', "Admin sent broadcast: $title");
            
            logMessage('INFO', "Broadcast message sent by admin {$currentUser['id']}");
            
            return ['message' => 'Broadcast sent to ' . count($users) . ' users'];
            
        } catch (Exception $e) {
            logMessage('ERROR', 'Error sending broadcast: ' . $e->getMessage());
            respondError('Failed to send broadcast', 500);
        }
    }
}
