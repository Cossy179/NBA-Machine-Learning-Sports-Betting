<?php
/**
 * Authentication class for JWT token management and user authentication
 */

class Auth {
    private $db;
    private $currentUser = null;
    
    public function __construct($database) {
        $this->db = $database;
        $this->loadCurrentUser();
    }
    
    private function loadCurrentUser() {
        $token = $this->getTokenFromRequest();
        if ($token) {
            $payload = $this->verifyJwtToken($token);
            if ($payload && isset($payload['user_id'])) {
                $user = $this->db->fetch(
                    'SELECT * FROM users WHERE id = ? AND status = ?',
                    [$payload['user_id'], 'active']
                );
                if ($user) {
                    $this->currentUser = $user;
                }
            }
        }
    }
    
    private function getTokenFromRequest() {
        // Try to get Authorization header in multiple ways
        $authHeader = null;
        
        // Method 1: Direct server variable
        if (isset($_SERVER['HTTP_AUTHORIZATION'])) {
            $authHeader = $_SERVER['HTTP_AUTHORIZATION'];
        }
        
        // Method 2: getallheaders() if available
        if (!$authHeader && function_exists('getallheaders')) {
            $headers = getallheaders();
            if (isset($headers['Authorization'])) {
                $authHeader = $headers['Authorization'];
            } else {
                // Fallback for case-insensitive headers
                foreach ($headers as $key => $value) {
                    if (strtolower($key) === 'authorization') {
                        $authHeader = $value;
                        break;
                    }
                }
            }
        }
        
        // Method 3: Check all possible server variables
        if (!$authHeader) {
            $possibleKeys = ['HTTP_AUTHORIZATION', 'REDIRECT_HTTP_AUTHORIZATION'];
            foreach ($possibleKeys as $key) {
                if (isset($_SERVER[$key])) {
                    $authHeader = $_SERVER[$key];
                    break;
                }
            }
        }
        
        if ($authHeader) {
            return str_replace('Bearer ', '', $authHeader);
        }
        
        return null;
    }
    
    public function getCurrentUser() {
        return $this->currentUser;
    }
    
    public function hashPassword($password, $salt = null) {
        if ($salt === null) {
            $salt = bin2hex(random_bytes(16));
        }
        
        $passwordHash = hash_pbkdf2('sha256', $password, $salt, 100000, 0, true);
        return [$passwordHash, $salt];
    }
    
    public function verifyPassword($password, $passwordHash, $salt) {
        list($computedHash, ) = $this->hashPassword($password, $salt);
        return hash_equals($passwordHash, $computedHash);
    }
    
    public function createJwtToken($userId) {
        $header = json_encode(['typ' => 'JWT', 'alg' => 'HS256']);
        $payload = json_encode([
            'user_id' => $userId,
            'exp' => time() + (JWT_EXPIRATION_HOURS * 3600),
            'iat' => time()
        ]);
        
        $base64Header = str_replace(['+', '/', '='], ['-', '_', ''], base64_encode($header));
        $base64Payload = str_replace(['+', '/', '='], ['-', '_', ''], base64_encode($payload));
        
        $signature = hash_hmac('sha256', $base64Header . "." . $base64Payload, JWT_SECRET, true);
        $base64Signature = str_replace(['+', '/', '='], ['-', '_', ''], base64_encode($signature));
        
        return $base64Header . "." . $base64Payload . "." . $base64Signature;
    }
    
    public function verifyJwtToken($token) {
        $tokenParts = explode('.', $token);
        if (count($tokenParts) !== 3) {
            return null;
        }
        
        $header = base64_decode(str_replace(['-', '_'], ['+', '/'], $tokenParts[0]));
        $payload = base64_decode(str_replace(['-', '_'], ['+', '/'], $tokenParts[1]));
        $signatureProvided = $tokenParts[2];
        
        $base64Header = str_replace(['+', '/', '='], ['-', '_', ''], base64_encode($header));
        $base64Payload = str_replace(['+', '/', '='], ['-', '_', ''], base64_encode($payload));
        
        $signature = hash_hmac('sha256', $base64Header . "." . $base64Payload, JWT_SECRET, true);
        $base64Signature = str_replace(['+', '/', '='], ['-', '_', ''], base64_encode($signature));
        
        if (!hash_equals($base64Signature, $signatureProvided)) {
            return null;
        }
        
        $payloadData = json_decode($payload, true);
        if (!$payloadData || $payloadData['exp'] < time()) {
            return null;
        }
        
        return $payloadData;
    }
    
    public function generateSessionToken() {
        return bin2hex(random_bytes(32));
    }
    
    public function logActivity($userId, $activityType, $description, $metadata = null) {
        $ipAddress = $_SERVER['REMOTE_ADDR'] ?? null;
        $userAgent = $_SERVER['HTTP_USER_AGENT'] ?? null;
        
        $this->db->execute(
            'INSERT INTO user_activity (user_id, activity_type, description, ip_address, user_agent, metadata) VALUES (?, ?, ?, ?, ?, ?)',
            [$userId, $activityType, $description, $ipAddress, $userAgent, $metadata ? json_encode($metadata) : null]
        );
    }
    
    public function validateEmail($email) {
        return filter_var($email, FILTER_VALIDATE_EMAIL) !== false;
    }
    
    public function validateUsername($username) {
        return preg_match('/^[a-zA-Z0-9_]{3,20}$/', $username);
    }
    
    public function parseBirthDate($value) {
        if (empty($value)) {
            return null;
        }
        
        $value = trim($value);
        $formats = ['Y-m-d', 'd/m/Y', 'm/d/Y'];
        
        foreach ($formats as $format) {
            $date = DateTime::createFromFormat($format, $value);
            if ($date !== false) {
                return $date->format('Y-m-d');
            }
        }
        
        // Accept year-only input
        if (strlen($value) === 4 && is_numeric($value)) {
            $year = intval($value);
            if ($year >= 1900 && $year <= date('Y')) {
                return $year . '-01-01';
            }
        }
        
        return null;
    }
    
    public function calculateAge($birthDate) {
        $birth = new DateTime($birthDate);
        $today = new DateTime();
        return $today->diff($birth)->y;
    }
    
    public function isAccountLocked($user) {
        if (!$user['locked_until']) {
            return false;
        }
        
        $lockTime = new DateTime($user['locked_until']);
        $now = new DateTime();
        
        return $now < $lockTime;
    }
    
    public function incrementLoginAttempts($userId) {
        $user = $this->db->fetch('SELECT login_attempts FROM users WHERE id = ?', [$userId]);
        $attempts = ($user['login_attempts'] ?? 0) + 1;
        
        $lockedUntil = null;
        if ($attempts >= 5) {
            $lockTime = new DateTime();
            $lockTime->add(new DateInterval('PT30M')); // 30 minutes
            $lockedUntil = $lockTime->format('Y-m-d H:i:s');
        }
        
        $this->db->execute(
            'UPDATE users SET login_attempts = ?, locked_until = ? WHERE id = ?',
            [$attempts, $lockedUntil, $userId]
        );
        
        return $attempts;
    }
    
    public function resetLoginAttempts($userId) {
        $this->db->execute(
            'UPDATE users SET login_attempts = 0, locked_until = NULL, last_login = ? WHERE id = ?',
            [date('Y-m-d H:i:s'), $userId]
        );
    }
    
    public function createSession($userId, $rememberMe = false) {
        $sessionToken = $this->generateSessionToken();
        $expiresAt = new DateTime();
        $expiresAt->add(new DateInterval($rememberMe ? 'P1D' : 'PT8H')); // 24h or 8h
        
        $this->db->execute(
            'INSERT INTO user_sessions (user_id, session_token, ip_address, user_agent, expires_at) VALUES (?, ?, ?, ?, ?)',
            [
                $userId,
                $sessionToken,
                $_SERVER['REMOTE_ADDR'] ?? null,
                $_SERVER['HTTP_USER_AGENT'] ?? null,
                $expiresAt->format('Y-m-d H:i:s')
            ]
        );
        
        return $sessionToken;
    }
    
    public function requireAuth() {
        if (!$this->currentUser) {
            respondError('Authentication required', 401);
        }
    }
    
    public function requireAdmin() {
        $this->requireAuth();
        if (!$this->currentUser['is_admin']) {
            respondError('Admin privileges required', 403);
        }
    }
}
