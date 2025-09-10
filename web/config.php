<?php
/**
 * Configuration file for GoonSteen PHP Backend
 */

// Database configuration
define('DB_HOST', 'localhost');
define('DB_NAME', 'web_database');
define('DB_USER', 'root');
define('DB_PASS', '');
define('DB_CHARSET', 'utf8mb4');

// Security configuration
define('JWT_SECRET', getenv('JWT_SECRET') ?: bin2hex(random_bytes(32)));
define('JWT_EXPIRATION_HOURS', getenv('JWT_EXPIRATION_HOURS') ?: 24);
define('SECRET_KEY', getenv('SECRET_KEY') ?: bin2hex(random_bytes(32)));

// Application configuration
define('APP_ENV', getenv('APP_ENV') ?: 'production');
define('APP_DEBUG', APP_ENV === 'development');

// Paths
define('ROOT_PATH', __DIR__);
define('WEB_PATH', ROOT_PATH);
define('DATABASE_SCHEMA_PATH', ROOT_PATH . '/database_schema.sql');

// Logging
define('LOG_LEVEL', APP_DEBUG ? 'DEBUG' : 'INFO');
define('LOG_FILE', ROOT_PATH . '/logs/app.log');

// Create logs directory if it doesn't exist
if (!is_dir(ROOT_PATH . '/logs')) {
    mkdir(ROOT_PATH . '/logs', 0755, true);
}

// Error handling
if (APP_DEBUG) {
    error_reporting(E_ALL);
    ini_set('display_errors', 1);
} else {
    error_reporting(E_ERROR | E_PARSE);
    ini_set('display_errors', 0);
    ini_set('log_errors', 1);
    ini_set('error_log', LOG_FILE);
}

// Timezone
date_default_timezone_set('UTC');

// Session configuration
ini_set('session.cookie_httponly', 1);
ini_set('session.cookie_secure', isset($_SERVER['HTTPS']));
ini_set('session.use_strict_mode', 1);

// Helper functions
function logMessage($level, $message, $context = []) {
    $timestamp = date('Y-m-d H:i:s');
    $contextStr = !empty($context) ? json_encode($context) : '';
    $logEntry = "[$timestamp] [$level] $message $contextStr" . PHP_EOL;
    file_put_contents(LOG_FILE, $logEntry, FILE_APPEND | LOCK_EX);
}

function respondJson($data, $statusCode = 200) {
    http_response_code($statusCode);
    header('Content-Type: application/json');
    echo json_encode($data);
    exit();
}

function respondError($message, $statusCode = 400) {
    respondJson(['error' => $message], $statusCode);
}

function getCurrentUserId() {
    global $auth;
    if ($auth && $auth->getCurrentUser()) {
        return $auth->getCurrentUser()['id'];
    }
    return null;
}

function requireAuth() {
    global $auth;
    if (!$auth || !$auth->getCurrentUser()) {
        respondError('Authentication required', 401);
    }
}

function requireAdmin() {
    requireAuth();
    global $auth;
    $user = $auth->getCurrentUser();
    if (!$user['is_admin']) {
        respondError('Admin privileges required', 403);
    }
}