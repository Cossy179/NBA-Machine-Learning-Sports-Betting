<?php
/**
 * GoonSteen Web Backend - PHP Version
 * NBA Sports Betting Platform Backend for Plesk
 * 
 * This is a complete PHP rewrite of the Python Flask backend
 * Supports all the same endpoints and functionality
 */

// Start output buffering to prevent unwanted output
ob_start();

// Error reporting for development - but don't display errors in JSON responses
error_reporting(E_ALL);
ini_set('display_errors', 0);
ini_set('log_errors', 1);

// Set timezone
date_default_timezone_set('UTC');

// Enable CORS for all requests
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, Authorization, X-Requested-With');
header('Access-Control-Max-Age: 86400');

// Only set JSON content type for API requests
if (isset($_GET['route']) && (strpos($_GET['route'], 'api/') === 0 || strpos($_GET['route'], 'ws/') === 0)) {
    header('Content-Type: application/json; charset=utf-8');
}

// Handle preflight OPTIONS request
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

// Include configuration and classes
try {
    require_once __DIR__ . '/config.php';
    require_once __DIR__ . '/classes/Database.php';
    require_once __DIR__ . '/classes/Auth.php';
    require_once __DIR__ . '/classes/Router.php';
    require_once __DIR__ . '/classes/UserController.php';
    require_once __DIR__ . '/classes/AdminController.php';
    require_once __DIR__ . '/classes/DashboardController.php';
} catch (Error $e) {
    http_response_code(500);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'Failed to load required files: ' . $e->getMessage()]);
    exit();
}

try {
    // Initialize database
    $database = new Database();
    $db = $database->getConnection();
    
    // Initialize authentication
    $auth = new Auth($db);
    
    // Initialize router
    $router = new Router();
    
    // Initialize controllers
    $userController = new UserController($db, $auth);
    $adminController = new AdminController($db, $auth);
    $dashboardController = new DashboardController($db, $auth);
    
    // Define routes
    
    // Health check
    $router->get('/api/health', function() {
        return ['status' => 'healthy', 'message' => 'GoonSteen PHP API is running'];
    });
    
    // Authentication routes
    $router->post('/api/signup', [$userController, 'signup']);
    $router->post('/api/login', [$userController, 'login']);
    $router->post('/api/logout', [$userController, 'logout']);
    $router->get('/api/session', [$userController, 'getSession']);
    $router->get('/api/check-username', [$userController, 'checkUsername']);
    
    // Dashboard routes
    $router->get('/api/dashboard/overview', [$dashboardController, 'getOverview']);
    $router->get('/api/dashboard/games', [$dashboardController, 'getGames']);
    $router->get('/api/dashboard/activity', [$dashboardController, 'getActivity']);
    
    // User routes
    $router->get('/api/user/profile', [$userController, 'getProfile']);
    $router->get('/api/user/bankroll', [$userController, 'getBankroll']);
    $router->post('/api/user/bankroll', [$userController, 'updateBankroll']);
    $router->post('/api/user/track-bet', [$userController, 'trackBet']);
    $router->post('/api/calculate-kelly', [$userController, 'calculateKelly']);
    $router->get('/api/user/analytics', [$userController, 'getAnalytics']);
    $router->get('/api/user/betting-history', [$userController, 'getBettingHistory']);
    
    // Notification routes
    $router->get('/api/notifications', [$userController, 'getNotifications']);
    $router->post('/api/notifications/{id}/read', [$userController, 'markNotificationRead']);
    
    // Admin routes
    $router->get('/api/admin/overview', [$adminController, 'getOverview']);
    $router->get('/api/admin/recent-users', [$adminController, 'getRecentUsers']);
    $router->get('/api/admin/activity', [$adminController, 'getActivity']);
    $router->get('/api/admin/users/{id}', [$adminController, 'getUser']);
    $router->put('/api/admin/users/{id}', [$adminController, 'updateUser']);
    $router->post('/api/admin/users/{id}/suspend', [$adminController, 'suspendUser']);
    $router->post('/api/admin/users/{id}/unsuspend', [$adminController, 'unsuspendUser']);
    $router->get('/api/admin/system-health', [$adminController, 'getSystemHealth']);
    $router->get('/api/admin/chart-data', [$adminController, 'getChartData']);
    $router->get('/api/admin/realtime', [$adminController, 'getRealtime']);
    $router->get('/api/admin/all-users', [$adminController, 'getAllUsers']);
    $router->get('/api/admin/model-performance', [$adminController, 'getModelPerformance']);
    $router->get('/api/admin/detailed-activity', [$adminController, 'getDetailedActivity']);
    $router->get('/api/admin/betting-analytics', [$adminController, 'getBettingAnalytics']);
    $router->get('/api/admin/recent-bets', [$adminController, 'getRecentBets']);
    $router->get('/api/admin/settings', [$adminController, 'getSettings']);
    $router->post('/api/admin/settings', [$adminController, 'updateSettings']);
    $router->post('/api/admin/cleanup', [$adminController, 'cleanup']);
    $router->post('/api/admin/optimize-db', [$adminController, 'optimizeDb']);
    $router->get('/api/admin/backup', [$adminController, 'backup']);
    $router->post('/api/admin/broadcast', [$adminController, 'broadcast']);
    
    // WebSocket stub endpoints
    $router->get('/ws/dashboard', function() {
        return ['message' => 'WebSocket not implemented'];
    });
    
    $router->get('/ws/admin', function() {
        return ['message' => 'WebSocket not implemented'];
    });
    
    // Get the request path
    $requestPath = $_GET['route'] ?? '';
    
    // If no route specified or empty, serve the main page
    if (empty($requestPath) || $requestPath === '/') {
        if (file_exists(__DIR__ . '/index.html')) {
            header('Content-Type: text/html');
            readfile(__DIR__ . '/index.html');
            exit();
        }
    }
    
    // Handle static files
    if (!empty($requestPath) && strpos($requestPath, 'api/') !== 0 && strpos($requestPath, 'ws/') !== 0) {
        $filePath = __DIR__ . '/' . $requestPath;
        if (file_exists($filePath) && !is_dir($filePath)) {
            // Serve the static file with appropriate content type
            $extension = pathinfo($filePath, PATHINFO_EXTENSION);
            switch ($extension) {
                case 'html':
                    header('Content-Type: text/html');
                    break;
                case 'css':
                    header('Content-Type: text/css');
                    break;
                case 'js':
                    header('Content-Type: application/javascript');
                    break;
                case 'json':
                    header('Content-Type: application/json');
                    break;
                default:
                    header('Content-Type: application/octet-stream');
            }
            readfile($filePath);
            exit();
        }
    }
    
    // Handle the API request
    $router->handleRequest();
    
} catch (Exception $e) {
    error_log("GoonSteen API Error: " . $e->getMessage());
    
    // Clear any output that might have been sent
    if (ob_get_length()) {
        ob_clean();
    }
    
    http_response_code(500);
    header('Content-Type: application/json');
    echo json_encode([
        'error' => 'Internal server error',
        'message' => APP_DEBUG ? $e->getMessage() : 'Server error'
    ]);
}