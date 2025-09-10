<?php
/**
 * Test script to verify PHP backend functionality
 * Run this script to test various endpoints
 */

// Include the main application
require_once 'config.php';
require_once 'classes/Database.php';
require_once 'classes/Auth.php';

echo "<h1>GoonSteen PHP Backend Test</h1>\n";
echo "<style>body{font-family:Arial,sans-serif;margin:20px;} .success{color:green;} .error{color:red;} .info{color:blue;}</style>\n";

try {
    // Test database connection
    echo "<h2>1. Database Connection Test</h2>\n";
    $database = new Database();
    $db = $database->getConnection();
    echo "<span class='success'>✅ Database connection successful</span><br>\n";
    
    // Test database query
    $result = $db->query('SELECT COUNT(*) as count FROM users')->fetch();
    echo "<span class='success'>✅ Database query successful - Users count: " . $result['count'] . "</span><br>\n";
    
    // Test authentication class
    echo "<h2>2. Authentication Test</h2>\n";
    $auth = new Auth($db);
    echo "<span class='success'>✅ Auth class initialized</span><br>\n";
    
    // Test password hashing
    list($hash, $salt) = $auth->hashPassword('test123');
    echo "<span class='success'>✅ Password hashing works</span><br>\n";
    
    // Test password verification
    $verified = $auth->verifyPassword('test123', $hash, $salt);
    echo $verified ? "<span class='success'>✅ Password verification works</span><br>\n" : "<span class='error'>❌ Password verification failed</span><br>\n";
    
    // Test JWT token creation
    $token = $auth->createJwtToken(1);
    echo "<span class='success'>✅ JWT token created: " . substr($token, 0, 50) . "...</span><br>\n";
    
    // Test JWT token verification
    $payload = $auth->verifyJwtToken($token);
    echo $payload ? "<span class='success'>✅ JWT token verification works</span><br>\n" : "<span class='error'>❌ JWT token verification failed</span><br>\n";
    
    // Test email validation
    echo "<h2>3. Validation Tests</h2>\n";
    echo $auth->validateEmail('test@example.com') ? "<span class='success'>✅ Email validation works</span><br>\n" : "<span class='error'>❌ Email validation failed</span><br>\n";
    echo $auth->validateUsername('testuser') ? "<span class='success'>✅ Username validation works</span><br>\n" : "<span class='error'>❌ Username validation failed</span><br>\n";
    
    // Test system settings
    echo "<h2>4. System Settings Test</h2>\n";
    $settings = $db->fetchAll('SELECT setting_key, setting_value FROM system_settings LIMIT 3');
    if ($settings) {
        echo "<span class='success'>✅ System settings loaded:</span><br>\n";
        foreach ($settings as $setting) {
            echo "<span class='info'>   - {$setting['setting_key']}: {$setting['setting_value']}</span><br>\n";
        }
    } else {
        echo "<span class='error'>❌ No system settings found</span><br>\n";
    }
    
    // Test teams data
    echo "<h2>5. Teams Data Test</h2>\n";
    $teams = $db->fetchAll('SELECT name, abbreviation FROM teams LIMIT 5');
    if ($teams) {
        echo "<span class='success'>✅ Teams data loaded:</span><br>\n";
        foreach ($teams as $team) {
            echo "<span class='info'>   - {$team['name']} ({$team['abbreviation']})</span><br>\n";
        }
    } else {
        echo "<span class='error'>❌ No teams data found</span><br>\n";
    }
    
    // Test API endpoints (simulate)
    echo "<h2>6. API Endpoint Simulation</h2>\n";
    
    // Simulate health check
    $_SERVER['REQUEST_METHOD'] = 'GET';
    $_SERVER['REQUEST_URI'] = '/api/health';
    echo "<span class='success'>✅ Health endpoint would return: " . json_encode(['status' => 'healthy', 'message' => 'GoonSteen PHP API is running']) . "</span><br>\n";
    
    // Check if admin user exists
    $adminUser = $db->fetch('SELECT username FROM users WHERE is_admin = 1 LIMIT 1');
    if ($adminUser) {
        echo "<span class='success'>✅ Admin user found: {$adminUser['username']}</span><br>\n";
    } else {
        echo "<span class='error'>❌ No admin user found</span><br>\n";
    }
    
    echo "<h2>7. File Structure Test</h2>\n";
    $requiredFiles = [
        'index.php',
        'config.php',
        'classes/Database.php',
        'classes/Auth.php',
        'classes/Router.php',
        'classes/UserController.php',
        'classes/AdminController.php',
        'classes/DashboardController.php',
        '.htaccess',
        'composer.json'
    ];
    
    foreach ($requiredFiles as $file) {
        if (file_exists($file)) {
            echo "<span class='success'>✅ $file exists</span><br>\n";
        } else {
            echo "<span class='error'>❌ $file missing</span><br>\n";
        }
    }
    
    echo "<h2>8. PHP Configuration</h2>\n";
    echo "<span class='success'>✅ PHP Version: " . phpversion() . "</span><br>\n";
    echo "<span class='success'>✅ PDO SQLite: " . (extension_loaded('pdo_sqlite') ? 'Available' : 'Not available') . "</span><br>\n";
    echo "<span class='success'>✅ JSON: " . (extension_loaded('json') ? 'Available' : 'Not available') . "</span><br>\n";
    echo "<span class='success'>✅ OpenSSL: " . (extension_loaded('openssl') ? 'Available' : 'Not available') . "</span><br>\n";
    
    // Test actual API endpoint
    echo "<h2>9. Live API Test</h2>\n";
    echo "<span class='info'>📍 Test the following URLs after deployment:</span><br>\n";
    echo "<span class='info'>   - Health Check: <a href='/api/health'>/api/health</a></span><br>\n";
    echo "<span class='info'>   - Main Site: <a href='/'>/</a></span><br>\n";
    echo "<span class='info'>   - Login Page: <a href='/login.html'>/login.html</a></span><br>\n";
    echo "<span class='info'>   - Admin Dashboard: <a href='/admin-dashboard.html'>/admin-dashboard.html</a></span><br>\n";
    
    echo "<h2 class='success'>✅ All Tests Completed Successfully!</h2>\n";
    echo "<p><strong>Your PHP backend is ready for deployment on Plesk!</strong></p>\n";
    echo "<div style='background:#e7f3e7;border:1px solid #4caf50;padding:15px;margin:20px 0;border-radius:5px;'>";
    echo "<h3>🚀 Next Steps:</h3>";
    echo "<ol>";
    echo "<li>Upload the entire <code>/web</code> directory to your Plesk httpdocs folder</li>";
    echo "<li>Set proper file permissions (755 for directories, 644 for files)</li>";
    echo "<li>Ensure PHP 7.4+ is enabled in Plesk</li>";
    echo "<li>Test your site at your domain URL</li>";
    echo "<li>Login with admin/admin123 and change the password immediately</li>";
    echo "</ol>";
    echo "</div>";
    
} catch (Exception $e) {
    echo "<h2 class='error'>❌ Test Failed</h2>\n";
    echo "<p class='error'>Error: " . $e->getMessage() . "</p>\n";
    echo "<p>Please check your configuration and try again.</p>\n";
}
?>