<?php
/**
 * Simple router class for handling HTTP requests
 */

class Router {
    private $routes = [];
    
    public function get($path, $handler) {
        $this->addRoute('GET', $path, $handler);
    }
    
    public function post($path, $handler) {
        $this->addRoute('POST', $path, $handler);
    }
    
    public function put($path, $handler) {
        $this->addRoute('PUT', $path, $handler);
    }
    
    public function delete($path, $handler) {
        $this->addRoute('DELETE', $path, $handler);
    }
    
    private function addRoute($method, $path, $handler) {
        $this->routes[] = [
            'method' => $method,
            'path' => $path,
            'handler' => $handler
        ];
    }
    
    public function handleRequest() {
        $method = $_SERVER['REQUEST_METHOD'];
        $path = $this->getCurrentPath();
        
        foreach ($this->routes as $route) {
            if ($route['method'] === $method && $this->matchPath($route['path'], $path, $params)) {
                try {
                    $result = $this->callHandler($route['handler'], $params);
                    if ($result !== null) {
                        respondJson($result);
                    }
                    return;
                } catch (Exception $e) {
                    logMessage('ERROR', 'Route handler error: ' . $e->getMessage(), [
                        'path' => $path,
                        'method' => $method
                    ]);
                    respondError('Internal server error', 500);
                }
            }
        }
        
        // No route found
        http_response_code(404);
        echo json_encode(['error' => 'Not found']);
    }
    
    private function getCurrentPath() {
        // Handle both direct access and with route parameter
        if (isset($_GET['route'])) {
            $path = '/' . ltrim($_GET['route'], '/');
        } else {
            $path = $_SERVER['REQUEST_URI'] ?? '/';
            
            // Remove query string
            if (($pos = strpos($path, '?')) !== false) {
                $path = substr($path, 0, $pos);
            }
        }
        
        return $path;
    }
    
    private function matchPath($routePath, $requestPath, &$params) {
        $params = [];
        
        // Convert route path to regex pattern
        $pattern = preg_replace('/\{([^}]+)\}/', '([^/]+)', $routePath);
        $pattern = '#^' . $pattern . '$#';
        
        if (preg_match($pattern, $requestPath, $matches)) {
            // Extract parameter names
            preg_match_all('/\{([^}]+)\}/', $routePath, $paramNames);
            
            // Map parameter values
            for ($i = 1; $i < count($matches); $i++) {
                if (isset($paramNames[1][$i - 1])) {
                    $params[$paramNames[1][$i - 1]] = $matches[$i];
                }
            }
            
            return true;
        }
        
        return false;
    }
    
    private function callHandler($handler, $params) {
        if (is_callable($handler)) {
            return call_user_func($handler, $params);
        }
        
        if (is_array($handler) && count($handler) === 2) {
            list($controller, $method) = $handler;
            
            if (is_object($controller) && method_exists($controller, $method)) {
                return call_user_func([$controller, $method], $params);
            }
        }
        
        throw new Exception('Invalid route handler');
    }
    
    public function getJsonInput() {
        $input = file_get_contents('php://input');
        return json_decode($input, true) ?: [];
    }
}
