<?php
/**
 * Dashboard controller for handling dashboard-related API endpoints
 */

class DashboardController {
    private $db;
    private $auth;
    
    public function __construct($database, $auth) {
        $this->db = $database;
        $this->auth = $auth;
    }
    
    public function getOverview($params = []) {
        $this->auth->requireAuth();
        $user = $this->auth->getCurrentUser();
        
        // Get bankroll info
        $bankroll = $this->db->fetch('SELECT * FROM bankrolls WHERE user_id = ?', [$user['id']]);
        
        // Get betting stats
        $betStats = $this->db->fetch(
            'SELECT 
                COUNT(*) as total_bets,
                COUNT(CASE WHEN status = "pending" THEN 1 END) as active_bets,
                COUNT(CASE WHEN status = "won" THEN 1 END) as won_bets,
                COUNT(CASE WHEN status = "lost" THEN 1 END) as lost_bets,
                SUM(CASE WHEN status = "won" THEN actual_payout - stake ELSE 0 END) as total_profit,
                COUNT(CASE WHEN placed_at >= date("now", "-7 days") AND status = "won" THEN 1 END) as week_wins,
                COUNT(CASE WHEN placed_at >= date("now", "-7 days") AND status = "lost" THEN 1 END) as week_losses
            FROM bets 
            WHERE user_id = ?',
            [$user['id']]
        );
        
        return [
            'bankroll' => $bankroll ? (float)$bankroll['total_balance'] : 1000.00,
            'bankroll_change' => 12.5, // Calculate actual change
            'active_bets' => (int)($betStats['active_bets'] ?? 0),
            'pending_results' => 3, // Calculate actual pending
            'week_wins' => (int)($betStats['week_wins'] ?? 0),
            'week_losses' => (int)($betStats['week_losses'] ?? 0),
            'profit_loss' => (float)($betStats['total_profit'] ?? 0.00),
            'roi' => 15.3 // Calculate actual ROI
        ];
    }
    
    public function getGames($params = []) {
        $this->auth->requireAuth();
        
        $games = $this->db->fetchAll(
            'SELECT 
                g.id, g.game_date, g.game_time, g.status,
                ht.name as home_team_name, ht.abbreviation as home_team_abbr,
                at.name as away_team_name, at.abbreviation as away_team_abbr,
                g.home_score, g.away_score,
                p.confidence, p.predicted_winner, p.predicted_home_score, p.predicted_away_score
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.id
            JOIN teams at ON g.away_team_id = at.id
            LEFT JOIN predictions p ON g.id = p.game_id AND p.model_name = "Ensemble_NBA_v1"
            WHERE g.game_date = date("now")
            ORDER BY g.game_time'
        );
        
        $gameList = [];
        foreach ($games as $game) {
            $gameData = [
                'id' => $game['id'],
                'start_time' => $game['game_date'] . ' ' . $game['game_time'],
                'confidence' => (float)($game['confidence'] ?? 75),
                'home_team' => [
                    'name' => $game['home_team_name'],
                    'abbreviation' => $game['home_team_abbr'],
                    'record' => '25-15', // Get from actual data
                    'odds' => '-110'
                ],
                'away_team' => [
                    'name' => $game['away_team_name'],
                    'abbreviation' => $game['away_team_abbr'],
                    'record' => '22-18', // Get from actual data
                    'odds' => '+105'
                ],
                'prediction' => [
                    'winner' => $game['predicted_winner'] ? $game['home_team_name'] : $game['away_team_name'],
                    'score' => intval($game['predicted_home_score'] ?? 110) . '-' . intval($game['predicted_away_score'] ?? 105),
                    'total' => '220.5',
                    'over_under' => 'Over'
                ]
            ];
            $gameList[] = $gameData;
        }
        
        return $gameList;
    }
    
    public function getActivity($params = []) {
        $this->auth->requireAuth();
        $user = $this->auth->getCurrentUser();
        
        // Get real user activity from bets and user_activity tables
        $activities = $this->db->fetchAll(
            'SELECT 
                "bet" as type,
                CASE 
                    WHEN b.status = "won" THEN "Bet Won"
                    WHEN b.status = "lost" THEN "Bet Lost"
                    WHEN b.status = "pending" THEN "Bet Pending"
                    ELSE "Bet Placed"
                END as title,
                CASE 
                    WHEN json_extract(b.bet_details, "$.bet_type") IS NOT NULL 
                    THEN json_extract(b.bet_details, "$.bet_type") || " • " || b.status
                    ELSE b.bet_type || " • " || b.status
                END as description,
                b.placed_at as timestamp,
                CASE 
                    WHEN b.status = "won" THEN b.actual_payout - b.stake
                    WHEN b.status = "lost" THEN -b.stake
                    ELSE b.stake
                END as amount,
                b.status
            FROM bets b
            WHERE b.user_id = ?
            
            UNION ALL
            
            SELECT 
                ua.activity_type as type,
                CASE 
                    WHEN ua.activity_type = "bankroll_updated" THEN "Bankroll Updated"
                    WHEN ua.activity_type = "login_success" THEN "Login"
                    WHEN ua.activity_type = "profile_updated" THEN "Profile Updated"
                    ELSE ua.activity_type
                END as title,
                ua.description,
                ua.created_at as timestamp,
                0 as amount,
                "info" as status
            FROM user_activity ua
            WHERE ua.user_id = ? 
            AND ua.activity_type IN ("bankroll_updated", "profile_updated")
            
            ORDER BY timestamp DESC
            LIMIT 10',
            [$user['id'], $user['id']]
        );
        
        // Convert amount to float
        foreach ($activities as &$activity) {
            $activity['amount'] = (float)$activity['amount'];
        }
        
        return $activities;
    }
}
