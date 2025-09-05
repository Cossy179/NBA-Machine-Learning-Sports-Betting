"""
Configuration manager for NBA prediction system.
Handles environment variables, API keys, and system settings.
"""
import os
from typing import Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import dotenv, fall back gracefully if not available
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    def load_dotenv(path=None):
        pass  # No-op fallback

class ConfigManager:
    def __init__(self, env_file: str = '.env'):
        """Initialize configuration manager"""
        self.env_file = env_file
        self.config = {}
        self.load_config()
        
    def load_config(self):
        """Load configuration from environment variables"""
        # Load .env file if it exists
        if os.path.exists(self.env_file) and HAS_DOTENV:
            load_dotenv(self.env_file)
            print(f"✅ Loaded configuration from {self.env_file}")
        elif os.path.exists(self.env_file):
            print(f"⚠️ .env file found but python-dotenv not installed. Install with: pip install python-dotenv")
        else:
            print(f"⚠️ No {self.env_file} file found, using environment variables only")
        
        # API Keys
        self.config['api_keys'] = {
            'the_odds_api': os.getenv('THE_ODDS_API_KEY'),
            'sportsradar': os.getenv('SPORTSRADAR_API_KEY'),
            'nba_stats': os.getenv('NBA_STATS_API_KEY', 'not_required'),
            'rapidapi': os.getenv('RAPIDAPI_KEY'),
            'news_api': os.getenv('NEWS_API_KEY')
        }
        
        # Database settings
        self.config['database'] = {
            'path': os.getenv('DATABASE_PATH', 'Data/dataset.sqlite'),
            'cache_duration': int(os.getenv('CACHE_DURATION', '300'))
        }
        
        # Betting settings
        self.config['betting'] = {
            'default_bet_size': float(os.getenv('DEFAULT_BET_SIZE', '100')),
            'max_bet_percentage': float(os.getenv('MAX_BET_PERCENTAGE', '0.05')),
            'kelly_fraction': float(os.getenv('KELLY_FRACTION', '0.25'))
        }
        
        # API endpoints
        self.config['endpoints'] = {
            'nba_stats_base': 'https://stats.nba.com/stats',
            'the_odds_api_base': 'https://api.the-odds-api.com/v4',
            'sportsradar_base': 'https://api.sportradar.us/nba/trial/v8/en',
            'rapidapi_base': 'https://api-nba-v1.p.rapidapi.com',
            'news_api_base': 'https://newsapi.org/v2'
        }
        
        # Rate limiting
        self.config['rate_limits'] = {
            'nba_stats': {'requests_per_minute': 20, 'requests_per_hour': 200},
            'the_odds_api': {'requests_per_minute': 10, 'requests_per_day': 500},
            'sportsradar': {'requests_per_minute': 5, 'requests_per_day': 1000},
            'rapidapi': {'requests_per_minute': 10, 'requests_per_day': 100},
            'news_api': {'requests_per_minute': 20, 'requests_per_day': 1000}
        }
        
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service"""
        return self.config['api_keys'].get(service)
    
    def has_api_key(self, service: str) -> bool:
        """Check if API key exists for service"""
        key = self.get_api_key(service)
        return key is not None and key != 'your_api_key_here' and key != ''
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.config['database']
    
    def get_betting_config(self) -> Dict[str, Any]:
        """Get betting configuration"""
        return self.config['betting']
    
    def get_endpoint(self, service: str) -> Optional[str]:
        """Get API endpoint for service"""
        return self.config['endpoints'].get(f'{service}_base')
    
    def get_rate_limit(self, service: str) -> Dict[str, int]:
        """Get rate limit configuration for service"""
        return self.config['rate_limits'].get(service, {'requests_per_minute': 10})
    
    def get_available_services(self) -> Dict[str, bool]:
        """Get list of available services based on API keys"""
        return {
            service: self.has_api_key(service) 
            for service in self.config['api_keys'].keys()
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'available_services': self.get_available_services()
        }
        
        # Check critical services
        critical_services = ['the_odds_api']
        for service in critical_services:
            if not self.has_api_key(service):
                validation['warnings'].append(f"Missing API key for {service} - betting odds will not be available")
        
        # Check optional services
        optional_services = ['sportsradar', 'rapidapi', 'news_api']
        missing_optional = [s for s in optional_services if not self.has_api_key(s)]
        if missing_optional:
            validation['warnings'].append(f"Optional services without API keys: {', '.join(missing_optional)}")
        
        # Check database
        db_path = self.config['database']['path']
        if not os.path.exists(db_path):
            validation['warnings'].append(f"Database file not found: {db_path}")
        
        return validation
    
    def print_status(self):
        """Print configuration status"""
        print("\n🔧 NBA Prediction System Configuration Status")
        print("=" * 50)
        
        validation = self.validate_config()
        
        print("\n📡 API Services:")
        for service, available in validation['available_services'].items():
            status = "✅ Available" if available else "❌ Not configured"
            print(f"  {service:15} {status}")
        
        print(f"\n💾 Database: {self.config['database']['path']}")
        print(f"⏱️  Cache Duration: {self.config['database']['cache_duration']} seconds")
        
        print(f"\n💰 Betting Settings:")
        betting = self.config['betting']
        print(f"  Default Bet Size: ${betting['default_bet_size']}")
        print(f"  Max Bet %: {betting['max_bet_percentage']*100}%")
        print(f"  Kelly Fraction: {betting['kelly_fraction']}")
        
        if validation['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"  • {warning}")
        
        if validation['errors']:
            print(f"\n❌ Errors:")
            for error in validation['errors']:
                print(f"  • {error}")
        
        print(f"\n🎯 System Status: {'Ready' if not validation['errors'] else 'Needs attention'}")

# Global configuration instance
config_manager = ConfigManager()

def get_config() -> ConfigManager:
    """Get global configuration instance"""
    return config_manager

if __name__ == "__main__":
    # Test configuration manager
    config = ConfigManager()
    config.print_status()
