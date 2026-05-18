import os
import sys

# Configuration
class Config:
    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Add Anthropic API key for client validation
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.anthropic_api_keys = [
            key.strip()
            for key in os.environ.get("ANTHROPIC_API_KEYS", "").split(",")
            if key.strip()
        ]
        if not self.anthropic_api_key:
            print("Warning: ANTHROPIC_API_KEY not set. Client API key validation will be disabled.")
        
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.azure_api_version = os.environ.get("AZURE_API_VERSION")  # For Azure OpenAI
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8082"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "4096"))
        self.min_tokens_limit = int(os.environ.get("MIN_TOKENS_LIMIT", "100"))
        
        # Connection settings
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "90"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))

        # PostHog traffic analytics
        self.posthog_api_key = os.environ.get("POSTHOG_API_KEY")
        self.posthog_host = os.environ.get("POSTHOG_HOST", "https://us.i.posthog.com").rstrip("/")
        self.posthog_logs_enabled = (
            os.environ.get("POSTHOG_LOGS_ENABLED", "false").lower()
            in ("1", "true", "yes", "on")
        )
        self.posthog_service_name = os.environ.get("POSTHOG_SERVICE_NAME", "claude-code-proxy")
        self.posthog_capture_bodies = (
            os.environ.get("POSTHOG_CAPTURE_BODIES", "false").lower()
            in ("1", "true", "yes", "on")
        )
        self.provider_compat = os.environ.get("PROVIDER_COMPAT", "").strip().lower()
        self.flatten_multimodal_content = (
            self.provider_compat == "zai" or "z.ai" in self.openai_base_url.lower()
        )
        
        # Model settings - BIG and SMALL models
        self.big_model = os.environ.get("BIG_MODEL", "gpt-4o")
        self.middle_model = os.environ.get("MIDDLE_MODEL", self.big_model)
        self.small_model = os.environ.get("SMALL_MODEL", "gpt-4o-mini")
        
    def validate_api_key(self):
        """Basic API key validation"""
        if not self.openai_api_key:
            return False
        # Basic format check for OpenAI API keys
        if not self.openai_api_key.startswith('sk-'):
            return False
        return True
        
    def validate_client_api_key(self, client_api_key):
        """Validate client's Anthropic API key"""
        # If no client API key allowlist is set in environment, skip validation
        allowed_keys = []
        if self.anthropic_api_key:
            allowed_keys.append(self.anthropic_api_key)
        allowed_keys.extend(self.anthropic_api_keys)

        if not allowed_keys:
            return True
            
        # Check if the client's API key matches the expected value
        return client_api_key in allowed_keys
    
    def get_custom_headers(self):
        """Get custom headers from environment variables"""
        custom_headers = {}
        
        # Get all environment variables
        env_vars = dict(os.environ)
        
        # Find CUSTOM_HEADER_* environment variables
        for env_key, env_value in env_vars.items():
            if env_key.startswith('CUSTOM_HEADER_'):
                # Convert CUSTOM_HEADER_KEY to Header-Key
                # Remove 'CUSTOM_HEADER_' prefix and convert to header format
                header_name = env_key[14:]  # Remove 'CUSTOM_HEADER_' prefix
                
                if header_name:  # Make sure it's not empty
                    # Convert underscores to hyphens for HTTP header format
                    header_name = header_name.replace('_', '-')
                    custom_headers[header_name] = env_value
        
        return custom_headers

try:
    config = Config()
    print(f" Configuration loaded: API_KEY={'*' * 20}..., BASE_URL='{config.openai_base_url}'")
except Exception as e:
    print(f"=4 Configuration Error: {e}")
    sys.exit(1)
