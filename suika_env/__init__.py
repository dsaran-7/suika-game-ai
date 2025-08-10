from gymnasium.envs.registration import register as register
print("registering suika env")

# Import the environment classes
try:
    from .suika_browser_env import SuikaBrowserEnv
    from .suika_ai_agent import SuikaAIAgent
    print("✅ All environment classes imported successfully")
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    # Fallback: import after registration
    pass

register(
    id="SuikaEnv-v0",
    entry_point='suika_env.suika_browser_env:SuikaBrowserEnv',
    max_episode_steps=100,
)

# Register the AI agent environment
register(
    id="SuikaAIAgent-v0",
    entry_point='suika_env.suika_ai_agent:SuikaAIAgent',
    max_episode_steps=100,
)

# Ensure classes are available as module attributes
try:
    from .suika_browser_env import SuikaBrowserEnv
    from .suika_ai_agent import SuikaAIAgent
    __all__ = ['SuikaBrowserEnv', 'SuikaAIAgent']
except ImportError:
    pass
