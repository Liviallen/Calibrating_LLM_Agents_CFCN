"""Load this pipeline's .env without replacing exported environment variables."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_environment(path=None):
    env_file = ROOT / '.env' if path is None else Path(path)
    if not env_file.is_file():
        return False
    from dotenv import load_dotenv
    return load_dotenv(dotenv_path=env_file, override=False)
