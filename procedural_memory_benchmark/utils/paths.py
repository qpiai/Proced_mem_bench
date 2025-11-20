"""
Path utilities for procedural memory benchmark.

Handles package data resolution and configurable paths for databases and results.
"""

import os
from pathlib import Path
from typing import Union

try:
    # Python 3.9+
    from importlib.resources import files
except ImportError:
    # Python 3.8 fallback
    import pkg_resources
    files = None


def get_query_bank_path(custom_path: Union[str, Path, None] = None) -> Path:
    """
    Get path to query bank JSON file.

    Args:
        custom_path: Optional custom path to query bank. If None, uses default
                    package data location.

    Returns:
        Path object pointing to query bank JSON file.

    Example:
        >>> path = get_query_bank_path()
        >>> with open(path) as f:
        ...     queries = json.load(f)
    """
    if custom_path is not None:
        return Path(custom_path)

    # Use package data
    if files is not None:
        # Python 3.9+
        try:
            return files('procedural_memory_benchmark.benchmark') / 'data' / 'query_bank.json'
        except (TypeError, AttributeError):
            # Fallback if files() doesn't work as expected
            pass

    # Python 3.8 fallback or if files() failed
    try:
        return Path(pkg_resources.resource_filename(
            'procedural_memory_benchmark.benchmark',
            'data/query_bank.json'
        ))
    except Exception:
        # Last resort: relative to this file
        benchmark_dir = Path(__file__).parent.parent / 'benchmark'
        return benchmark_dir / 'data' / 'query_bank.json'


def get_corpus_path(custom_path: Union[str, Path, None] = None) -> Path:
    """
    Get path to AgentInstruct trajectory corpus JSON file.

    Args:
        custom_path: Optional custom path to corpus. If None, uses default
                    package data location.

    Returns:
        Path object pointing to corpus JSON file (~900KB).

    Example:
        >>> path = get_corpus_path()
        >>> with open(path) as f:
        ...     corpus = json.load(f)
    """
    if custom_path is not None:
        return Path(custom_path)

    # Use package data
    if files is not None:
        # Python 3.9+
        try:
            return files('procedural_memory_benchmark') / 'data' / 'corpus' / 'agentinstruct_trajectories.json'
        except (TypeError, AttributeError):
            pass

    # Python 3.8 fallback
    try:
        return Path(pkg_resources.resource_filename(
            'procedural_memory_benchmark',
            'data/corpus/agentinstruct_trajectories.json'
        ))
    except Exception:
        # Last resort: relative to this file
        return Path(__file__).parent.parent / 'data' / 'corpus' / 'agentinstruct_trajectories.json'


def get_default_db_path(db_name: str = "databases") -> Path:
    """
    Get default database path in user cache directory.

    Creates directory if it doesn't exist. Uses platform-appropriate cache location:
    - Linux/Mac: ~/.cache/procedural_memory_benchmark/
    - Windows: %LOCALAPPDATA%/procedural_memory_benchmark/

    Can be overridden with PROCEDURAL_MEMORY_DB_PATH environment variable.

    Args:
        db_name: Subdirectory name for databases (default: "databases")

    Returns:
        Path object pointing to database directory.

    Example:
        >>> db_path = get_default_db_path()
        >>> print(db_path)
        /home/user/.cache/procedural_memory_benchmark/databases
    """
    # Check for environment variable override
    env_path = os.environ.get('PROCEDURAL_MEMORY_DB_PATH')
    if env_path:
        return Path(env_path)

    # Use platform-appropriate cache directory
    if os.name == 'nt':  # Windows
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    else:  # Linux/Mac
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))

    db_path = base / 'procedural_memory_benchmark' / db_name
    db_path.mkdir(parents=True, exist_ok=True)

    return db_path


def get_results_dir(custom_dir: Union[str, Path, None] = None) -> Path:
    """
    Get directory for saving benchmark results.

    Args:
        custom_dir: Optional custom directory. If None, uses current working directory.

    Returns:
        Path object pointing to results directory.

    Example:
        >>> results_dir = get_results_dir()
        >>> result_file = results_dir / "benchmark_results.json"
    """
    if custom_dir is not None:
        path = Path(custom_dir)
    else:
        path = Path.cwd() / "results"

    path.mkdir(parents=True, exist_ok=True)
    return path
