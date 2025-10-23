import subprocess
from pathlib import Path
from typing import Optional


def get_git_sha(short: bool = False, repo_path: Optional[Path] = None) -> str:
    """Get current git commit SHA.

    Args:
        short: If True, return short SHA (7 chars)
        repo_path: Path to git repository (default: current directory)

    Returns:
        str: Git SHA or 'unknown' if not in a git repo
    """
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        if repo_path:
            cmd = ["git", "-C", str(repo_path), "rev-parse", "HEAD"]

        sha = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()

        return sha[:7] if short else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_branch(repo_path: Optional[Path] = None) -> str:
    """Get current git branch name.

    Args:
        repo_path: Path to git repository (default: current directory)

    Returns:
        str: Branch name or 'unknown' if not in a git repo
    """
    try:
        cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        if repo_path:
            cmd = ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"]

        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True
        ).strip()

    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def is_git_dirty(repo_path: Optional[Path] = None) -> bool:
    """Check if git repository has uncommitted changes.

    Args:
        repo_path: Path to git repository (default: current directory)

    Returns:
        bool: True if there are uncommitted changes
    """
    try:
        cmd = ["git", "status", "--porcelain"]
        if repo_path:
            cmd = ["git", "-C", str(repo_path), "status", "--porcelain"]

        output = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True
        ).strip()

        return len(output) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_git_diff(repo_path: Optional[Path] = None) -> str:
    """Get git diff of uncommitted changes.

    Args:
        repo_path: Path to git repository (default: current directory)

    Returns:
        str: Git diff output or empty string if no changes
    """
    try:
        cmd = ["git", "diff", "HEAD"]
        if repo_path:
            cmd = ["git", "-C", str(repo_path), "diff", "HEAD"]

        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)

    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def get_git_info(repo_path: Optional[Path] = None) -> dict:
    """Get comprehensive git information.

    Args:
        repo_path: Path to git repository (default: current directory)

    Returns:
        dict: Git information including sha, branch, dirty status, etc.
    """
    return {
        "sha": get_git_sha(short=False, repo_path=repo_path),
        "sha_short": get_git_sha(short=True, repo_path=repo_path),
        "branch": get_git_branch(repo_path=repo_path),
        "dirty": is_git_dirty(repo_path=repo_path),
    }


def save_git_info(
    run_dir: Path, include_diff: bool = True, repo_path: Optional[Path] = None
):
    """Save git information to run directory.

    Args:
        run_dir: Directory to save git info
        include_diff: If True and repo is dirty, save diff to file
        repo_path: Path to git repository (default: current directory)
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    info = get_git_info(repo_path)

    # Save git info as text file
    git_info_file = run_dir / "git_info.txt"
    with open(git_info_file, "w") as f:
        f.write(f"SHA: {info['sha']}\n")
        f.write(f"Branch: {info['branch']}\n")
        f.write(f"Dirty: {info['dirty']}\n")

    # Save diff if dirty
    if include_diff and info["dirty"]:
        diff = get_git_diff(repo_path)
        if diff:
            diff_file = run_dir / "git_diff.patch"
            with open(diff_file, "w") as f:
                f.write(diff)
