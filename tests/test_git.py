"""Tests for Git integration module."""

import pytest

try:
    from expmate.git import get_git_info, save_git_info, is_git_repo

    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False


pytestmark = pytest.mark.skipif(not GIT_AVAILABLE, reason="Git module not available")


class TestGitIntegration:
    """Test Git integration functionality."""

    def test_is_git_repo_true(self, mock_git_repo):
        assert is_git_repo(mock_git_repo) is True

    def test_is_git_repo_false(self, temp_dir):
        non_git_dir = temp_dir / "not_a_repo"
        non_git_dir.mkdir()
        assert is_git_repo(non_git_dir) is False

    def test_get_git_info(self, mock_git_repo):
        info = get_git_info(mock_git_repo)

        assert "sha" in info
        assert "branch" in info
        assert "is_dirty" in info
        assert isinstance(info["sha"], str)
        assert len(info["sha"]) > 0

    def test_get_git_info_with_dirty_repo(self, mock_git_repo):
        # Make repo dirty
        test_file = mock_git_repo / "test.txt"
        test_file.write_text("modified content")

        info = get_git_info(mock_git_repo)
        assert info["is_dirty"] is True

    def test_get_git_info_with_clean_repo(self, mock_git_repo):
        info = get_git_info(mock_git_repo)
        assert info["is_dirty"] is False

    def test_save_git_info(self, mock_git_repo, temp_dir):
        save_path = temp_dir / "git_info.txt"
        save_git_info(mock_git_repo, save_path)

        assert save_path.exists()
        content = save_path.read_text()
        assert "SHA:" in content or "Commit:" in content

    def test_get_git_info_non_git_dir(self, temp_dir):
        non_git_dir = temp_dir / "not_a_repo"
        non_git_dir.mkdir()

        info = get_git_info(non_git_dir)
        # Should return empty or None for non-git directories
        assert info is None or len(info) == 0 or all(v is None for v in info.values())
