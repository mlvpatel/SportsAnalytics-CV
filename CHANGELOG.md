# Changelog

All notable changes to SportsAnalytics-CV will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.1.0] - 2026-02-13

### Added
- FastAPI REST API with `/health`, `/analyze`, `/status/{job_id}`, `/jobs/{job_id}` endpoints
- Pydantic request/response models for API type safety
- Configuration system with YAML config file and environment variable overrides
- Comprehensive test suite (71+ tests across 10 test files)
- Shared test fixtures in `conftest.py`
- `Makefile` with `lint`, `format`, `test`, `test-cov`, `run`, `api`, `docker-build`, `clean` commands
- `.pre-commit-config.yaml` for automatic code formatting on commits
- `.dockerignore` to exclude unnecessary files from Docker builds
- `.env.example` with documented environment variables
- `CHANGELOG.md` (this file)
- `SECURITY.md` with vulnerability reporting policy
- `pyproject.toml` centralising tool configuration (black, isort, pytest, coverage)
- `requirements-dev.txt` for development dependencies
- `.github/dependabot.yml` for automated dependency updates
- `.github/workflows/codeql.yml` for GitHub code scanning
- GitHub issue and PR templates (bug report, feature request, PR checklist)
- API security headers middleware (X-Content-Type-Options, X-Frame-Options, etc.)
- Per-IP rate limiting on API (configurable via `RATE_LIMIT_RPM`)
- Optional API key authentication via `X-API-Key` header
- Security scanning with `pip-audit` in CI
- Multi-Python version CI matrix (3.10, 3.11, 3.12)

### Changed
- Added type hints to all 33+ functions across all source modules
- Added Google-style docstrings to all classes and functions
- Added error handling (try/except) throughout all source modules
- Renamed `SpeedAndDistance_Estimator` to `SpeedDistanceEstimator` (PEP 8)
- Fixed typo `draw_traingle` → `draw_triangle` in tracker
- Fixed typo `sekf` → `self` in tracker
- Fixed typo `persepctive_trasnformer` → `perspective_transformer` in view_transformer
- Made `ViewTransformer` configurable (pixel_vertices, court dimensions)
- Made `CameraMovementEstimator` configurable (minimum_distance)
- Made `SpeedDistanceEstimator` configurable (frame_window, frame_rate)
- Made `PlayerBallAssigner` configurable (max_player_ball_distance)
- Added input validation for model file existence in Tracker
- Added graceful error recovery for corrupt pickle caches
- Added null-safe dictionary access throughout tracking pipeline
- Added division-by-zero guards in speed and ball control calculations
- Removed hardcoded player_id == 91 override in team_assigner
- Updated `.gitignore` to exclude `.pkl` files (security)
- Hardened `main.py` with path validation, log sanitization, and file-size limits
- CORS restricted to configurable origins (no more wildcard `*`)
- Docker Compose: added `read_only`, `tmpfs`, `no-new-privileges`, models volume read-only
- Removed deprecated `version: "3.9"` from docker-compose.yml
- Added `isort` to Docker development stage
- Requirements pinned to real installable versions
- Removed duplicate `opencv-python` (kept headless only)
- Extended lint checks to cover `tests/` and `main.py`
- Made `torch` import optional in API (graceful fallback for CI)

### Removed
- Pickle cache files from git tracking (security risk)
- Unused imports across test files

### Security
- Log injection prevention via `SanitizedFormatter`
- File path traversal protection with extension allow-lists
- Null-byte injection prevention
- API key authentication (opt-in)
- Rate limiting (60 RPM default)
- Security headers on all API responses
- Dependabot for automated dependency updates
- CodeQL analysis for code scanning

## [1.0.0] - 2026-01-01

### Added
- Initial release with YOLO-based player/ball tracking
- Camera movement estimation via optical flow
- Team assignment via K-Means color clustering
- Ball possession analysis
- Speed and distance estimation
- Streamlit dashboard UI
- Docker support with CUDA
- CI/CD pipeline with GitHub Actions
