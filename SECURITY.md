# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.1.x   | :white_check_mark: |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of SportsAnalytics-CV seriously.

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT open a public issue.** Security vulnerabilities should be reported privately.
2. **Email:** Send a detailed report to [malav.patel203@gmail.com](mailto:malav.patel203@gmail.com)
3. **Include:**
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

| Stage | Timeline |
|-------|----------|
| Acknowledgment | Within 48 hours |
| Initial assessment | Within 1 week |
| Patch release | Within 2 weeks (critical), 4 weeks (moderate) |

## Security Practices

This project follows these security practices:

- **Dependency scanning** — Automated via Dependabot and `pip-audit`
- **Code scanning** — GitHub CodeQL analysis on every push/PR
- **Input validation** — All file paths and user inputs are validated and sanitized
- **Log sanitization** — Control characters stripped to prevent log injection
- **Docker hardening** — Non-root user, read-only filesystem, no-new-privileges
- **API security** — Optional API key auth, rate limiting, security headers, configurable CORS
- **No secrets in code** — Credentials managed via environment variables (`.env`)
