# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it by emailing **security@brick.so** (or your preferred contact).

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

**Do not open a public GitHub issue for security vulnerabilities.**

We will acknowledge your report within 48 hours and provide a detailed response within 7 days, including next steps and timeline for a fix.

## Security Best Practices

When using thalamus-serve:

1. **Always set `THALAMUS_API_KEY`** - Never run without authentication in production
2. **Use HTTPS** - Deploy behind a reverse proxy with TLS
3. **Limit network access** - Restrict access to trusted clients only
4. **Keep dependencies updated** - Regularly update to the latest version
