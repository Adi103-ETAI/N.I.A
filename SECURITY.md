# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in N.I.A, please report it responsibly:

1. **Do NOT open a public GitHub issue.**
2. Email: security@adi103.dev (or DM via GitHub)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment:** within 48 hours
- **Initial assessment:** within 1 week
- **Fix or mitigation:** within 30 days for critical, 90 days for high

## Security Features

N.I.A includes the following security measures:

- **Shell hardening:** Deobfuscation of shell commands (ANSI stripping, Unicode normalization, $IFS expansion, backslash-escape removal) before pattern matching
- **Hardline blocklist:** Unconditional blocks for `rm -rf /`, `mkfs`, `dd to /dev/sd*`, fork bombs, `shutdown`/`reboot` — fires even under FULL_AUTO mode
- **Sudo stdin guard:** Blocks `sudo -S` (password guessing) when `SUDO_PASSWORD` is not configured
- **API key sanitization:** All API keys are stripped from subprocess environment variables before executing bash commands
- **MCP security validation:** Blocks shell interpreters with network egress or OS persistence patterns in MCP stdio commands
- **Skills guard:** Security scan (80+ threat patterns) before installing skills from the hub
- **AST sandbox:** Execute_code tool blocks dunder attribute access (`__class__`, `__subclasses__`, etc.) to prevent sandbox escape
- **SSRF guard:** MCP HTTP/SSE connections to private/localhost IPs are blocked by default
- **Circuit breaker:** MCP servers that fail 5 consecutive connection attempts are temporarily disabled

## Scope

This policy covers the N.I.A codebase at `https://github.com/Adi103-ETAI/N.I.A`.
