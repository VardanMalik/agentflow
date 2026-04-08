# Contributing to AgentFlow

Thank you for your interest in contributing to AgentFlow.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/agentflow.git`
3. Set up the development environment: `make install-dev`
4. Create a feature branch: `git checkout -b feature/your-feature`

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed setup instructions.

## Development Workflow

1. Make your changes on a feature branch
2. Write or update tests for your changes
3. Ensure all checks pass:
   ```bash
   make test       # All 119 tests pass
   make lint       # No lint issues
   make typecheck  # No type errors
   ```
4. Commit with a clear, descriptive message
5. Push to your fork and open a pull request

## Code Standards

- **Python 3.11+** — use modern syntax (type hints, `match` statements, `|` unions)
- **Ruff** — linting and formatting (`make lint`, `make format`)
- **mypy strict** — all code must pass strict type checking (`make typecheck`)
- **Line length** — 100 characters
- **Tests required** — all new features and bug fixes must include tests

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Write a clear description of what changed and why
- Reference any related issues
- Ensure CI passes before requesting review
- Rebase on `main` if your branch is behind

## Commit Messages

Use clear, imperative-mood commit messages:

```
feat: add summarization agent
fix: handle circuit breaker timeout in resilient engine
docs: update API reference for workflow endpoints
test: add integration tests for DLQ retry
```

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include reproduction steps for bugs
- Include your environment (Python version, OS, Docker version)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
