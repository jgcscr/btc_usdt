# Contributing to BTC/USDT Trading Pipeline

## Development Workflow
1. Fork the repository and create a feature branch.
2. Make your changes with clear, well-documented code and descriptive commit messages.
3. Add or update unit tests for any new features or bug fixes.
4. Run all tests locally with `pytest` before submitting a pull request.
5. Submit a pull request (PR) to the `main` branch with a clear description of your changes.
6. Participate in code review and address any requested changes.

## Code Style Guidelines
- Follow [PEP8](https://www.python.org/dev/peps/pep-0008/) for Python code.
- Use type annotations and comprehensive docstrings for all functions and classes.
- Keep functions small and focused; prefer composition over inheritance.
- Use logging (not print) for all runtime messages.
- Organize imports: standard library, third-party, then local modules.

## Testing Requirements
- All new code must include unit tests covering edge cases and error handling.
- Tests should be placed in the appropriate `tests/` subdirectory.
- Run `pytest` and ensure all tests pass before submitting a PR.
- If adding new modules, include at least one test file for them.

## Submitting Pull Requests
- Ensure your branch is up to date with `main` before opening a PR.
- Provide a clear, descriptive title and summary for your PR.
- Reference any related issues in the PR description.
- Be responsive to code review feedback and make requested changes promptly.

Thank you for contributing to the BTC/USDT Trading Pipeline!
