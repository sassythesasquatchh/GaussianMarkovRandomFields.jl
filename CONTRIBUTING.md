# Contributing to GaussianMarkovRandomFields (Rust)

Thank you for your interest in contributing! The following guidelines will help
you get started.

## Getting Started

1. **Fork and clone** the repository.
2. **Build and test**:
   ```sh
   cargo test
   cargo test --examples
   ```

## Code Style

- Follow standard Rust style (`cargo fmt`) and linting (`cargo clippy`).
- Use meaningful variable names and avoid excessive abbreviations.

## Making Changes

- **Open an issue** before implementing new features to discuss your idea.
- **Document your code** with Rust doc comments and module-level docs.
- **Write tests** for new functionality (see next section).
- **Ensure tests pass** before submitting your changes.

## Testing

Run tests with:

```sh
cargo test
```

When adding a new feature:
- Place integration tests in `tests/` and unit tests alongside modules.
- Write small, focused tests that validate correctness.
- Add edge cases and performance benchmarks where appropriate.

## Submitting a Pull Request

1. Push your changes to your fork and create a pull request (PR) against the `main` branch.
2. Ensure your PR:
   - Passes all tests.
   - Includes appropriate documentation and tests.
   - Provides a clear description of the changes.
3. Be open to feedback and revisions during the review process.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on the
repository. When reporting bugs:
- Provide a **minimal reproducible example**.
- Include Rust and crate version information.
- Describe expected vs. actual behavior.

## License

By contributing, you agree that your contributions will be licensed under the same license as the repository.

Thank you for contributing to GaussianMarkovRandomFields!

