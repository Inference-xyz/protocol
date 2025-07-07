# Contributing to Inference Protocol

Thank you for your interest in contributing to the Inference Protocol! This document provides guidelines for contributors.

## Getting Started

### Prerequisites

- Foundry (latest version)
- Node.js (for development tools)
- Git
- Basic understanding of Solidity and smart contract development

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/Inference-xyz/protocol.git
   cd protocol
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/Inference-xyz/protocol.git
   ```
4. **Install dependencies**:
   ```bash
   forge install
   ```
5. **Build the project**:
   ```bash
   forge build
   ```

## Development Workflow

### Branch Strategy

- **main**: Production-ready code
- **feature/***: Feature development branches
- **bugfix/***: Bug fix branches

### Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

## Code Style Guidelines

### Solidity Code Style

- Use 4 spaces for indentation (no tabs)
- Use underscores instead of dashes in file names
- Follow the [Solidity Style Guide](https://docs.soliditylang.org/en/latest/style-guide.html)
- Use descriptive variable and function names
- Add comprehensive NatSpec documentation

### File Naming Conventions

- Use PascalCase for contract files: `MyContract.sol`
- Use PascalCase for interface files: `IMyInterface.sol`
- Use lowercase with underscores for test files: `my_contract.t.sol`
- Use lowercase with underscores for script files: `my_script.s.sol`

## Testing Guidelines

### Writing Tests

- Write comprehensive tests for all new functionality
- Use descriptive test names that explain the scenario
- Test both positive and negative cases
- Test edge cases and boundary conditions

### Running Tests

```bash
# Run all tests
forge test

# Run tests with verbose output
forge test -vv

# Run specific test file
forge test --match-contract MyContract
```

## Pull Request Process

### Before Submitting a PR

1. **Ensure your code builds successfully**:
   ```bash
   forge build
   ```

2. **Run all tests and ensure they pass**:
   ```bash
   forge test
   ```

3. **Check for linting issues**:
   ```bash
   forge fmt --check
   ```

4. **Update documentation** as needed

### PR Description Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] New tests added for new functionality

## Security Considerations
- [ ] Security implications reviewed
- [ ] Access controls properly implemented
```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Foundry version, etc.)

### Security Issues

For security issues:

- **DO NOT** create a public issue
- Email security issues to the maintainers
- Include detailed vulnerability information

Thank you for contributing to the Inference Protocol! 