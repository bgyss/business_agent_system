# CI/CD Guide - Business Agent Management System

This guide covers the comprehensive CI/CD pipeline implementation, including the new incremental validation workflow that mirrors our local development process.

## Overview

Our CI/CD system consists of two main workflows:

1. **Incremental CI** (`ci-incremental.yml`) - Primary workflow for efficient validation
2. **Legacy CI** (`ci.yml`) - Comprehensive backup workflow for full validation

## 🚀 Incremental CI Workflow

The incremental CI workflow is designed to provide fast feedback while maintaining quality gates. It only runs checks on changed files, dramatically reducing CI execution time.

### Key Features

- **Smart Change Detection**: Identifies Python files in critical paths
- **Incremental Validation**: Validates only changed files individually
- **Quality Gate Integration**: Uses our local quality gate scripts
- **Matrix Testing**: Tests across Python 3.8-3.11
- **Security Integration**: Runs security checks on changed files only

### Workflow Jobs

#### 1. Change Detection (`detect-changes`)
```yaml
# Detects changed files and categorizes them
- Python files: agents/, models/, simulation/, dashboard/, utils/, main.py, scripts/
- Config files: *.yml, *.yaml, *.toml, *.json
- Test files: tests/**/*.py
```

#### 2. Incremental Validation (`incremental-validation`)
- Runs for each changed Python file
- Uses `scripts/validate-file.py` for targeted validation
- Matrix across Python versions 3.8-3.11
- Fast feedback on file-level issues

#### 3. Quality Gate Enforcement (`quality-gate-enforcement`)
- Runs comprehensive quality gate on all changed files
- Uses `scripts/quality-gate.py` in normal mode
- Provides detailed validation reports
- Blocks merges on quality violations

#### 4. Comprehensive Testing (`comprehensive-testing`)
- Runs unit tests with coverage reporting
- Executes integration smoke tests
- Tests on minimum (3.8) and maximum (3.11) Python versions
- Uploads coverage data to Codecov

#### 5. Security Scanning (`security-scan`)
- Runs Bandit security analysis on changed files
- Performs dependency vulnerability scanning with Safety
- Non-blocking but reports security findings

#### 6. Configuration Validation (`config-validation`)
- Validates YAML, JSON, and TOML configuration files
- Ensures syntax correctness and structural integrity
- Prevents deployment of broken configurations

#### 7. Final Validation Gate (`final-validation-gate`)
- Aggregates results from all previous jobs
- Provides comprehensive success/failure reporting
- Blocks merges if any critical jobs fail

### Triggering

The incremental CI runs on:
- Push to `main` or `develop` branches
- Pull requests targeting `main`
- Automatic cancellation of previous runs when new commits are pushed

## 📋 Legacy CI Workflow

The legacy CI workflow provides comprehensive validation and serves as a backup when incremental CI is insufficient.

### When to Use Legacy CI

- Manual testing via `workflow_dispatch`
- When incremental CI is not providing sufficient coverage
- For comprehensive validation before major releases
- When the workflow file itself is modified

### Key Features

- **Full Project Validation**: Runs quality gate in strict mode on entire codebase
- **Comprehensive Testing**: Complete test suite execution
- **Security Scanning**: Full security analysis
- **Manual Triggering**: Can be run on-demand via GitHub UI

## 🔒 Branch Protection Rules

Our branch protection system works with the incremental CI to enforce quality:

### Required Status Checks

- `final-validation-gate`
- `incremental-validation (3.8)` and `incremental-validation (3.11)`
- `quality-gate-enforcement (3.11)`
- `comprehensive-testing (3.8)` and `comprehensive-testing (3.11)`
- `security-scan`

### Setup Branch Protection

```bash
# Run the setup script
./scripts/setup-branch-protection.sh

# For dry-run (see what would be changed)
./scripts/setup-branch-protection.sh --dry-run

# For different branch
./scripts/setup-branch-protection.sh --branch develop
```

### Review Requirements

- At least 1 approving review required
- Stale reviews dismissed when new commits are pushed
- Code owner reviews required (see `.github/CODEOWNERS`)
- Conversation resolution required before merge

## 🛠️ Quality Gate Integration

Both CI workflows integrate with our quality gate system:

### Quality Gate Modes

- **Quick**: Format, imports, lint, type checking (for incremental validation)
- **Normal**: Adds unit tests (for standard enforcement)
- **Strict**: Comprehensive checks including security and coverage (for legacy CI)
- **Emergency**: Minimal checks for urgent fixes

### Local Development Alignment

The CI system mirrors our local development workflow:

```bash
# What CI runs on individual files
make validate-file FILE=agents/base_agent.py

# What CI runs for quality enforcement
make quality-gate

# What legacy CI runs for comprehensive validation
make quality-gate-strict
```

## 📊 Performance Optimization

### Incremental CI Benefits

- **~70% faster execution**: Only validates changed files
- **Parallel validation**: Each file validated independently
- **Smart caching**: uv provides fast dependency installation
- **Early failure detection**: File-level validation catches issues quickly

### Efficiency Metrics

| Workflow Type | Typical Runtime | Files Checked | Resource Usage |
|---------------|----------------|---------------|----------------|
| Incremental CI | 5-15 minutes | Changed only | Low |
| Legacy CI | 20-45 minutes | All files | High |
| Traditional CI | 15-30 minutes | All files | Medium |

## 🔧 Configuration

### Environment Variables

```yaml
# Required for integration tests
ANTHROPIC_API_KEY: test-api-key-for-testing

# For project structure validation
PYTHONPATH: ${{ github.workspace }}
```

### Matrix Strategy

```yaml
strategy:
  fail-fast: false  # Allow other versions to continue if one fails
  matrix:
    python-version: ["3.8", "3.9", "3.10", "3.11"]
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Change Detection Not Working
```bash
# Check if git fetch-depth is correct
- uses: actions/checkout@v4
  with:
    fetch-depth: 0  # Ensure full history for comparison
```

#### 2. Quality Gate Failures
- Check local development with same commands
- Review quality gate configuration in `scripts/quality-gate.py`
- Ensure all dependencies are properly installed

#### 3. Matrix Job Failures
- Individual Python version failures don't block others
- Check specific Python version compatibility
- Review error logs in job artifacts

#### 4. Status Check Failures
- Ensure all required status checks are correctly named
- Verify branch protection rules match job names
- Check for typos in status check requirements

### Debugging Commands

```bash
# Test change detection locally
git diff --name-only HEAD~1 HEAD | grep -E '^(agents/|models/|simulation/).*\.py$'

# Test quality gate locally
uv run python scripts/quality-gate.py --mode normal --files agents/base_agent.py

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci-incremental.yml'))"

# Test branch protection settings
gh api repos/:owner/:repo/branches/main/protection
```

## 📈 Monitoring and Metrics

### Artifact Collection

- **Test Results**: JUnit XML files for each job
- **Coverage Reports**: XML format for Codecov integration
- **Quality Gate Results**: JSON reports for analysis
- **Validation Logs**: Detailed logs for troubleshooting

### Success Metrics

- CI execution time reduction
- Failure detection rate
- Time to feedback for developers
- Resource usage optimization

## 🔄 Migration Strategy

### Gradual Rollout

1. **Phase 1**: Deploy incremental CI alongside existing CI
2. **Phase 2**: Enable branch protection with incremental CI
3. **Phase 3**: Monitor performance and adjust as needed
4. **Phase 4**: Fully migrate to incremental CI as primary

### Rollback Plan

- Legacy CI workflow remains available for manual triggering
- Branch protection can be temporarily disabled if needed
- Quality gate system remains independent and functional

## 🤝 Best Practices

### For Developers

1. **Local Testing**: Run quality gates locally before pushing
2. **Small Commits**: Keep changes focused for efficient CI runs
3. **Fix Quickly**: Address CI failures promptly to avoid blocking others
4. **Review Carefully**: Use branch protection as a safety net, not a replacement for code review

### For Maintainers

1. **Monitor Performance**: Track CI execution times and success rates
2. **Update Dependencies**: Keep uv and other tools updated
3. **Review Failures**: Investigate systematic failures
4. **Adjust Configuration**: Fine-tune quality gate settings based on team needs

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [Quality Gate System](../scripts/quality-gate.py)
- [Local Development Workflow](../CLAUDE.md#development-workflow)

---

*This guide should be updated when CI/CD configurations change. Keep it current to ensure effective development and deployment practices.*