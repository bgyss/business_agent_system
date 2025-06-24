# Scripts Directory

This directory contains utility scripts for the Business Agent Management System.

## Watch Mode Validation (`watch-validate.py`)

Continuous validation system that monitors file changes and automatically runs validation steps.

### Features

- **Real-time monitoring**: Uses watchdog to monitor `agents/`, `models/`, and `simulation/` directories
- **Incremental validation**: Runs validation steps on changed files only
- **Smart test discovery**: Automatically finds and runs related tests based on file names
- **Timeout protection**: 30-second timeout per validation step prevents hanging
- **Graceful error handling**: Continues monitoring even after validation failures
- **Visual feedback**: Emoji-enhanced output with color coding

### Usage

#### Using Make (Recommended)
```bash
make watch
```

#### Direct execution
```bash
uv run python scripts/watch-validate.py
```

### Validation Sequence

For each changed `.py` file, the system runs:

1. **Lint Check** - `ruff check file.py`
2. **Auto-fix** - `ruff check --fix file.py`
3. **Format Check** - `black --check file.py`
4. **Import Sorting** - `isort --check-only file.py`
5. **Type Check** - `mypy file.py`
6. **Test Discovery** - Find and run related tests

### Smart Test Discovery

The system intelligently discovers related tests based on file paths:

- **Direct tests**: `test_module.py` for `module.py`
- **Enhanced tests**: `test_module_enhanced.py`, `test_module_coverage.py`, etc.
- **Advanced patterns**: Handles complex test naming patterns in the project

### Examples

```bash
# Start watch mode
make watch

# Output when a file is changed:
🔍 Validating: agents/inventory_agent.py
==================================================
⏳ Lint Check... ✅
⏳ Auto-fix... ✅
⏳ Format Check... ✅
⏳ Import Sorting... ✅
⏳ Type Check... ✅
⏳ Test Discovery... 🧪 Found 8 related test(s)
==================================================
✨ Validation complete for agents/inventory_agent.py
```

### Configuration

- **Debounce time**: 2 seconds (prevents rapid re-processing)
- **Timeout**: 30 seconds per validation step
- **Monitored directories**: `agents/`, `models/`, `simulation/`
- **File patterns**: `*.py` files only

### Requirements

All validation tools must be available:
- `ruff` - Linting and auto-fixing
- `black` - Code formatting
- `isort` - Import sorting
- `mypy` - Type checking
- `pytest` - Test running

Run `make install` or `uv sync --all-extras` to install all dependencies.

## Testing Scripts

### `test-validation.py`

Validates that all required tools are available and the project structure is correct.

```bash
uv run python scripts/test-validation.py
```

### `test-discovery.py`

Tests the smart test discovery functionality on existing project files.

```bash
uv run python scripts/test-discovery.py
```

## Development

### Adding New Validation Steps

To add a new validation step to the watch mode:

1. Add a new method to the `FileValidator` class in `watch-validate.py`
2. Add the step to the `steps` list in `validate_single_file()`
3. Ensure the new tool is added to `required_tools` in `main()`

### Customizing Test Discovery

Modify the `_discover_related_tests()` method in `FileValidator` to add new test discovery patterns.

## Quality Gate System

The Business Agent Management System includes a comprehensive quality gate system that enforces development workflow standards and prevents problematic code from being committed or deployed.

### Quality Gate (`quality-gate.py`)

The main quality gate system supports multiple execution modes with different check intensities and provides comprehensive validation of the entire codebase.

#### Features

- **Multiple Modes**: normal, strict, quick, emergency - each with different check configurations
- **Comprehensive Checks**: formatting, imports, linting, type checking, tests, security, coverage
- **Timeout Protection**: Prevents hanging on long-running checks
- **Detailed Reporting**: Clear error messages with actionable suggestions
- **JSON Output**: Results can be exported for CI/CD integration
- **Exit Codes**: Proper exit codes for automation and scripts

#### Execution Modes

1. **Normal Mode** (default)
   - Checks: format, imports, lint, type, tests
   - Timeout: 180 seconds
   - Use case: Regular development workflow

2. **Strict Mode**
   - Checks: All available checks including security and coverage
   - Timeout: 300 seconds
   - Use case: Pre-commit, CI/CD, production deployment

3. **Quick Mode**
   - Checks: format, imports, lint, type
   - Timeout: 60 seconds
   - Use case: Rapid iteration during development

4. **Emergency Mode**
   - Checks: format, lint (minimal)
   - Timeout: 30 seconds
   - Use case: Emergency fixes, hotfixes

#### Usage Examples

```bash
# Run normal quality gate
make quality-gate
# or
uv run python scripts/quality-gate.py

# Run specific modes
make quality-gate-strict
make quality-gate-quick
make quality-gate-emergency

# Check specific files only
uv run python scripts/quality-gate.py --files agents/base_agent.py models/financial.py

# Export results for CI/CD
uv run python scripts/quality-gate.py --mode strict --output results.json

# Disable colored output
uv run python scripts/quality-gate.py --no-color
```

#### Configuration

The quality gate can be customized with a JSON configuration file:

```json
{
  "timeout_seconds": 240,
  "checks": ["format_check", "lint_check", "type_check", "test_run"]
}
```

### Individual File Validation (`validate-file.py`)

A focused validation tool for checking individual files quickly during development.

#### Features

- **Fast Validation**: Optimized for single-file checks
- **Smart Test Discovery**: Automatically finds and runs related tests
- **Customizable Checks**: Choose specific validation steps
- **Editor Integration**: Perfect for IDE/editor integration
- **Targeted Feedback**: File-specific suggestions and recommendations

#### Available Checks

- **format**: Code formatting with Black
- **imports**: Import sorting with isort  
- **lint**: Code quality with Ruff
- **type**: Type checking with MyPy
- **syntax**: Python syntax validation

#### Usage Examples

```bash
# Validate with default checks
make validate-file FILE=agents/base_agent.py
# or
uv run python scripts/validate-file.py agents/base_agent.py

# Include related tests
make validate-file-with-tests FILE=agents/base_agent.py
# or
uv run python scripts/validate-file.py agents/base_agent.py --with-tests

# Run specific checks only
uv run python scripts/validate-file.py models/financial.py --checks format,lint

# List available checks
uv run python scripts/validate-file.py --list-checks

# Custom timeout
uv run python scripts/validate-file.py agents/base_agent.py --timeout 60
```

#### Smart Check Selection

The validator automatically selects appropriate checks based on file location:

- **agents/**: format, imports, lint, type
- **models/**: format, imports, lint, type
- **simulation/**: format, imports, lint, type
- **dashboard/**: format, imports, lint (no type checking for UI code)
- **tests/**: format, imports, lint (no type checking for tests)

### Make Targets

The quality gate system integrates with the existing Makefile:

```bash
# Quality gate modes
make quality-gate              # Normal mode
make quality-gate-strict       # Strict comprehensive checks
make quality-gate-quick        # Quick iteration checks
make quality-gate-emergency    # Emergency minimal checks

# File validation
make validate-file FILE=path/to/file.py
make validate-file-with-tests FILE=path/to/file.py

# CI/CD integration
make pre-commit-quality        # Pre-commit hook quality gate
make ci-quality-gate          # CI/CD quality gate with JSON output
```

### Integration Examples

#### Pre-commit Hook Integration

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: quality-gate
      name: Quality Gate
      entry: make pre-commit-quality
      language: system
      pass_filenames: false
```

#### CI/CD Integration

```yaml
# GitHub Actions example
- name: Quality Gate
  run: |
    make ci-quality-gate
    
- name: Upload Quality Results
  uses: actions/upload-artifact@v3
  if: always()
  with:
    name: quality-gate-results
    path: quality-gate-results.json
```

#### Editor Integration

For VS Code, add to `tasks.json`:

```json
{
  "label": "Validate Current File",
  "type": "shell",
  "command": "make",
  "args": ["validate-file", "FILE=${relativeFile}"],
  "group": "build",
  "presentation": {
    "echo": true,
    "reveal": "always",
    "focus": false,
    "panel": "shared"
  }
}
```

### Error Handling and Recovery

The quality gate system includes comprehensive error handling:

- **Timeout Protection**: Prevents hanging on long-running operations
- **Tool Availability**: Checks for required tools and provides installation guidance
- **Graceful Degradation**: Continues with available tools if some are missing
- **Detailed Diagnostics**: Clear error messages with suggested fixes

### Performance Considerations

- **Parallel Execution**: Checks run sequentially but are optimized for speed
- **File Filtering**: Smart file discovery excludes irrelevant files
- **Incremental Validation**: File validator focuses on single files for speed
- **Timeout Management**: Prevents resource exhaustion on large codebases

## Troubleshooting

### Common Issues

1. **"Missing required tools"**: Run `make install` to install dependencies
2. **"Could not find pyproject.toml"**: Ensure you're running from the project root
3. **Timeout errors**: Increase the timeout value or use --mode quick
4. **No tests found**: Check that test files follow the expected naming patterns
5. **Type checking failures**: Review MyPy configuration and add missing annotations
6. **Linting failures**: Run auto-fixes with `uv run ruff check --fix .`

### Quality Gate Specific Issues

1. **Security scan failures**: Install bandit and safety with `uv sync --all-extras`
2. **Coverage failures**: Ensure tests exist and meet coverage thresholds
3. **Emergency mode still failing**: Check for critical syntax or formatting errors
4. **File validation timeout**: Reduce timeout or check for infinite loops in type checking

### Debug Mode

For debugging, you can:
- Use `--verbose` flag for detailed output
- Export results with `--output` for analysis
- Run individual checks with validate-file
- Adjust timeout values for slower systems