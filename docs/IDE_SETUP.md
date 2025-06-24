# IDE Setup Guide - Business Agent Management System

This guide provides comprehensive instructions for setting up your IDE for optimal development experience with the Business Agent Management System.

## Table of Contents

- [VS Code Setup (Recommended)](#vs-code-setup-recommended)
- [Alternative IDEs](#alternative-ides)
- [Features Overview](#features-overview)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## VS Code Setup (Recommended)

### Quick Start

1. **Open the project workspace:**
   ```bash
   code business-agent-system.code-workspace
   ```

2. **Install recommended extensions:**
   - VS Code will prompt you to install recommended extensions
   - Click "Install" to install all recommended extensions
   - Or manually install from the Extensions tab

3. **Set up the Python environment:**
   ```bash
   # Ensure you're in the project root
   make dev-setup
   
   # Or manually:
   uv sync --all-extras
   source .venv/bin/activate
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

### Essential Extensions

The following extensions are automatically recommended and configured:

#### Core Python Development
- **Python** (`ms-python.python`) - Core Python support
- **Pylance** (`ms-python.vscode-pylance`) - Fast Python language server
- **Black Formatter** (`ms-python.black-formatter`) - Code formatting
- **isort** (`ms-python.isort`) - Import sorting
- **Ruff** (`charliermarsh.ruff`) - Fast linting
- **MyPy Type Checker** (`ms-python.mypy-type-checker`) - Type checking

#### Testing
- **Python Test Explorer** (`hbenl.vscode-test-explorer`) - Test discovery and running
- **pytest** (`ms-python.pytest`) - pytest integration

#### Code Quality
- **GitLens** (`eamodio.gitlens`) - Enhanced Git capabilities
- **GitHub Copilot** (`github.copilot`) - AI-powered code completion
- **Todo Tree** (`gruntfuggly.todo-tree`) - TODO/FIXME highlighting

#### Project Tools
- **Nix IDE** (`jnoortheen.nix-ide`) - Nix language support
- **Makefile Tools** (`ms-vscode.makefile-tools`) - Makefile support
- **YAML** (`redhat.vscode-yaml`) - YAML support with schema validation

### Real-time Validation Features

The IDE is configured for real-time feedback:

#### Format on Save
- **Black** automatically formats Python code on save
- **isort** organizes imports on save
- Line length set to 100 characters (matching project standards)

#### Lint on Type
- **Ruff** provides real-time linting as you type
- **MyPy** shows type errors in the Problems panel
- Custom problem matchers for quality gate integration

#### Import Organization
- Automatic import sorting with isort profile
- Unused import detection and removal
- Import statement formatting

## Features Overview

### Quality Gate Integration

The IDE is deeply integrated with the project's quality gate system:

#### Quick Tasks (Ctrl+Shift+P → Tasks)
- **Quality Gate - Quick**: Fast validation for iteration
- **Quality Gate - Normal**: Standard pre-commit checks
- **Quality Gate - Strict**: Comprehensive validation
- **Validate Current File**: Check only the current file

#### Real-time Feedback
- Problems panel shows all quality issues
- Inline error highlighting
- Suggested fixes via Code Actions (Ctrl+.)

### Testing Integration

#### Test Discovery
- Automatic test discovery in `tests/` directory
- Test Explorer integration
- Individual test running and debugging

#### Debug Configurations
- **Debug Current Test File**: Debug the currently open test
- **Debug Unit Tests**: Debug all unit tests
- **Debug Integration Tests**: Debug integration test suite
- **Debug Main Application**: Debug the main application with various configs

### Development Workflow

#### File Validation
```bash
# Keyboard shortcut: Ctrl+Shift+V
Validate Current File
```

#### Watch Mode
```bash
# Keyboard shortcut: Ctrl+Shift+W
Start Watch Mode
```

#### Application Running
```bash
# F5: Run restaurant system (fast)
# Ctrl+F5: Run restaurant system (normal)
# Shift+F5: Run retail system
# Ctrl+Shift+D: Launch dashboard
```

## Keyboard Shortcuts

### Primary Shortcuts

| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+Shift+Q` | Quality Gate - Quick | Fast validation |
| `Ctrl+Alt+Q` | Quality Gate - Normal | Standard validation |
| `Ctrl+Shift+V` | Validate Current File | Check current file |
| `Ctrl+Shift+T` | Run Unit Tests | Execute unit tests |
| `Ctrl+Alt+T` | Run All Tests | Execute all tests |
| `F5` | Run Restaurant (Fast) | Quick simulation |
| `Ctrl+Shift+D` | Launch Dashboard | Start Streamlit dashboard |
| `Ctrl+Shift+W` | Start Watch Mode | Continuous validation |

### Code Quality Shortcuts

| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+Shift+F` | Format Current File | Format with Black |
| `Ctrl+Alt+F` | Format All Code | Format entire project |
| `Ctrl+Shift+L` | Lint Code | Run linting checks |
| `Ctrl+Shift+Y` | Type Check | Run MyPy type checking |
| `Ctrl+Shift+O` | Organize Imports | Sort imports with isort |

### Navigation Shortcuts

| Shortcut | Action | Description |
|----------|--------|-------------|
| `F8` | Next Problem | Navigate to next issue |
| `Shift+F8` | Previous Problem | Navigate to previous issue |
| `Ctrl+.` | Quick Fix | Show available fixes |
| `Ctrl+Shift+P` | Command Palette | Access all commands |

## Debugging Support

### Available Debug Configurations

1. **Main Application Debugging**
   - Debug Main Application (restaurant config)
   - Debug Main Application (fast test)
   - Debug Retail Application

2. **Agent Debugging**
   - Debug Accounting Agent
   - Debug Base Agent
   - Debug individual agent components

3. **Dashboard Debugging**
   - Debug Dashboard
   - Debug Streamlit Dashboard

4. **Testing Debugging**
   - Debug Current Test File
   - Debug Unit Tests
   - Debug Integration Tests
   - Debug Web Tests

5. **Script Debugging**
   - Debug Quality Gate
   - Debug File Validation
   - Debug Watch Mode

### Debug Features

- **Breakpoints**: Set breakpoints in any Python file
- **Variable Inspection**: Inspect variables in debug console
- **Call Stack**: Navigate through function calls
- **Step Through**: Step over, into, and out of functions
- **Console Evaluation**: Evaluate expressions in debug console

## Alternative IDEs

### PyCharm Setup

1. **Install PyCharm Professional or Community**
2. **Open project directory**
3. **Configure interpreter:**
   - File → Settings → Project → Python Interpreter
   - Add Local Interpreter → Existing environment
   - Select `.venv/bin/python`

4. **Configure tools:**
   - Settings → Tools → External Tools
   - Add tools for make commands:
     - Name: Quality Gate Quick
     - Program: make
     - Arguments: quality-gate-quick
     - Working directory: $ProjectFileDir$

### Vim/Neovim Setup

1. **Install language server support:**
   ```bash
   # Install language servers
   uv add --dev python-lsp-server[all]
   uv add --dev mypy
   uv add --dev ruff-lsp
   ```

2. **Configure with your preferred plugin manager:**
   ```vim
   " Example with vim-plug
   Plug 'neovim/nvim-lspconfig'
   Plug 'hrsh7th/nvim-cmp'
   Plug 'hrsh7th/cmp-nvim-lsp'
   ```

3. **Configure LSP:**
   ```lua
   -- init.lua
   local lspconfig = require('lspconfig')
   
   lspconfig.pylsp.setup({
     settings = {
       pylsp = {
         plugins = {
           black = {enabled = true},
           isort = {enabled = true},
           mypy = {enabled = true},
           ruff = {enabled = true},
         }
       }
     }
   })
   ```

### Emacs Setup

1. **Install packages:**
   ```elisp
   (use-package lsp-mode
     :ensure t
     :commands lsp)
   
   (use-package lsp-ui
     :ensure t
     :commands lsp-ui-mode)
   
   (use-package lsp-pyright
     :ensure t
     :hook (python-mode . (lambda ()
                            (require 'lsp-pyright)
                            (lsp))))
   ```

2. **Configure Python environment:**
   ```elisp
   (setq lsp-pyright-venv-path ".venv")
   ```

## Troubleshooting

### Common Issues

#### Extension Not Working

**Problem**: Python extension not recognizing virtual environment

**Solution**:
1. Check Python interpreter path: `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Select `.venv/bin/python`
3. Reload window: `Ctrl+Shift+P` → "Developer: Reload Window"

#### Type Checking Issues

**Problem**: MyPy not finding modules

**Solution**:
1. Ensure PYTHONPATH is set correctly
2. Check `.vscode/settings.json` for correct paths
3. Run `make type-check` in terminal to verify setup

#### Linting Not Working

**Problem**: Ruff not showing linting errors

**Solution**:
1. Verify Ruff extension is installed and enabled
2. Check output panel: View → Output → Select "Ruff"
3. Ensure `pyproject.toml` has correct Ruff configuration

#### Testing Issues

**Problem**: Tests not discovered

**Solution**:
1. Check test discovery settings in `.vscode/settings.json`
2. Ensure pytest is installed: `uv run pytest --version`
3. Reload window and check Test Explorer

### Performance Issues

#### Slow IntelliSense

**Solutions**:
- Reduce `python.analysis.userFileIndexingLimit`
- Exclude large directories from indexing
- Use `python.analysis.autoImportCompletions` selectively

#### High CPU Usage

**Solutions**:
- Disable unnecessary extensions
- Reduce file watching scope
- Use `python.analysis.diagnosticMode` = "workspace" instead of "openFilesOnly"

### Debugging Issues

#### Debugger Not Stopping at Breakpoints

**Solutions**:
1. Ensure `justMyCode` is set to `false` in launch configuration
2. Check if code is being executed in the expected path
3. Verify Python interpreter is correct

#### Environment Variables Not Loaded

**Solutions**:
1. Check `.env` file exists and is readable
2. Verify `envFile` setting in launch configuration
3. Manually set environment variables in launch configuration

## Advanced Configuration

### Custom Problem Matchers

The project includes custom problem matchers for quality gate integration:

```json
{
  "name": "quality-gate",
  "owner": "quality-gate",
  "fileLocation": "relative",
  "pattern": {
    "regexp": "^(.*):(\\d+):(\\d+):\\s+(warning|error):\\s+(.*)$",
    "file": 1,
    "line": 2,
    "column": 3,
    "severity": 4,
    "message": 5
  }
}
```

### Custom Snippets

The project includes custom code snippets for:
- Agent class templates
- Test class templates
- Database model templates
- Configuration schemas

Access via: `Ctrl+Space` → type snippet prefix → Tab

### Task Customization

Add custom tasks to `.vscode/tasks.json`:

```json
{
  "label": "My Custom Task",
  "type": "shell",
  "command": "make",
  "args": ["my-target"],
  "group": "build",
  "presentation": {
    "echo": true,
    "reveal": "always"
  }
}
```

### Workspace Settings Override

Customize workspace-specific settings in `business-agent-system.code-workspace`:

```json
{
  "settings": {
    "python.analysis.logLevel": "Trace",
    "python.analysis.diagnosticMode": "workspace"
  }
}
```

## Integration with Development Workflow

### Pre-commit Integration

The IDE integrates with pre-commit hooks:
- Real-time validation matches pre-commit checks
- Quality gate tasks align with git hooks
- Consistent formatting and linting

### CI/CD Integration

IDE configuration matches CI/CD pipeline:
- Same tools (Black, Ruff, MyPy, pytest)
- Same configuration files
- Same quality standards

### Nix Integration

For Nix users:
- Nix IDE extension provides syntax highlighting
- Terminal profiles for `nix develop`
- Integration with Nix-based development environment

## Best Practices

### Development Workflow

1. **Start with watch mode**: `Ctrl+Shift+W`
2. **Validate frequently**: `Ctrl+Shift+V` on current file
3. **Run quality gate before commit**: `Ctrl+Shift+Q`
4. **Use debugging for complex issues**: Set breakpoints and debug

### Code Quality

1. **Format on save**: Enabled automatically
2. **Fix linting issues immediately**: Use Quick Fix (`Ctrl+.`)
3. **Add type hints**: Use MyPy feedback
4. **Write tests**: Use test templates from snippets

### Performance

1. **Use specific configurations**: Debug with appropriate launch configs
2. **Limit file watching**: Exclude unnecessary directories
3. **Profile when needed**: Use performance test tasks

## Support and Resources

### Getting Help

1. **Check this documentation first**
2. **Review VS Code Python documentation**
3. **Check project's quality gate output**
4. **Review extension logs**: View → Output → Select extension

### Useful Commands

| Command | Description |
|---------|-------------|
| `Developer: Reload Window` | Reload VS Code window |
| `Python: Select Interpreter` | Choose Python interpreter |
| `Tasks: Run Task` | Run project tasks |
| `Debug: Start Debugging` | Start debugging session |
| `Problems: Focus on Problems View` | Show problems panel |

### Resources

- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [Python Testing in VS Code](https://code.visualstudio.com/docs/python/testing)
- [Debugging Python in VS Code](https://code.visualstudio.com/docs/python/debugging)
- [Project Quality Gate Documentation](../scripts/README.md)

---

*This documentation is updated regularly. For the latest information, check the project's main documentation.*