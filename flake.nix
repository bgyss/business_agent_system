{
  description = "Business Agent Management System - AI-powered autonomous business management";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonVersion = pkgs.python311;
        
        # Python environment with uv
        pythonEnv = pythonVersion.withPackages (ps: with ps; [
          pip
          setuptools
          wheel
        ]);

        # Build inputs for the application
        nativeBuildInputs = with pkgs; [
          pythonEnv
          uv
          sqlite
          postgresql
          redis
          git
          curl
          jq
        ];

        # Runtime dependencies
        buildInputs = with pkgs; [
          sqlite
          postgresql
          redis
        ];

        # Business Agent System package
        business-agent-system = pkgs.python311Packages.buildPythonApplication {
          pname = "business-agent-system";
          version = "1.0.0";
          
          src = ./.;
          
          format = "pyproject";
          
          nativeBuildInputs = with pkgs.python311Packages; [
            setuptools
            wheel
          ];
          
          propagatedBuildInputs = with pkgs.python311Packages; [
            anthropic
            sqlalchemy
            pydantic
            streamlit
            fastapi
            uvicorn
            python-dotenv
            pandas
            numpy
            python-dateutil
            pyyaml
            aiofiles
            redis
            plotly
            pytest
            pytest-asyncio
          ];

          # Skip tests during build for now
          doCheck = false;

          meta = with pkgs.lib; {
            description = "AI-powered autonomous business management system";
            homepage = "https://github.com/your-org/business-agent-system";
            license = licenses.mit;
            maintainers = [ ];
          };
        };

      in
      {
        # Development shell
        devShells.default = pkgs.mkShell {
          buildInputs = nativeBuildInputs ++ buildInputs ++ [
            # Development tools
            pkgs.python311Packages.black
            pkgs.python311Packages.isort
            pkgs.python311Packages.flake8
            pkgs.python311Packages.mypy
            pkgs.python311Packages.pytest
            pkgs.python311Packages.pytest-asyncio
            pkgs.python311Packages.pytest-cov
            
            # Additional development utilities
            pkgs.nodePackages.pyright
            pkgs.ruff
            pkgs.act  # For local GitHub Actions testing
          ];

          shellHook = ''
            echo "🏢 Business Agent Management System Development Environment"
            echo "=================================================="
            echo "Python: $(python --version)"
            echo "uv: $(uv --version)"
            echo "SQLite: $(sqlite3 --version)"
            echo ""
            echo "Available commands:"
            echo "  uv sync                    - Install/update dependencies"
            echo "  uv run python main.py      - Run the main application"
            echo "  uv run streamlit run dashboard/app.py - Run dashboard"
            echo "  uv run pytest              - Run tests"
            echo "  nix build                  - Build the package"
            echo ""
            echo "Configuration files:"
            echo "  config/restaurant_config.yaml"
            echo "  config/retail_config.yaml"
            echo ""
            
            # Create necessary directories
            mkdir -p logs
            
            # Set up environment variables
            export PYTHONPATH="$PWD:$PYTHONPATH"
            
            # Ensure .env file exists
            if [[ ! -f .env ]]; then
              echo "📝 Creating .env file from template..."
              cp .env.example .env
              echo "⚠️  Please edit .env and add your ANTHROPIC_API_KEY"
            fi
            
            # Initialize uv project if not already done
            if [[ ! -f pyproject.toml ]]; then
              echo "🔧 Initializing uv project..."
              uv init --no-readme
            fi
            
            # Check if dependencies are synced
            if [[ ! -d .venv ]]; then
              echo "📦 Installing dependencies with uv..."
              uv sync
            fi
            
            echo "🚀 Ready to develop!"
          '';

          # Environment variables for development
          ANTHROPIC_API_KEY = ""; # Will be set from .env file
          PYTHONPATH = ".";
        };

        # Package outputs
        packages = {
          default = business-agent-system;
          business-agent-system = business-agent-system;
        };

        # Application runners
        apps = {
          default = {
            type = "app";
            program = "${business-agent-system}/bin/business-agent-system";
          };
          
          restaurant = {
            type = "app";
            program = pkgs.writeShellScript "run-restaurant" ''
              ${business-agent-system}/bin/python ${business-agent-system}/bin/main.py \
                --config ${business-agent-system}/share/config/restaurant_config.yaml
            '';
          };
          
          retail = {
            type = "app";
            program = pkgs.writeShellScript "run-retail" ''
              ${business-agent-system}/bin/python ${business-agent-system}/bin/main.py \
                --config ${business-agent-system}/share/config/retail_config.yaml
            '';
          };
          
          dashboard = {
            type = "app";
            program = pkgs.writeShellScript "run-dashboard" ''
              ${business-agent-system}/bin/streamlit run ${business-agent-system}/share/dashboard/app.py
            '';
          };
        };

        # Formatter for `nix fmt`
        formatter = pkgs.alejandra;

        # Development checks
        checks = {
          # Python linting and formatting checks
          lint = pkgs.runCommand "lint-check" {
            buildInputs = [ pythonEnv pkgs.python311Packages.black pkgs.python311Packages.isort pkgs.ruff ];
          } ''
            cd ${./.}
            black --check .
            isort --check-only .
            ruff check .
            touch $out
          '';
          
          # Type checking
          typecheck = pkgs.runCommand "type-check" {
            buildInputs = [ pythonEnv pkgs.python311Packages.mypy ];
          } ''
            cd ${./.}
            mypy --ignore-missing-imports agents/ models/ simulation/
            touch $out
          '';
        };
      }
    );
}