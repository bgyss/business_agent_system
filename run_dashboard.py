#!/usr/bin/env python3
"""
Convenience script to run the Streamlit dashboard
"""
import os
import subprocess
import sys


def main():
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    # Run streamlit
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")

if __name__ == "__main__":
    main()
