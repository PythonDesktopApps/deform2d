#!/usr/bin/env python
"""
Convenience script to run tests with common options
"""
import sys
import subprocess

def main():
    """Run pytest with sensible defaults"""
    
    # Default arguments
    args = [
        'pytest',
        '-v',  # Verbose
        '--tb=short',  # Short traceback format
        '--color=yes',  # Colored output
    ]
    
    # Add any command-line arguments passed to this script
    args.extend(sys.argv[1:])
    
    print("Running tests with pytest...")
    print(f"Command: {' '.join(args)}\n")
    
    # Run pytest
    result = subprocess.run(args)
    
    return result.returncode

if __name__ == '__main__':
    sys.exit(main())
