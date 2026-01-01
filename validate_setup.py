#!/usr/bin/env python3
"""
Environment Validation Script for Multimodal EV RAG Assistant

This script checks if your environment is properly configured before running the application.
Run this after setting up your .env file and installing dependencies.
"""

import sys
import os
from typing import Tuple, List

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is 3.8 or higher"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (need 3.8+)"

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, f"{package_name} ({version})"
    except ImportError:
        return False, f"{package_name} (not installed)"

def check_env_file() -> Tuple[bool, str]:
    """Check if .env file exists"""
    if os.path.exists('.env'):
        return True, ".env file found"
    return False, ".env file not found"

def check_env_variable(var_name: str) -> Tuple[bool, str]:
    """Check if an environment variable is set"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # If dotenv is not installed, just check os.environ
        pass
    
    value = os.getenv(var_name)
    if value and len(value) > 0:
        masked = value[:8] + "..." if len(value) > 8 else value
        return True, f"{var_name}: {masked}"
    return False, f"{var_name}: not set"

def check_docker() -> Tuple[bool, str]:
    """Check if Docker is available"""
    try:
        import subprocess
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, version
        return False, "Docker command failed"
    except FileNotFoundError:
        return False, "Docker not found"
    except Exception as e:
        return False, f"Docker check failed: {e}"

def check_docker_compose() -> Tuple[bool, str]:
    """Check if Docker Compose is available"""
    try:
        import subprocess
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, version
        return False, "Docker Compose command failed"
    except FileNotFoundError:
        return False, "Docker Compose not found"
    except Exception as e:
        return False, f"Docker Compose check failed: {e}"

def check_kafka_running() -> Tuple[bool, str]:
    """Check if Kafka is running"""
    try:
        import subprocess
        result = subprocess.run(['docker-compose', 'ps', 'kafka'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if 'Up' in result.stdout:
            return True, "Kafka container is running"
        return False, "Kafka container is not running"
    except Exception as e:
        return False, f"Could not check Kafka status: {e}"

def main():
    """Run all validation checks"""
    print_header("🔍 Environment Validation for Multimodal EV RAG")
    
    checks_passed = 0
    checks_failed = 0
    warnings = 0
    
    # Critical checks
    print(f"{BLUE}Critical Requirements:{RESET}")
    
    critical_checks = [
        ("Python Version", check_python_version()),
        ("Environment File", check_env_file()),
    ]
    
    for name, (passed, message) in critical_checks:
        if passed:
            print(f"  {GREEN}✓{RESET} {name}: {message}")
            checks_passed += 1
        else:
            print(f"  {RED}✗{RESET} {name}: {message}")
            checks_failed += 1
    
    # Package checks
    print(f"\n{BLUE}Required Packages:{RESET}")
    
    packages = [
        ("streamlit", "streamlit"),
        ("python-dotenv", "dotenv"),
        ("requests", "requests"),
        ("kafka-python", "kafka"),
        ("pinecone", "pinecone"),
        ("langchain", "langchain"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
    ]
    
    for package_name, import_name in packages:
        passed, message = check_package(package_name, import_name)
        if passed:
            print(f"  {GREEN}✓{RESET} {message}")
            checks_passed += 1
        else:
            print(f"  {RED}✗{RESET} {message}")
            checks_failed += 1
    
    # Environment variables
    print(f"\n{BLUE}API Keys:{RESET}")
    
    env_vars = [
        "NEWS_API_KEY",
        "PINECONE_API_KEY",
        "OPENROUTER_API_KEY",
    ]
    
    for var in env_vars:
        passed, message = check_env_variable(var)
        if passed:
            print(f"  {GREEN}✓{RESET} {message}")
            checks_passed += 1
        else:
            print(f"  {RED}✗{RESET} {message}")
            checks_failed += 1
    
    # Optional checks
    print(f"\n{BLUE}Optional Components:{RESET}")
    
    optional_checks = [
        ("Docker", check_docker()),
        ("Docker Compose", check_docker_compose()),
    ]
    
    for name, (passed, message) in optional_checks:
        if passed:
            print(f"  {GREEN}✓{RESET} {name}: {message}")
            checks_passed += 1
        else:
            print(f"  {YELLOW}⚠{RESET} {name}: {message}")
            warnings += 1
    
    # Kafka status check
    kafka_passed, kafka_msg = check_kafka_running()
    if kafka_passed:
        print(f"  {GREEN}✓{RESET} Kafka Status: {kafka_msg}")
        checks_passed += 1
    else:
        print(f"  {YELLOW}⚠{RESET} Kafka Status: {kafka_msg}")
        warnings += 1
    
    # Summary
    print_header("📊 Validation Summary")
    
    print(f"  {GREEN}Passed:{RESET} {checks_passed}")
    print(f"  {RED}Failed:{RESET} {checks_failed}")
    print(f"  {YELLOW}Warnings:{RESET} {warnings}")
    
    # Recommendations
    if checks_failed > 0:
        print(f"\n{RED}❌ Setup is incomplete!{RESET}")
        print("\n📝 Next steps:")
        
        if check_env_file()[0] == False:
            print("  1. Create .env file with your API keys")
            print("     cp .env.example .env  # If example exists")
        
        print("  2. Install missing packages:")
        print("     pip install -r requirements.txt")
        
        print("  3. Set up your API keys in .env file")
        print("     NEWS_API_KEY=your_key")
        print("     PINECONE_API_KEY=your_key")
        print("     OPENROUTER_API_KEY=your_key")
        
        sys.exit(1)
    
    elif warnings > 0:
        print(f"\n{YELLOW}⚠️  Setup is mostly complete, but some optional components are missing{RESET}")
        
        if not check_docker()[0]:
            print("\n📝 To use the full pipeline:")
            print("  1. Install Docker: https://docs.docker.com/get-docker/")
            print("  2. Install Docker Compose: https://docs.docker.com/compose/install/")
        
        if not check_kafka_running()[0]:
            print("\n📝 To start Kafka:")
            print("  docker-compose up -d")
        
        print(f"\n{GREEN}✓ You can still run the Streamlit app!{RESET}")
        print("  streamlit run app.py")
    
    else:
        print(f"\n{GREEN}✅ All checks passed! Your environment is ready!{RESET}")
        
        print(f"\n{BLUE}🚀 Quick Start:{RESET}")
        print("  1. Setup Pinecone index:")
        print("     python setup_multimodal_index.py")
        print("\n  2. Start data pipeline (in separate terminals):")
        print("     python ingestion_scripts/producer.py")
        print("     python data_processor/consumer_and_embedder.py")
        print("\n  3. Launch the app:")
        print("     streamlit run app.py")
    
    print()

if __name__ == "__main__":
    main()
