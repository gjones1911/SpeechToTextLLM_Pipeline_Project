#!/bin/bash

# Update ngrok URL Script
# This script updates all crucial instances of ngrok URLs in the project
# Usage: ./update_ngrok_url.sh <new_ngrok_url>
# Example: ./update_ngrok_url.sh https://abc123-new-url.ngrok-free.app

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to validate ngrok URL format
validate_ngrok_url() {
    local url="$1"
    if [[ ! "$url" =~ ^https://[a-zA-Z0-9-]+\.ngrok-free\.app$ ]]; then
        return 1
    fi
    return 0
}

# Function to backup a file before modification
backup_file() {
    local file="$1"
    if [ -f "$file" ]; then
        cp "$file" "$file.backup.$(date +%Y%m%d_%H%M%S)"
        print_status "Backed up: $file"
    fi
}

# Function to update ngrok URL in a file
update_file() {
    local file="$1"
    local old_pattern="$2"
    local new_url="$3"
    local description="$4"
    
    if [ ! -f "$file" ]; then
        print_warning "File not found: $file"
        return 1
    fi
    
    # Check if file contains any ngrok URLs
    if ! grep -q "ngrok-free\.app" "$file"; then
        print_status "No ngrok URLs found in: $file"
        return 0
    fi
    
    print_status "Updating $description: $file"
    
    # Backup the file
    backup_file "$file"
    
    # Count matches before update
    local matches_before=$(grep -c "$old_pattern" "$file" 2>/dev/null || echo "0")
    
    # Perform the update using sed
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS sed syntax
        sed -i '' "s|$old_pattern|$new_url|g" "$file"
    else
        # Linux/Windows (Git Bash) sed syntax
        sed -i "s|$old_pattern|$new_url|g" "$file"
    fi
    
    # Count matches after update
    local matches_after=$(grep -c "$new_url" "$file" 2>/dev/null || echo "0")
    
    if [ "$matches_before" -gt 0 ] && [ "$matches_after" -gt 0 ]; then
        print_success "Updated $matches_before instance(s) in $file"
    elif [ "$matches_before" -eq 0 ]; then
        print_status "No matching URLs found in $file"
    else
        print_warning "Update may have failed for $file"
    fi
}

# Main function
main() {
    echo "🔄 ngrok URL Update Script"
    echo "=========================="
    
    # Check if new URL is provided
    if [ $# -ne 1 ]; then
        print_error "Usage: $0 <new_ngrok_url>"
        echo "Example: $0 https://abc123-new-url.ngrok-free.app"
        exit 1
    fi
    
    local new_url="$1"
    
    # Validate the new URL format
    if ! validate_ngrok_url "$new_url"; then
        print_error "Invalid ngrok URL format: $new_url"
        print_error "Expected format: https://xxxx-xxx.ngrok-free.app"
        exit 1
    fi
    
    print_success "New ngrok URL: $new_url"
    echo
    
    # Get current directory
    local project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$project_root"
    
    print_status "Project root: $project_root"
    echo
    
    # Pattern to match any ngrok-free.app URL
    local ngrok_pattern="https://[a-zA-Z0-9-]*\.ngrok-free\.app"
    
    # List of files to update with descriptions
    declare -a files_to_update=(
        "test_ngrok_connection.py:ngrok connection test script"
        "debug_connection.py:debug connection script"
        "test_history.py:history test script"
        "code/llm_agent_linux_package/README.md:LLM agent documentation"
        "code/llm_agent_linux_package/quick_test.sh:quick test script"
        "code/llm_agent_linux_package/package_info.txt:package info file"
    )
    
    # Track update statistics
    local total_files=0
    local updated_files=0
    local failed_files=0
    
    # Update each file
    for file_info in "${files_to_update[@]}"; do
        IFS=':' read -r file description <<< "$file_info"
        total_files=$((total_files + 1))
        
        if update_file "$file" "$ngrok_pattern" "$new_url" "$description"; then
            updated_files=$((updated_files + 1))
        else
            failed_files=$((failed_files + 1))
        fi
    done
    
    echo
    echo "📊 Update Summary"
    echo "================="
    print_status "Total files processed: $total_files"
    print_success "Successfully updated: $updated_files"
    
    if [ "$failed_files" -gt 0 ]; then
        print_warning "Failed updates: $failed_files"
    fi
    
    echo
    print_success "✅ ngrok URL update completed!"
    print_status "New URL: $new_url"
    
    # Show backup files created
    local backup_count=$(find . -name "*.backup.*" -newer "$0" 2>/dev/null | wc -l)
    if [ "$backup_count" -gt 0 ]; then
        echo
        print_status "📁 Backup files created: $backup_count"
        print_status "   To clean up backups: find . -name '*.backup.*' -delete"
    fi
    
    echo
    print_status "🧪 To test the new URL, run:"
    echo "   python test_transcriber_agent.py $new_url --test-only"
}

# Run main function with all arguments
main "$@"
