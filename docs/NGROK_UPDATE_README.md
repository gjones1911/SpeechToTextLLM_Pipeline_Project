# ngrok URL Update Scripts

This directory contains scripts to automatically update all crucial ngrok URL instances in the project. This allows you to quickly update everything with a single command whenever you get a new ngrok URL.

## Available Scripts

### 1. Bash Script (Recommended for Git Bash/Linux/macOS)
- **File**: `update_ngrok_url.sh`
- **Usage**: `./update_ngrok_url.sh <new_ngrok_url>`

### 2. Windows Batch Script
- **File**: `update_ngrok_url.bat`
- **Usage**: `update_ngrok_url.bat <new_ngrok_url>`

## Usage Examples

```bash
# Using bash script (Git Bash, Linux, macOS)
./update_ngrok_url.sh https://abc123-new-url.ngrok-free.app

# Using Windows batch script
update_ngrok_url.bat https://abc123-new-url.ngrok-free.app
```

## What Gets Updated

The scripts automatically update ngrok URLs in these crucial files:

1. **`test_ngrok_connection.py`** - Connection testing script
2. **`debug_connection.py`** - Debug connection script
3. **`test_history.py`** - History testing script
4. **`code/llm_agent_linux_package/README.md`** - LLM agent documentation
5. **`code/llm_agent_linux_package/quick_test.sh`** - Quick test script
6. **`code/llm_agent_linux_package/package_info.txt`** - Package info file

## Features

- ✅ **URL Validation**: Ensures the new URL is in correct ngrok format
- 💾 **Automatic Backups**: Creates timestamped backups before modification
- 📊 **Update Summary**: Shows detailed results of the update process
- 🛡️ **Error Handling**: Stops on errors and provides clear error messages
- 🎨 **Colored Output**: Easy-to-read status messages

## Safety Features

- **Backup Creation**: All files are backed up before modification with timestamp
- **Validation**: URL format is validated before any changes are made
- **Rollback**: You can restore from backups if needed

## Backup Management

Backup files are created with the format: `filename.backup.YYYYMMDD_HHMMSS`

To clean up old backups:
```bash
# Remove all backup files
find . -name "*.backup.*" -delete

# Or remove backups older than 7 days
find . -name "*.backup.*" -mtime +7 -delete
```

## Troubleshooting

### Permission Denied (Linux/macOS)
```bash
chmod +x update_ngrok_url.sh
```

### Windows Execution Policy
If you get execution policy errors with PowerShell:
```cmd
powershell -ExecutionPolicy Bypass -File update_ngrok_url.bat <url>
```

### Invalid URL Format
The script expects URLs in this format:
- ✅ `https://abc123-def456.ngrok-free.app`
- ❌ `http://localhost:8080`
- ❌ `https://example.com`

## Quick Test After Update

After updating the URL, test it quickly:
```bash
python test_transcriber_agent.py <new_url> --test-only
```

## Integration with Development Workflow

You can integrate this into your development workflow:

1. **Get new ngrok URL**: `ngrok http 1234`
2. **Update project**: `./update_ngrok_url.sh https://new-url.ngrok-free.app`
3. **Test connection**: `python test_transcriber_agent.py https://new-url.ngrok-free.app --test-only`
4. **Start development**: Ready to go!

## Files NOT Updated

These files contain placeholder URLs and are intentionally not updated:
- Documentation with example URLs (`https://your-ngrok-url.ngrok-free.app`)
- Template files meant for user customization

This ensures that documentation remains generic and usable for different users.
