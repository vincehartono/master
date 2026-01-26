# Alpaca Crypto Trading Bot - Deployment Guide

## Option 1: Build as Windows EXE

### Prerequisites
- Python 3.10+
- PyInstaller (will be installed automatically)

### Build Steps

1. **From PowerShell in the project directory:**
   ```powershell
   .\build_exe.ps1
   ```

2. **Or manually:**
   ```bash
   pip install pyinstaller
   pyinstaller trade.spec --onefile
   ```

3. **Output:** `dist\AlpacaTradingBot.exe`

### Run the EXE

1. Copy `.env` file to the same folder as `AlpacaTradingBot.exe`
2. Double-click `AlpacaTradingBot.exe`
3. Answer the prompts:
   - Trading mode: `p` for paper (or `l` for live)
   - Backtest: `n` to use existing results or `y` to run backtest
4. Open browser to `http://localhost:5000`

---

## Option 2: GitHub Actions (Cloud-Based)

### Setup

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add CI/CD and EXE build support"
   git push
   ```

2. **Add GitHub Secrets:**
   - Go to repo → Settings → Secrets and variables → Actions
   - Add these secrets:
     - `ALPACA_API_KEY` - Your Alpaca API key
     - `ALPACA_SECRET_KEY` - Your Alpaca secret key
     - `SLACK_WEBHOOK` (optional) - For notifications

3. **Enable Actions:**
   - Go to Actions tab → Enable workflows

### Schedules

The bot runs automatically:
- **Daily at 9 AM UTC** (configurable in `.github/workflows/crypto-trading-bot.yml`)
- **Manual trigger** - Can run anytime via Actions tab

### Monitor Runs

1. Go to your repo's **Actions** tab
2. Click on "Alpaca Crypto Trading Bot"
3. View live logs and download trade results

### View Results

After each run:
- Download logs from **Artifacts** section
- Includes: trade logs, trade history, P&L tracking, backtest results
- Retained for 30 days

### Optional: Slack Notifications

Add Slack webhook to get notified when trades complete:
1. Create Slack webhook: https://api.slack.com/messaging/webhooks
2. Add as `SLACK_WEBHOOK` secret in GitHub
3. Receive notifications on each run

---

## Docker Option (Advanced)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "trade.py"]
```

Build and run:
```bash
docker build -t alpaca-bot .
docker run -e APCA_API_KEY_ID=xxx -e APCA_API_SECRET_KEY=yyy alpaca-bot
```

---

## Troubleshooting

### EXE issues:
- If antivirus blocks: PyInstaller files are safe, add exception
- If .env not found: Make sure it's in same folder as EXE
- If API key error: Verify .env format and keys

### GitHub Actions issues:
- Workflows not running: Check Actions are enabled in settings
- Secrets not loading: Verify secret names match exactly
- Build errors: Check logs in Actions tab for details

---

## Security Notes

- **Never commit `.env` file** - Already in `.gitignore`
- GitHub secrets are encrypted and only available to Actions
- Paper trading is safe for testing (no real money)
- Live trading requires explicit `(l)ive` mode selection

---

## Files Created

```
.github/
  └─ workflows/
     └─ crypto-trading-bot.yml    # GitHub Actions workflow
trade.spec                        # PyInstaller spec file
build_exe.ps1                     # Build script for Windows
requirements.txt                  # Python dependencies
DEPLOYMENT_GUIDE.md              # This file
```
