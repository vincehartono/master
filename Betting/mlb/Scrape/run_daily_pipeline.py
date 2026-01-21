import subprocess
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# === Step 1: Run Scripts === #
scripts = [
    r"C:\Users\Vince\master\Betting\mlb\Scrape\scrape_recs.py",
    r"C:\Users\Vince\master\Betting\mlb\Scrape\get_todays_games.py",
    r"C:\Users\Vince\master\Betting\mlb\Scrape\combine_with_today's.py"
]

for script in scripts:
    print(f"🔄 Running {script}...")
    subprocess.run(["python", script], check=True)

# === Step 2: Load Results === #
csv_path = r"C:\Users\Vince\master\Betting\mlb\Scrape\combined_schedule_with_picks.csv"
df = pd.read_csv(csv_path)

# === Step 3: Build Summary === #
top_matchups = df.reindex(df['pick_margin'].abs().sort_values(ascending=False).index).head(4)

summary_lines = ["📊 Top Expert Pick Matchups:\n"]
for _, row in top_matchups.iterrows():
    summary_lines.append(
        f"- {row['Home Team']} ({row['home_pick_count']} picks) vs "
        f"{row['Away Team']} ({row['away_pick_count']} picks) | "
        f"Favor: {row['most_favored_team']} (Margin: {abs(row['pick_margin'])})"
    )

zero_picks = df[(df['home_pick_count'] == 0) | (df['away_pick_count'] == 0)]

if not zero_picks.empty:
    summary_lines.append("\n❗ Games with 0 picks on one or both teams:\n")
    for _, row in zero_picks.iterrows():
        summary_lines.append(
            f"- {row['Home Team']} ({row['home_pick_count']}) vs {row['Away Team']} ({row['away_pick_count']})"
        )

email_body = "\n".join(summary_lines)

# === Step 4: Gmail Send Function === #
def send_email_with_attachment(sender_email, app_password, recipient_email, subject, body, attachment_path):
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    filename = attachment_path.split("\\")[-1]
    with open(attachment_path, "rb") as f:
        part = MIMEApplication(f.read(), Name=filename)
        part["Content-Disposition"] = f'attachment; filename="{filename}"'
        msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, app_password)
        server.send_message(msg)

# === Step 5: Credentials and Send === #
sender_email = "hartono.vince@gmail.com"
app_password = "ontk xwcl dltn fris"
recipient_email = "yeozoman@hotmail.com"

send_email_with_attachment(
    sender_email=sender_email,
    app_password=app_password,
    recipient_email=recipient_email,
    subject="📝 Daily MLB Expert Pick Summary",
    body=email_body,
    attachment_path=csv_path
)

print("✅ Email sent with expert pick summary and full CSV.")