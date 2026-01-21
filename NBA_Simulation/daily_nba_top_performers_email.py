import os
import sys
import subprocess
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime


def run_cmd(cmd, cwd=None):
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def send_email_with_attachment(sender_email, app_password, recipient_email, subject, body, attachment_path):
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    filename = attachment_path.split("\\")[-1].split("/")[-1]
    with open(attachment_path, "rb") as f:
        part = MIMEApplication(f.read(), Name=filename)
        part["Content-Disposition"] = f'attachment; filename="{filename}"'
        msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, app_password)
        server.send_message(msg)


def build_email_body(csv_path):
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        # Build a compact text table
        cols = [c for c in ["player_name","team_display","stat","mae","games","pred_pts","pred_reb","pred_ast"] if c in df.columns]
        head = df[cols].head(20)  # include top 20 rows across stats
        lines = ["Top Performers (last 5 games)\n", "", ", ".join(cols)]
        for _, r in head.iterrows():
            row = [str(r.get(c, "")) for c in cols]
            lines.append(", ".join(row))
        return "\n".join(lines)
    except Exception as e:
        return f"Top performers attached. Failed to render preview: {e}"


def main():
    repo_dir = os.path.dirname(os.path.dirname(__file__))
    nba_dir = os.path.join(repo_dir, "NBA_Simulation")

    # 1) Update players and inputs
    run_cmd([sys.executable, os.path.join(nba_dir, "check_and_pull_players.py")])

    # 2) Run full pipeline: backtest + calibration + today's sims + report
    # First pass tuning defaults (can be overridden by env vars):
    #  - no calibration during backtest to avoid circularity
    #  - pace scale to lift totals slightly
    #  - gentler make nerf to increase scoring
    run_cmd([
        sys.executable,
        os.path.join(nba_dir, "run_calibrated_sim.py"),
        "--bt-days", os.environ.get("BT_DAYS", "7"),
        "--bt-sims", os.environ.get("BT_SIMS", "10"),
        "--sims", os.environ.get("TODAY_SIMS", "10"),
        "--report-last-n", os.environ.get("REPORT_LAST_N", "5"),
        "--no-calib-backtest",
        "--pace-scale", os.environ.get("PACE_SCALE", "1.08"),
        "--make-nerf", os.environ.get("MAKE_NERF", "0.04"),
    ])

    # 3) Locate report CSV
    report_csv = os.path.join(nba_dir, "top_performers_all_last5.csv")
    if not os.path.exists(report_csv):
        raise FileNotFoundError(f"Report CSV not found: {report_csv}")

    # 4) Compose and send email
    sender_email = os.environ.get("SENDER_EMAIL", "hartono.vince@gmail.com")
    app_password = os.environ.get("SENDER_APP_PASSWORD", "ontk xwcl dltn fris")
    recipient_email = os.environ.get("RECIPIENT_EMAIL", "yeozoman@hotmail.com")

    today = datetime.now().strftime("%Y-%m-%d")
    subject = f"NBA Top Performers Report - {today}"
    body = build_email_body(report_csv)

    send_email_with_attachment(
        sender_email=sender_email,
        app_password=app_password,
        recipient_email=recipient_email,
        subject=subject,
        body=body,
        attachment_path=report_csv,
    )
    print(f"[email] Sent report to {recipient_email} with attachment {report_csv}")


if __name__ == "__main__":
    main()
