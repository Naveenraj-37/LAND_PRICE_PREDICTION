import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cx_Oracle

# Email configuration
FROM_EMAIL = 'nootherfutureplot2025@gmail.com'
FROM_PASSWORD = 'wdbt qftb tfyn dkru'

# -----------------------------
# Database Connection Handling
# -----------------------------
def get_db_cursor():
    dsn = cx_Oracle.makedsn("localhost", 1521, service_name="XE")
    conn = cx_Oracle.connect(user="system", password="mydbms1027", dsn=dsn)
    return conn.cursor(), conn

# -----------------------------
# Send Email Function
# -----------------------------
def send_email(to_email, subject, body, reply_to=None):
    msg = MIMEMultipart()
    msg['From'] = FROM_EMAIL
    msg['To'] = to_email
    msg['Subject'] = subject
    if reply_to:
        msg['Reply-To'] = reply_to

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(FROM_EMAIL, FROM_PASSWORD)
        server.sendmail(FROM_EMAIL, to_email, msg.as_string())
        server.quit()
        print(f"✅ Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")

# ---------------------------------------------
# Admin Posts News & Notifies Users in Region
# ---------------------------------------------
def post_news_and_notify_users(title, content, region):
    cursor, conn = get_db_cursor()
    try:
        # Insert the news into the database
        cursor.execute("""
            INSERT INTO news (title, content, region)
            VALUES (:1, :2, :3)
        """, (title, content, region))
        conn.commit()

        # Fetch all users in the selected region
        cursor.execute("SELECT email FROM users WHERE region = :1", (region,))
        emails = [row[0] for row in cursor.fetchall()]

        subject = f"📰 Regional Update: {title}"

        for email in emails:
            send_email(email, subject, content)

        return True

    except Exception as e:
        print("❌ Error posting news or sending emails:", e)
        return False
    finally:
        cursor.close()
        conn.close()

# -------------------------------
# User Sends Message to Admin
# -------------------------------
def user_message_to_admin(from_user_email, subject, message_body):
    try:
        full_subject = f"📬 User Message: {subject}"
        full_body = f"From: {from_user_email}\n\n{message_body}"
        send_email(FROM_EMAIL, full_subject, full_body, reply_to=from_user_email)
        return True
    except Exception as e:
        print("❌ Error sending email to admin:", e)
        return False
