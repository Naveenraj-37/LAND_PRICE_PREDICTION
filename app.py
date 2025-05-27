import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, flash, session, send_file,jsonify
from model import LandPricePredictor
import cx_Oracle
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import logging
from io import BytesIO
from search import search_bp
from fpdf import FPDF
import tempfile
import io
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configuration
FROM_EMAIL = 'nootherfutureplot2025@gmail.com'
FROM_PASSWORD = 'wdbt qftb tfyn dkru'


logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your_secret_key')
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Dataset Configuration
dataset_dir = 'data'
os.makedirs(dataset_dir, exist_ok=True)
DATASET_PATH = os.path.join(dataset_dir, 'land_prediction_Dataset.csv')
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("Dataset file not found!")

# Initialize predictor
predictor = LandPricePredictor(DATASET_PATH)

# Read data
df = pd.read_csv(DATASET_PATH)
land_data = df[['Area', 'Land Use Type', 'Water Table Depth (m)', 'Land Price (?/sq.ft)']].dropna().to_dict(orient='records')

# Database Connection
def get_db_connection():
    try:
        dsn = cx_Oracle.makedsn("localhost", 1521, service_name="XE")
        return cx_Oracle.connect(user="system", password="mydbms1027", dsn=dsn)
    except cx_Oracle.DatabaseError as e:
        logging.error("Database connection error: %s", e)
        return None

conn = get_db_connection()
if conn is None:
    exit(1)
cursor = conn.cursor()
@app.route('/', methods=['GET', 'POST'])
def first_page(): 
    return render_template('first_pg.html')


@app.route('/home')
def home():
    if 'user' not in session:
        flash("Please log in first", "danger")
        return redirect(url_for('login'))
    return render_template('index.html', predicted_prices=None, results=getattr(predictor, 'results', []))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        region = request.form.get('region', '').strip() or None

        if not all([name, email, password]):
            flash("All fields are required", "danger")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        try:
            if region:
                cursor.execute("INSERT INTO users (name, email, password, region) VALUES (:1, :2, :3, :4)",
                             (name, email, hashed_password, region))
            else:
                cursor.execute("INSERT INTO users (name, email, password) VALUES (:1, :2, :3)",
                             (name, email, hashed_password))
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except cx_Oracle.IntegrityError:
            flash('Email already registered. Try another one.', 'danger')
            return redirect(url_for('register'))
        except Exception as e:
            flash(f'Registration failed: {str(e)}', 'danger')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/predict_search', methods=['POST'])
def predict_search():
    try:
        # Get form inputs
        area = request.form.get('area')

        # Predict prices for the specified area
        predictions = predictor.predict_for_area(area)

        # Find matching areas based on the predicted price
        matching_areas = []
        for _, row in df.iterrows():
            if row['Land Price (?/sq.ft)'] >= predictions['Current Price'] * 0.9 and row['Land Price (?/sq.ft)'] <= predictions['Current Price'] * 1.1:
                matching_areas.append({'name': row['Area'], 'price': row['Land Price (?/sq.ft)']})

        return render_template('predict.html', 
                             area=area, 
                             inflated_prices=predictions['Corrected Average with Inflation'], 
                             years=list(range(2025, 2037)), 
                             matching_areas=matching_areas)
        
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        return render_template('error.html', message="An error occurred processing your request"), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not email or not password:
            flash("Email and password are required", "danger")
            return redirect(url_for('login'))

        if email == "nootherfutureplot2025@gmail.com" and password == "Admin@12345":
            session['user'] = email
            flash("Admin login successful", "success")
            return redirect(url_for('admin_dashboard'))

        try:
            cursor.execute("SELECT password FROM users WHERE email = :1", (email,))
            user = cursor.fetchone()

            if user and check_password_hash(user[0], password):
                session['user'] = email
                flash("Login successful", "success")
                return redirect(url_for('home'))
            
            flash("Invalid credentials", "danger")
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"Login error: {str(e)}", "danger")
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if session.get('user') != 'nootherfutureplot2025@gmail.com':
        flash("Unauthorized access", "danger")
        return redirect(url_for('login'))
    return render_template('admin_dashboard.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        flash("Please log in first", "danger")
        return redirect(url_for('login'))

    # Define all supported land types
    land_types = [
        'Commercial', 'Residential', 'Industrial', 
        'Recreational', 'Agricultural', 'Forest',
        'Wetlands', 'Institutional', 'Barren'
    ]

    if request.method == 'POST':
        area = request.form.get('area', '').strip()
        land_type = request.form.get('land_type', 'Commercial').strip()
        
        # Validate land type
        if land_type not in land_types:
            flash("Invalid land type selected", "danger")
            return redirect(url_for('predict'))
        
        if not area:
            flash("Please enter an area name", "danger")
            return redirect(url_for('predict'))
        
        try:
            # Get predictions with land_type
            predictions = predictor.predict_for_area(area, land_type)
            session['predictions'] = predictions
            session['area'] = area
            session['land_type'] = land_type
            
            # Insert prediction into database
            email = session['user']
            predicted_price = predictions['Current Price']
            insert_prediction(email, area, predicted_price, land_type)
            
            # Process predictions
            if callable(predictions.get('Corrected Average with Inflation')):
                predictions['Corrected Average with Inflation'] = predictions['Corrected Average with Inflation']()

            # Get analysis factors
            factors = predictor.analyze_data(area, land_type)

            # Format factors for display
            descriptive_factors = [
                {
                    'name': ('Previous Year (2024) Price' if f['Feature'] == '1 prev land price'
                             else 'Two Years Prior (2023) Price' if f['Feature'] == '2 prev land price'
                             else f"{f['Feature']} Price" if f['Feature'].isdigit() and len(f['Feature']) == 4
                             else f['Feature']),
                    'importance': float(f['Importance'])
                }
                for f in factors
            ]

            # Generate plot data
            prediction_length = len(predictions['Corrected Average with Inflation'])
            years = list(range(2025, 2025 + prediction_length))
            plot_data = None
            
            try:
                plot_buf = predictor.plot_price_comparison(
                    years,
                    [float(predictions['Current Price'])] * prediction_length,
                    [float(v) for v in predictions['Corrected Average with Inflation']],
                    area,
                    land_type
                )
                if plot_buf:
                    plot_data = base64.b64encode(plot_buf.getvalue()).decode('utf-8')
            except Exception as plot_error:
                logging.error("Plot generation failed: %s", plot_error)

            # Prepare data for template
            prediction_data = {
                'area': area,
                'land_type': land_type,
                'years': years,
                'current_price': float(predictions['Current Price']),
                'predictions': [
                    {
                        'name': model_name,
                        'values': [float(x) for x in predictions[model_name]],
                        'color': color
                    }
                    for model_name, color in {
                        'Linear Regression': '#3498db',
                        'Random Forest': '#e74c3c',
                        'KNN Regressor': '#f39c12',
                        'Corrected Average with Inflation': '#27ae60'
                    }.items()
                    if model_name in predictions
                ],
                'factors': descriptive_factors,
                'plot_data': plot_data
            }

            return render_template('result.html', data=prediction_data, factors=descriptive_factors)
        except Exception as e:
            logging.error("Prediction error: %s", e)
            flash(f"Error generating predictions: {str(e)}", "danger")
            return redirect(url_for('predict'))

    # For GET requests, pass all land types to template
    return render_template('predict.html', land_types=land_types)

app.register_blueprint(search_bp, url_prefix='/search')

import os
import tempfile
from flask import send_file, session, flash, redirect, url_for
import matplotlib.pyplot as plt

@app.route('/generate_pdf', methods=['GET'])
def generate_pdf():
    # Initialize variables
    temp_files = []
    pdf_path = None
    chart_path = None
    
    try:
        # Retrieve data from session
        predictions = session.get('predictions', {})
        area = session.get('area', 'Unknown Area')

        if not predictions:
            flash("No predictions available to generate PDF.", "danger")
            return redirect(url_for('home'))

        # Create PDF document
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.cell(200, 10, txt=f"Land Price Prediction for {area}", ln=True, align='C')
        pdf.ln(10)

        # Current Price
        current_price = predictions.get('Current Price', 'N/A')
        pdf.cell(200, 10, txt=f"Current Average Price: Rs{current_price}/sq.ft", ln=True)
        pdf.ln(10)

        # Generate the price trend plot
        years = list(range(2025, 2037))
        prices = []
        
        # Find prediction data
        for model, values in predictions.items():
            if model != 'Current Price' and isinstance(values, (list, tuple)) and len(values) >= 12:
                prices = values[:12]
                break
        
        if prices:
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(years, prices, marker='o', color='blue', linewidth=2)
            plt.title(f'Price Trend Prediction for {area} (2025-2036)')
            plt.xlabel('Year')
            plt.ylabel('Price (Rs/sq.ft)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(years, rotation=45)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as chart_file:
                chart_path = chart_file.name
                temp_files.append(chart_path)
                plt.savefig(chart_path, bbox_inches='tight', dpi=100)
                plt.close()

            # Add chart to PDF
            pdf.cell(200, 10, txt="Price Trend Prediction:", ln=True)
            pdf.image(chart_path, x=10, w=180)
            pdf.ln(90)
        else:
            pdf.cell(200, 10, txt="No prediction data available for plotting.", ln=True)
            pdf.ln(10)

        # Create predictions table
        pdf.cell(200, 10, txt="Detailed Price Predictions:", ln=True)
        pdf.ln(10)
        
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(40, 10, "Year", 1, 0, 'C', 1)
        pdf.cell(40, 10, "Price (Rs/sq.ft)", 1, 1, 'C', 1)
        for model, values in predictions.items():
            if model != 'Current Price' and isinstance(values, (list, tuple)) and len(values) >= 12:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(80, 10, f"{model} Predictions:", ln=True, fill=True)
                pdf.set_font("Arial", size=12)
                
                for year, price in zip(years, values[:12]):
                    pdf.cell(40, 10, str(year), 1)
                    pdf.cell(40, 10, f"Rs{price:,.2f}", 1, 1)

        # Add explanatory methodology section
        pdf.ln(15)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(200, 10, txt="How We Predict Future Land Prices:", ln=True)
        pdf.ln(8)
                
        # Set up for bullet points
        line_height = 8
        
        # 1. Past Price Trends
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(200, line_height, txt="Past Price Trends:", ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, line_height, 
                      "We analyze how land prices have changed over the years to identify long-term trends in pricing. "
                      "Historical data helps establish baseline growth patterns.")
        pdf.ln(4)
        
        # 2. Inflation Impact
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(200, line_height, txt="Inflation Rate Impact:", ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, line_height, 
                      "Land prices correlate strongly with inflation rates (the general increase in costs). "
                      "We study historical inflation to understand its effect on land values.")
        pdf.ln(4)
        
        # 3. Political Year Influence  
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(200, line_height, txt="Political Year Influence:", ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, line_height,
                      "Election years significantly impact prices. Typically:\n"
                      "- Pre-election: Inflation drops, prices may stabilize\n"
                      "- Post-election: Inflation and prices often rise\n"
                      "Our models adjust for these political cycles.")
        pdf.ln(4)
        
        # 4. Environmental Factors
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(200, line_height, txt="Environmental & Development Factors:", ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, line_height,
                      "We also consider:\n"
                      "- Air quality and water availability\n"
                      "- New infrastructure (roads, buildings)\n"
                      "- Overall area development\n"
                      "Areas with better amenities and growth potential see faster price appreciation.")
        pdf.ln(10)

        # Save PDF to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as pdf_file:
            pdf_path = pdf_file.name
            temp_files.append(pdf_path)
            pdf.output(pdf_path)

        # Create response with cleanup callback
        response = send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"{area}_land_price_predictions.pdf",
            mimetype='application/pdf'
        )

        @response.call_on_close
        def cleanup():
            for file_path in temp_files:
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting temporary file {file_path}: {e}")

        return response
        
    except Exception as e:
        # Clean up any created files if error occurs
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error during cleanup: {e}")
        
        flash(f"Error generating PDF: {str(e)}", "danger")
        return redirect(url_for('home'))
def alter_predictions_table():
    try:
        cursor.execute("""
            ALTER TABLE predictions 
            ADD (land_type VARCHAR2(50))
        """)
        conn.commit()
        print("Successfully added land_type column")
    except cx_Oracle.DatabaseError as e:
        print(f"Error altering table: {e}")
        # Check if column already exists
        if "column already exists" in str(e).lower():
            print("Column already exists")
        else:
            raise
        
def insert_prediction(email, area_name, current_price, land_type):
    try:
        # First check if the column exists
        cursor.execute("""
            SELECT COUNT(*) FROM user_tab_columns 
            WHERE table_name = 'PREDICTIONS' 
            AND column_name = 'LAND_TYPE'
        """)
        has_column = cursor.fetchone()[0] > 0
        
        if not has_column:
            try:
                cursor.execute("""
                    ALTER TABLE predictions 
                    ADD (land_type VARCHAR2(50))
                """)
                conn.commit()
                logging.info("Added land_type column to predictions table")
            except cx_Oracle.DatabaseError as e:
                if "column already exists" not in str(e).lower():
                    raise

        # Now insert the record
        cursor.execute("""
            INSERT INTO predictions (
                email_id, 
                area_name, 
                current_price, 
                land_type, 
                prediction_date
            ) VALUES (
                :1, :2, :3, :4, SYSTIMESTAMP
            )
        """, (email, area_name, current_price, land_type))
        conn.commit()
        logging.info("Prediction inserted successfully for %s in %s (%s)", 
                    email, area_name, land_type)
    except cx_Oracle.DatabaseError as e:
        logging.error("Error inserting prediction: %s", e)
        logging.error("Parameters were: email=%s, area=%s, price=%s, type=%s",
                    email, area_name, current_price, land_type)
        raise RuntimeError(f"Database error: {str(e)}")
    
# Function to fetch user emails from the predictions table based on area
def fetch_user_emails_by_area(area_name):
    try:
        cursor.execute("SELECT EMAIL_ID FROM predictions WHERE AREA_NAME = :1", (area_name,))
        return [row[0] for row in cursor.fetchall()]
    except cx_Oracle.DatabaseError as e:
        logging.error("Error fetching user emails: %s", e)
        return []

def send_email(to_email, subject, body, reply_to=None):
    msg = MIMEMultipart()
    msg['From'] = FROM_EMAIL
    msg['To'] = to_email
    msg['Subject'] = subject
    if reply_to:
        msg['Reply-To'] = reply_to

    msg.attach(MIMEText(body, 'plain'))

    try:
        # Set up the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
        server.login(FROM_EMAIL, FROM_PASSWORD)  # Log in to your email account
        server.sendmail(FROM_EMAIL, to_email, msg.as_string())  # Send the email
        server.quit()  # Close the connection
        print(f"✅ Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")
        
# Function to post news and notify users
def post_news_and_notify_users(title, content, area_name):
    # Fetch emails from the predictions table based on the area name
    emails = fetch_user_emails_by_area(area_name)
    subject = f"📰 Regional Update: {title}"

    for email in emails:
        send_email(email, subject, content)

    return True

# Route to handle posting news
@app.route('/notification', methods=['GET', 'POST'])
def notification():
    if request.method == 'POST':
        title = request.form["title"]
        content = request.form["content"]
        area_name = request.form["region"]  # Assuming the region is passed as area_name

        if post_news_and_notify_users(title, content, area_name):
            flash("✅ News added and emails sent!", "success")
        else:
            flash("❌ Failed to send emails.", "danger")

        return redirect(url_for("notification"))

    return render_template("notifications.html")
  
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('first_page'))
faqs = [
    {
        "id": 1,
        "question": "What is the purpose of predicting future land prices?",
        "answer": "The primary goal is to provide insights into potential price trends."
    },
    {
        "id": 2,
        "question": "What factors influence land price predictions?",
        "answer": "Economic indicators, market trends, and location are key factors."
    },
    {
        "id": 3,
        "question": "What methods are commonly used for land price prediction?",
        "answer": "Common methods include time series analysis and machine learning models."
    },
    {
        "id": 4,
        "question": "How accurate are land price predictions?",
        "answer": "While predictions can provide valuable insights, they are inherently uncertain."
    },
    {
        "id": 5,
        "question": "Who can benefit from land price predictions?",
        "answer": "Investors, developers, and policymakers can all benefit from land price predictions."
    },
    {
        "id": 6,
        "question": "Where can I find more information on land price prediction?",
        "answer": "You can explore academic papers and real estate market reports."
    },
    {
        "id": 7,
        "question": "How do we predict future land prices?",
        "answer": "We analyze past price trends, inflation rates, political cycles, and environmental factors to make predictions."
    },
    {
        "id": 8,
        "question": "What role do past price trends play in predictions?",
        "answer": "We analyze how land prices have changed over the years to identify long-term trends in pricing. Historical data helps establish baseline growth patterns."
    },
    {
        "id": 9,
        "question": "How does the inflation rate impact land prices?",
        "answer": "Land prices correlate strongly with inflation rates. We study historical inflation to understand its effect on land values."
    },
    {
        "id": 10,
        "question": "How do political cycles influence land prices?",
        "answer": "Election years significantly impact prices. Typically, inflation drops before elections, stabilizing prices, while post-election, inflation and prices often rise. Our models adjust for these political cycles."
    },
    {
        "id": 11,
        "question": "What environmental factors do we consider in our predictions?",
        "answer": "We consider air quality, water availability, new infrastructure, and overall area development. Areas with better amenities and growth potential see faster price appreciation."
    },
    {
        "id": 12,
        "question": "How does new infrastructure affect land prices?",
        "answer": "New infrastructure, such as roads and buildings, can significantly increase land value by improving accessibility and attracting development."
    }
]
@app.route('/faqs', methods=['GET'])
def get_faqs():
    return jsonify(faqs)
@app.route('/faqs_page', methods=['GET'])
def faqs_page():
    return render_template('faqs.html', faqs=faqs)
if __name__ == '__main__':
    try:
       app.run(debug=True, port=5050)
    finally:
        if conn:
            conn.close()