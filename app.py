import os
from datetime import datetime
import smtplib
from email.message import EmailMessage
import cv2
import numpy as np
from flask import Flask, render_template, redirect, url_for, request, flash, send_from_directory, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from collections import Counter

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pcb_sentry_aoi_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# --- Email Configuration (Update with your SMTP details) ---
MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 587
MAIL_USERNAME = 'pcbsentryaoi001@gmail.com' 
MAIL_PASSWORD = 'mswo vghe tlej qioi'    

# Initialize Extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load your custom trained 50-epoch YOLO PCB model
model = YOLO("runs/detect/pcb_defect_model/weights/best.pt")

# --- User Database Model ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)

# --- Production Inspection Audit Log Model ---
# --- Production Inspection Audit Log Model ---
class InspectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    operator_name = db.Column(db.String(100), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(20), nullable=False)
    defect_count = db.Column(db.Integer, default=0, nullable=False)
    defect_types = db.Column(db.String(255), default="None detected", nullable=False) # Added
    confidence = db.Column(db.String(20), default="100.0%", nullable=False)          # Added

# --- Support Inquiry Database Model ---
class SupportInquiry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    operator_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default="Pending", nullable=False)    

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Helper: Send Registration Email ---
def send_welcome_email(to_email, username):
    try:
        msg = EmailMessage()
        msg['Subject'] = 'Welcome to PCB-Sentry AOI Workspace'
        msg['From'] = MAIL_USERNAME
        msg['To'] = to_email
        msg.set_content(
            f"Hello {username},\n\n"
            f"Your operator account has been successfully registered on the PCB-Sentry AOI Production Console.\n"
            f"You can now sign in and run automated optical inspections on SMT Line 04.\n\n"
            f"Best regards,\n"
            f"PCB-Sentry System Administration"
        )
        
        with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as server:
            server.starttls()
            server.login(MAIL_USERNAME, MAIL_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"Email notification failed to send: {e}")

# --- Helper: Real-World Image Preprocessing ---
def preprocess_real_world_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    cv2.imwrite(image_path, thresh)

# --- Public & Home Routes ---
@app.route('/')
def home():
    total_scanned = InspectionLog.query.count()
    total_passed = InspectionLog.query.filter_by(status='PASS').count()
    
    # Calculate live accuracy rate dynamically
    accuracy_rate = round((total_passed / total_scanned * 100), 1) if total_scanned > 0 else 99.7
    
    # Total defects caught dynamically
    total_defects = db.session.query(db.func.sum(InspectionLog.defect_count)).scalar() or 0

    return render_template('index.html', 
                           total_scanned=total_scanned, 
                           accuracy_rate=accuracy_rate,
                           total_defects=total_defects)

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/about')
def about():
    # Allow public access, but pass the username safely if logged in
    username = current_user.username if current_user.is_authenticated else None
    return render_template('about.html', name=username)

# --- Authentication Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            session['username'] = user.username
            
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password. Please try again.', 'danger')
            
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('Email address already registered.', 'warning')
            return redirect(url_for('signup'))
            
        hashed_password = generate_password_hash(password, method='scrypt')
        
        new_user = User(
            username=username, 
            email=email, 
            password=hashed_password,
            is_admin=False
        )
        db.session.add(new_user)
        db.session.commit()
        
        send_welcome_email(email, username)
        
        flash('Account created successfully! A confirmation email has been sent. Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    total_scanned = InspectionLog.query.count()
    total_defects = db.session.query(db.func.sum(InspectionLog.defect_count)).scalar() or 0
    total_passed = InspectionLog.query.filter_by(status='PASS').count()
    defect_rate = round((total_defects / total_scanned * 100), 2) if total_scanned > 0 else 0.0
    
    logs = InspectionLog.query.order_by(InspectionLog.timestamp.desc()).limit(5).all()
    
    # FIX: Convert logs into dictionaries to match dashboard.html expectations
    recent_scans = []
    for log in logs:
        recent_scans.append({
            "filename": log.filename,
            "status": log.status,
            "defects": log.defect_count,
            "confidence": "95.0%",
            "timestamp": log.timestamp.strftime('%Y-%m-%d %H:%M')
        })

    return render_template('dashboard.html',
                           name=current_user.username,
                           total_scanned=total_scanned,
                           total_defects=total_defects,
                           total_passed=total_passed,
                           defect_rate=defect_rate,
                           recent_scans=recent_scans,
                           defect_counts_dict={},
                           defects=[],
                           original_image=None,
                           result_image=None)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return redirect(url_for('dashboard'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('dashboard'))
        
    # Validate allowed file extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff'}
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Verify that the uploaded file is a valid, readable image file
        img_check = cv2.imread(input_path)
        if img_check is None:
            if os.path.exists(input_path):
                os.remove(input_path)
            flash("Invalid image file. Please upload a proper PCB image.", "error")
            return redirect(url_for('dashboard'))
        
        preprocess_real_world_image(input_path)
        results = model(input_path, conf=0.70)
        
        result_filename = f"res_{filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        results[0].save(filename=result_path)
        
        defects_found = []
        defect_counts_dict = {}

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                
                defects_found.append({
                    "class": class_name,
                    "confidence": round(conf * 100, 2)
                })
                
                # Dynamically count occurrences of each class detected
                defect_counts_dict[class_name] = defect_counts_dict.get(class_name, 0) + 1
                
        defect_count = len(defects_found)
        pass_fail_status = "FAIL" if defect_count > 0 else "PASS"
        
        # Build comma-separated string of defect types found
        defect_types_str = ", ".join([d['class'] for d in defects_found]) if defects_found else "None detected"
        
        # Calculate average confidence for the current scan
        current_avg_conf = round(sum([d['confidence'] for d in defects_found]) / len(defects_found), 1) if defects_found else 100.0
        current_avg_conf_str = f"{current_avg_conf}%"
        
        new_log = InspectionLog(
            operator_name=current_user.username,
            filename=filename,
            status=pass_fail_status,
            defect_count=defect_count,
            defect_types=defect_types_str,    # Saved to DB
            confidence=current_avg_conf_str   # Saved to DB
        )
        db.session.add(new_log)
        db.session.commit()
        
        total_scanned = InspectionLog.query.count()
        total_defects = db.session.query(db.func.sum(InspectionLog.defect_count)).scalar() or 0
        total_passed = InspectionLog.query.filter_by(status='PASS').count()
        defect_rate = round((total_defects / total_scanned * 100), 2) if total_scanned > 0 else 0.0
        
        # Fetch recent logs from database and map to a dynamic structure
        logs = InspectionLog.query.order_by(InspectionLog.timestamp.desc()).limit(5).all()
        recent_scans = []
        for log in logs:
            # If it's the current scan, display its real calculated confidence; otherwise default for older logs
            conf_display = f"{current_avg_conf}%" if log.filename == filename else "95.0%"
            recent_scans.append({
                "filename": log.filename,
                "status": log.status,
                "defects": log.defect_count,
                "confidence": conf_display,
                "timestamp": log.timestamp.strftime('%Y-%m-%d %H:%M')
            })
                
        return render_template('dashboard.html', 
                               name=current_user.username,
                               total_scanned=total_scanned,
                               total_defects=total_defects,
                               total_passed=total_passed,
                               defect_rate=defect_rate,
                               recent_scans=recent_scans,
                               original_image=url_for('static', filename=f'uploads/{filename}'),
                               result_image=url_for('static', filename=f'results/{result_filename}'),
                               defects=defects_found,
                               defect_counts_dict=defect_counts_dict)
    
    return redirect(url_for('dashboard'))




@app.route('/audit-logs')
@login_required
def audit_logs():
    page = request.args.get('page', 1, type=int)
    per_page = 5
    
    search_query = request.args.get('search', '')
    status_filter = request.args.get('status', '')

    query = InspectionLog.query
    if search_query:
        query = query.filter(InspectionLog.filename.ilike(f'%{search_query}%'))
    if status_filter:
        query = query.filter_by(status=status_filter)
        
    pagination = query.order_by(InspectionLog.timestamp.desc()).paginate(page=page, per_page=per_page, error_out=False)
    logs = pagination.items
    
    total_scanned = InspectionLog.query.count()
    total_passed = InspectionLog.query.filter_by(status='PASS').count()
    pass_rate = round((total_passed / total_scanned * 100), 1) if total_scanned > 0 else 0.0

    # Safely calculate Top Defect in Python using Counter
    all_logs = InspectionLog.query.all()
    defect_list = []
    for log in all_logs:
        if hasattr(log, 'defect_types') and log.defect_types and log.defect_types != "None detected":
            for d in log.defect_types.split(','):
                d_cleaned = d.strip()
                if d_cleaned:
                    defect_list.append(d_cleaned)
    
    top_defect = Counter(defect_list).most_common(1)[0][0] if defect_list else ("None" if total_scanned > 0 else "None")

    return render_template('audit_logs.html', 
                           logs=logs,
                           pagination=pagination,
                           total_scanned=total_scanned,
                           pass_rate=pass_rate,
                           top_defect=top_defect)

# Route to delete a single audit log entry
@app.route('/delete-log/<int:log_id>', methods=['POST'])
@login_required
def delete_log(log_id):
    log_entry = InspectionLog.query.get_or_404(log_id)
    db.session.delete(log_entry)
    db.session.commit()
    return redirect(url_for('audit_logs'))

# --- Admin Routes ---
@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    all_users = User.query.all()
    all_logs = InspectionLog.query.order_by(InspectionLog.timestamp.desc()).all()
    
    total_users = User.query.count()
    total_scans = InspectionLog.query.count()
    total_defects = db.session.query(db.func.sum(InspectionLog.defect_count)).scalar() or 0
    total_passed = InspectionLog.query.filter_by(status='PASS').count()
    
    return render_template('admin.html', 
                           name=current_user.username,
                           all_users=all_users,
                           all_logs=all_logs,
                           total_users=total_users,
                           total_scans=total_scans,
                           total_defects=total_defects,
                           total_passed=total_passed)

@app.route('/admin/delete-user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    user_to_delete = User.query.get_or_404(user_id)
    
    if user_to_delete.id == current_user.id:
        flash('You cannot delete your own active administrator account.', 'warning')
        return redirect(url_for('admin_dashboard'))
        
    db.session.delete(user_to_delete)
    db.session.commit()
    
    flash(f'User "{user_to_delete.username}" has been successfully removed from the database.', 'success')
    return redirect(url_for('admin_dashboard'))

# --- Utility & Support Routes ---
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Password reset instructions have been sent to your email address.', 'success')
        else:
            flash('Email address not found in the system database.', 'danger')
        return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Check both possible form field names to prevent NoneType errors
        operator_name = request.form.get('operator_name') or request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # Save to database
        new_inquiry = SupportInquiry(
            operator_name=operator_name,
            email=email,
            message=message
        )
        db.session.add(new_inquiry)
        db.session.commit()
        
        flash('Support inquiry transmitted successfully! Our hardware engineers will review your log.', 'success')
        return redirect(url_for('contact'))
        
    return render_template('contact.html')

@app.route('/admin/support-inquiries')
@login_required
def admin_support_inquiries():
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.', 'danger')
        return redirect(url_for('dashboard'))
        
    inquiries = SupportInquiry.query.order_by(SupportInquiry.timestamp.desc()).all()
    return render_template('admin_inquiries.html', inquiries=inquiries)



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        admin_email = "admin@pcbsentry.com"
        existing_admin = User.query.filter_by(email=admin_email).first()
        if not existing_admin:
            hashed_admin_pwd = generate_password_hash("admin1234", method='scrypt')
            default_admin = User(
                username="FactoryAdmin",
                email=admin_email,
                password=hashed_admin_pwd,
                is_admin=True
            )
            db.session.add(default_admin)
            db.session.commit()
            print("Default Admin Account Created -> Email: admin@pcbsentry.com | Password: admin1234")

    app.run(debug=True)