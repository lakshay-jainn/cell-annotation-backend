import os
import sys
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_cors import CORS
from flask_admin import Admin
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from utilities.auth_utility import hash_password,verify_password
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_session import Session



# import your models and admin views
from db.models import db, User, Sample, SampleAnnotation, PatientAnnotation, UserRole, Patient, UserActivityLog
from admin_views import UserAdmin, SampleAdmin, SampleAnnotationAdmin, PatientAnnotationAdmin, AuthIndexView, PatientAdmin, ModelProcessingView, UserActivityLogAdmin

# blueprint(s)
try:
    from api_prefix import api_bp
except ImportError:
    api_bp = None

# processing_route.py removed - functionality moved to admin_views.py

# config from env
SQL_URI = os.environ.get("SQL_URI", "sqlite:///test.db")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "sqlite:///test.db")
# Auto-generate SECRET_KEY if not provided
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    import secrets
    SECRET_KEY = secrets.token_hex(32)
    print(f"Generated SECRET_KEY for this session: {SECRET_KEY[:10]}...")

ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

limiter = Limiter(key_func=get_remote_address, default_limits=["1000 per day", "200 per hour"])


def create_app():
    app = Flask(__name__, template_folder="templates")
    # keep the simple CORS config as before
    CORS(app, origins=[FRONTEND_URL])

    # core config
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["SQLALCHEMY_DATABASE_URI"] = SQL_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    
    # Session configuration for better debugging
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_PERMANENT'] = True
    app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour session lifetime
    
    # Server-side session storage for multiple workers
    app.config['SESSION_TYPE'] = 'sqlalchemy'
    app.config['SESSION_SQLALCHEMY'] = db
    app.config['SESSION_SQLALCHEMY_TABLE'] = 'sessions'

    print(f"App SECRET_KEY: {SECRET_KEY[:10]}...")
    print(f"Database URI: {SQL_URI}")

    # init optional limiter
    if limiter:
        try:
            limiter.init_app(app)
            # Exempt OPTIONS requests from rate limiting
            limiter.request_filter(lambda: request.method == 'OPTIONS')
        except Exception:
            pass

    # initialize DB
    db.init_app(app)
    
    # Initialize server-side sessions (after db.init_app)
    with app.app_context():
        try:
            Session(app)
            print("✅ Server-side sessions initialized")
        except Exception as e:
            print(f"⚠️  Session initialization failed: {e}")
            # Fallback to basic sessions
            app.config['SESSION_TYPE'] = 'filesystem'
    with app.app_context():
        # Ensure AWS infrastructure is ready
        print("🔧 Checking AWS infrastructure...", flush=True)
        
        # Check S3 bucket and CORS
        print("🔧 Checking S3 bucket and CORS configuration...", flush=True)
        bucket_name = os.environ.get('S3_BUCKET_NAME', 'NOT_SET')
        print(f"🔧 S3_BUCKET_NAME env var: {bucket_name}", flush=True)
        
        try:
            from utilities.aws_utility import ensure_bucket_cors
            print("🔧 Importing ensure_bucket_cors successful", flush=True)
            print(f"🔧 Calling ensure_bucket_cors() for bucket: {bucket_name}", flush=True)
            bucket_result = ensure_bucket_cors()
            print(f"🔧 ensure_bucket_cors() returned: {bucket_result}", flush=True)
            if bucket_result:
                print("✅ S3 bucket exists and CORS is configured", flush=True)
            else:
                print("⚠️  S3 bucket setup may have issues", flush=True)
        except Exception as e:
            print(f"❌ Could not initialize S3 bucket: {e}", flush=True)
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}", flush=True)
        
        # Check SQS queue
        print("🔧 Checking SQS queue...", flush=True)
        try:
            from sqs_processor import ensure_sqs_queue
            queue_url = ensure_sqs_queue()
            if queue_url:
                print(f"✅ SQS queue exists/created: {queue_url}", flush=True)
            else:
                print("❌ Failed to create/access SQS queue", flush=True)
        except Exception as e:
            print(f"❌ Could not initialize SQS queue: {e}", flush=True)
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}", flush=True)
        
        print("🔧 AWS infrastructure check complete", flush=True)
        
        # Create database tables
        print("� Creating database tables...", flush=True)
        try:
            # db.drop_all()
            db.create_all()
            print("✅ Database tables created successfully", flush=True)
        except Exception as e:
            print(f"❌ Error creating database tables: {e}", flush=True)
            raise
        
        # seed super admin user if env vars present
        if ADMIN_EMAIL and ADMIN_PASSWORD:
            try: 
                existing = db.session.execute(db.select(User).filter_by(email=ADMIN_EMAIL)).scalar()
                if not existing:
                    print(f"Seeding admin user: {ADMIN_EMAIL}")
                    hashed = hash_password(ADMIN_PASSWORD)
                    admin_user = User(
                        name="Super Admin",
                        email=ADMIN_EMAIL,
                        password=hashed,
                        role=UserRole.ADMIN,
                        hospital="NA",
                        location="NA"
                    )
                    db.session.add(admin_user)
                    db.session.commit()
                    print("Admin user created.")
                else:
                    print("Admin user already exists.")
            except Exception as e:
                print(f"⚠️  Warning: Could not seed admin user: {e}")
                # Don't fail the application if seeding fails
                db.session.rollback()

    # --- Flask-Login setup ---
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "admin_login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "info"

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, str(user_id))

    # ----------------- ADMIN LOGIN -----------------
    @app.route("/admin/login", methods=["GET", "POST"])
    def admin_login():
        if current_user.is_authenticated and current_user.role == UserRole.ADMIN:
            return redirect("/admin")

        if request.method == "POST":
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "")

            if not email or not password:
                flash("Please enter both email and password", "error")
                return render_template("admin_login.html")

            user = db.session.execute(db.select(User).filter_by(email=email)).scalar()
            
            if user and user.role == UserRole.ADMIN and verify_password(password, user.password):
                login_user(user)
                flash("Login successful!", "success")
                next_page = request.args.get("next")
                return redirect(next_page) if next_page else redirect("/admin")
            else:
                flash("Invalid email or password", "error")
        
        return render_template("admin_login.html")

    @app.route("/admin/logout")
    @login_required
    def admin_logout():
        logout_user()
        flash("You have been logged out", "info")
        return redirect(url_for("admin_login"))

    # --- Setup Flask-Admin ---
    admin = Admin(app, name="Cell Annotation Admin", template_mode="bootstrap3", index_view=AuthIndexView())
    admin.add_view(UserAdmin(User, db.session, name="Users"))
    admin.add_view(PatientAdmin(Patient, db.session, name="Patients"))
    admin.add_view(SampleAdmin(Sample, db.session, name="Samples"))
    admin.add_view(SampleAnnotationAdmin(SampleAnnotation, db.session, name="Sample Annotations"))
    admin.add_view(PatientAnnotationAdmin(PatientAnnotation, db.session, name="Patient Annotations"))
    admin.add_view(UserActivityLogAdmin(UserActivityLog, db.session, name="Activity Logs"))
    admin.add_view(ModelProcessingView(name="Model Processing", endpoint="modelprocessing"))

    # register other blueprints if present
    if api_bp is not None:
        app.register_blueprint(api_bp)
    
    # Processing routes now handled by admin_views.py - no separate blueprint needed
 
    # Health check endpoint for background processes
    @app.route('/health/background-processes')
    def background_processes_health():
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, 'process_monitor.py', 'status'
            ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
            
            return {
                'status': 'ok',
                'background_processes': result.stdout.strip(),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    # Manual cleanup endpoint for stuck processing samples
    @app.route('/admin/cleanup-stuck-samples', methods=['POST'])
    @login_required
    def cleanup_stuck_samples():
        if not (current_user.is_authenticated and current_user.role == UserRole.ADMIN):
            return {'error': 'Unauthorized'}, 403
        
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, 'background_processor.py', 'cleanup'
            ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
            
            if result.returncode == 0:
                # Parse the output to extract the reset count
                output = result.stdout.strip()
                print(f"Cleanup output: {output}")
                
                # Try to extract number from output like "Reset 3 stuck processing samples"
                try:
                    # Look for a number in the output
                    import re
                    match = re.search(r'Reset (\d+) stuck', output)
                    if match:
                        reset_count = int(match.group(1))
                    else:
                        reset_count = 0
                except:
                    reset_count = 0
                
                return {
                    'status': 'success',
                    'message': output or f'Reset {reset_count} stuck processing samples',
                    'reset_count': reset_count
                }
            else:
                error_msg = result.stderr.strip() or "Unknown error"
                print(f"Cleanup failed: {error_msg}")
                return {
                    'status': 'error',
                    'message': f'Cleanup failed: {error_msg}'
                }, 500
                
        except Exception as e:
            print(f"Cleanup endpoint error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }, 500

    return app

# create app
app = create_app()

if __name__ == "__main__":
    # run dev server
    app.run(host="0.0.0.0", port=8000, debug=False)
