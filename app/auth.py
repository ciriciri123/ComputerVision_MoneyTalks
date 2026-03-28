"""
app/auth.py — MoneyTalks Admin Authentication
=============================================
Handles all authentication logic for the administrator portal:
    - Flask-Login user_loader (reconstructs the user object from session)
    - AdminUser class (the Flask-Login User model backed by Supabase)
    - Login Blueprint (GET /admin/login, POST /admin/login)
    - Logout route (POST /admin/logout)

How authentication flows through the stack:

    1. Admin submits the login form → POST /admin/login
    2. auth.py calls supabase_client.get_db().get_admin_by_email(email)
    3. bcrypt.checkpw() compares the submitted password to the stored hash
    4. On success, Flask-Login calls login_user(AdminUser(record))
    5. Flask sets an encrypted session cookie containing admin_id
    6. On every subsequent request to an @login_required route:
         a. Flask-Login reads admin_id from the session cookie
         b. Calls load_admin_user(admin_id) → supabase_client.get_admin_by_id()
         c. Reconstructs the AdminUser object and attaches it to current_user

Why bcrypt here and not in supabase_client.py?
    bcrypt is a business-logic concern (password policy), not a data concern.
    supabase_client.py only handles storage and retrieval — it returns the raw
    hash and lets auth.py decide how to verify it.
"""

import logging

import bcrypt
from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import UserMixin, current_user, login_required, login_user, logout_user

from app import login_manager, limiter
from app.services.supabase_client import get_db, AdminRecord, SupabaseError

logger = logging.getLogger(__name__)

# Blueprint: all routes in this file are prefixed with /admin (set in __init__.py)
auth_bp = Blueprint("auth", __name__, template_folder="templates")


# =============================================================================
# FLASK-LOGIN USER MODEL
# =============================================================================

class AdminUser(UserMixin):
    """
    Flask-Login User model backed by an AdminRecord from Supabase.

    Flask-Login requires this class to implement four properties:
        is_authenticated, is_active, is_anonymous, get_id()
    UserMixin provides sensible defaults for all four; we only need to
    supply the data and override get_id() to return admin_id.

    This class is intentionally thin. It wraps the AdminRecord dataclass
    and exposes its fields as attributes. Business logic stays in auth.py.

    Attributes:
        admin_id: Primary key from the Administrator table.
        email:    Administrator's email address.
        username: Display name shown on the dashboard.
        phone:    Optional phone number.

    Note: The password hash is NOT stored on this object. It is only read
    during the login check and discarded immediately after bcrypt.checkpw().
    """

    def __init__(self, record: AdminRecord) -> None:
        self.admin_id = record.admin_id
        self.email    = record.email
        self.username = record.username
        self.phone    = record.phone
        # password hash is NOT stored — only used once during login

    def get_id(self) -> str:
        """
        Return the unique identifier stored in the session cookie.

        Flask-Login stores this string in the session and passes it back
        to load_admin_user() on every subsequent request.
        """
        return self.admin_id


# =============================================================================
# FLASK-LOGIN USER LOADER
# =============================================================================

@login_manager.user_loader
def load_admin_user(admin_id: str) -> AdminUser | None:
    """
    Called by Flask-Login on EVERY request to a protected route.

    Flask-Login reads the admin_id from the session cookie and calls this
    function to reconstruct the current_user object. If this returns None,
    Flask-Login treats the session as invalid and forces a re-login.

    Performance note: this makes one Supabase round-trip per protected request.
    For MoneyTalks' expected admin traffic (1–3 concurrent admins), this is
    acceptable. If it becomes a bottleneck, add a short-lived in-memory cache
    keyed by admin_id with a TTL of ~30 seconds.

    Args:
        admin_id: The value returned by AdminUser.get_id() at login time.

    Returns:
        AdminUser if the admin_id exists in the database, None otherwise.
    """
    try:
        record = get_db().get_admin_by_id(admin_id)
        if record is None:
            logger.warning("user_loader: admin_id '%s' not found in database.", admin_id)
            return None
        return AdminUser(record)
    except SupabaseError as exc:
        # Log the error but return None so Flask-Login degrades gracefully
        # (redirects to login) rather than returning a 500 error.
        logger.error("user_loader: Supabase error for admin_id '%s': %s", admin_id, exc)
        return None


# =============================================================================
# LOGIN / LOGOUT ROUTES
# =============================================================================

@auth_bp.route("/login", methods=["GET"])
def login():
    """
    GET /admin/login — Render the admin login page.

    If the admin already has a valid session, skip the login form and redirect
    directly to the dashboard.
    """
    if current_user.is_authenticated:
        return redirect(url_for("auth.dashboard"))
    return render_template("admin/login.html")


@auth_bp.route("/login", methods=["POST"])
@limiter.limit("5 per minute")   # Rate-limit: 5 login attempts per IP per minute
def login_post():
    """
    POST /admin/login — Validate credentials and create a session.

    Security measures applied here:
        - Rate limiting (5/min/IP) via @limiter.limit (Flask-Limiter).
        - Constant-time password comparison via bcrypt.checkpw().
        - Generic error message ("Invalid email or password") to prevent
          email enumeration (attacker cannot tell if email exists).
        - Supabase query errors are caught and logged server-side; the user
          sees the same generic error, not internal details.
    """
    email    = request.form.get("email",    "").strip().lower()
    password = request.form.get("password", "").strip()

    # Basic presence check before hitting the database
    if not email or not password:
        flash("Please enter both email and password.", "error")
        return render_template("admin/login.html"), 400

    try:
        record = get_db().get_admin_by_email(email)
    except SupabaseError as exc:
        logger.error("Login: Supabase error fetching admin '%s': %s", email, exc)
        flash("A server error occurred. Please try again.", "error")
        return render_template("admin/login.html"), 500

    # Deliberate: use the same branch for "not found" and "wrong password"
    # so the response time is similar and no information is leaked.
    if record is None or not _verify_password(password, record.password):
        logger.warning("Failed login attempt for email: %s", email)
        flash("Invalid email or password.", "error")
        return render_template("admin/login.html"), 401

    # ── Login successful ──────────────────────────────────────────────────────
    admin = AdminUser(record)
    login_user(admin, remember=False)   # no persistent "remember me" cookie
    logger.info("Admin logged in: %s (%s)", record.username, record.admin_id)

    # Redirect to the page the admin originally tried to access (if any),
    # defaulting to the dashboard. Validate next to prevent open redirects.
    next_page = request.args.get("next")
    if next_page and _is_safe_redirect(next_page):
        return redirect(next_page)
    return redirect(url_for("auth.dashboard"))


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    """
    POST /admin/logout — Destroy the admin session.

    Uses POST (not GET) to prevent logout via prefetching or CSRF.
    The template must include a CSRF-protected form with method="POST".
    """
    logger.info("Admin logged out: %s", current_user.admin_id)
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))


# =============================================================================
# DASHBOARD & ADMIN PAGE ROUTES
# =============================================================================

@auth_bp.route("/dashboard")
@login_required
def dashboard():
    """
    GET /admin/dashboard — Render the admin overview page.

    Fetches summary data from Supabase:
        - Total number of scanned images
        - Currently active model version info
    """
    try:
        db          = get_db()
        scan_count  = db.get_scan_count()
        active_model = db.get_active_model()
    except SupabaseError as exc:
        logger.error("Dashboard: Supabase error: %s", exc)
        scan_count   = "N/A"
        active_model = None

    return render_template(
        "admin/dashboard.html",
        admin        = current_user,
        scan_count   = scan_count,
        active_model = active_model,
    )


@auth_bp.route("/images")
@login_required
def images():
    """
    GET /admin/images — Paginated list of ScannedMoney records.

    Query parameters:
        from  (str, YYYY-MM-DD): Start of date range filter.
        to    (str, YYYY-MM-DD): End of date range filter.
        page  (int, default 1): Page number.
    """
    from datetime import date as date_type

    def _parse_date(param: str | None) -> date_type | None:
        if not param:
            return None
        try:
            return date_type.fromisoformat(param)
        except ValueError:
            return None

    from_date = _parse_date(request.args.get("from"))
    to_date   = _parse_date(request.args.get("to"))
    page      = max(1, int(request.args.get("page", 1)))

    try:
        records, total = get_db().get_scan_records(
            from_date=from_date,
            to_date=to_date,
            page=page,
            page_size=50,
        )
    except SupabaseError as exc:
        logger.error("Images page: Supabase error: %s", exc)
        records, total = [], 0

    return render_template(
        "admin/images.html",
        records   = records,
        total     = total,
        page      = page,
        from_date = from_date,
        to_date   = to_date,
        admin     = current_user,
    )


@auth_bp.route("/models")
@login_required
def models():
    """
    GET /admin/models — List all model versions.
    """
    try:
        model_list = get_db().get_all_models()
    except SupabaseError as exc:
        logger.error("Models page: Supabase error: %s", exc)
        model_list = []

    return render_template(
        "admin/models.html",
        models = model_list,
        admin  = current_user,
    )


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _verify_password(plain: str, hashed: str) -> bool:
    """
    Compare a plain-text password against a bcrypt hash.

    Uses bcrypt.checkpw() which is constant-time and safe against timing
    attacks. The comparison takes ~100-200ms at cost factor 12 — this is
    intentional and desirable as it limits brute-force speed.

    Args:
        plain:  The password string submitted in the login form.
        hashed: The bcrypt hash string stored in the Administrator table.

    Returns:
        True if the password matches, False otherwise.
    """
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception as exc:
        # bcrypt.checkpw raises on malformed hashes (e.g. if the DB column is
        # accidentally storing a plain-text password). Log and treat as failure.
        logger.error("bcrypt.checkpw raised an exception: %s", exc)
        return False


def _is_safe_redirect(url: str) -> bool:
    """
    Return True if the 'next' redirect URL is safe (relative path only).

    Prevents open redirect attacks where an attacker crafts a login URL like:
        /admin/login?next=https://evil.com
    and tricks the admin into being redirected to an external site.

    Only relative paths (starting with /) are allowed.
    """
    return url.startswith("/") and not url.startswith("//")
