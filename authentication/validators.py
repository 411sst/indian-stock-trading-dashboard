import re
import validators

def validate_email(email):
    """Validate email format and basic checks"""
    if not email:
        return False, "Email cannot be empty"
        
    if len(email) < 5:
        return False, "Email too short"
        
    if not validators.email(email):
        return False, "Please enter a valid email address"
    
    # Check for common email providers
    valid_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'icloud.com']
    domain = email.split('@')[1].lower() if '@' in email else ''
    
    if domain not in valid_domains and not any(d in domain for d in ['.com', '.org', '.net', '.edu']):
        return False, "Please use a valid email provider"
    
    return True, ""

def validate_password(password, confirm_password=None):
    """Validate password strength with detailed feedback"""
    if not password:
        return False, "Password cannot be empty"
        
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
        
    if len(password) > 128:
        return False, "Password too long (max 128 characters)"
        
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter (A-Z)"
        
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter (a-z)"
        
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number (0-9)"
        
    # Check for special characters (optional but recommended)
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        # This is optional, so we'll just give a warning in the UI
        pass
        
    # Check for common weak passwords
    weak_patterns = [
        r"^password",
        r"^123456",
        r"^qwerty",
        r"^abc123",
        r"^admin"
    ]
    
    for pattern in weak_patterns:
        if re.search(pattern, password.lower()):
            return False, "Password is too common. Please choose a stronger password"
    
    # Confirm password check
    if confirm_password and password != confirm_password:
        return False, "Passwords do not match"
        
    return True, ""

def validate_username(username):
    """Validate username format and requirements"""
    if not username:
        return False, "Username cannot be empty"
        
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
        
    if len(username) > 30:
        return False, "Username too long (max 30 characters)"
        
    if not re.match(r"^[a-zA-Z0-9_-]+$", username):
        return False, "Username can only contain letters, numbers, underscore, and dash"
        
    if username.lower() in ['admin', 'root', 'user', 'test', 'guest', 'null', 'undefined']:
        return False, "Username not allowed. Please choose a different one"
        
    return True, ""

def get_password_strength_score(password):
    """Calculate password strength score (0-100)"""
    score = 0
    
    # Length bonus
    if len(password) >= 8:
        score += 20
    if len(password) >= 12:
        score += 10
    if len(password) >= 16:
        score += 10
        
    # Character variety
    if re.search(r"[a-z]", password):
        score += 10
    if re.search(r"[A-Z]", password):
        score += 10
    if re.search(r"\d", password):
        score += 10
    if re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        score += 15
        
    # Patterns and complexity
    if not re.search(r"(.)\1{2,}", password):  # No 3+ repeated chars
        score += 10
    if not re.search(r"(012|123|234|345|456|567|678|789|890)", password):  # No sequences
        score += 5
    if not re.search(r"(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)", password.lower()):
        score += 5
        
    return min(100, score)

def get_password_strength_text(score):
    """Get text description of password strength"""
    if score < 30:
        return "Very Weak", "red"
    elif score < 50:
        return "Weak", "orange"
    elif score < 70:
        return "Moderate", "yellow"
    elif score < 85:
        return "Strong", "lightgreen"
    else:
        return "Very Strong", "green"
