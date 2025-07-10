import streamlit as st
import sqlite3
import bcrypt
import validators
import re
import os
from datetime import datetime

class AuthHandler:
    def __init__(self):
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        self.db_path = 'data/users.db'
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # User portfolios table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                symbol TEXT,
                quantity REAL,
                buy_price REAL,
                buy_date DATE,
                notes TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        
        # User watchlists table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_watchlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                symbol TEXT,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                alert_price REAL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        
        # User preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER PRIMARY KEY,
                theme TEXT DEFAULT 'dark',
                default_mode TEXT DEFAULT 'Beginner',
                email_notifications BOOLEAN DEFAULT TRUE,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_user(self, username, email, password):
        """Register new user with validation"""
        # Input validation
        if not username or not email or not password:
            return False, "All fields are required"
            
        if len(username) < 3:
            return False, "Username must be at least 3 characters"
            
        if not validators.email(email):
            return False, "Invalid email format"
            
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
            
        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"
            
        if not re.search(r"\d", password):
            return False, "Password must contain at least one number"
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute("SELECT * FROM users WHERE username=? OR email=?", (username, email))
            if cursor.fetchone():
                conn.close()
                return False, "Username or email already exists"
                
            # Hash password and create user
            hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, hashed_pw)
            )
            
            user_id = cursor.lastrowid
            
            # Create default preferences
            cursor.execute(
                "INSERT INTO user_preferences (user_id) VALUES (?)",
                (user_id,)
            )
            
            conn.commit()
            conn.close()
            
            return True, "Registration successful! Please login."
            
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def verify_user(self, username, password):
        """Verify user credentials"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT id, password_hash, is_active FROM users WHERE username=?", 
                (username,)
            )
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return None, "User not found"
                
            user_id, stored_hash, is_active = result
            
            if not is_active:
                conn.close()
                return None, "Account is deactivated"
                
            if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                # Update last login
                cursor.execute(
                    "UPDATE users SET last_login=CURRENT_TIMESTAMP WHERE id=?",
                    (user_id,)
                )
                conn.commit()
                conn.close()
                return user_id, "Login successful"
            else:
                conn.close()
                return None, "Incorrect password"
                
        except Exception as e:
            return None, f"Login failed: {str(e)}"
    
    def get_user_info(self, user_id):
        """Get user information including preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT u.username, u.email, u.created_at, u.last_login,
                       p.theme, p.default_mode, p.email_notifications
                FROM users u
                LEFT JOIN user_preferences p ON u.id = p.user_id
                WHERE u.id = ?
            """, (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    "id": user_id,
                    "username": result[0],
                    "email": result[1],
                    "created_at": result[2],
                    "last_login": result[3],
                    "theme": result[4] or 'dark',
                    "default_mode": result[5] or 'Beginner',
                    "email_notifications": result[6] if result[6] is not None else True
                }
            return None
            
        except Exception as e:
            st.error(f"Error getting user info: {str(e)}")
            return None
    
    def update_user_preferences(self, user_id, theme=None, default_mode=None, email_notifications=None):
        """Update user preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if theme is not None:
                updates.append("theme = ?")
                params.append(theme)
            if default_mode is not None:
                updates.append("default_mode = ?")
                params.append(default_mode)
            if email_notifications is not None:
                updates.append("email_notifications = ?")
                params.append(email_notifications)
            
            if updates:
                params.append(user_id)
                cursor.execute(
                    f"UPDATE user_preferences SET {', '.join(updates)} WHERE user_id = ?",
                    params
                )
                conn.commit()
            
            conn.close()
            return True
            
        except Exception as e:
            st.error(f"Error updating preferences: {str(e)}")
            return False
    
    def save_user_portfolio(self, user_id, symbol, quantity, buy_price, buy_date, notes=""):
        """Save portfolio entry for user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO user_portfolios (user_id, symbol, quantity, buy_price, buy_date, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, symbol, quantity, buy_price, buy_date, notes))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            st.error(f"Error saving portfolio: {str(e)}")
            return False
    
    def get_user_portfolio(self, user_id):
        """Get user's portfolio"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, symbol, quantity, buy_price, buy_date, notes
                FROM user_portfolios
                WHERE user_id = ?
                ORDER BY buy_date DESC
            """, (user_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            portfolio = []
            for row in results:
                portfolio.append({
                    "id": row[0],
                    "symbol": row[1],
                    "quantity": row[2],
                    "buy_price": row[3],
                    "buy_date": row[4],
                    "notes": row[5] or ""
                })
            
            return portfolio
            
        except Exception as e:
            st.error(f"Error getting portfolio: {str(e)}")
            return []
    
    def delete_portfolio_entry(self, user_id, entry_id):
        """Delete a portfolio entry"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "DELETE FROM user_portfolios WHERE id = ? AND user_id = ?",
                (entry_id, user_id)
            )
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            st.error(f"Error deleting portfolio entry: {str(e)}")
            return False
    
    def add_to_watchlist(self, user_id, symbol, alert_price=None):
        """Add stock to user's watchlist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if already in watchlist
            cursor.execute(
                "SELECT id FROM user_watchlists WHERE user_id = ? AND symbol = ?",
                (user_id, symbol)
            )
            
            if cursor.fetchone():
                conn.close()
                return False, "Stock already in watchlist"
            
            cursor.execute("""
                INSERT INTO user_watchlists (user_id, symbol, alert_price)
                VALUES (?, ?, ?)
            """, (user_id, symbol, alert_price))
            
            conn.commit()
            conn.close()
            return True, "Added to watchlist successfully"
            
        except Exception as e:
            return False, f"Error adding to watchlist: {str(e)}"
    
    def get_user_watchlist(self, user_id):
        """Get user's watchlist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, symbol, added_date, alert_price
                FROM user_watchlists
                WHERE user_id = ?
                ORDER BY added_date DESC
            """, (user_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            watchlist = []
            for row in results:
                watchlist.append({
                    "id": row[0],
                    "symbol": row[1],
                    "added_date": row[2],
                    "alert_price": row[3]
                })
            
            return watchlist
            
        except Exception as e:
            st.error(f"Error getting watchlist: {str(e)}")
            return []
