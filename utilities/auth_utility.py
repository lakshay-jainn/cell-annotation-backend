import jwt
from datetime import datetime, timezone, timedelta
from flask import make_response,jsonify,request
import os
from functools import wraps
import bcrypt
from db.models import User
import requests

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
RECAPTCHA_SECRET = os.environ.get("RECAPTCHA_SECRET")

def create_token(data,secret,expiration):
    issued_at=datetime.now(timezone.utc).timestamp()
    exp=int(issued_at) + int(expiration)
    
    return jwt.encode({**data,'iat':int(issued_at),'exp':exp},secret,algorithm='HS256')


def verify_recaptcha(token, remote_ip=None):
    payload = {
        'secret': RECAPTCHA_SECRET,
        'response': token
    }
    if remote_ip:
        payload['remoteip'] = remote_ip
    r = requests.post('https://www.google.com/recaptcha/api/siteverify', data=payload, timeout=5)
    return r.json()

def verify_token(data,secret):
    try:
        decoded_token=jwt.decode(data,secret,algorithms=['HS256'])
        return decoded_token
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:    
        return None

def send_new_tokens(message,access_token,responsecode):
    response=make_response(jsonify({**message,'token':access_token}),responsecode)

    return response

def protected(func):
    @wraps(func)
    def checker_function(*args,**kargs):
        auth_header=request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"message":"Authorization header missing", "token":False}),401

        bearer_token=auth_header.split()

        if len(bearer_token)!=2 or bearer_token[0].lower()!='bearer':
            return jsonify({"message":"invalid Authorization format",'actual':bearer_token, "token":False}),400

        token=bearer_token[1]
        
        decoded_token=verify_token(token,JWT_SECRET_KEY)
        if not decoded_token:
            return jsonify({"message":"Access token expired or invalid", "token":False}),401
        
        if isinstance(decoded_token, dict) and "user_id" in decoded_token:
            user_id = decoded_token["user_id"]
        else:
            user_id = None

        if not user_id:
            return jsonify({"message":"User Doesnt Exist", "token":False}),401

        user = User.query.filter_by(id=user_id).first()

        if not user:
            return jsonify({"message":"User Doesnt Exist", "token":False}),401

        return func(decoded_token,*args,**kargs)
    
    return checker_function

def hash_password(password):
    """Hash a password using bcrypt and return the hash as a UTF-8 string."""
    hashed_bytes = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed_bytes.decode('utf-8')


def verify_password(password, hashed):
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))