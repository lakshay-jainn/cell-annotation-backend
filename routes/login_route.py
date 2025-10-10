from flask import Blueprint,jsonify,request
import os
from db.models import db,User
from utilities.auth_utility import create_token,send_new_tokens, protected
from utilities.auth_utility import verify_password,hash_password,verify_recaptcha
from utilities.logging_utility import ActivityLogger
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
ACCESS_TOKEN_EXPIRY = os.environ.get("ACCESS_TOKEN_EXPIRY")

login_route_bp = Blueprint("login",__name__)


@login_route_bp.route("/register",methods=["POST"])
def register():
    
    data=request.json


    role=data['role']
    name = data['name']
    hospital = data['hospital']
    location = data['location']
    email=data['email']
    password=data['password']
    gtoken=data['gtoken']
    gresp = verify_recaptcha(gtoken, remote_ip=request.remote_addr)
    if not gresp.get('success'):
        ActivityLogger.log_activity(
            user_id=None,
            user_role=role,
            action_type="auth_event",
            action_details="register_failed_recaptcha",
            status="error",
            metadata={'email': email, 'recaptcha_response': gresp}
        )
        return jsonify({'success': False, 'message': 'recaptcha failed', 'details': gresp}), 403

    user=User.query.filter_by(email=email,role=role).first()

    if not user:
        hashed_password=hash_password(password)
        newuser=User(email=email,password=hashed_password,role=role, name=name,hospital=hospital,location=location)
        try:
            db.session.add(newuser)
            db.session.commit()
            
            ActivityLogger.log_activity(
                user_id=newuser.id,
                user_role=newuser.role.value,
                action_type="auth_event",
                action_details="register_success",
                status="success",
                metadata={'email': email, 'name': name, 'hospital': hospital, 'location': location}
            )
            
        except Exception as e:
            ActivityLogger.log_activity(
                user_id=None,
                user_role=role,
                action_type="auth_event",
                action_details="register_failed_db_error",
                status="error",
                metadata={'email': email, 'error': str(e)}
            )
            return jsonify({'message':'Internal Server Error'}),409
        
        access_token = create_token({
        'role': newuser.role.name,
        'user_id': newuser.id,
        'name': newuser.name,
        'location': newuser.location,
        'hospital': newuser.hospital
        }, JWT_SECRET_KEY, int(ACCESS_TOKEN_EXPIRY))

        message = {'message': 'User Registered Successfully'}
        response = send_new_tokens(message, access_token, 200)
        
        return response
    
    ActivityLogger.log_activity(
        user_id=None,
        user_role=role,
        action_type="auth_event",
        action_details="register_failed_user_exists",
        status="warning",
        metadata={'email': email}
    )
    return jsonify({'message':'User Already Registered'}),409 



@login_route_bp.route("/login", methods=["POST"])
def login():

    data=request.json
    
    role=data['role']
    email=data['email']
    password=data['password']

    user=User.query.filter_by(email=email).first()

    if user is None:
        ActivityLogger.log_activity(
            user_id=None,
            user_role=role,
            action_type="auth_event",
            action_details="login_failed_user_not_found",
            status="warning",
            metadata={'email': email}
        )
        print('no user')
        return jsonify({'message':'User doesnt exist'}),404
    
    print(user.email==email)
    print(verify_password(password, user.password))
    print(user.role.name == role)
    print(user.role.name)
    print(role)
    
    if (user.email==email) and verify_password(password, user.password) and (user.role.name == role):

        access_token=create_token({'role':user.role.name,'user_id':user.id, 'name': user.name, 'location': user.location, 'hospital':user.hospital},JWT_SECRET_KEY,int(ACCESS_TOKEN_EXPIRY))

        message={'message':'Login successFul'}
        response=send_new_tokens(message,access_token,200)
        
        ActivityLogger.log_activity(
            user_id=user.id,
            user_role=user.role.value,
            action_type="auth_event",
            action_details="login_success",
            status="success",
            metadata={'email': email, 'login_method': 'password'}
        )
        
        try:
            db.session.commit()
        except Exception as e:
            return jsonify({'message':'Internal Server Error'}),409
        
        return response
        
    ActivityLogger.log_activity(
        user_id=user.id if user else None,
        user_role=user.role.value if user else role,
        action_type="auth_event",
        action_details="login_failed_invalid_credentials",
        status="warning",
        metadata={'email': email, 'reason': 'invalid_credentials'}
    )
    return jsonify({'message':'Invalid Credentials'}),401 

@login_route_bp.route("/log-activity", methods=["POST"])
@protected
def log_user_activity(decoded_token):
    """Log user activity from frontend"""
    try:
        data = request.get_json()
        
        ActivityLogger.log_activity(
            user_id=decoded_token.get('user_id'),
            user_role=decoded_token.get('role'),
            action_type=data.get('action_type', 'user_action'),
            action_details=data.get('action_details', ''),
            status=data.get('status', 'info'),
            metadata=data.get('metadata', {})
        )
        
        return jsonify({'success': True}), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500