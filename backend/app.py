import os
import sys
import io
import base64
import json
import secrets
import datetime
import urllib.parse
import requests
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from torchvision import transforms

# Load environment variables from .env file
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# Add project root to sys.path
sys.path.append(BASE_DIR)
from notebooks.data_preprocessing import CLAHETransform
from notebooks.model_training import BoneCancerClassifier, GradCAMEngine

app = FastAPI(
    title="Bone Cancer Detection & 3D Explainability API",
    description="Advanced Deep Learning REST API for Bone Cancer Detection using PyTorch & Grad-CAM",
    version="1.0.0"
)

# Enable CORS for frontend dynamic communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))

MODEL_PATH = os.path.join(BASE_DIR, 'saved_models', 'bone_cancer_model.pth')
SAMPLE_DIR = os.path.join(BASE_DIR, 'dataset', 'processed', 'val')
USERS_FILE = os.path.join(BASE_DIR, 'saved_models', 'users.json')

# Global variables for model and grad-cam engine
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = None
GRAD_CAM = None
CLASS_NAMES = ['chondrosarcoma', 'ewing_sarcoma', 'normal', 'osteosarcoma']

# OAuth 2.0 Credentials (Server-side configuration loaded from .env)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/api/auth/google/callback")

# User Database Persistence Functions
def load_users_db():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading users database: {e}")
            return {}
    return {}

def save_users_db(users_data):
    try:
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users_data, f, indent=2)
    except Exception as e:
        print(f"Error saving users database: {e}")

def find_or_create_user(google_id: str, email: str, name: str, picture: str):
    users = load_users_db()
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    if google_id in users:
        user = users[google_id]
        user['last_login'] = now_iso
        if name:
            user['name'] = name
        if picture:
            user['picture'] = picture
    else:
        user = {
            'id': google_id,
            'email': email,
            'name': name or (email.split('@')[0] if email else "Google User"),
            'picture': picture or "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
            'created_at': now_iso,
            'last_login': now_iso
        }
        users[google_id] = user
        
    save_users_db(users)
    return user

@app.get("/api/config")
def get_config():
    """
    Exposes only the public Client ID for client SDK initialization.
    Client secret is never exposed.
    """
    return {
        "google_client_id": GOOGLE_CLIENT_ID
    }

@app.get("/api/auth/google/login")
def google_login_redirect(request: Request):
    """
    Initiates standard Google OAuth 2.0 Authorization Code Flow.
    Redirects user browser to Google's official account selection & login screen.
    """
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth environment variables (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET) are missing from .env."
        )
        
    state = secrets.token_urlsafe(32)
    google_auth_endpoint = "https://accounts.google.com/o/oauth2/v2/auth"
    
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
        "access_type": "online"
    }
    
    auth_url = f"{google_auth_endpoint}?{urllib.parse.urlencode(params)}"
    response = RedirectResponse(url=auth_url, status_code=307)
    response.set_cookie(key="oauth_state", value=state, httponly=True, max_age=600, samesite="lax")
    return response

@app.get("/api/auth/google/callback")
async def google_auth_callback(
    request: Request,
    code: str = None,
    state: str = None,
    error: str = None,
    oauth_state: str = Cookie(default=None)
):
    """
    OAuth 2.0 Redirect Callback Endpoint.
    Exchanges code for access token server-side, validates identity, creates user session.
    """
    if error:
        print(f"Google OAuth Cancellation/Error: {error}")
        return RedirectResponse(url=f"/?auth_error={urllib.parse.quote(error)}", status_code=307)
        
    if not code:
        return RedirectResponse(url="/?auth_error=missing_code", status_code=307)
        
    # Server-Side Code Exchange with Google Token Endpoint
    token_url = "https://oauth2.googleapis.com/token"
    token_payload = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET, # Server-side only!
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": GOOGLE_REDIRECT_URI
    }
    
    try:
        token_resp = requests.post(token_url, data=token_payload, timeout=10)
        if token_resp.status_code != 200:
            print(f"Token Exchange Error: {token_resp.text}")
            return RedirectResponse(url="/?auth_error=token_exchange_failed", status_code=307)
            
        tokens = token_resp.json()
        access_token = tokens.get("access_token")
        
        # Fetch Google user info securely using access token
        userinfo_resp = requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10
        )
        if userinfo_resp.status_code != 200:
            return RedirectResponse(url="/?auth_error=userinfo_failed", status_code=307)
            
        google_user = userinfo_resp.json()
        
        # Save or update user profile
        user = find_or_create_user(
            google_id=google_user.get("sub"),
            email=google_user.get("email"),
            name=google_user.get("name"),
            picture=google_user.get("picture")
        )
        
        user_json_str = urllib.parse.quote(json.dumps(user))
        response = RedirectResponse(url=f"/?auth_success=1&user={user_json_str}", status_code=307)
        response.set_cookie(
            key="user_session_id",
            value=user['id'],
            httponly=True,
            max_age=86400 * 30,
            samesite="lax"
        )
        response.delete_cookie("oauth_state")
        return response
    except Exception as e:
        print(f"OAuth Callback Exception: {e}")
        return RedirectResponse(url=f"/?auth_error={urllib.parse.quote(str(e))}", status_code=307)

@app.post("/api/auth/google")
async def verify_google_token(payload: dict, response: Response):
    """
    Verifies Google ID Token from Google Identity Services client SDK.
    """
    token = payload.get("credential")
    if not token:
        raise HTTPException(status_code=400, detail="Missing credential token.")
        
    try:
        resp = requests.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={token}", timeout=5)
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid Google OAuth token.")
            
        user_info = resp.json()
        aud = user_info.get("aud")
        if aud and GOOGLE_CLIENT_ID and aud != GOOGLE_CLIENT_ID:
            print(f"Warning: Token audience {aud} differs from GOOGLE_CLIENT_ID {GOOGLE_CLIENT_ID}")
            
        user = find_or_create_user(
            google_id=user_info.get("sub"),
            email=user_info.get("email"),
            name=user_info.get("name"),
            picture=user_info.get("picture")
        )
        
        res = JSONResponse({
            "authenticated": True,
            "user": user
        })
        res.set_cookie(
            key="user_session_id",
            value=user['id'],
            httponly=True,
            max_age=86400 * 30,
            samesite="lax"
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")

@app.get("/api/auth/me")
def get_current_user(user_session_id: str = Cookie(default=None)):
    """
    Returns current authenticated user details from session cookie.
    """
    if not user_session_id:
        return {"authenticated": False, "user": None}
        
    users = load_users_db()
    user = users.get(user_session_id)
    if not user:
        return {"authenticated": False, "user": None}
        
    return {"authenticated": True, "user": user}

@app.post("/api/auth/logout")
def logout_user():
    """
    Logs out the user by clearing the session cookie.
    """
    res = JSONResponse({"status": "success", "message": "Logged out successfully"})
    res.delete_cookie("user_session_id")
    return res

CLINICAL_RECOMMENDATIONS = {
    'normal': {
        'risk_level': 'Low Risk',
        'badge_color': '#10b981', # Green
        'description': 'No significant radiological evidence of osteolytic or sclerotic malignant bone lesions detected.',
        'action': 'Routine clinical follow-up as indicated by symptoms.'
    },
    'osteosarcoma': {
        'risk_level': 'Critical / High Risk',
        'badge_color': '#ef4444', # Red
        'description': 'Sclerotic lesion with characteristic sunburst periosteal reaction and Codman triangle features suspicious for Osteosarcoma.',
        'action': 'Urgent Orthopedic Oncology referral, Contrast MRI of affected limb, and Biopsy.'
    },
    'ewing_sarcoma': {
        'risk_level': 'Critical / High Risk',
        'badge_color': '#f97316', # Orange
        'description': 'Permeative lytic lesion with concentric onion-skin periosteal lamination in diaphyseal bone.',
        'action': 'Immediate Specialist consultation, Staging CT/PET scan, and Core needle biopsy.'
    },
    'chondrosarcoma': {
        'risk_level': 'High Risk',
        'badge_color': '#eab308', # Yellow
        'description': 'Cartilaginous matrix calcification with characteristic ring-and-arc calcification and endosteal scalloping.',
        'action': 'Orthopedic Surgical evaluation, CT chest/pelvis for staging, and Surgical resection planning.'
    }
}

def load_model_pipeline():
    global MODEL, GRAD_CAM, CLASS_NAMES
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            CLASS_NAMES = checkpoint.get('class_names', CLASS_NAMES)
            
            MODEL = BoneCancerClassifier(num_classes=len(CLASS_NAMES), pretrained=False)
            MODEL.load_state_dict(checkpoint['model_state_dict'])
            MODEL.to(DEVICE)
            MODEL.eval()
            
            GRAD_CAM = GradCAMEngine(MODEL, target_layer_name='layer4')
            print(f"Loaded trained model successfully from {MODEL_PATH}")
            return
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")
            
    # Fallback to untrained model if checkpoint not ready yet
    MODEL = BoneCancerClassifier(num_classes=len(CLASS_NAMES), pretrained=True)
    MODEL.to(DEVICE)
    MODEL.eval()
    GRAD_CAM = GradCAMEngine(MODEL, target_layer_name='layer4')
    print("Initialized default model architecture.")

@app.on_event("startup")
async def startup_event():
    load_model_pipeline()

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
        "model_loaded": MODEL is not None
    }

def process_image_tensor(pil_img):
    """
    Applies CLAHE enhancement & PyTorch image normalization.
    """
    clahe_enhancer = CLAHETransform(clip_limit=2.0)
    enhanced_pil = clahe_enhancer(pil_img)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(enhanced_pil).unsqueeze(0).to(DEVICE)
    return enhanced_pil, tensor

@app.post("/api/predict")
async def predict_bone_xray(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not a valid image.")
        
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Ensure model is ready
        if MODEL is None or GRAD_CAM is None:
            load_model_pipeline()
            
        enhanced_pil, input_tensor = process_image_tensor(pil_img)
        
        # Grad-CAM heatmap & probabilities
        heatmap, pred_idx, probs = GRAD_CAM.generate_heatmap(input_tensor)
        pred_class = CLASS_NAMES[pred_idx]
        top_confidence = float(probs[pred_idx])
        
        # Create visual Grad-CAM heatmap overlay
        orig_np = np.array(enhanced_pil.resize((224, 224)))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(orig_np, 0.55, heatmap_rgb, 0.45, 0)
        
        # Base64 encode images for browser rendering
        def pil_to_b64(img_arr):
            buffered = io.BytesIO()
            Image.fromarray(img_arr).save(buffered, format="JPEG", quality=90)
            return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')
            
        enhanced_b64 = pil_to_b64(orig_np)
        overlay_b64 = pil_to_b64(overlay)
        
        clinical_info = CLINICAL_RECOMMENDATIONS.get(pred_class, CLINICAL_RECOMMENDATIONS['normal'])
        
        probabilities_dict = {
            cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)
        }
        
        return JSONResponse(content={
            "prediction": pred_class,
            "confidence": round(top_confidence * 100, 2),
            "risk_level": clinical_info['risk_level'],
            "badge_color": clinical_info['badge_color'],
            "description": clinical_info['description'],
            "recommended_action": clinical_info['action'],
            "probabilities": probabilities_dict,
            "heatmap_overlay_b64": overlay_b64,
            "enhanced_xray_b64": enhanced_b64
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic error: {str(e)}")

@app.get("/api/samples")
def get_sample_xrays():
    """
    Returns list of sample X-ray images for immediate browser demo.
    """
    samples = []
    if os.path.exists(SAMPLE_DIR):
        for cls in CLASS_NAMES:
            cls_dir = os.path.join(SAMPLE_DIR, cls)
            if os.path.exists(cls_dir):
                files = [f for f in os.listdir(cls_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    img_path = os.path.join(cls_dir, files[0])
                    with open(img_path, "rb") as f:
                        b64_data = "data:image/png;base64," + base64.b64encode(f.read()).decode('utf-8')
                    samples.append({
                        "id": cls,
                        "class_name": cls,
                        "title": f"Sample {cls.replace('_', ' ').title()}",
                        "image_b64": b64_data
                    })
    return {"samples": samples}

# ==========================================
# BONEAI MEDICAL ASSISTANT CHAT SERVICE
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class AssistantChatMessage(BaseModel):
    role: str
    content: str

class AssistantChatPayload(BaseModel):
    message: str
    analysis_context: Optional[Dict[str, Any]] = None
    conversation: Optional[List[AssistantChatMessage]] = []

MEDICAL_ASSISTANT_SYSTEM_PROMPT = """
You are BoneAI Assistant, an empathetic, supportive, calm, and medically responsible AI information assistant integrated into OsteoVision AI (a Bone Cancer Detection & Explainability application).

CRITICAL MEDICAL RESPONSIBILITY & SAFETY GUARDRAILS:
1. EMPATHY & EMOTIONAL ADAPTABILITY: Respond with warmth, patient understanding, and non-judgmental support. If the user expresses fear, anxiety, or panic ("I'm scared", "Am I going to die?", "I'm panicking"), acknowledge their concern first with empathy. Avoid robotic or dismissive phrasing.
2. NON-DIAGNOSTIC & NO FALSE CERTAINTY: You are an educational screening assistant, NOT an autonomous doctor or diagnostic system. NEVER claim that an AI scan classification is a confirmed medical diagnosis. Explicitly distinguish model prediction confidence from medical diagnosis certainty (e.g. explain that a 94% model confidence describes algorithm classification certainty, NOT a 94% medical probability of having cancer).
3. NO PRESCRIBING & NO MEDICAL TREATMENT PLANS: Never prescribe medications, dosages, or personalized treatment regimens. Explain generally that medical management depends on confirmed pathology biopsy, tumor stage, location, specialist evaluation, and individual patient health.
4. URGENT TRIAGE SAFEGUARD: If the user describes severe or emergency symptoms (severe sudden pain, uncontrolled bleeding, shortness of breath, fainting, sudden loss of limb sensation/movement), immediately advise them to seek emergency/urgent medical care. Do not attempt diagnostic evaluation of emergencies.
5. APPOINTMENT PREPARATION: Help users prepare for doctor appointments by explaining medical terms in simple plain language, explaining scan results accurately based ONLY on the provided scan context, and offering useful questions for them to ask their physician.
6. NO HALLUCINATIONS: Discuss scan results ONLY using the provided analysis context. Do not invent abnormalities, visual findings, or fake pathology reports.
"""

def generate_medical_fallback_response(user_msg: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Intelligent Medical Knowledge Fallback Engine when Gemini API Key is unconfigured or offline.
    """
    msg_lower = user_msg.lower()
    
    # 1. Emergency Triage Check
    emergency_keywords = ['chest pain', 'bleeding', 'can\'t breathe', 'difficulty breathing', 'fainted', 'fainting', 'unconscious', 'severe sudden pain', 'paralysis']
    if any(k in msg_lower for k in emergency_keywords):
        return {
            "reply": "⚠️ **Urgent Medical Attention Recommended**\n\nThe symptoms you've described could represent a serious medical emergency requiring immediate evaluation. Please contact emergency services (such as 911 or your local emergency room) or go to the nearest urgent care facility right away rather than waiting for an online conversation.\n\nYour health and immediate safety are the highest priority.",
            "suggested_questions": ["What emergency contacts should I call?", "When should I seek urgent care?"]
        }
        
    # 2. Emotional Anxiety Handling
    if any(k in msg_lower for k in ['scared', 'worried', 'die', 'panic', 'panicking', 'afraid', 'frightened']):
        reply = "I can completely understand why seeing imaging results or thinking about bone cancer feels frightening. It is very natural to feel anxious right now.\n\n"
        if context and context.get("prediction"):
            pred = context.get("prediction", "").replace("_", " ").title()
            conf = context.get("confidence", "")
            reply += f"The most important thing to remember is that this AI analysis classified your scan as **{pred}** with **{conf}% model confidence**. This is an automated screening prediction—it is **NOT a confirmed medical diagnosis**.\n\n"
        else:
            reply += "Please remember that AI screening tools provide automated analysis to assist doctors, but **only a qualified healthcare professional** (like an orthopedic oncologist or radiologist) can evaluate your health and make a diagnosis.\n\n"
        reply += "Taking things one step at a time is the best path forward. Would you like me to help you make a list of questions to take to a doctor?"
        return {
            "reply": reply,
            "suggested_questions": ["What questions should I ask my doctor?", "What does my result mean?", "What are reasonable next steps?"]
        }

    # 3. Result & Confidence Explanation
    if any(k in msg_lower for k in ['result', 'mean', 'confidence', 'score', 'definitely', 'cancer']):
        if context and context.get("prediction"):
            pred = context.get("prediction", "").replace("_", " ").title()
            conf = context.get("confidence", 0)
            risk = context.get("risk_level", "Clinical Review Recommended")
            desc = context.get("description", "")
            action = context.get("recommended_action", "")
            
            reply = f"**Understanding Your Scan Analysis:**\n\n"
            reply += f"• **AI Model Prediction:** {pred}\n"
            reply += f"• **Model Confidence:** {conf}%\n"
            reply += f"• **Clinical Severity Assessment:** {risk}\n\n"
            reply += f"**What Model Confidence Means:**\n"
            reply += f"A model confidence of {conf}% describes how strongly the deep learning algorithm matched visual patterns in your image to known training cases. **It does NOT mean there is a {conf}% medical probability that you have cancer.**\n\n"
            reply += f"**Medical Interpretation:**\n{desc}\n\n"
            reply += f"**Recommended Next Step:**\n{action}"
            return {
                "reply": reply,
                "suggested_questions": ["What questions should I ask my doctor?", "Why did the AI classify my scan this way?", "What does the heatmap mean?"]
            }
        else:
            return {
                "reply": "To explain your specific scan result, please upload a bone X-Ray scan first or select a pathology sample from the dashboard! Once uploaded, I can explain the AI prediction, confidence score, and clinical recommendations in simple terms.",
                "suggested_questions": ["How do I upload a scan?", "What bone conditions does this AI analyze?"]
            }

    # 4. Pathology Definitions
    if 'osteosarcoma' in msg_lower:
        return {
            "reply": "**What is Osteosarcoma?**\n\nOsteosarcoma is the most common type of primary bone cancer, most frequently developing in the growing osteoblast cells of long bones (such as around the knee or upper arm).\n\n• **Key Features:** It often presents on radiological imaging with osteolytic or sclerotic lesions, sometimes displaying a characteristic 'sunburst' periosteal pattern.\n• **How It Is Evaluated:** Diagnosis requires specialized evaluation by an orthopedic oncologist, contrast MRI/CT scans, and a tissue biopsy.\n• **Important Context:** An AI screening pattern suggests visual features associated with osteosarcoma, but a biopsy and physician review are required for verification.",
            "suggested_questions": ["What questions should I ask my doctor?", "What tests does a doctor perform to diagnose bone cancer?"]
        }

    if 'ewing' in msg_lower:
        return {
            "reply": "**What is Ewing Sarcoma?**\n\nEwing Sarcoma is a rare type of bone or soft tissue tumor that most commonly affects children, teenagers, and young adults.\n\n• **Key Features:** It typically arises in the pelvis, thigh bone (femur), or shin bone (tibia), and often shows an 'onion-skin' periosteal reaction on radiological imaging.\n• **Evaluation:** Diagnosis involves comprehensive diagnostic imaging (MRI/PET-CT), tissue biopsy, and evaluation by a pediatric or adult oncology team.",
            "suggested_questions": ["What should I do next?", "What questions should I ask my doctor?"]
        }

    if 'chondrosarcoma' in msg_lower:
        return {
            "reply": "**What is Chondrosarcoma?**\n\nChondrosarcoma is a primary bone cancer that develops in the cartilage cells (chondrocytes), most commonly affecting adults in the pelvis, femur, or shoulder blade.\n\n• **Key Features:** It often presents with ring-and-arc calcifications or cartilage matrix changes on radiological scans.\n• **Evaluation:** Radiologists assess lesion grade using MRI/CT imaging, followed by specialist evaluation and biopsy when indicated.",
            "suggested_questions": ["What questions should I ask my doctor?", "What does the AI confidence score mean?"]
        }

    if 'heatmap' in msg_lower or 'grad-cam' in msg_lower or 'cam' in msg_lower:
        return {
            "reply": "**Understanding the Grad-CAM AI Heatmap:**\n\nGrad-CAM (Gradient-Weighted Class Activation Mapping) is an explainable AI technology that highlights the specific regions of your X-Ray that influenced the neural network's classification.\n\n• **Warm Colors (Red / Yellow):** Indicate high-attention regions where the AI detected structural or radiological patterns resembling bone pathology.\n• **Cool Colors (Blue / Transparent):** Indicate background or normal tissue regions.\n\nThis visual map helps radiologists verify that the AI focused on valid anatomical features rather than image artifacts.",
            "suggested_questions": ["What does my result mean?", "What questions should I ask my doctor?"]
        }

    # 5. Doctor Appointment Preparation
    if any(k in msg_lower for k in ['doctor', 'questions', 'appointment', 'ask', 'prepare', 'next step', 'what to do']):
        pred_text = context.get("prediction", "bone imaging").replace("_", " ").title() if context else "my scan"
        return {
            "reply": f"**Recommended Questions for Your Doctor Appointment:**\n\nHere is a list of structured questions you can print or take to your medical appointment concerning {pred_text}:\n\n1. *\"Can you review this bone X-Ray scan with me and explain what structural features you observe?\"*\n2. *\"Does this image show any evidence of osteolytic or sclerotic lesions, or periosteal reaction?\"*\n3. *\"Would additional diagnostic imaging (such as an MRI, CT scan, or bone scan) be appropriate?\"*\n4. *\"What non-cancerous conditions (such as bone infections, fractures, or benign lesions) could cause similar imaging findings?\"*\n5. *\"Is a consultation with an orthopedic specialist or a tissue biopsy recommended at this stage?\"*\n6. *\"What specific symptoms or changes should prompt me to seek urgent medical care?\"*",
            "suggested_questions": ["What does my result mean?", "Can you explain osteosarcoma in simple terms?"]
        }

    # 6. Default Medical Support Reply
    return {
        "reply": "I am here to help you understand your bone imaging scan, explain medical terminology in simple language, discuss your concerns, and help you prepare for a medical appointment.\n\n**Please Note:** I am an educational AI assistant and cannot provide a medical diagnosis or prescribe treatments. All scan findings should be evaluated by a qualified healthcare professional.\n\nHow can I help you today?",
        "suggested_questions": ["What does my result mean?", "Explain my confidence score", "What should I ask my doctor?", "What is osteosarcoma?"]
    }

@app.post("/api/assistant/chat")
async def assistant_chat_endpoint(payload: AssistantChatPayload):
    """
    BoneAI Medical Assistant Chat Endpoint.
    Sends user query and scan context to Gemini API if configured,
    or falls back seamlessly to the intelligent medical response engine.
    """
    user_msg = payload.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    ctx = payload.analysis_context or {}
    
    # 1. Check if Gemini API Key is configured for direct LLM invocation
    if GEMINI_API_KEY:
        try:
            gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
            
            prompt_content = f"{MEDICAL_ASSISTANT_SYSTEM_PROMPT}\n\nCURRENT ANALYSIS CONTEXT:\n{json.dumps(ctx, indent=2)}\n\nUSER QUESTION: {user_msg}"
            
            contents = [{"parts": [{"text": prompt_content}]}]
            
            resp = requests.post(
                gemini_url,
                headers={"Content-Type": "application/json"},
                json={"contents": contents},
                timeout=12
            )
            
            if resp.status_code == 200:
                resp_json = resp.json()
                candidates = resp_json.get("candidates", [])
                if candidates:
                    text_out = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    if text_out:
                        return JSONResponse({
                            "reply": text_out,
                            "suggested_questions": ["What does my result mean?", "What questions should I ask my doctor?", "What is osteosarcoma?"]
                        })
        except Exception as e:
            print(f"Gemini API request note (falling back to engine): {e}")

    # 2. Fallback to Built-In Medical Knowledge Engine
    fallback_result = generate_medical_fallback_response(user_msg, ctx)
    return JSONResponse(fallback_result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

