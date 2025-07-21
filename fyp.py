import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import easyocr
import requests
import smtplib
from email.message import EmailMessage
import time
from datetime import datetime
import logging
import os
import torch.serialization
from ultralytics.nn.tasks import DetectionModel
import torch.nn
from ultralytics.nn.modules import Conv # <--- NEW: Import Conv module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
LICENSE_MODEL_DETECTION_DIR = 'models/license_plate_detector.pt'
COCO_MODEL_DIR = 'yolov8n.pt'
API_BASE_URL = "https://quantisbroker.com/vehicle-payment-api"
EMAIL_CONFIG = {
    'sender_email': 'googlotanzania@gmail.com',
    'app_password': 'qyln pcco sedy gewj',
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 465
}

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        logger.info(f"COCO model path (will be downloaded if not found): {COCO_MODEL_DIR}")
        logger.info(f"License plate model path (must be in repo): {LICENSE_MODEL_DETECTION_DIR}")
        st.write(f"Attempting to load COCO model from: {COCO_MODEL_DIR}")
        st.write(f"Attempting to load License plate model from: {LICENSE_MODEL_DETECTION_DIR}")

        # --- CORRECTED FIX ---
        # Add ultralytics.nn.tasks.DetectionModel, torch.nn.modules.container.Sequential,
        # and ultralytics.nn.modules.Conv to safe globals for PyTorch loading.
        torch.serialization.add_safe_globals([
            DetectionModel,
            torch.nn.modules.container.Sequential,
            Conv # <--- NEW: Add Conv module
        ])
        # --- END OF CORRECTED FIX ---

        coco_model = YOLO(COCO_MODEL_DIR)
        license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)
        reader = easyocr.Reader(['en'], gpu=False)
        return coco_model, license_plate_detector, reader
    except Exception as e:
        st.error(f"Error loading models: {e}")
        logger.error(f"Error loading models: {e}")
        return None, None, None

coco_model, license_plate_detector, reader = load_models()

# --- Helper Functions ---
def format_license_plate(text):
    if not text:
        return None
    cleaned = text.strip().upper()
    no_spaces = cleaned.replace(" ", "")
    return no_spaces

def get_vehicle_payment_status_from_api(license_plate):
    """Fetches vehicle payment status from the PHP API."""
    formatted_license = format_license_plate(license_plate)
    if not formatted_license:
        return {'found': False, 'error': 'Invalid license plate format'}
    try:
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
        }
        response = requests.get(
            f"{API_BASE_URL}/get_vehicle_by_id.php?vehicle_id={formatted_license}",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        if data.get('success') and data.get('data'):
            vehicle_data = data['data']
            return {
                'found': True,
                'vehicle_id': vehicle_data['vehicle_ID'],
                'owner_name': vehicle_data['owner_name'],
                'phone_number': vehicle_data['phone_number'],
                'fine_payment': float(vehicle_data['fine_payment']),
                'command_status': vehicle_data['command_status'],
                'last_updated': vehicle_data['last_updated']
            }
        else:
            logger.warning(f"API response for {formatted_license}: {data.get('message', 'No success or data')}")
            return {'found': False, 'error': data.get('message', 'Vehicle not found via API')}
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for {formatted_license}: {e}")
        return {'found': False, 'error': f"API connection error: {e}"}
    except ValueError as e:
        logger.error(f"JSON decoding error for {formatted_license}: {e}")
        return {'found': False, 'error': f"API response format error: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error fetching vehicle status: {e}")
        return {'found': False, 'error': f"Unexpected error: {e}"}

def read_license_plate(license_plate_crop, img):
    detections = reader.readtext(license_plate_crop)
    logger.info(f"OCR detections: {detections}")

    if not detections:
        return None, 0

    plate = []
    scores = 0
    rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]

    for result in detections:
        pts = np.array(result[0]).astype(int)
        text = result[1]
        score = result[2]

        length = np.linalg.norm(np.array(result[0][1]) - np.array(result[0][0]))
        height = np.linalg.norm(np.array(result[0][2]) - np.array(result[0][1]))
        area_ratio = (length * height) / rectangle_size

        logger.info(f"Detection box area ratio: {area_ratio:.3f} for text: {text}")

        if area_ratio > 0.05:
            plate.append(text.upper())
            scores += score
    if plate:
        avg_score = scores / len(plate)
        combined_plate = " ".join(plate)
        logger.info(f"Accepted plate text: {combined_plate} with avg score: {avg_score:.3f}")
        return combined_plate, avg_score

    logger.info("No license plate text passed the area filter.")
    return None, 0

def model_prediction(img):
    if license_plate_detector is None:
        st.error("License plate detection model failed to load. Please check the model path and restart the app.")
        logger.error("License plate detection model is None.")
        return []
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    license_detections = license_plate_detector(img_bgr)[0]
    results = []
    if len(license_detections.boxes.cls.tolist()) != 0:
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            margin = 10
            h, w = img_bgr.shape[:2]
            x1 = max(int(x1) - margin, 0)
            y1 = max(int(y1) - margin, 0)
            x2 = min(int(x2) + margin, w)
            y2 = min(int(y2) + margin, h)
            license_plate_crop = img_bgr[y1:y2, x1:x2, :]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            license_plate_text, license_plate_text_score = read_license_plate(
                license_plate_crop_gray, img_bgr
            )
            payment_status = None
            if license_plate_text:
                payment_status = get_vehicle_payment_status_from_api(license_plate_text)
            results.append({
                'bbox': [x1, y1, x2, y2],
                'text': license_plate_text,
                'crop': cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB),
                'payment_status': payment_status,
                'confidence': license_plate_text_score
            })
    return results

def send_sms_notification(phone_number, vehicle_id, fine_amount, status):
    try:
        payment_link = f"https://v0-payment-simulation-page.vercel.app/?vehicle_id={vehicle_id}"
        message = (
            f"ALERT: Vehicle {vehicle_id} status: {status}. Fine: ${fine_amount}. "
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
            f"Pay now: {payment_link}"
        )
        response = requests.post(
            "http://localhost/QONTE/sms_notification.php",
            data={
                "phone_number": phone_number,
                "vehicle_id": vehicle_id,
                "fine_amount": fine_amount,
                "message": message
            },
            timeout=10
        )
        if response.status_code == 200:
            st.success(f"✅ SMS notification sent to {phone_number}")
            logger.info(f"SMS sent successfully to {phone_number}")
            return True
        else:
            st.warning(f"⚠️ SMS service responded with status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ SMS service unavailable: {e}")
        logger.error(f"SMS error: {e}")
        return False
    except Exception as e:
        st.error(f"❌ Unexpected SMS error: {e}")
        logger.error(f"Unexpected SMS error: {e}")
        return False

def send_email_fast(recipient_email, subject, body):
    try:
        msg = EmailMessage()
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.set_content(body)

        with smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as smtp:
            smtp.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['app_password'])
            smtp.send_message(msg)

        st.success(f"✅ Email sent successfully to {recipient_email}")
        logger.info(f"Email sent successfully to {recipient_email}")
        return True

    except smtplib.SMTPAuthenticationError:
        st.error("❌ Email authentication failed. Check credentials.")
        logger.error("Email authentication failed")
        return False
    except smtplib.SMTPException as e:
        st.error(f"❌ SMTP error: {e}")
        logger.error(f"SMTP error: {e}")
        return False
    except Exception as e:
        st.error(f"❌ Email failed to send: {e}")
        logger.error(f"Email error: {e}")
        return False

def process_vehicle_detection(detection_result):
    """Process vehicle detection with real-time status from API."""
    payment_status = detection_result['payment_status']

    if not payment_status or not payment_status.get('found', False):
        st.warning(f"⚠️ Vehicle not found in database or API error: {payment_status.get('error', 'Unknown error')}. Access DENIED by default.")
        return

    vehicle_id = payment_status.get('vehicle_id')
    status = payment_status.get('command_status', 'UNKNOWN')
    owner_name = payment_status.get('owner_name', 'Unknown')
    phone_number = payment_status.get('phone_number', 'Unknown')
    fine_amount = payment_status.get('fine_payment', 0)
    last_updated = payment_status.get('last_updated', 'Unknown')

    st.info(f"**Vehicle ID:** {vehicle_id}")
    st.info(f"**Owner:** {owner_name}")
    st.info(f"**Phone:** {phone_number}")
    st.info(f"**Fine Amount:** ${fine_amount}")
    st.info(f"**Last Updated:** {last_updated}")
    st.info(f"**Detection Confidence:** {detection_result['confidence']:.2f}")

    if status == 'NOT PAID':
        st.error("🚫 **VEHICLE BLOCKED** - Payment Required!")
        st.error("**ACCESS DENIED** - Vehicle cannot proceed until payment is made.")
        payment_link = f"https://v0-payment-simulation-page.vercel.app/?vehicle_id={vehicle_id}"
        st.info(f"[🔗 Pay Fine Now]({payment_link})")
        email_subject = f"🚨 HARAKA: Gari {vehicle_id} LIMEZUIWA - Malipo Yanahitajika"
        email_body = f"""TAARIFA YA UFUATILIAJI WA GARI - HATUA YA HARAKA INAHITAJIKAGari hili limezuiwa kiotomatiki kutokana na faini ambazo hazijalipwa.Gari haliwezi kuendelea hadi malipo yakamilike.Lipa faini yako sasa: {payment_link}Tafadhali wasiliana na mmiliki mara moja au kamilisha malipo ili kufungua gari.
        """
        send_sms_notification(phone_number, vehicle_id, fine_amount, "BLOCKED - NOT PAID")
        send_email_fast(EMAIL_CONFIG['sender_email'], email_subject, email_body)

    elif status == 'PAID':
        st.success("✅ **VEHICLE CLEARED** - Payment Verified!")
        st.success("**ACCESS GRANTED** - Vehicle is free to proceed.")
        email_subject = f"✅ Gari {vehicle_id} - Ufikiaji Umeruhusiwa"
        email_body = f"""UTHIBITISHO WA UFUATILIAJI WA GARIGari hili limethibitishwa kuwa LIMELIPWA na linaruhusiwa kuendelea.
        """
        send_sms_notification(phone_number, vehicle_id, fine_amount, "CLEARED - PAID")
        send_email_fast(EMAIL_CONFIG['sender_email'], email_subject, email_body)

    else:
        st.warning(f"⚠️ **UNKNOWN STATUS: {status}**")
        st.warning("**ACCESS DENIED** - Status verification required.")

# --- Streamlit UI ---
def main():
    st.set_page_config(
        page_title="Real-Time Vehicle Control System",
        page_icon="🚗",
        layout="wide"
    )

    st.title("🚗 REAL-TIME VEHICLE LICENSE PLATE DETECTION & CONTROL SYSTEM")
    st.markdown("---")

    with st.sidebar:
        st.header("📊 System Status")
        st.info("This app fetches real-time vehicle status from your deployed PHP API.")
        st.markdown(f"**PHP API Base URL:** `{API_BASE_URL}`")
        st.markdown("---")
        st.subheader("Model Paths (Local)")
        st.write(f"COCO: `{COCO_MODEL_DIR}`")
        st.write(f"License Plate: `{LICENSE_MODEL_DETECTION_DIR}`")

    st.write("📸 **Capture or upload a vehicle image for real-time license plate detection and payment verification.**")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Camera Input")
        st.info("**Tip:** On mobile, tap the camera icon in your browser's address bar or settings to select the back (environment) camera for best results.")
        camera_img = st.camera_input("Take a Photo")
    with col2:
        st.subheader("📁 File Upload")
        uploaded_img = st.file_uploader("Upload Vehicle Image", type=["jpg", "png", "jpeg"])

    image = None
    if camera_img is not None:
        image = np.array(Image.open(camera_img))
        st.success("📷 Camera image captured!")
    elif uploaded_img is not None:
        image = np.array(Image.open(uploaded_img))
        st.success("📁 Image uploaded successfully!")

    if image is not None:
        st.subheader("🖼️ Input Image")
        st.image(image, width=600, caption="Vehicle Image for Analysis")

        with st.spinner("🔍 Detecting license plate and checking status via API..."):
            results = model_prediction(image)

        if not results:
            st.warning("⚠️ No license plate detected in the image.")
            st.info("💡 **Tips for better detection:**")
            st.info("- Ensure the license plate is clearly visible")
            st.info("- Good lighting conditions")
            st.info("- Minimal blur or distortion")
        else:
            st.success(f"✅ Detected {len(results)} license plate(s)")

            for i, result in enumerate(results):
                st.markdown("---")
                st.subheader(f"🚗 Vehicle Detection #{i+1}")

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(result['crop'], caption="License Plate Crop", width=300)

                with col2:
                    if result['text']:
                        st.success(f"**License Number:** {result['text']}")
                        process_vehicle_detection(result)
                    else:
                        st.error("❌ Could not read license plate text")
    st.markdown("---")
    st.markdown("**🔧 Real-Time Vehicle Control System** | Status fetched from PHP API | Instant notifications")

if __name__ == "__main__":
    main()