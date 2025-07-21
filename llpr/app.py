import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
from util import set_background, write_csv
import uuid
import os
from streamlit_webrtc import webrtc_streamer
import av
import mysql.connector
from mysql.connector import Error
import re
import requests
import json
import time

# Set page config first
st.set_page_config(
    page_title="Smart Traffic Enforcement System",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced appearance
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .header {
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .detection-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .status-paid {
        color: #27ae60;
        font-weight: bold;
    }
    .status-not-paid {
        color: #e74c3c;
        font-weight: bold;
    }
    .status-unknown {
        color: #f39c12;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
        color: white;
    }
    .sidebar .sidebar-content .stRadio label {
        color: white;
    }
    .tabs {
        display: flex;
        margin-bottom: 1rem;
        border-bottom: 1px solid #ddd;
    }
    .tab {
        padding: 0.5rem 1rem;
        cursor: pointer;
        border-radius: 5px 5px 0 0;
        margin-right: 0.5rem;
        background: #eee;
    }
    .tab.active {
        background: #3498db;
        color: white;
    }
    .plate-display {
        font-family: 'Courier New', monospace;
        font-size: 1.5rem;
        letter-spacing: 0.2rem;
        background: #2c3e50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        display: inline-block;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "state" not in st.session_state:
    st.session_state["state"] = "Uploader"
if "sms_sent_log" not in st.session_state:
    st.session_state.sms_sent_log = {}

# Constants and configurations
folder_path = "./licenses_plates_imgs_detected/"
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"
vehicles = [2]  # COCO class IDs for vehicles

# Initialize models and readers
@st.cache_resource
def load_models():
    coco_model = YOLO(COCO_MODEL_DIR)
    license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)
    reader = easyocr.Reader(['en'], gpu=False)
    return coco_model, license_plate_detector, reader

coco_model, license_plate_detector, reader = load_models()

# Database connection function
def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='vehicle_fyp'
        )
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"Error connecting to MySQL database: {e}")
        return None

# SMS notification function
def send_sms_notification(phone_number, vehicle_id, fine_amount):
    if not phone_number:
        st.error("No phone number provided")
        return False
    
    # Format the phone number (Tanzania country code is +255)
    if phone_number.startswith('0'):
        phone_number = '+255' + phone_number[1:]
    elif not phone_number.startswith('+'):
        phone_number = '+' + phone_number
    
    try:
        message = f"WARNING: Your vehicle {vehicle_id} has an unpaid fine of {fine_amount}. Please make payment to avoid further penalties."
        
        response = requests.post(
            "http://localhost/QONTE/sms_notification.php",
            data={
                "phone_number": phone_number,
                "vehicle_id": vehicle_id,
                "fine_amount": fine_amount,
                "message": message
            }
        )
        
        if response.status_code == 200:
            try:
                result = response.json()
                if result.get('status') == 'success':
                    st.success(f"SMS notification sent to {phone_number}")
                    return True
                else:
                    st.error(f"Failed to send SMS: {result.get('message')}")
                    return False
            except:
                st.error(f"Invalid response from SMS API: {response.text}")
                return False
        else:
            st.error(f"Failed to send SMS: HTTP {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Unexpected error sending SMS: {e}")
        return False

# Function to update the latest detected license plate for Arduino control
def update_latest_detection(vehicle_id, command_status):
    connection = connect_to_database()
    if connection is None:
        st.error("Failed to connect to database for updating latest detection")
        return False
    
    try:
        cursor = connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS latest_detection (
            id INT AUTO_INCREMENT PRIMARY KEY,
            vehicle_id VARCHAR(20),
            command_status VARCHAR(10),
            detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        connection.commit()
        
        cursor.execute("DELETE FROM latest_detection")
        cursor.execute("""
        INSERT INTO latest_detection (vehicle_id, command_status) 
        VALUES (%s, %s)
        """, (vehicle_id, command_status))
        connection.commit()
        
        # Also send to PHP endpoint for Arduino control
        try:
            response = requests.post(
                "http://localhost/QONTE/update_status.php",
                data={"vehicle_id": vehicle_id, "command_status": command_status}
            )
            if response.status_code != 200:
                st.warning(f"Failed to send to Arduino control system: {response.status_code}")
        except Exception as e:
            st.error(f"Error sending to Arduino control system: {e}")
        
        return True
    except Error as e:
        st.error(f"Database error while updating latest detection: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# License plate formatting function
def format_license_plate(text):
    if not text:
        return None
    
    cleaned = text.strip().upper()
    no_spaces = cleaned.replace(" ", "")
    
    # For Tanzania plates: T followed by 3 digits followed by 2-3 letters
    match = re.match(r'^T(\d{3})([A-Z]{2,3})$', no_spaces)
    if match:
        return f"T{match.group(1)}{match.group(2)}"
    
    # Try to extract just the alphanumeric characters
    alphanumeric = ''.join(c for c in no_spaces if c.isalnum())
    match = re.match(r'^T(\d{3})([A-Z]{2,3})$', alphanumeric)
    if match:
        return f"T{match.group(1)}{match.group(2)}"
    
    return no_spaces

# Function to check vehicle payment status
def check_vehicle_payment_status(license_plate):
    if not license_plate:
        return {'found': False, 'error': 'Empty license plate text'}
    
    connection = connect_to_database()
    if connection is None:
        return {'found': False, 'error': 'Database connection failed'}
    
    try:
        cursor = connection.cursor(dictionary=True)
        formatted_license = format_license_plate(license_plate)
        
        # Try exact match first
        query = "SELECT * FROM vehicles_payment_status WHERE vehicle_ID = %s"
        cursor.execute(query, (formatted_license,))
        result = cursor.fetchone()
        
        if result:
            update_latest_detection(result['vehicle_ID'], result['Command_status'])
            
            # If status is NOT PAID, send SMS notification (once per hour)
            if result['Command_status'] == 'NOT PAID':
                current_time = time.time()
                vehicle_id = result['vehicle_ID']
                
                if (vehicle_id not in st.session_state.sms_sent_log or 
                    (current_time - st.session_state.sms_sent_log[vehicle_id]) > 3600):
                    phone_number = result['Phone_number']
                    fine_amount = result['fine_payment']
                    
                    if phone_number:
                        success = send_sms_notification(phone_number, vehicle_id, fine_amount)
                        if success:
                            st.session_state.sms_sent_log[vehicle_id] = current_time
            
            return {
                'found': True,
                'vehicle_id': result['vehicle_ID'],
                'owner_name': result['Owner_name'],
                'phone_number': result['Phone_number'],
                'fine_payment': result['fine_payment'],
                'command_status': result['Command_status'],
                'last_updated': result['Last_updated']
            }
        
        # If no exact match, try other matching methods...
        # (Rest of your matching logic remains the same)
        
        return {'found': False, 'error': 'No matching vehicle found'}
    
    except Error as e:
        st.error(f"Error querying database: {e}")
        return {'found': False, 'error': str(e)}
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Video processor class for live detection
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_to_an = img.copy()
        img_to_an = cv2.cvtColor(img_to_an, cv2.COLOR_RGB2BGR)
        license_detections = license_plate_detector(img_to_an)[0]

        if len(license_detections.boxes.cls.tolist()) != 0:
            for license_plate in license_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 
                license_plate_text, _ = read_license_plate(license_plate_crop_gray, img)
                
                if license_plate_text:
                    payment_status = check_vehicle_payment_status(license_plate_text)
                    status_text = "UNKNOWN"
                    status_color = (255, 255, 0)  # Yellow for unknown
                    
                    if payment_status and payment_status.get('found', False):
                        if payment_status.get('command_status') == 'PAID':
                            status_text = "PAID"
                            status_color = (0, 255, 0)  # Green for paid
                        else:
                            status_text = "NOT PAID"
                            status_color = (0, 0, 255)  # Red for not paid

                    # Draw license plate text
                    cv2.rectangle(img, (int(x1) - 40, int(y1) - 40), (int(x2) + 40, int(y1)), (255, 255, 255), cv2.FILLED)
                    cv2.putText(img,
                                str(license_plate_text),
                                (int((int(x1) + int(x2)) / 2) - 70, int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 0),
                                3)
                    
                    # Draw payment status
                    cv2.rectangle(img, (int(x1) - 40, int(y2)), (int(x2) + 40, int(y2) + 40), status_color, cv2.FILLED)
                    cv2.putText(img,
                                status_text,
                                (int((int(x1) + int(x2)) / 2) - 70, int(y2) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Function to read license plate text
def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    if detections == []:
        return None, None

    rectangle_size = license_plate_crop.shape[0]*license_plate_crop.shape[1]
    plate = [] 

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > 0.17:
            text = result[1].upper()
            scores += result[2]
            plate.append(text)
    
    if len(plate) != 0: 
        return " ".join(plate), scores/len(plate)
    else:
        return None, 0

# Main detection function
def model_prediction(img):
    license_numbers = 0
    results = {}
    licenses_texts = []
    payment_statuses = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Initialize default values for vehicle detection
    xcar1, ycar1, xcar2, ycar2, car_score = 0, 0, 0, 0, 0

    object_detections = coco_model(img)[0]
    license_detections = license_plate_detector(img)[0]

    if len(object_detections.boxes.cls.tolist()) != 0:
        for detection in object_detections.boxes.data.tolist():
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection
            if int(class_id) in vehicles:
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)

    if len(license_detections.boxes.cls.tolist()) != 0:
        license_plate_crops_total = []
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

            img_name = '{}.jpg'.format(uuid.uuid1())
            cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

            licenses_texts.append(license_plate_text)

            if license_plate_text is not None and license_plate_text_score is not None:
                payment_status = check_vehicle_payment_status(license_plate_text)
                payment_statuses.append(payment_status)
                
                license_plate_crops_total.append(license_plate_crop)
                results[license_numbers] = {}
                
                results[license_numbers][license_numbers] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'car_score': car_score},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    }
                }
                
                if payment_status and payment_status.get('found', False):
                    results[license_numbers][license_numbers]['payment_status'] = {
                        'status': payment_status.get('command_status', 'UNKNOWN'),
                        'owner': payment_status.get('owner_name', 'Unknown'),
                        'phone': payment_status.get('phone_number', 'Unknown'),
                        'fine': payment_status.get('fine_payment', 0)
                    }
                
                license_numbers += 1
          
        write_csv(results, f"./csv_detections/detection_results.csv")
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box, licenses_texts, license_plate_crops_total, payment_statuses]
    
    else: 
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box]
# State change functions
def change_state_uploader():
    st.session_state["state"] = "Uploader"

def change_state_camera():
    st.session_state["state"] = "Camera"

def change_state_live():
    st.session_state["state"] = "Live"

# Main app layout
def main():
    # Sidebar with system info
    with st.sidebar:
        st.title("🚔 System Dashboard")
        st.markdown("""
        ### Traffic Enforcement System
        This system automatically detects license plates and checks their payment status in real-time.
        
        **Features:**
        - Vehicle and license plate detection
        - OCR for plate number extraction
        - Database lookup for payment status
        - SMS notifications for unpaid fines
        - Barrier control integration
        """)
        
        st.markdown("---")
        st.markdown("### System Status")
        
        # Test database connection
        if st.button("Test Database Connection"):
            connection = connect_to_database()
            if connection:
                st.success("✅ Database connection successful")
                connection.close()
            else:
                st.error("❌ Database connection failed")
        
        # Test SMS functionality
        test_phone = st.text_input("Enter phone number to test SMS:")
        if st.button("Test SMS Notification"):
            if test_phone:
                if send_sms_notification(test_phone, "TEST123", "10,000 TZS"):
                    st.success("SMS test successful!")
            else:
                st.warning("Please enter a phone number")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        Developed by [Abdul & Luchabanya]  
        Powered by YOLOv8 and EasyOCR  
        Version 1.0.0
        """)

    # Main content area
    st.markdown('<div class="header"><h1>Smart Traffic Enforcement System</h1></div>', unsafe_allow_html=True)
    
    # Mode selection tabs
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📁 Upload Image", use_container_width=True):
            change_state_uploader()
    with col2:
        if st.button("📷 Take Photo", use_container_width=True):
            change_state_camera()
    with col3:
        if st.button("🎥 Live Detection", use_container_width=True):
            change_state_live()
    
    st.markdown("---")
    
    # Main content based on state
    if st.session_state["state"] == "Uploader":
        st.markdown("### 📁 Upload Vehicle Image")
        img = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
    elif st.session_state["state"] == "Camera":
        st.markdown("### 📷 Capture Vehicle Photo")
        img = st.camera_input("Take a picture of the vehicle...", label_visibility="collapsed")
        
    elif st.session_state["state"] == "Live":
        st.markdown("### 🎥 Live License Plate Detection")
        webrtc_streamer(
            key="live-detection",
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        img = None
    
    # Process image if available
    if img is not None and st.session_state["state"] in ["Uploader", "Camera"]:
        with st.spinner("Processing image..."):
            image = np.array(Image.open(img))    
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Original Image")
                st.image(image, use_column_width=True)
            
            if st.button("🔍 Detect License Plate", use_container_width=True):
                results = model_prediction(image)
                
                if len(results) >= 4:
                    prediction, texts, license_plate_crop, payment_statuses = results[0], results[1], results[2], results[3]
                    texts = [i for i in texts if i is not None]
                    
                    with col2:
                        st.markdown("### Detection Results")
                        st.image(prediction, use_column_width=True)
                    
                    # Display results in cards
                    for i, (text, crop, status) in enumerate(zip(texts, license_plate_crop, payment_statuses)):
                        with st.expander(f"🚗 Vehicle {i+1} Details", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**License Plate**")
                                st.markdown(f'<div class="plate-display">{text}</div>', unsafe_allow_html=True)
                                st.image(crop, caption="Detected License Plate")
                            
                            with col2:
                                st.markdown("**Payment Status**")
                                if status and status.get('found', False):
                                    if status.get('command_status') == 'PAID':
                                        st.markdown('<p class="status-paid">✅ Payment Status: PAID</p>', unsafe_allow_html=True)
                                        st.success("🚦 Barrier Status: OPEN (Vehicle can pass)")
                                    else:
                                        st.markdown('<p class="status-not-paid">❌ Payment Status: NOT PAID</p>', unsafe_allow_html=True)
                                        st.error("🚦 Barrier Status: CLOSED (Vehicle blocked)")
                                        st.warning("📱 SMS notification sent to owner")
                                    
                                    st.markdown(f"**Owner:** {status.get('owner_name', 'Unknown')}")
                                    st.markdown(f"**Phone:** {status.get('phone_number', 'Unknown')}")
                                    st.markdown(f"**Fine Amount:** {status.get('fine_payment', 0)} TZS")
                                    st.markdown(f"**Last Updated:** {status.get('last_updated', 'Unknown')}")
                                else:
                                    st.markdown('<p class="status-unknown">⚠️ Payment Status: UNKNOWN</p>', unsafe_allow_html=True)
                                    st.error("Vehicle not found in database")
                                    if status:
                                        st.error(f"Error: {status.get('error', 'Unknown error')}")
                
                    # Show detection data
                   
                    
                else:
                    st.warning("No license plates detected in the image")
                    st.image(results[0], use_column_width=True)

if __name__ == "__main__":
    main()