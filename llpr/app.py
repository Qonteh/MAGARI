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

set_background("./imgs/background.png")

folder_path = "./licenses_plates_imgs_detected/"
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"

reader = easyocr.Reader(['en'], gpu=False)

vehicles = [2]

header = st.container()
body = st.container()

coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

threshold = 0.15

state = "Uploader"

if "state" not in st.session_state:
    st.session_state["state"] = "Uploader"

# Track SMS sent status to avoid sending multiple SMS for the same detection
sms_sent_log = {}

# Function to send SMS notification using Africa's Talking API via PHP endpoint
def send_sms_notification(phone_number, vehicle_id, fine_amount):
    """
    Send SMS notification using Africa's Talking API via PHP endpoint
    """
    if not phone_number:
        st.error("No phone number provided")
        return False
    
    # Format the phone number (ensure it has country code)
    # Tanzania country code is +255
    if phone_number.startswith('0'):
        phone_number = '+255' + phone_number[1:]
    elif not phone_number.startswith('+'):
        phone_number = '+' + phone_number
    
    try:
        # Create the message
        message = f"WARNING: Your vehicle {vehicle_id} has an unpaid fine of {fine_amount}. Please make payment to avoid further penalties."
        
        # Send to PHP endpoint that handles the SMS API
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

# Database connection function
def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',  # Your MySQL username
            password='',  # Your MySQL password
            database='vehicle_fyp'
        )
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"Error connecting to MySQL database: {e}")
        return None

# Function to update the latest detected license plate for Arduino control
def update_latest_detection(vehicle_id, command_status):
    connection = connect_to_database()
    if connection is None:
        st.error("Failed to connect to database for updating latest detection")
        return False
    
    try:
        cursor = connection.cursor()
        
        # First, check if the table exists, if not create it
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS latest_detection (
            id INT AUTO_INCREMENT PRIMARY KEY,
            vehicle_id VARCHAR(20),
            command_status VARCHAR(10),
            detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        connection.commit()
        
        # Clear previous detections and insert the new one
        cursor.execute("DELETE FROM latest_detection")
        cursor.execute("""
        INSERT INTO latest_detection (vehicle_id, command_status) 
        VALUES (%s, %s)
        """, (vehicle_id, command_status))
        connection.commit()
        
        st.success(f"Updated latest detection: {vehicle_id} - {command_status}")
        
        # Also send to PHP endpoint for Arduino control
        try:
            response = requests.post(
                "http://localhost/QONTE/update_status.php",
                data={"vehicle_id": vehicle_id, "command_status": command_status}
            )
            if response.status_code == 200:
                st.success("Successfully sent to Arduino control system")
            else:
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

# Function to test database connection
def test_database_connection():
    st.write("### Testing Database Connection")
    connection = connect_to_database()
    
    if connection is None:
        st.error("Failed to connect to database")
        return
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Test query to get all vehicle IDs
        cursor.execute("SELECT vehicle_ID FROM vehicles_payment_status")
        vehicles = cursor.fetchall()
        
        if vehicles:
            st.success(f"Successfully connected to database. Found {len(vehicles)} vehicles.")
            st.write("Sample vehicle IDs:")
            for i, vehicle in enumerate(vehicles[:10]):  # Show first 10
                st.write(f"{i+1}. {vehicle['vehicle_ID']}")
        else:
            st.warning("Connected to database but no vehicles found in the table.")
        
    except Error as e:
        st.error(f"Database query error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Function to test SMS functionality
def test_sms_functionality():
    st.write("### Testing SMS Functionality")
    
    test_phone = st.text_input("Enter a phone number to test SMS (e.g., 0712345678):")
    
    if st.button("Send Test SMS"):
        if test_phone:
            success = send_sms_notification(test_phone, "TEST123", "10,000 TZS")
            if success:
                st.success(f"Test SMS sent successfully to {test_phone}")
            else:
                st.error("Failed to send test SMS")
        else:
            st.warning("Please enter a phone number")

# License plate formatting function for non-spaced format
def format_license_plate(text):
    if not text:
        return None
    
    # Clean the text - remove extra spaces and normalize
    cleaned = text.strip().upper()
    
    # Remove all spaces to ensure non-spaced format
    no_spaces = cleaned.replace(" ", "")
    
    # For Tanzania plates, ensure it follows the pattern T followed by 3 digits followed by 2-3 letters
    match = re.match(r'^T(\d{3})([A-Z]{2,3})$', no_spaces)
    if match:
        # Return in non-spaced format: T135ABD
        return f"T{match.group(1)}{match.group(2)}"
    
    # Try to extract just the alphanumeric characters and format them
    alphanumeric = ''.join(c for c in no_spaces if c.isalnum())
    match = re.match(r'^T(\d{3})([A-Z]{2,3})$', alphanumeric)
    if match:
        # Return in non-spaced format: T135ABD
        return f"T{match.group(1)}{match.group(2)}"
    
    # If all else fails, return the cleaned text without spaces
    return no_spaces

# Debug function to help troubleshoot license plate formatting
def debug_license_plate_format(original_text, formatted_text):
    st.write("---")
    st.write("### License Plate Format Debugging")
    st.write(f"Original OCR text: '{original_text}'")
    st.write(f"Formatted text (non-spaced): '{formatted_text}'")
    
    # Check if the formatted text exists in the database
    connection = connect_to_database()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            
            # Get all vehicle IDs for comparison
            cursor.execute("SELECT vehicle_ID FROM vehicles_payment_status")
            all_ids = [row['vehicle_ID'] for row in cursor.fetchall()]
            
            # Try exact match
            if formatted_text in all_ids:
                st.success(f"Exact match found in database: '{formatted_text}'")
            else:
                st.warning(f"No exact match found for '{formatted_text}'")
                
                # Try direct match without case sensitivity
                for vehicle_id in all_ids:
                    if vehicle_id.upper() == formatted_text.upper():
                        st.success(f"Case-insensitive match found: '{vehicle_id}'")
                        return
                
                # Try matching without spaces for database entries that might have spaces
                for vehicle_id in all_ids:
                    vehicle_id_no_spaces = vehicle_id.replace(" ", "")
                    if vehicle_id_no_spaces.upper() == formatted_text.upper():
                        st.success(f"Match found ignoring spaces: '{vehicle_id}'")
                        return
                
                # Find closest matches
                st.write("Closest matches in database:")
                for vehicle_id in all_ids:
                    # Calculate similarity (simple character-by-character comparison)
                    vehicle_id_no_spaces = vehicle_id.replace(" ", "")
                    similarity = sum(a == b for a, b in zip(formatted_text, 
                                                          vehicle_id_no_spaces)) / max(len(formatted_text), 
                                                                                    len(vehicle_id_no_spaces))
                    if similarity > 0.7:  # Show matches with >70% similarity
                        st.write(f"- '{vehicle_id}' (similarity: {similarity:.2f})")
            
        except Error as e:
            st.error(f"Database query error: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# Improved function to check vehicle payment status
def check_vehicle_payment_status(license_plate):
    if not license_plate:
        return {'found': False, 'error': 'Empty license plate text'}
    
    connection = connect_to_database()
    if connection is None:
        return {'found': False, 'error': 'Database connection failed'}
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Format the license plate to match database format (non-spaced)
        formatted_license = format_license_plate(license_plate)
        
        # Debug the formatting
        debug_license_plate_format(license_plate, formatted_license)
        
        # Try exact match first
        query = "SELECT * FROM vehicles_payment_status WHERE vehicle_ID = %s"
        cursor.execute(query, (formatted_license,))
        result = cursor.fetchone()
        
        if result:
            # Update the latest detection for Arduino control
            update_latest_detection(result['vehicle_ID'], result['Command_status'])
            
            # If status is NOT PAID, send SMS notification
            if result['Command_status'] == 'NOT PAID':
                # Check if we've already sent an SMS for this vehicle recently
                current_time = time.time()
                vehicle_id = result['vehicle_ID']
                
                # Only send SMS if we haven't sent one in the last hour (3600 seconds)
                if vehicle_id not in sms_sent_log or (current_time - sms_sent_log[vehicle_id]) > 3600:
                    phone_number = result['Phone_number']
                    fine_amount = result['fine_payment']
                    
                    if phone_number:
                        success = send_sms_notification(phone_number, vehicle_id, fine_amount)
                        if success:
                            # Log the time we sent the SMS
                            sms_sent_log[vehicle_id] = current_time
                    else:
                        st.warning(f"No phone number available for {vehicle_id}")
            
            return {
                'found': True,
                'vehicle_id': result['vehicle_ID'],
                'owner_name': result['Owner_name'],
                'phone_number': result['Phone_number'],
                'fine_payment': result['fine_payment'],
                'command_status': result['Command_status'],
                'last_updated': result['Last_updated']
            }
        
        # If no exact match, try without case sensitivity
        query = "SELECT * FROM vehicles_payment_status WHERE UPPER(vehicle_ID) = UPPER(%s)"
        cursor.execute(query, (formatted_license,))
        result = cursor.fetchone()
        
        if result:
            st.info(f"Found case-insensitive match: '{result['vehicle_ID']}' for '{formatted_license}'")
            
            # Update the latest detection for Arduino control
            update_latest_detection(result['vehicle_ID'], result['Command_status'])
            
            # If status is NOT PAID, send SMS notification
            if result['Command_status'] == 'NOT PAID':
                # Check if we've already sent an SMS for this vehicle recently
                current_time = time.time()
                vehicle_id = result['vehicle_ID']
                
                # Only send SMS if we haven't sent one in the last hour (3600 seconds)
                if vehicle_id not in sms_sent_log or (current_time - sms_sent_log[vehicle_id]) > 3600:
                    phone_number = result['Phone_number']
                    fine_amount = result['fine_payment']
                    
                    if phone_number:
                        success = send_sms_notification(phone_number, vehicle_id, fine_amount)
                        if success:
                            # Log the time we sent the SMS
                            sms_sent_log[vehicle_id] = current_time
                    else:
                        st.warning(f"No phone number available for {vehicle_id}")
            
            return {
                'found': True,
                'vehicle_id': result['vehicle_ID'],
                'owner_name': result['Owner_name'],
                'phone_number': result['Phone_number'],
                'fine_payment': result['fine_payment'],
                'command_status': result['Command_status'],
                'last_updated': result['Last_updated']
            }
        
        # Try matching without spaces for database entries that might have spaces
        query = "SELECT * FROM vehicles_payment_status WHERE REPLACE(vehicle_ID, ' ', '') = %s"
        cursor.execute(query, (formatted_license,))
        result = cursor.fetchone()
        
        if result:
            st.info(f"Found match ignoring spaces: '{result['vehicle_ID']}' for '{formatted_license}'")
            
            # Update the latest detection for Arduino control
            update_latest_detection(result['vehicle_ID'], result['Command_status'])
            
            # If status is NOT PAID, send SMS notification
            if result['Command_status'] == 'NOT PAID':
                # Check if we've already sent an SMS for this vehicle recently
                current_time = time.time()
                vehicle_id = result['vehicle_ID']
                
                # Only send SMS if we haven't sent one in the last hour (3600 seconds)
                if vehicle_id not in sms_sent_log or (current_time - sms_sent_log[vehicle_id]) > 3600:
                    phone_number = result['Phone_number']
                    fine_amount = result['fine_payment']
                    
                    if phone_number:
                        success = send_sms_notification(phone_number, vehicle_id, fine_amount)
                        if success:
                            # Log the time we sent the SMS
                            sms_sent_log[vehicle_id] = current_time
                    else:
                        st.warning(f"No phone number available for {vehicle_id}")
            
            return {
                'found': True,
                'vehicle_id': result['vehicle_ID'],
                'owner_name': result['Owner_name'],
                'phone_number': result['Phone_number'],
                'fine_payment': result['fine_payment'],
                'command_status': result['Command_status'],
                'last_updated': result['Last_updated']
            }
        
        # If still no match, try a more flexible search with LIKE
        # This handles minor OCR errors by looking for similar plates
        query = "SELECT * FROM vehicles_payment_status WHERE REPLACE(vehicle_ID, ' ', '') LIKE %s"
        
        # Extract the pattern (e.g., T135ABC -> T%135%AB%)
        if len(formatted_license) >= 6:  # Minimum length for a valid plate (T + 3 digits + 2 letters)
            # Try to match the first letter, the 3 digits, and first 2 letters of the last part
            pattern = f"{formatted_license[0]}%{formatted_license[1:4]}%{formatted_license[4:6]}%"
            cursor.execute(query, (pattern,))
            result = cursor.fetchone()
            
            if result:
                st.info(f"Found similar match: '{result['vehicle_ID']}' for '{formatted_license}'")
                
                # Update the latest detection for Arduino control
                update_latest_detection(result['vehicle_ID'], result['Command_status'])
                
                # If status is NOT PAID, send SMS notification
                if result['Command_status'] == 'NOT PAID':
                    # Check if we've already sent an SMS for this vehicle recently
                    current_time = time.time()
                    vehicle_id = result['vehicle_ID']
                    
                    # Only send SMS if we haven't sent one in the last hour (3600 seconds)
                    if vehicle_id not in sms_sent_log or (current_time - sms_sent_log[vehicle_id]) > 3600:
                        phone_number = result['Phone_number']
                        fine_amount = result['fine_payment']
                        
                        if phone_number:
                            success = send_sms_notification(phone_number, vehicle_id, fine_amount)
                            if success:
                                # Log the time we sent the SMS
                                sms_sent_log[vehicle_id] = current_time
                        else:
                            st.warning(f"No phone number available for {vehicle_id}")
                
                return {
                    'found': True,
                    'vehicle_id': result['vehicle_ID'],
                    'owner_name': result['Owner_name'],
                    'phone_number': result['Phone_number'],
                    'fine_payment': result['fine_payment'],
                    'command_status': result['Command_status'],
                    'last_updated': result['Last_updated']
                }
        
        # DIRECT LOOKUP FOR SPECIFIC CASE: T135ABD
        if license_plate == "T135ABD" or formatted_license == "T135ABD":
            # Try to find T135ABD directly
            cursor.execute("SELECT * FROM vehicles_payment_status WHERE vehicle_ID = 'T135ABD' OR REPLACE(vehicle_ID, ' ', '') = 'T135ABD'")
            result = cursor.fetchone()
            if result:
                st.success("Found direct match for T135ABD")
                
                # Update the latest detection for Arduino control
                update_latest_detection(result['vehicle_ID'], result['Command_status'])
                
                # If status is NOT PAID, send SMS notification
                if result['Command_status'] == 'NOT PAID':
                    # Check if we've already sent an SMS for this vehicle recently
                    current_time = time.time()
                    vehicle_id = result['vehicle_ID']
                    
                    # Only send SMS if we haven't sent one in the last hour (3600 seconds)
                    if vehicle_id not in sms_sent_log or (current_time - sms_sent_log[vehicle_id]) > 3600:
                        phone_number = result['Phone_number']
                        fine_amount = result['fine_payment']
                        
                        if phone_number:
                            success = send_sms_notification(phone_number, vehicle_id, fine_amount)
                            if success:
                                # Log the time we sent the SMS
                                sms_sent_log[vehicle_id] = current_time
                        else:
                            st.warning(f"No phone number available for {vehicle_id}")
                
                return {
                    'found': True,
                    'vehicle_id': result['vehicle_ID'],
                    'owner_name': result['Owner_name'],
                    'phone_number': result['Phone_number'],
                    'fine_payment': result['fine_payment'],
                    'command_status': result['Command_status'],
                    'last_updated': result['Last_updated']
                }
        
        # Add a direct database check to see what's in the database
        cursor.execute("SELECT vehicle_ID FROM vehicles_payment_status")
        all_ids = [row['vehicle_ID'] for row in cursor.fetchall()]
        st.write("All vehicle IDs in database:", all_ids)
        
        return {'found': False, 'error': 'No matching vehicle found'}
    
    except Error as e:
        st.error(f"Error querying database: {e}")
        return {'found': False, 'error': str(e)}
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

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

                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)
                
                # Check payment status if license plate is detected
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

def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    width = img.shape[1]
    height = img.shape[0]
    
    if detections == []:
        return None, None

    rectangle_size = license_plate_crop.shape[0]*license_plate_crop.shape[1]

    plate = [] 

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > 0.17:
            bbox, text, score = result
            text = result[1]
            text = text.upper()
            scores += score
            plate.append(text)
    
    if len(plate) != 0: 
        return " ".join(plate), scores/len(plate)
    else:
        return None, 0

def model_prediction(img):
    license_numbers = 0
    results = {}
    licenses_texts = []
    payment_statuses = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    object_detections = coco_model(img)[0]
    license_detections = license_plate_detector(img)[0]

    if len(object_detections.boxes.cls.tolist()) != 0:
        for detection in object_detections.boxes.data.tolist():
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection

            if int(class_id) in vehicles:
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
    else:
        xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
        car_score = 0

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
                # Check payment status in database
                payment_status = check_vehicle_payment_status(license_plate_text)
                payment_statuses.append(payment_status)
                
                # Draw payment status on the image
                if payment_status and payment_status.get('found', False):
                    status_text = payment_status.get('command_status', 'UNKNOWN')
                    status_color = (0, 255, 0) if status_text == 'PAID' else (0, 0, 255)
                    
                    cv2.rectangle(img, (int(x1) - 40, int(y2)), (int(x2) + 40, int(y2) + 40), status_color, cv2.FILLED)
                    cv2.putText(img,
                                status_text,
                                (int((int(x1) + int(x2)) / 2) - 70, int(y2) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2)
                
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
                
                # Add payment status to results
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

def change_state_uploader():
    st.session_state["state"] = "Uploader"

def change_state_camera():
    st.session_state["state"] = "Camera"

def change_state_live():
    st.session_state["state"] = "Live"
    
with header:
    _, col1, _ = st.columns([0.2,1,0.1])
    col1.title("💥 License Car Plate Detection 🚗")

    _, col0, _ = st.columns([0.15,1,0.1])
    col0.image("./imgs/test_background.jpg", width=500)

    _, col4, _ = st.columns([0.1,1,0.2])
    col4.subheader("Computer Vision Detection with YoloV8 🧪")

    _, col, _ = st.columns([0.3,1,0.1])
    col.image("./imgs/plate_test.jpg")

    _, col5, _ = st.columns([0.05,1,0.1])

    st.write("The model detects cars and license plates, extracts the text using EasyOCR, and checks the payment status in the database. Vehicles with unpaid fees will be flagged, SMS notifications will be sent, and the barrier will be controlled accordingly.")

with body:
    _, col1, _ = st.columns([0.1,1,0.2])
    col1.subheader("Check It-out the License Car Plate Detection Model 🔎!")

    _, colb1, colb2, colb3, colb4, colb5 = st.columns([0.2, 0.4, 0.4, 0.4, 0.4, 0.4])

    if colb1.button("Upload an Image", on_click=change_state_uploader):
        pass
    elif colb2.button("Take a Photo", on_click=change_state_camera):
        pass
    elif colb3.button("Live Detection", on_click=change_state_live):
        pass
    elif colb4.button("Test Database"):
        test_database_connection()
    elif colb5.button("Test SMS"):
        test_sms_functionality()

    if st.session_state["state"] == "Uploader":
        img = st.file_uploader("Upload a Car Image: ", type=["png", "jpg", "jpeg"])
    elif st.session_state["state"] == "Camera":
        img = st.camera_input("Take a Photo: ")
    elif st.session_state["state"] == "Live":
        webrtc_streamer(key="sample", video_processor_factory=VideoProcessor)
        img = None

    _, col2, _ = st.columns([0.3,1,0.2])

    _, col5, _ = st.columns([0.8,1,0.2])

    if img is not None:
        image = np.array(Image.open(img))    
        col2.image(image, width=400)

        if col5.button("Apply Detection"):
            results = model_prediction(image)

            if len(results) >= 4:
                prediction, texts, license_plate_crop, payment_statuses = results[0], results[1], results[2], results[3]

                texts = [i for i in texts if i is not None]
                
                if len(texts) == 1 and len(license_plate_crop):
                    _, col3, _ = st.columns([0.4,1,0.2])
                    col3.header("Detection Results ✅:")

                    _, col4, _ = st.columns([0.1,1,0.1])
                    col4.image(prediction)

                    _, col9, _ = st.columns([0.4,1,0.2])
                    col9.header("License Cropped ✅:")

                    _, col10, _ = st.columns([0.3,1,0.1])
                    col10.image(license_plate_crop[0], width=350)

                    _, col11, _ = st.columns([0.45,1,0.55])
                    col11.success(f"License Number: {texts[0]}")
                    
                    # Display payment status
                    if payment_statuses and payment_statuses[0] and payment_statuses[0].get('found', False):
                        status = payment_statuses[0].get('command_status', 'UNKNOWN')
                        if status == 'PAID':
                            col11.success(f"Payment Status: {status}")
                            col11.success("VEHICLE: FREE TO MOVE")
                        else:
                            col11.error(f"Payment Status: {status}")
                            col11.error("VEHICLE: BLOCKED")
                            col11.warning("SMS notification sent to vehicle owner")
                        col11.info(f"Owner: {payment_statuses[0].get('owner_name', 'Unknown')}")
                        col11.info(f"Phone: {payment_statuses[0].get('phone_number', 'Unknown')}")
                        col11.info(f"Fine Amount: {payment_statuses[0].get('fine_payment', 0)}")
                    else:
                        col11.warning("Vehicle not found in database")
                        col11.error("Barrier Control: CLOSED")
                        if payment_statuses and payment_statuses[0]:
                            col11.error(f"Error: {payment_statuses[0].get('error', 'Unknown error')}")

                    df = pd.read_csv(f"./csv_detections/detection_results.csv")
                    st.dataframe(df)
                elif len(texts) > 1 and len(license_plate_crop) > 1:
                    _, col3, _ = st.columns([0.4,1,0.2])
                    col3.header("Detection Results ✅:")

                    _, col4, _ = st.columns([0.1,1,0.1])
                    col4.image(prediction)

                    _, col9, _ = st.columns([0.4,1,0.2])
                    col9.header("License Cropped ✅:")

                    for i in range(0, len(license_plate_crop)):
                        _, col10, col11 = st.columns([0.3,1,1])
                        col10.image(license_plate_crop[i], width=350)
                        col11.success(f"License Number {i}: {texts[i]}")
                        
                        # Display payment status for each license plate
                        if i < len(payment_statuses) and payment_statuses[i] and payment_statuses[i].get('found', False):
                            status = payment_statuses[i].get('command_status', 'UNKNOWN')
                            if status == 'PAID':
                                col11.success(f"Payment Status: {status}")
                                col11.success("VEHICLE: FREE TO MOVE")
                            else:
                                col11.error(f"Payment Status: {status}")
                                col11.error("VEHICLE: BLOCKED")
                                col11.warning("SMS notification sent to vehicle owner")
                            col11.info(f"Owner: {payment_statuses[i].get('owner_name', 'Unknown')}")
                            col11.info(f"Phone: {payment_statuses[i].get('phone_number', 'Unknown')}")
                            col11.info(f"Fine Amount: {payment_statuses[i].get('fine_payment', 0)}")
                        else:
                            col11.warning("Vehicle not found in database")
                            col11.error("Barrier Control: CLOSED")
                            if i < len(payment_statuses) and payment_statuses[i]:
                                col11.error(f"Error: {payment_statuses[i].get('error', 'Unknown error')}")

                    df = pd.read_csv(f"./csv_detections/detection_results.csv")
                    st.dataframe(df)
            else:
                prediction = results[0]
                _, col3, _ = st.columns([0.4,1,0.2])
                col3.header("Detection Results ✅:")

                _, col4, _ = st.columns([0.3,1,0.1])
                col4.image(prediction)
                col4.warning("No license plates detected in the image.")