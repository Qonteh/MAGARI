import requests
import serial
import time
import json
import os

# COM port where COMPIM is configured in Proteus (e.g., COM3)
ser = serial.Serial('COM3', 9600, timeout=1)  # Replace with your actual COM port

# URL to fetch data from PHP (running on XAMPP server)
url = "http://localhost/QONTE/fetch_data.php"

# URL to fetch detailed data (optional)
detail_url = "http://localhost/QONTE/latest_detection.json"

def fetch_command_status():
    try:
        response = requests.get(url)
        if response.status_code == 200:
            status = response.text.strip()  # Get the command status (PAID or NOT PAID)
            print(f"Command status fetched: {status}")
            
            # Try to get detailed information
            try:
                detail_response = requests.get(detail_url)
                if detail_response.status_code == 200:
                    data = json.loads(detail_response.text)
                    print(f"Vehicle ID: {data.get('vehicle_id', 'Unknown')}")
                    print(f"Timestamp: {data.get('timestamp', 'Unknown')}")
            except Exception as e:
                print(f"Error fetching detailed data: {e}")
                
            return status
        else:
            print(f"Error: Received status code {response.status_code}")
            return "NOT PAID"  # Default status if fetch fails
    except Exception as e:
        print(f"Error fetching data: {e}")
        return "NOT PAID"

def send_to_proteus(command):
    if command == "PAID":
        ser.write(b'1')  # Send '1' for PAID - OPEN BARRIER
        print("Sent to Proteus: 1 (PAID) - VEHICLE FREE TO MOVE")
    elif command == "NOT PAID":
        ser.write(b'2')  # Send '2' for NOT PAID - BLOCK VEHICLE
        print("Sent to Proteus: 2 (NOT PAID) - VEHICLE BLOCKED")
    else:
        ser.write(b'2')  # Default to blocked for unknown status
        print(f"Unknown status '{command}', defaulting to VEHICLE BLOCKED")

def main():
    print("Starting License Plate Detection Arduino Control System")
    print("------------------------------------------------------")
    print("This system will check for vehicle payment status and control the barrier")
    print("PAID status = VEHICLE FREE TO MOVE")
    print("NOT PAID status = VEHICLE BLOCKED")
    print("------------------------------------------------------")
    
    last_status = None
    
    while True:
        command_status = fetch_command_status()  # Fetch the command status
        
        # Only send to Arduino if status has changed (to reduce serial traffic)
        if command_status != last_status:
            send_to_proteus(command_status)  # Send the status to Proteus
            last_status = command_status
        
        time.sleep(2)  # Check every 2 seconds

if __name__ == "__main__":
    main()