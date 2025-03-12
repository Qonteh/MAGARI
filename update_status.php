<?php
// update_status.php - Receives vehicle_id and command_status from app.py
// and updates a text file that will be read by fyp.py

// Get the data from POST request
$vehicle_id = isset($_POST['vehicle_id']) ? $_POST['vehicle_id'] : '';
$command_status = isset($_POST['command_status']) ? $_POST['command_status'] : 'NOT PAID';

// Validate data
if (empty($vehicle_id)) {
    echo "Error: No vehicle ID provided";
    exit;
}

// Create data array
$data = [
    'vehicle_id' => $vehicle_id,
    'command_status' => $command_status,
    'timestamp' => date('Y-m-d H:i:s')
];

// Save to a JSON file
file_put_contents('latest_detection.json', json_encode($data));

// Also save just the command status for simple reading by fyp.py
file_put_contents('command_status.txt', $command_status);

echo "Success: Updated status for vehicle $vehicle_id to $command_status";
?>