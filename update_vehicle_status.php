<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *'); // Allow requests from any origin (for Vercel app)
header('Access-Control-Allow-Methods: POST, GET, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With');

// Database configuration (ensure this matches your Hostinger MySQL details)
$servername = "localhost";
$username = "mhzspamy_fyp"; // Your Hostinger database username
$password = "Qontetina051@"; // Your Hostinger database password
$dbname = "mhzspamy_vehicle"; // Your Hostinger database name

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    echo json_encode(["success" => false, "message" => "Database connection failed: " . $conn->connect_error]);
    exit();
}

// Handle preflight OPTIONS request
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

// Expecting POST request for updates
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $input = json_decode(file_get_contents('php://input'), true);

    $vehicle_id = isset($input['vehicle_id']) ? $conn->real_escape_string($input['vehicle_id']) : '';
    $new_status = isset($input['new_status']) ? $conn->real_escape_string($input['new_status']) : '';

    if (empty($vehicle_id) || empty($new_status)) {
        echo json_encode(["success" => false, "message" => "Missing vehicle_id or new_status."]);
        exit();
    }

    // Format license plate by removing spaces for consistency with database storage
    $formatted_vehicle_id = str_replace(' ', '', strtoupper($vehicle_id));

    // Update query
    $sql = "UPDATE vehicles_payment_status SET command_status = ?, Last_updated = NOW() WHERE REPLACE(vehicle_ID, ' ', '') = ?";
    $stmt = $conn->prepare($sql);

    if ($stmt === false) {
        echo json_encode(["success" => false, "message" => "Prepare failed: " . $conn->error]);
        exit();
    }

    $stmt->bind_param("ss", $new_status, $formatted_vehicle_id);

    if ($stmt->execute()) {
        if ($stmt->affected_rows > 0) {
            echo json_encode(["success" => true, "message" => "Vehicle status updated successfully."]);
        } else {
            echo json_encode(["success" => false, "message" => "No vehicle found with ID: " . $vehicle_id . " or status already " . $new_status . "."]);
        }
    } else {
        echo json_encode(["success" => false, "message" => "Execute failed: " . $stmt->error]);
    }

    $stmt->close();
} else {
    echo json_encode(["success" => false, "message" => "Invalid request method. Only POST is allowed."]);
}

$conn->close();
?>
