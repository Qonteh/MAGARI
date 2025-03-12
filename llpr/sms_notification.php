<?php
// sms_notification.php - Handles SMS notifications via Africa's Talking API
// This is an alternative to using Twilio directly in Python

// Get the data from POST request
$phone_number = isset($_POST['phone_number']) ? $_POST['phone_number'] : '';
$vehicle_id = isset($_POST['vehicle_id']) ? $_POST['vehicle_id'] : '';
$fine_amount = isset($_POST['fine_amount']) ? $_POST['fine_amount'] : '';
$message = isset($_POST['message']) ? $_POST['message'] : '';

// Validate data
if (empty($phone_number)) {
    echo json_encode(['status' => 'error', 'message' => 'No phone number provided']);
    exit;
}

if (empty($message)) {
    // Create default message if none provided
    $message = "WARNING: Your vehicle $vehicle_id has an unpaid fine of $fine_amount. Please make payment to avoid further penalties.";
}

// Format the phone number (ensure it has country code)
// Tanzania country code is +255
if (substr($phone_number, 0, 1) === '0') {
    $phone_number = '+255' . substr($phone_number, 1);
} elseif (substr($phone_number, 0, 1) !== '+') {
    $phone_number = '+' . $phone_number;
}

// Africa's Talking API credentials
$username = "sandbox"; // Using sandbox
$apiKey = "atsk_f094aef34b8a2f66708d6061c4af0ea42553e30f93c73a9f2a87dc73be1d2e76e056370d";
$sender = "VEHICLE_FYP"; // Your sender ID

// Log the request details for debugging
file_put_contents('sms_request_log.txt', date('Y-m-d H:i:s') . " - Attempting to send SMS to $phone_number for vehicle $vehicle_id\n", FILE_APPEND);

// Initialize the SDK
$AT = new AfricasTalking($username, $apiKey);

// Get the SMS service
$sms = $AT->sms();

try {
    // Send the message
    $result = $sms->send([
        'to' => $phone_number,
        'message' => $message,
        'from' => $sender
    ]);

    // Log the SMS
    file_put_contents('sms_log.txt', date('Y-m-d H:i:s') . " - Sent SMS to $phone_number for vehicle $vehicle_id\n", FILE_APPEND);
    file_put_contents('sms_response_log.txt', date('Y-m-d H:i:s') . " - API Response: " . json_encode($result) . "\n", FILE_APPEND);
    
    echo json_encode(['status' => 'success', 'message' => 'SMS sent successfully', 'response' => $result]);
} catch (Exception $e) {
    file_put_contents('sms_error_log.txt', date('Y-m-d H:i:s') . " - Error: " . $e->getMessage() . "\n", FILE_APPEND);
    echo json_encode(['status' => 'error', 'message' => 'Failed to send SMS: ' . $e->getMessage()]);
}

// Africa's Talking SDK class
class AfricasTalking
{
    protected $username;
    protected $apiKey;
    
    public function __construct($username, $apiKey)
    {
        $this->username = $username;
        $this->apiKey = $apiKey;
    }
    
    public function sms()
    {
        return new SMS($this->username, $this->apiKey);
    }
}

class SMS
{
    protected $username;
    protected $apiKey;
    protected $apiHost;
    
    public function __construct($username, $apiKey)
    {
        $this->username = $username;
        $this->apiKey = $apiKey;
        
        // Use the sandbox API host if using the sandbox username
        if ($username == 'sandbox') {
            $this->apiHost = 'api.sandbox.africastalking.com';
        } else {
            $this->apiHost = 'api.africastalking.com';
        }
    }
    
    public function send($options)
    {
        $to = $options['to'];
        $message = $options['message'];
        $from = isset($options['from']) ? $options['from'] : null;
        
        $data = [
            'username' => $this->username,
            'to' => $to,
            'message' => $message
        ];
        
        if ($from !== null && $this->username != 'sandbox') {
            // Sender ID is not used in sandbox mode
            $data['from'] = $from;
        }
        
        $ch = curl_init();
        $url = 'https://' . $this->apiHost . '/version1/messaging';
        
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, http_build_query($data));
        curl_setopt($ch, CURLOPT_HTTPHEADER, [
            'Accept: application/json',
            'Content-Type: application/x-www-form-urlencoded',
            'apiKey: ' . $this->apiKey
        ]);
        
        // Log the request URL and data for debugging
        file_put_contents('sms_request_details.txt', date('Y-m-d H:i:s') . " - URL: $url, Data: " . json_encode($data) . "\n", FILE_APPEND);
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);
        
        // Log the curl response and error for debugging
        file_put_contents('sms_curl_log.txt', date('Y-m-d H:i:s') . " - HTTP Code: $httpCode, Response: $response, Error: $error\n", FILE_APPEND);
        
        curl_close($ch);
        
        if ($httpCode == 201) {
            return json_decode($response, true);
        } else {
            throw new Exception("Failed to send SMS: HTTP $httpCode - $response - $error");
        }
    }
}
?>