<?php
// fetch_data.php - Returns the current command status for Arduino control

// Check if the command status file exists
if (file_exists('command_status.txt')) {
    // Read the command status
    $command_status = file_get_contents('command_status.txt');
    
    // Output just the command status (PAID or NOT PAID)
    echo trim($command_status);
} else {
    // Default to NOT PAID if file doesn't exist
    echo "NOT PAID";
}
?>