<?php
// Database connection
$host = 'localhost';
$user = 'root';
$password = '';
$database = 'vehicle_fyp';

$conn = new mysqli($host, $user, $password, $database);

// Check connection
if ($conn->connect_error) {
    die('Connection failed: ' . $conn->connect_error);
}

// Add new record
if (isset($_POST['add'])) {
    $vehicle_ID = $_POST['vehicle_ID'];
    $Owner_name = $_POST['Owner_name'];
    $Phone_number = $_POST['Phone_number'];
    $Fine_payment = $_POST['Fine_payment'];
    $Last_updated = date('Y-m-d H:i:s');

    $sql = "INSERT INTO vehicles_payment_status (vehicle_ID, Owner_name, Phone_number, fine_payment, Command_status, Last_updated) 
            VALUES ('$vehicle_ID', '$Owner_name', '$Phone_number', '$Fine_payment', 'NOT PAID', '$Last_updated')";
    $conn->query($sql);
}

// Delete record
if (isset($_GET['delete'])) {
    $vehicle_ID = $_GET['delete'];
    $sql = "DELETE FROM vehicles_payment_status WHERE vehicle_ID = '$vehicle_ID'";
    $conn->query($sql);
}

// Update record
if (isset($_POST['update'])) {
    $vehicle_ID = $_POST['vehicle_ID'];
    $Owner_name = $_POST['Owner_name'];
    $Phone_number = $_POST['Phone_number'];
    $Fine_payment = $_POST['Fine_payment'];
    $Last_updated = date('Y-m-d H:i:s');

    $sql = "UPDATE vehicles_payment_status SET Owner_name='$Owner_name', Phone_number='$Phone_number', fine_payment='$Fine_payment', Last_updated='$Last_updated' WHERE vehicle_ID='$vehicle_ID'";
    $conn->query($sql);
}

// Fetch data
$sql = "SELECT vehicle_ID, Owner_name, Phone_number, fine_payment, Command_status, Last_updated FROM vehicles_payment_status";
$result = $conn->query($sql);
?>

<!DOCTYPE html>
<html>
<head>
    <title>Vehicle Payment Status CRUD</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #282c34;
            color: #fff;
            margin: 0;
            padding: 20px;
        }

        h2 {
            text-align: center;
            margin-bottom: 20px;
            color: #61dafb;
        }

        table, th, td {
            border: 1px solid #444;
            padding: 10px;
            margin: 0 auto;
            border-collapse: collapse;
            background-color: #333;
        }

        th {
            background-color: #444;
        }

        button, input[type='text'] {
            margin: 5px;
            padding: 8px;
            border-radius: 5px;
            border: none;
            color: #fff;
            background-color: #555;
            cursor: pointer;
        }

        button:hover {
            background-color: #777;
        }

        .btn-delete {
            background-color: #e74c3c;
        }

        .btn-edit {
            background-color: #3498db;
        }

        .btn-add {
            background-color: #2ecc71;
        }

        .footer {
            text-align: center;
            margin-top: 20px;
        }
    </style>
</head>
<body>

<h2>🚗 Vehicle Payment Status CRUD 🚦</h2>

<form method="POST">
    <input type="text" name="vehicle_ID" placeholder="Vehicle ID" required>
    <input type="text" name="Owner_name" placeholder="Owner Name" required>
    <input type="text" name="Phone_number" placeholder="Phone Number" required>
    <input type="text" name="Fine_payment" placeholder="Fine Payment" required>
    <button type="submit" name="add" class="btn-add">Add New</button>
</form>

<table>
    <tr>
        <th>Vehicle ID</th>
        <th>Owner Name</th>
        <th>Phone Number</th>
        <th>Fine Payment</th>
        <th>Command Status</th>
        <th>Last Updated</th>
        <th>Actions</th>
    </tr>

    <?php
    if ($result->num_rows > 0) {
        while ($row = $result->fetch_assoc()) {
            echo "<tr>";
            echo "<td>{$row['vehicle_ID']}</td>";
            echo "<td>{$row['Owner_name']}</td>";
            echo "<td>{$row['Phone_number']}</td>";
            echo "<td>{$row['fine_payment']}</td>";
            echo "<td>{$row['Command_status']}</td>";
            echo "<td>{$row['Last_updated']}</td>";
            echo "<td>
                <a href='?delete={$row['vehicle_ID']}' class='btn-delete'>Delete</a>
                <form method='POST' style='display:inline;'>
                    <input type='hidden' name='vehicle_ID' value='{$row['vehicle_ID']}'>
                    <input type='text' name='Owner_name' value='{$row['Owner_name']}' required>
                    <input type='text' name='Phone_number' value='{$row['Phone_number']}' required>
                    <input type='text' name='Fine_payment' value='{$row['fine_payment']}' required>
                    <button type='submit' name='update' class='btn-edit'>Update</button>
                </form>
            </td>";
            echo "</tr>";
        }
    } else {
        echo "<tr><td colspan='7'>No data available</td></tr>";
    }
    $conn->close();
    ?>
</table>

<div class="footer">
    © <?php echo date('Y'); ?> Vehicle Payment System. All Rights Reserved.<br>
    <div >
            Designed by Brother_Qonte | <a href="tel:+255692438585">Call: 0692438585</a>
    </div>
</div>

</body>
</html>
