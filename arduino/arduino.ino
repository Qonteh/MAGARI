const int ledPin = 12; // LED connected to pin 12
int incomingNumber;    // Variable to store received number

void setup() {
  Serial.begin(9600);   // Start serial communication
  pinMode(ledPin, OUTPUT); // Set LED pin as output
}

void loop() {
  if (Serial.available() > 0) {  // Check if data is received
    incomingNumber = Serial.parseInt(); // Read the number from Serial

    Serial.print("Received: ");
    Serial.println(incomingNumber); // Print received number for debugging

    if (incomingNumber == 1) {
      digitalWrite(ledPin, HIGH);  // Turn LED ON
      Serial.println("LED ON");
    } 
    else if (incomingNumber == 2) {
      digitalWrite(ledPin, LOW);   // Turn LED OFF
      Serial.println("LED OFF");
    }
  }
}
