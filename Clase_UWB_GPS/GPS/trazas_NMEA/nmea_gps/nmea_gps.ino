#define SERIAL1_RX 34 // GPS_TX -> 34
#define SERIAL1_TX 12 // 12 -> GPS_RX

void setup(){
  Serial.begin(115200);
  Serial.println("TTGO GPS TEST");
  delay(2000);
  Serial1.begin(9600, SERIAL_8N1, SERIAL1_RX, SERIAL1_TX);
}

void loop() {
  if (Serial1.available()) {
    Serial.write(Serial1.read());
    Serial1.println();
  }
}
