#include <Wire.h>
#include<Energia.h>
#include "accelerometer.h"
#include <math.h>
#include <stdint.h>
#include <SPI.h>
#include <WiFi.h>
#include <WifiIPStack.h>
#include <Countdown.h>
#include <MQTTClient.h>

// your network name also called SSID
char ssid[] = "UCL_EE_IOT";
// your network password
char password[] = "PI3+IOT+eduroam";
// IBM IoT Foundation Cloud Settings
// When adding a device on internetofthings.ibmcloud.com the following
// information will be generated:
// org=<org>
// type=iotsample-ti-energia
// id=<mac>
// auth-method=token
// auth-token=<password>

#define MQTT_MAX_PACKET_SIZE 100
#define IBMSERVERURLLEN  64
#define IBMIOTFSERVERSUFFIX "messaging.internetofthings.ibmcloud.com"

char organization[] = "vs51te";
char typeId[] = "cc3200";
char pubtopic[] = "iot-2/evt/status/fmt/json";
char subTopic[] = "iot-2/cmd/+/fmt/json";
char deviceId[] = "d4f51303f038";
char clientId[64];

char mqttAddr[IBMSERVERURLLEN];
int mqttPort = 1883;

// Authentication method. Should be use-toke-auth
// When using authenticated mode
char authMethod[] = "use-token-auth";
// The auth-token from the information above
char authToken[] = "l0e32j7XTgJMfgPNoT";
int ledPin = RED_LED;
MACAddress mac;

WifiIPStack ipstack;  
MQTT::Client<WifiIPStack, Countdown, MQTT_MAX_PACKET_SIZE> client(ipstack);

// The function to call when a message arrives
void callback(char* topic, byte* payload, unsigned int length);
void messageArrived(MQTT::MessageData& md);

void setup() {
  uint8_t macOctets[6];
  
  Serial.begin(115200);
  Wire.begin();
  pinMode(RED_LED, OUTPUT);
  digitalWrite(RED_LED, LOW);
  //pinMode(YELLOW_LED,OUTPUT);
  //digitalWrite(YELLOW_LED, LOW);
//  pinMode(GREEN_LED,OUTPUT);
 // digitalWrite(GREEN_LED, LOW);
   // attempt to connect to Wifi network:
  Serial.print("Attempting to connect to Network named: ");
  // print the network name (SSID);
  Serial.println(ssid); 
  // Connect to WPA/WPA2 network. Change this line if using open or WEP network:
  WiFi.begin(ssid, password);
  while ( WiFi.status() != WL_CONNECTED) {
    // print dots while we wait to connect
    Serial.print(".");
    delay(300);
  }
  
  Serial.println("\nYou're connected to the network");
  Serial.println("Waiting for an ip address");
  
  while (WiFi.localIP() == INADDR_NONE) {
    // print dots while we wait for an ip addresss
    Serial.print(".");
    delay(300);
  }

  // We are connected and have an IP address.
  Serial.print("\nIP Address obtained: ");
  Serial.println(WiFi.localIP());

  mac = WiFi.macAddress(macOctets);
  Serial.print("MAC Address: ");
  Serial.println(mac);
  
  // Use MAC Address as deviceId
  sprintf(deviceId, "%02x%02x%02x%02x%02x%02x", macOctets[0], macOctets[1], macOctets[2], macOctets[3], macOctets[4], macOctets[5]);
  Serial.print("deviceId: ");
  Serial.println(deviceId);

  sprintf(clientId, "d:%s:%s:%s", organization, typeId, deviceId);
  sprintf(mqttAddr, "%s.%s", organization, IBMIOTFSERVERSUFFIX);
}

void loop() {  
  int rc = -1;
  if (!client.isConnected()) {
    Serial.print("Connecting to ");
    Serial.print(mqttAddr);
    Serial.print(":");
    Serial.println(mqttPort);
    Serial.print("With client id: ");
    Serial.println(clientId);
    
    while (rc != 0) {
      rc = ipstack.connect(mqttAddr, mqttPort);
    }

    MQTTPacket_connectData connectData = MQTTPacket_connectData_initializer;
    connectData.MQTTVersion = 3;
    connectData.clientID.cstring = clientId;
    connectData.username.cstring = authMethod;
    connectData.password.cstring = authToken;
    connectData.keepAliveInterval = 10;
    
    rc = -1;
    while ((rc = client.connect(connectData)) != 0)
      ;
    Serial.println("Connected\n");
    
    Serial.print("Subscribing to topic: ");
    Serial.println(subTopic);
    
    // Unsubscribe the topic, if it had subscribed it before.
    client.unsubscribe(subTopic);
    // Try to subscribe for commands
    if ((rc = client.subscribe(subTopic, MQTT::QOS0, messageArrived)) != 0) {
      Serial.print("Subscribe failed with return code : ");
      Serial.println(rc);
    } else {
      Serial.println("Subscribe success\n");
    }
  }
   char json[56] = "{\"d\":{\"myName\":\"TILaunchPad\",\"acceleration\":";
//accelerometer
  AccData acc = readAccelerometer();//change made = float changed to double
  double z_gravity= 66.0;
  double gravity = 9.81; //in m/s^2
  double x_range = 127.0; //digital unit 
  double y_range = 127.0; //digital unit 
  double z_range = 130.0; //digital unit
  
  double para_z = acc.z-z_gravity; //to eliminate gravity element in z-axis of accelerometer
  
  double x = (acc.x/x_range)*2*gravity; //convert digital unit to m/s^2
  double y = (acc.y/y_range)*2*gravity; 
  double z = (para_z/z_range)*2*gravity;
  
  double force_acceleration= sqrt(sq(x)+sq(y)+sq(z)); //in m/s^2
  //float force_acceleration = acc.z;
  double shock_boundary = 2.0; // in m/s^2 ,   to be set by users
  
  
//print 
  Serial.println("Current acceleration onto the package");  
  Serial.print(force_acceleration);
  Serial.println(" m/s^2");

//delay(1000);
  
  if (force_acceleration > shock_boundary) { 
    digitalWrite(RED_LED, HIGH);
    Serial.println("A serious shock has just occurred!");
    delay(1000);// 1s
  } 

  digitalWrite(RED_LED, LOW);
  
  dtostrf(force_acceleration,1,2, &json[43]);
  json[48] = '}';
  json[49] = '}';
  json[50] = '\0';
  Serial.print("Publishing: ");
  Serial.println(json);
  MQTT::Message message;
  message.qos = MQTT::QOS0; 
  message.retained = false;
  message.payload = json; 
  message.payloadlen = strlen(json);
  rc = client.publish(pubtopic, message);
  if (rc != 0) {
    Serial.print("Message publish failed with return code : ");
    Serial.println(rc);
  }
  
  // Wait for one second before publishing again
  // This will also service any incoming messages
  client.yield(5000);
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.println("Message has arrived");
  
  char * msg = (char *)malloc(length * sizeof(char));
  int count = 0;
  for(count = 0 ; count < length ; count++) {
    msg[count] = payload[count];
  }
  msg[count] = '\0';
  Serial.println(msg);
  
  if(length > 0) {
    digitalWrite(ledPin, HIGH);
    delay(1000);
    digitalWrite(ledPin, LOW);  
  }

  free(msg);
}

void messageArrived(MQTT::MessageData& md) {
  Serial.print("Message Received\t");
    MQTT::Message &message = md.message;
    int topicLen = strlen(md.topicName.lenstring.data) + 1;
//    char* topic = new char[topicLen];
    char * topic = (char *)malloc(topicLen * sizeof(char));
    topic = md.topicName.lenstring.data;
    topic[topicLen] = '\0';
    
    int payloadLen = message.payloadlen + 1;
//    char* payload = new char[payloadLen];
    char * payload = (char*)message.payload;
    payload[payloadLen] = '\0';
    
    String topicStr = topic;
    String payloadStr = payload;
    
    //Command topic: iot-2/cmd/blink/fmt/json

    if(strstr(topic, "/cmd/blink") != NULL) {
      Serial.print("Command IS Supported : ");
      Serial.print(payload);
      Serial.println("\t.....");
      
      pinMode(ledPin, OUTPUT);
      
      //Blink twice
      for(int i = 0 ; i < 2 ; i++ ) {
        digitalWrite(ledPin, HIGH);
        delay(250);
        digitalWrite(ledPin, LOW);
        delay(250);
      }
    } else {
      Serial.println("Command Not Supported:");            
    }
}

  
//LDR  
//int sensorValue = 0;
//int sensorPin = 2; //a.k.a P58 select the input pin for LDR
//sensorValue = analogRead(sensorPin); // variable to store the value coming from the sensor
 // Serial.println("LDR Sensor Value");
  //Serial.println(sensorValue);
    
 //if(sensorValue<4090){ //it's 4095 when there's shadow or LDR is covered
   // digitalWrite(RED_LED, HIGH);
    //Serial.println("The package is opened or not entirely sealed!");
    //delay(1000);//1s
  //}
 //digitalWrite(RED_LED, LOW);
    
//}
