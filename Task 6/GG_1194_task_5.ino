▪ /*Team Id: GG_1194
▪ * Author List: Ashish Rathore, Keshav Joshi, Disha Chhabra
▪ * Filename: Task_6 Bot_code
▪ * Theme: <Eyrc Geo Guide (GG)>
▪ * Functions: <determineMotion, determinePosition, stopCondition, moveForwardPWM>
▪ * Global Variables: <
Motion, prevNode, currentNode, nextNode, isTurning, turnNow, pathIndex, lastIndex, durationInSeconds, startTime, 
currentTime, interval, currentState, detected, moveOnLine, boxPathIndex, currentPathIndex, numRows, sensorActiveCount, 
isBotReached, changeState, stopIf, currentPath>
▪ */


#include <WiFi.h>
#define LED 2

char incomingPacket[80];
WiFiClient client;

String msg = "0";
int counter = 0;
int countBlink=0;
bool loopBroken = false; // Flag to indicate if the loop has been broken

const char* ssid = "ASUS";                    
const char* password =  "ASDFGHJKL";               
const uint16_t port = 8002;
const char * host = "192.168.110.58"; 

#define SIDE_LEFT 5
#define LEFT_SENSOR  18
#define CENTER_SENSOR 19
#define RIGHT_SENSOR 21
#define SIDE_RIGHT 22
#define BUZZER 13
#define LED_GREEN 14
#define LED_RED 23

#define LEFT_MOTOR_IN1 16
#define LEFT_MOTOR_IN2 4
#define RIGHT_MOTOR_IN3 15
#define RIGHT_MOTOR_IN4 2

// PWM values for different motions
int FULL_SPEED = 230;
int STOP_SPEED = 0;
int TURN_SPEED = 255;
int EXTREME_TURN_SPEED = 200; // Adjust this value for reduced speed during turns

enum Detected {
  centerOfLine,
  leftOfLine,
  rightOfLine,
  node,
  bothSideLine,
  sideRightLine,
  sideLeftLine,
  allThreeLine,
  none,
};

enum State{
  pathPlanning,
  waitingAtStart,
  followLine,
  onNode,
  botTurning,
  updatePosition,
  endRun,
  buzzerLed,
  reachedEventLocation,
  stopAtEvent,
  turningAround,
  turningAroundLine
};

enum Motion {
  forward,
  left,
  right,
  eRight,
  eLeft,
  sideLeft,
  sideRight,
  stop,
  aboutTurn  
};


Motion movement;
int prevNode = 0; 
int currentNode = 0;
int nextNode = 1;
bool isTurning = false;
bool turnNow=false;
int pathIndex=2;
int lastIndex=0;
const int durationInSeconds = 10;  // Set the duration of the timer in seconds
unsigned long startTime;  // Variable to store the start time
unsigned long currentTime;
unsigned long interval;
State currentState;
Detected detected;
Motion moveOnLine;

int boxPathIndex=0;
int currentPathIndex=0;
int numRows=0;
int sensorActiveCount=0;
bool isBotReached=false;
bool changeState=false;
Detected stopIf;

int currentPath[55];

void moveForwardPWM(int leftSpeed, int reverseLSpeed, int rightSpeed, int reverseRSpeed) {
  analogWrite(LEFT_MOTOR_IN1, leftSpeed);
  analogWrite(LEFT_MOTOR_IN2, reverseLSpeed);  // Assume LOW for forward motion
  analogWrite(RIGHT_MOTOR_IN3, rightSpeed);
  analogWrite(RIGHT_MOTOR_IN4, reverseRSpeed);  // Assume LOW for forward motion
}

Motion myArray[12][12][12];


▪ /*
▪ * Function Name:<determineMotion>
▪ * Input: <detect - the detected positon of bot w.r.t lines>
▪ * Output: <motion followed for each position>
▪ * Logic: <this function is used to make bot follow line, by sending the motion required to turn the bot to the middle line>
▪ * Example Call: <determineMotion(leftOfLine)>
▪ */ 

Motion determineMotion(Detected detect){

  if (detect == centerOfLine || detect == bothSideLine || detect == allThreeLine ) {
    moveOnLine = forward;

} else if (detect == leftOfLine) {
    moveOnLine = left;

} else if (detect == rightOfLine) {
    moveOnLine = right;

} else if (detect == sideRightLine && (currentNode == 9 || nextNode==9)) {
    moveOnLine = sideLeft;

} else if (detect == sideRightLine) {
    moveOnLine = left;

} else if (detect == sideLeftLine && ((currentNode==4 && nextNode==9) || (currentNode==9 && nextNode==4) ))  {
    moveOnLine = sideRight;

} else if (detect == sideLeftLine ) {
    moveOnLine = right;

} else if (detect == node) {
    moveOnLine = stop;
    
} else moveOnLine=moveOnLine;

 return moveOnLine;

}

▪ /*
▪ * Function Name:<determinePosition>
▪ * Input: <the value of all 5 IR sensors on the bot>
▪ * Output: <the positon of bot w.r.t to lines>
▪ * Logic: <this function is used to make detect the positon of bot w.r.t lines by using sensor values>
▪ * Example Call: <determinePosition(!leftSensor && rightSensor)>
▪ */ 

Detected determinePosition(bool sideLeftSensor, bool leftSensor, bool centerSensor, bool rightSensor, bool sideRightSensor) {

  if (!sideLeftSensor && !leftSensor && centerSensor && !rightSensor && !sideRightSensor) {
    // Node (center detected)
    detected = centerOfLine;

  } else if (!leftSensor && rightSensor) {
    // Right of the line
    detected = rightOfLine;

  } else if (leftSensor && !rightSensor) {
    // Left of the line
    detected = leftOfLine;

  } else if (leftSensor && centerSensor && rightSensor ) {
    // Left of the line
    detected = node;

  } else if (sideLeftSensor && !leftSensor && !centerSensor && !rightSensor && sideRightSensor) {
    // Both sides of the line
    detected = bothSideLine;

  } else if (!sideLeftSensor && !leftSensor &&  !centerSensor &&!rightSensor && sideRightSensor) {
    // Right side of the line
    detected = sideRightLine;

  } else if (sideLeftSensor && !leftSensor && !centerSensor && !rightSensor && !sideRightSensor) {
    // Left side of the line
    detected = sideLeftLine;

  } else if (!sideLeftSensor && !leftSensor && !centerSensor && !rightSensor && !sideRightSensor) {
    // Right side of the line
    detected = allThreeLine;

  } else {
    // Default Case
     detected = detected;
  }

  return detected;
}

▪ /*
▪ * Function Name:<stopCondition>
▪ * Input: <node>
▪ * Output: <the condition to stop>
▪ * Logic: <this function is used to find the positon of bot with after which it has to stop so that it reaches the bot location>
▪ * Example Call: <determinePosition(!leftSensor && rightSensor)>
▪ */

Detected stopCondition(int node) {
    if(node == 1 || node == 3 || node==11 || node==10){
      return allThreeLine;
    }

    if(node == 6 || node ==7 || node == 8){
      return centerOfLine;
    }

    if(node==4){
      return allThreeLine;
    }
  }



void setup() {
  Serial.begin(115200);
  currentState=pathPlanning;

  pinMode(LED, OUTPUT);
  digitalWrite(LED, LOW);
  WiFi.begin(ssid, password);

  pinMode(LEFT_SENSOR, INPUT);
  pinMode(CENTER_SENSOR, INPUT);
  pinMode(RIGHT_SENSOR, INPUT);

  pinMode(LEFT_MOTOR_IN1, OUTPUT);
  pinMode(LEFT_MOTOR_IN2, OUTPUT);
  pinMode(RIGHT_MOTOR_IN3, OUTPUT);
  pinMode(RIGHT_MOTOR_IN4, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_RED, OUTPUT);

  digitalWrite(BUZZER, HIGH);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_RED, LOW);



//Setting direction for bot
myArray[0][0][1]=forward;

myArray[0][1][2]=forward; myArray[2][1][0]=forward; myArray[0][1][8]=eRight; myArray[8][1][0]=eLeft;
myArray[8][1][2]=eRight; myArray[2][1][8]=eLeft;

myArray[1][2][3]=forward; myArray[3][2][1]=forward; myArray[1][2][7]=eRight; myArray[7][2][1]=eLeft;
myArray[1][2][7]=eRight; myArray[7][2][1]=eLeft; myArray[3][2][7]=eLeft; myArray[7][2][3]=eRight;

myArray[6][3][4]=eRight; myArray[4][3][6]=eLeft;
myArray[2][3][4]=forward; myArray[4][3][2]=forward; myArray[2][3][6]=eRight; myArray[6][3][2]=eLeft;

myArray[3][4][5]=eRight; myArray[5][4][3]=eLeft; myArray[3][4][9]=forward; myArray[9][4][3]=forward; myArray[5][4][9]=eRight; myArray[9][4][5]=eLeft;

myArray[4][5][6]=eRight; myArray[6][5][4]=eLeft; myArray[9][5][6]=eLeft; myArray[6][5][9]=eRight;
myArray[4][5][9]=forward; myArray[9][5][4]=forward;

myArray[3][6][7]=eRight; myArray[7][6][3]=eLeft; myArray[5][6][7]=forward; myArray[7][6][5]=forward; myArray[3][6][5]=eLeft; myArray[5][6][3]=eRight; 
myArray[10][6][7]=eLeft; myArray[10][6][5]=eRight; myArray[7][6][10]=eRight; myArray[5][6][10]=eLeft; myArray[3][6][10]=forward; myArray[10][6][3]=forward;

myArray[2][7][8]=eRight; myArray[11][7][8]=eLeft; myArray[2][7][11]=forward; myArray[6][7][8]=forward; myArray[11][7][6]=eRight; myArray[2][7][6]=eLeft;
myArray[8][7][2]=eLeft; myArray[8][7][11]=eRight; myArray[11][7][2]=forward; myArray[8][7][6]=forward; myArray[6][7][11]=eLeft; myArray[6][7][2]=eRight;

myArray[1][8][7]= eLeft ; myArray[7][8][1]=eRight; myArray[11][8][7]=eRight; myArray[7][8][11]=eLeft; myArray[1][8][11]=forward; myArray[11][8][1]=forward; 

myArray[4][9][10]=forward; myArray[10][9][4]=forward; myArray[5][9][4]=eLeft; myArray[4][9][5]=eRight; myArray[5][9][10]=eRight; myArray[10][9][5]=eLeft;

myArray[9][10][11]=forward; myArray[9][10][6]=eRight; myArray[6][10][11]=eRight; myArray[11][10][9]=forward; myArray[6][10][9]=eLeft; myArray[11][10][6]=eLeft;
myArray[9][10][11]=forward; myArray[9][10][6]=eRight; myArray[6][10][11]=eRight; myArray[11][10][9]=forward; myArray[6][10][9]=eLeft; myArray[11][10][6]=eLeft;

myArray[7][11][8]=eRight; myArray[8][11][7]=eLeft; myArray[10][11][8]=forward; myArray[8][11][10]=forward; myArray[10][11][7]=eRight; myArray[7][11][10]=eLeft;

  while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.println("...");
    }
  
    Serial.print("WiFi connected with IP: ");
    Serial.println(WiFi.localIP());
    currentState = pathPlanning;


}




void loop() {
    // Read sensor values
  bool leftSensor = digitalRead(LEFT_SENSOR);
  bool centerSensor = digitalRead(CENTER_SENSOR);
  bool rightSensor = digitalRead(RIGHT_SENSOR);
  bool sideLeftSensor = digitalRead(SIDE_LEFT);
  bool sideRightSensor = digitalRead(SIDE_RIGHT);

  // Determine the detected position and the corresponding motion
  detected = determinePosition(sideLeftSensor, leftSensor, centerSensor, rightSensor, sideRightSensor);
  moveOnLine = determineMotion(detected);

  // State machine to control the movement of the bot and changing the state
 switch(currentState) {
    case pathPlanning  :      
      if (!client.connect(host, port)) {
        delay(200);
        return;
      }

      while(1) {
        msg = client.readStringUntil('\n');         
        client.print("hey!");                       

        if(msg[0] == '[') {
          loopBroken = true;
          break;
        }          
      }

      if (loopBroken) {
        int i = 0;
        char tempMsg[msg.length() + 1];
        strcpy(tempMsg, msg.c_str()); 
        char* ptr = strtok(tempMsg, ",[] ");
        while (ptr != NULL && i < 55) {
          currentPath[i++] = atoi(ptr);
          ptr = strtok(NULL, ",[] ");
        }
        
        currentState = waitingAtStart;
      }

    break;

    // Start the bot when it detects the center of the line
  case waitingAtStart :
    movement=stop;
    if(detected==centerOfLine){
      currentState=followLine;
      movement=moveOnLine;
    }

    break;
  
  case followLine :
    // Continue following the line
    movement=moveOnLine;
    if(isBotReached==true && detected==allThreeLine){
      currentState=reachedEventLocation; 
      startTime=millis();
    }
    
    if(detected==node){
      // If a node is detected, transition to the updatePosition state
      currentState=updatePosition;
    }
    
    if(detected==allThreeLine && currentPath[pathIndex]==27) {
      //marks the end of the run if node 27 detected
      movement=stop;
      digitalWrite(BUZZER, LOW);
      digitalWrite(LED_RED, HIGH);
      currentState=endRun;
      movement=stop;
    }
  
    break;

  case updatePosition :
    //At each node update current position of bot with previousNode, currentNode and nextNode
    prevNode=currentNode;
    currentNode=nextNode;
    if(currentPath[pathIndex+1] == 23){
      nextNode=currentPath[pathIndex] ;
      Serial.print("Next Node is set to");
      Serial.println(nextNode);
      isBotReached=true;
    }
    else {
      nextNode=currentPath[pathIndex];
      }

    pathIndex++;
    currentState=onNode;
    startTime=millis();
    break;
  
  case onNode :
  //find the direction of turns based on currentNode , nextNode, previousNode
    currentTime=millis();
    interval=currentTime-startTime;
    if(detected==node){
      movement=forward;
  } else {
    movement=moveOnLine;
    if((detected == centerOfLine || detected==allThreeLine) && interval >150){
      currentState=botTurning;
      movement=myArray[prevNode][currentNode][nextNode];
    }

  } 
  
  break;

  case botTurning :
    //move a bit forward before starting to turn for better turning accurcy
    //turning the bot left, right, forward on reaching node
    if(changeState==false && movement==eRight && detected==rightOfLine) {
    movement=moveOnLine;
    startTime=millis();
    changeState=true;

  } else if(changeState==false && movement==eLeft && detected==leftOfLine) {
    movement=moveOnLine;
    startTime=millis();
    changeState=true;

  } else if(changeState==false && movement==forward && detected!=node){
    movement=moveOnLine;
    startTime=millis();
    changeState=true;
  }

  if(changeState==true){
      if(detected==centerOfLine && interval>70){
        currentState=followLine;
        changeState=false;
      }else
      {
        currentTime=millis();
        interval=currentTime-startTime;
        movement=moveOnLine;
      }
  }
  break;

  case reachedEventLocation:
    //this states is when the bot reaches the nearest node from the event
    //We check the stopCondition and use it to stop the bot
    movement=moveOnLine;
    stopIf = stopCondition(currentNode);
    currentTime=millis();
    interval=currentTime-startTime;

    if(detected==stopIf && currentNode != 4 && interval>400) {
      movement=stop;
      currentState=stopAtEvent;
      digitalWrite(BUZZER, LOW);
      startTime=millis();

    }

    if(detected==stopIf && currentNode == 4 ) {
      if(interval>3500) {
        movement=stop;
        currentState=stopAtEvent;
        digitalWrite(BUZZER, LOW);
        startTime=millis();
        }
    }

  break;

  case stopAtEvent:
    //stop the bot for desired time of 1sec and checks if it hass to go forward or turnaround
    currentTime=millis();
    interval=currentTime-startTime;
    isBotReached=false;
    if(interval>=1000){
      digitalWrite(BUZZER, HIGH);
      pathIndex++;
      if(currentNode == currentPath[pathIndex]) {
      prevNode=nextNode;
      currentNode=nextNode;
      nextNode=currentPath[pathIndex];
      pathIndex++;
        if(stopIf == centerOfLine || currentNode == 9){
        currentState=turningAroundLine;
        } else {
          currentState = turningAround;
        }
      movement = aboutTurn;
      startTime=millis();
      } else {
      pathIndex++;
      currentState=followLine;
      }
    }
  break;

  case turningAround :
    // turning around in places of 3 lines
    if(sideRightSensor && sensorActiveCount==0) {
      sensorActiveCount++;
    }

    if(sideLeftSensor && sensorActiveCount==1){
      sensorActiveCount++;
    }

    if(leftSensor && sensorActiveCount==2){
      movement=moveOnLine;
      sensorActiveCount=0;
      currentState=followLine;
    }
  break;

  case turningAroundLine :
  //turniong around in places of two sidelines
    currentTime=millis();
    interval=currentTime-startTime;

    if(interval<600 && currentNode == 9){
      sensorActiveCount=9;
      movement=moveOnLine;
    } 
    else if(interval>600 && sensorActiveCount == 9){
        sensorActiveCount=0;
        movement=aboutTurn;
    }

    if(sideLeftSensor && sensorActiveCount==0){
      sensorActiveCount++;
    }

    if(leftSensor && sensorActiveCount==1){
      sensorActiveCount++;
    }

    if(rightSensor && sensorActiveCount==2){
      sensorActiveCount++;
    }

    if(sideRightSensor && sensorActiveCount==3){
      sensorActiveCount++;
    }

    if(leftSensor && sensorActiveCount==4){
      sensorActiveCount++;
    }

    if(sideRightSensor && sensorActiveCount==5){
      movement=moveOnLine;
      sensorActiveCount=0;
      currentState=followLine;
    }

  break;

  case endRun :
    movement=stop;
      // this stops the bot and play the buzzer for 5 secs.
      delay(5000);
      digitalWrite(BUZZER, HIGH);
      digitalWrite(LED_RED, LOW);
    break;

 }

    //this is used to control the motors based on the value of movement from the state machine
    switch (movement){
        case forward:
          moveForwardPWM(FULL_SPEED,STOP_SPEED,FULL_SPEED,STOP_SPEED);
          break;

        case left:
          moveForwardPWM(FULL_SPEED,STOP_SPEED,TURN_SPEED,STOP_SPEED);
          break;

        case right:
          moveForwardPWM(TURN_SPEED,STOP_SPEED,FULL_SPEED,STOP_SPEED);
          break;
        
        case eLeft:
          moveForwardPWM(STOP_SPEED,EXTREME_TURN_SPEED,EXTREME_TURN_SPEED+20,STOP_SPEED);
        break;
        
        case eRight:
          moveForwardPWM(EXTREME_TURN_SPEED,STOP_SPEED,STOP_SPEED,EXTREME_TURN_SPEED-20);
          break; 

        case sideLeft:
          moveForwardPWM(STOP_SPEED,TURN_SPEED,TURN_SPEED-30,STOP_SPEED); 
          break;

        case sideRight:
           moveForwardPWM(TURN_SPEED-30,STOP_SPEED,STOP_SPEED,TURN_SPEED);
           break;                                                                                     

        case stop:
          moveForwardPWM(STOP_SPEED,STOP_SPEED,STOP_SPEED,STOP_SPEED);
          break;

        case aboutTurn:
          moveForwardPWM(STOP_SPEED,TURN_SPEED-20,TURN_SPEED-20,STOP_SPEED);
          break;
    }
}