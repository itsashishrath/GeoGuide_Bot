//ESP32 Module programming

#define LEFT_SENSOR 19
#define CENTER_SENSOR 18
#define RIGHT_SENSOR 21
#define SIDE_LEFT 22
#define SIDE_RIGHT 5

#define LEFT_MOTOR_IN1 16
#define LEFT_MOTOR_IN2 4
#define RIGHT_MOTOR_IN3 15
#define RIGHT_MOTOR_IN4 2

// PWM values for different motions
#define FULL_SPEED 165
#define STOP_SPEED 0
#define TURN_SPEED 155
#define EXTREME_TURN_SPEED 180 // Adjust this value for reduced speed during turns

enum Detected {
  centerOfLine,
  leftOfLine,
  rightOfLine,
  node,
  bothSideLine,
  sideRightLine,
  sideLeftLine,
  allThreeLine,
  none
};

enum State{
  waitingAtStart,
  followLine,
  onNode,
  botTurning,
  updatePosition
};

enum Motion {
  forward,
  left,
  right,
  eRight,
  eLeft,
  stop  
};

const char* detectedStrings[] = {
  "centerOfLine",
  "leftOfLine",
  "rightOfLine",
  "node",
  "bothSideLine",
  "sideRightLine",
  "sideLeftLine",
  "allThreeLine"
};

const char* motionStrings[] = {
  "forward",
  "left",
  "right",
  "eLeft",
  "eRight",
  "stop"
};

const char* stateStrings[] = {
  "waitingAtStart",
  "followLine",
  "onNode",
  "botTurning",
  "updatePosition"
};



Motion movement;
int prevNode = 0; 
int currentNode = 0;
int nextNode = 1;
bool isTurning = false;
bool turnNow=false;
int pathIndex=2;
const int durationInSeconds = 10;  // Set the duration of the timer in seconds
unsigned long startTime;  // Variable to store the start time
unsigned long currentTime;
State currentState;


int completePath [14]={0,1,2,3,6,5,9,10,11,7,2,1,0};

void moveForward(int leftSpeed, int rightSpeed) {
  analogWrite(LEFT_MOTOR_IN1, leftSpeed);
  analogWrite(LEFT_MOTOR_IN2, 0);  // Assume LOW for forward motion
  analogWrite(RIGHT_MOTOR_IN3, rightSpeed);
  analogWrite(RIGHT_MOTOR_IN4, 0);  // Assume LOW for forward motion
}

// Assuming myArray is a three-dimensional array
Motion myArray[12][12][12];

Motion determineMotion(Detected detect){
  Motion moveOnLine;

  if (detect == centerOfLine || detect == bothSideLine || detect == allThreeLine ) {
    moveOnLine = forward;

} else if (detect == leftOfLine) {
    moveOnLine = left;

} else if (detect == rightOfLine) {
    moveOnLine = right;

} else if (detect == sideRightLine ) {
    moveOnLine = eLeft;

} else if (detect == sideLeftLine ) {
    moveOnLine = eRight;

}else if (detect == node) {
    moveOnLine = stop;
    
}  return moveOnLine;

}

// Function to determine detected position based on sensor readings
Detected determinePosition(bool sideLeftSensor, bool leftSensor, bool centerSensor, bool rightSensor, bool sideRightSensor) {
  Detected detected;

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
     detected = none;
  }

  return detected;
}



void setup() {
  Serial.begin(115200);
  currentState=waitingAtStart;

  pinMode(LEFT_SENSOR, INPUT);
  pinMode(CENTER_SENSOR, INPUT);
  pinMode(RIGHT_SENSOR, INPUT);

  pinMode(LEFT_MOTOR_IN1, OUTPUT);
  pinMode(LEFT_MOTOR_IN2, OUTPUT);
  pinMode(RIGHT_MOTOR_IN3, OUTPUT);
  pinMode(RIGHT_MOTOR_IN4, OUTPUT);


//Setting direction for bot
myArray[0][0][1]=forward;

myArray[0][1][2]=forward; myArray[2][1][0]=forward; myArray[0][1][8]=eRight; myArray[8][1][0]=eLeft;

myArray[1][2][3]=forward; myArray[3][2][1]=forward; myArray[1][2][7]=eRight; myArray[7][2][1]=eLeft;
myArray[1][2][7]=eRight; myArray[7][2][1]=eLeft;

myArray[6][3][4]=eRight; myArray[4][3][6]=eLeft;
myArray[2][3][4]=forward; myArray[4][3][2]=forward; myArray[2][3][6]=eRight; myArray[6][3][2]=eLeft;

myArray[3][4][5]=eRight; myArray[5][4][3]=eLeft; myArray[3][4][9]=forward; myArray[9][4][3]=forward;

myArray[4][5][6]=eRight; myArray[6][5][4]=eLeft; myArray[9][5][6]=eLeft; myArray[6][5][9]=eRight;

myArray[3][6][7]=eRight; myArray[7][6][3]=eLeft; myArray[5][6][7]=forward; myArray[7][6][5]=forward; myArray[3][6][5]=eLeft; myArray[5][6][3]=eRight; 
myArray[10][6][7]=eLeft; myArray[10][6][5]=eRight; myArray[7][6][10]=eRight; myArray[5][6][10]=eLeft; myArray[3][6][10]=forward; myArray[10][6][3]=forward;

myArray[2][7][8]=eRight; myArray[11][7][8]=eLeft; myArray[2][7][11]=forward; myArray[6][7][8]=forward; myArray[11][7][6]=eRight; myArray[2][7][6]=eLeft;
myArray[8][7][2]=eLeft; myArray[8][7][11]=eRight; myArray[11][7][2]=forward; myArray[8][7][6]=forward; myArray[6][7][11]=eLeft; myArray[6][7][2]=eRight;

myArray[1][8][7]= left ; myArray[7][8][1]=eRight; myArray[11][8][7]=eRight; myArray[7][8][11]=eLeft; myArray[1][8][11]=forward; myArray[11][8][1]=forward; 

myArray[4][9][10]=forward; myArray[10][9][4]=forward; myArray[5][9][4]=eLeft; myArray[4][9][5]=eRight; myArray[5][9][10]=eRight; myArray[10][9][5]=eLeft;

myArray[9][10][11]=forward; myArray[9][10][6]=eRight; myArray[6][10][11]=eRight; myArray[11][10][9]=forward; myArray[6][10][9]=eLeft; myArray[11][10][6]=eLeft;
myArray[9][10][11]=forward; myArray[9][10][6]=eRight; myArray[6][10][11]=eRight; myArray[11][10][9]=forward; myArray[6][10][9]=eLeft; myArray[11][10][6]=eLeft;

myArray[7][11][8]=eRight; myArray[8][11][7]=eLeft; myArray[10][11][8]=forward; myArray[8][11][10]=forward; myArray[10][11][7]=eRight; myArray[7][11][10]=eLeft;

}

void loop() {
  bool leftSensor = digitalRead(LEFT_SENSOR);
  bool centerSensor = digitalRead(CENTER_SENSOR);
  bool rightSensor = digitalRead(RIGHT_SENSOR);
  bool sideLeftSensor = digitalRead(SIDE_LEFT);
  bool sideRightSensor = digitalRead(SIDE_RIGHT);

  Detected detected = determinePosition(sideLeftSensor, leftSensor, centerSensor, rightSensor, sideRightSensor);
  Motion moveOnLine = determineMotion(detected);

 switch(currentState) {

  case waitingAtStart :
    movement=stop;
    if(detected==centerOfLine){
      movement=moveOnLine;
      currentState=followLine;
    }

    break;
  
  case followLine :
    movement=moveOnLine;
    if(detected==node){
      currentState=updatePosition;
    }

    break;

  case updatePosition :
    startTime=millis();
    prevNode=currentNode;
    currentNode=nextNode;
    nextNode=completePath[pathIndex];
    pathIndex++;
    currentState=onNode;

    break;
  
  case onNode :
    if(detected==node){
      movement=forward;

  } else if(detected!=node){
    movement=moveOnLine;
    if(detected==centerOfLine){
      currentState=botTurning;
    }

  } break;

  case botTurning :
    movement=myArray[prevNode][currentNode][nextNode];
    if(movement==eRight && detected==rightOfLine) {
    currentState=followLine;

  } else if(movement==eLeft && detected==leftOfLine) {
    currentState=followLine;

  } else if(movement==forward && detected!=node){
    currentState=followLine;

  } break;
 
 }

  Serial.print(prevNode);
  Serial.print(currentNode);
  Serial.println(nextNode);
  Serial.print("Detected: ");
  Serial.println(detectedStrings[detected]);
  Serial.print("Movement: ");
  Serial.println(motionStrings[movement]);
  Serial.println(stateStrings[currentState]);
  // delay(100);


  switch (movement) {
    case forward:
      // Perform forward motion action
      // digitalWrite(LEFT_MOTOR_IN1, HIGH);
      // digitalWrite(LEFT_MOTOR_IN2, LOW);
      // digitalWrite(RIGHT_MOTOR_IN3, HIGH);
      // digitalWrite(RIGHT_MOTOR_IN4, LOW);

      moveForward(FULL_SPEED, FULL_SPEED);
      break;

    case left:
      // Perform left motion action
      // digitalWrite(LEFT_MOTOR_IN1, LOW);
      // digitalWrite(LEFT_MOTOR_IN2, HIGH);
      // digitalWrite(RIGHT_MOTOR_IN3, HIGH);
      // digitalWrite(RIGHT_MOTOR_IN4, LOW);
      moveForward(STOP_SPEED, TURN_SPEED);
      break;
    
    case eLeft:
      
      moveForward(STOP_SPEED, EXTREME_TURN_SPEED);

      break;

    case right:
      // Perform right motion action
      // digitalWrite(LEFT_MOTOR_IN1, HIGH);
      // digitalWrite(LEFT_MOTOR_IN2, LOW);
      // digitalWrite(RIGHT_MOTOR_IN3, LOW);
      // digitalWrite(RIGHT_MOTOR_IN4, HIGH);
      moveForward(TURN_SPEED, STOP_SPEED);

      break;
    
    case eRight:

      moveForward(EXTREME_TURN_SPEED, STOP_SPEED);

      break;

    case stop:
      // Perform stop motion action
      digitalWrite(LEFT_MOTOR_IN1, LOW);
      digitalWrite(LEFT_MOTOR_IN2, LOW);
      digitalWrite(RIGHT_MOTOR_IN3, LOW);
      digitalWrite(RIGHT_MOTOR_IN4, LOW);
      break;
    // Add cases for other detected positions if needed
  }
}