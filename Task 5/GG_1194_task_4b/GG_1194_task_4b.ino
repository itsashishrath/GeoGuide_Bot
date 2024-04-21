//ESP32 Module programming

#define LEFT_SENSOR 19
#define CENTER_SENSOR 18
#define RIGHT_SENSOR 21
#define SIDE_LEFT 22
#define SIDE_RIGHT 5
#define BUZZER 13
#define LED_GREEN 14
#define LED_RED 23

#define LEFT_MOTOR_IN1 16
#define LEFT_MOTOR_IN2 4
#define RIGHT_MOTOR_IN3 15
#define RIGHT_MOTOR_IN4 2

// PWM values for different motions
#define FULL_SPEED 165
#define STOP_SPEED 0
#define TURN_SPEED 155
#define EXTREME_TURN_SPEED 190 // Adjust this value for reduced speed during turns

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
  stop,
  aboutTurn  
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
  "eRight",
  "eLeft",
  "stop",
  "aboutTurn" 
};

const char* stateStrings[] = {
  "pathPlanning",
  "waitingAtStart",
  "followLine",
  "onNode",
  "botTurning",
  "updatePosition",
  "endRun",
  "buzzerLed",
  "reachedEventLocation",
  "stopAtEvent",
  "turningAround",
  "turningAroundLine"
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

// int storedTurningPath[12]={
//     23,8,23,6,23,23,3,11,1,23,6,7
//   };

// int currentPath[55] = {0, 1, 2, 3, 4, 9, 23, 4, 3, 2, 7, 11, 23, 7, 8, 1, 23, 8, 7, 6, 10, 23, 6, 3, 23, 3, 2, 1, 0, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27};
  int currentPath[55] = {0, 1, 8, 23, 1, 2, 3, 6, 23, 3, 2, 1, 0, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27};


// int currentPath [14]= {0,1,2,3,6,5,9,10,11,7,2,1,0} ;


void moveForward(int leftSpeed, int rightSpeed, int reverseSpeed) {
  analogWrite(LEFT_MOTOR_IN1, leftSpeed);
  analogWrite(LEFT_MOTOR_IN2, reverseSpeed);  // Assume LOW for forward motion
  analogWrite(RIGHT_MOTOR_IN3, rightSpeed);
  analogWrite(RIGHT_MOTOR_IN4, 0);  // Assume LOW for forward motion
}


//Map storing the nodes of 
Motion myArray[12][12][12];

Motion determineMotion(Detected detect){

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
    
}else moveOnLine=moveOnLine;

 return moveOnLine;

}

// Function to determine detected position based on sensor readings
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

myArray[3][4][5]=eRight; myArray[5][4][3]=eLeft; myArray[3][4][9]=forward; myArray[9][4][3]=forward;

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

  Serial.print(prevNode);
  Serial.print(currentNode);
  Serial.println(nextNode);
  Serial.print("Detected: ");
  Serial.println(detectedStrings[detected]);
  Serial.print("Movement: ");
  Serial.println(motionStrings[movement]);
  Serial.print("State: ");
  Serial.println(stateStrings[currentState]);

  // State machine to control the behavior of the bot
  // delay(1000);
 switch(currentState) {

  // Start the bot when it detects the center of the line
  case waitingAtStart :
    movement=stop;
    if(detected==centerOfLine){
    digitalWrite(BUZZER, LOW);
    digitalWrite(LED_RED, HIGH);
    delay(1000);
    digitalWrite(BUZZER, HIGH);
    digitalWrite(LED_RED, LOW);
    movement=stop;
    currentState=pathPlanning;
    }

    break;

    case pathPlanning  :

    // digitalWrite(LED_RED, HIGH);
    // delay(500);
    // digitalWrite(LED_RED, LOW);

    // Serial.print("BoxPathIndex :");
    // Serial.println(boxPathIndex);
    // lastIndex=0;

    // for (int i = 0; i < 10; i++) {
    // currentPath[i] = boxPath[boxPathIndex][i];
    // }


    // boxPathIndex++;

    // for(int i=0; i<10; i++) {
    //   if(currentPath[i]==23) {
    //   break;
    //   }
    // lastIndex++;
    // }

    // // for (int i = 0; i < 10; i++) {
    // // Serial.print(currentPath[i]);
    // // Serial.print(" ");
    // // }
    // // Serial.println();
    // // Serial.println(lastIndex);

    // // if(boxPathIndex<4){
    // //   boxPathIndex++;
    // //   lastIndex=0;
    // // }
    delay(3000);
    currentState=followLine;

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


    // if(pathIndex==lastIndex && detected==allThreeLine){
    //   // If the end of the path is reached, stop the bot
    //   movement=stop;
    //   currentState=endRun;
    // }
  
    break;

  case updatePosition :
    //update current position of bot
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
    // digitalWrite(BUZZER, LOW);
    // digitalWrite(LED_GREEN, HIGH);
    // delay(500);
    // digitalWrite(BUZZER, HIGH);
    // digitalWrite(LED_GREEN, LOW);

    break;
  
  case onNode :
    if(detected==node){
      movement=forward;
  } else {
    movement=moveOnLine;
    if(detected==centerOfLine || detected==allThreeLine){
      currentState=botTurning;


    }

  } 
  
  break;

  case botTurning :

    movement=myArray[prevNode][currentNode][nextNode];

    if(movement==eRight && detected==rightOfLine) {
    movement=moveOnLine;
    changeState=true;

  } else if(movement==eLeft && detected==leftOfLine) {
    movement=moveOnLine;
    changeState=true;

  } else if(movement==forward && detected!=node){
    movement=moveOnLine;
    changeState=true;

  }

  if(changeState==true && detected==centerOfLine){
    currentState=followLine;
    changeState=false;
  }

  break;

  case reachedEventLocation:
    movement=moveOnLine;
    stopIf = stopCondition(currentNode);
    currentTime=millis();
    interval=currentTime-startTime;


    if(detected==stopIf && currentNode != 4 && interval>700) {
      movement=stop;
      currentState=stopAtEvent;
      startTime=millis();

    }

    if(detected==stopIf && currentNode == 4 ) {

      if(interval>7000) {
        movement=stop;
        currentState=stopAtEvent;
        startTime=millis();
        }
    }

  break;

  case stopAtEvent:
    currentTime=millis();
    interval=currentTime-startTime;
    isBotReached=false;
    if(interval>=3000){
      pathIndex++;
      if(currentNode == currentPath[pathIndex]) {
      prevNode=nextNode;
      currentNode=nextNode;
      nextNode=currentPath[pathIndex];
      pathIndex++;
      currentState = turningAround;
      movement = aboutTurn;
      // pathIndex=1;
      startTime=millis();
      } else {
      pathIndex++;
      currentState=followLine;
      }
    }
  break;

  case turningAround :
    // currentTime=millis();

    if(stopIf == centerOfLine || currentNode==9){
      currentState=turningAroundLine;
    }

    if(sideRightSensor && sensorActiveCount==0) {
      sensorActiveCount++;
      digitalWrite(LED_RED,HIGH);
    }

    if(sideLeftSensor && sensorActiveCount==1){
      sensorActiveCount++;
    }

    // interval=currentTime-startTime;
    if(leftSensor && sensorActiveCount==2){
      movement=moveOnLine;
      sensorActiveCount=0;
      currentState=followLine;
      digitalWrite(LED_RED,LOW);
      // lastIndex=0;
    }

  break;

  case turningAroundLine :

    if(detected==sideLeftLine && sensorActiveCount==0){
      sensorActiveCount++;
    }

    if(detected==leftOfLine && sensorActiveCount==1){
      sensorActiveCount++;
    }

    if(detected==sideLeftLine && sensorActiveCount==2){
      sensorActiveCount++;
    }

    if(detected==sideRightLine && sensorActiveCount==3){
      movement=moveOnLine;
      sensorActiveCount=0;
      currentState=followLine;
    }

  break;

  case endRun :
    movement=stop;
    if(pathIndex==lastIndex && detected==allThreeLine){
      // If the end of the path is reached, stop the bot
      digitalWrite(BUZZER, LOW);
      digitalWrite(LED_RED, HIGH);
      delay(5000);
      digitalWrite(BUZZER, HIGH);
      digitalWrite(LED_RED, LOW);
      pathIndex++;
    }
    break;

 
 }

  switch (movement) {
    case forward:
      moveForward(FULL_SPEED, FULL_SPEED,0);
      break;

    case left:
      moveForward(STOP_SPEED, TURN_SPEED,0);
      break;
    
    case eLeft:
      moveForward(STOP_SPEED, EXTREME_TURN_SPEED,0);
      break;

    case right:
      moveForward(TURN_SPEED, STOP_SPEED,0);

      break;
    
    case eRight:
      moveForward(EXTREME_TURN_SPEED, STOP_SPEED,0);
      break;

    case stop:
      moveForward(STOP_SPEED, STOP_SPEED,0);
      break;

    case aboutTurn:
      moveForward(0,TURN_SPEED,TURN_SPEED);
      break;
  }
}