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

enum Detected {
  centerOfLine,
  leftOfLine,
  rightOfLine,
  node,
  bothSideLine,
  sideRightLine,
  sideLeftLine,
  allThreeLine
};

enum Motion {
  forward,
  left,
  right,
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
  "stop"
};



Motion movement;
int prevNode = 0; 
int currentNode = 0;
int nextNode = 1;
bool isTurning = false;
int pathIndex=2;

int completePath [13]={0,1,2,3,6,5,9,10,11,7,2,1,0};

// Assuming myArray is a three-dimensional array
Motion myArray[12][12][12];

Motion determineMotion(Detected detect){
  Motion moveOnLine;

  if (detect == centerOfLine || detect == bothSideLine || detect == allThreeLine) {
    moveOnLine = forward;

} else if (detect == leftOfLine || detect == sideLeftLine ) {
    moveOnLine = left;

} else if (detect == rightOfLine || detect == sideRightLine ) {
    moveOnLine = right;

} else if (detect == node) {
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

  } else if (!sideLeftSensor && leftSensor && centerSensor && rightSensor && !sideRightSensor) {
    // Left of the line
    detected = node;

  } else if (sideLeftSensor && !leftSensor && !centerSensor && !rightSensor && sideRightSensor) {
    // Both sides of the line
    detected = bothSideLine;

  } else if (!sideLeftSensor && !leftSensor && !rightSensor && sideRightSensor) {
    // Right side of the line
    detected = sideRightLine;

  } else if (sideLeftSensor && !leftSensor && !rightSensor && !sideRightSensor) {
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




void setup() {
  Serial.begin(9600);

  pinMode(LEFT_SENSOR, INPUT);
  pinMode(CENTER_SENSOR, INPUT);
  pinMode(RIGHT_SENSOR, INPUT);

  pinMode(LEFT_MOTOR_IN1, OUTPUT);
  pinMode(LEFT_MOTOR_IN2, OUTPUT);
  pinMode(RIGHT_MOTOR_IN3, OUTPUT);
  pinMode(RIGHT_MOTOR_IN4, OUTPUT);


//Setting direction for bot

myArray[0][0][1]=forward;

myArray[0][1][2]=forward; myArray[2][1][0]=forward; myArray[0][1][8]=right; myArray[8][1][0]=left;

myArray[1][2][3]=forward; myArray[3][2][1]=forward; myArray[1][2][7]=right; myArray[7][2][1]=left;
myArray[1][2][7]=right; myArray[7][2][1]=left;

myArray[6][3][4]=right; myArray[4][3][6]=left;
myArray[2][3][4]=forward; myArray[4][3][2]=forward; myArray[2][3][6]=right; myArray[6][3][2]=left;

myArray[3][4][5]=right; myArray[5][4][3]=left; myArray[3][4][9]=forward; myArray[9][4][3]=forward;

myArray[4][5][6]=right; myArray[6][5][4]=left; myArray[9][5][6]=left; myArray[6][5][9]=right;

myArray[3][6][7]=right; myArray[7][6][3]=left; myArray[5][6][7]=forward; myArray[7][6][5]=forward; myArray[3][6][5]=left; myArray[5][6][3]=right; 
myArray[10][6][7]=left; myArray[10][6][5]=right; myArray[7][6][10]=right; myArray[5][6][10]=left; myArray[3][6][10]=forward; myArray[10][6][3]=forward;

myArray[2][7][8]=right; myArray[11][7][8]=left; myArray[2][7][11]=forward; myArray[6][7][8]=forward; myArray[11][7][6]=right; myArray[2][7][6]=left;
myArray[8][7][2]=left; myArray[8][7][11]=right; myArray[11][7][2]=forward; myArray[8][7][6]=forward; myArray[6][7][11]=left; myArray[6][7][2]=right;

myArray[1][8][7]= left ; myArray[7][8][1]=right; myArray[11][8][7]=right; myArray[7][8][11]=left; myArray[1][8][11]=forward; myArray[11][8][1]=forward; 

myArray[4][9][10]=forward; myArray[10][9][4]=forward; myArray[5][9][4]=left; myArray[4][9][5]=right; myArray[5][9][10]=right; myArray[10][9][5]=left;

myArray[9][10][11]=forward; myArray[9][10][6]=right; myArray[6][10][11]=right; myArray[11][10][9]=forward; myArray[6][10][9]=left; myArray[11][10][6]=left;
myArray[9][10][11]=forward; myArray[9][10][6]=right; myArray[6][10][11]=right; myArray[11][10][9]=forward; myArray[6][10][9]=left; myArray[11][10][6]=left;

myArray[7][11][8]=right; myArray[8][11][7]=left; myArray[10][11][8]=forward; myArray[8][11][10]=forward; myArray[10][11][7]=right; myArray[7][11][10]=left;
}

void loop() {
  bool leftSensor = digitalRead(LEFT_SENSOR);
  bool centerSensor = digitalRead(CENTER_SENSOR);
  bool rightSensor = digitalRead(RIGHT_SENSOR);
  bool sideLeftSensor = digitalRead(SIDE_LEFT);
  bool sideRightSensor = digitalRead(SIDE_RIGHT);

  Detected detected = determinePosition(sideLeftSensor, leftSensor, centerSensor, rightSensor, sideRightSensor);

  Motion moveOnLine = determineMotion(detected);

  if(isTurning==false){
    movement = moveOnLine;
  }

  if (detected==node && isTurning==false) {
    movement = forward;
    delay(800);
    isTurning = true;
    // movement=myArray[prevNode][currentNode][nextNode];
  }

  if(isTurning && detected == centerOfLine) {
    isTurning = false;
    prevNode = currentNode; 
    currentNode = nextNode;
    nextNode = completePath[pathIndex];
    pathIndex++;
  }

  if(isTurning) {
    movement=myArray[prevNode][currentNode][nextNode];
  }


  Serial.print(prevNode);
  Serial.print(currentNode);
  Serial.println(nextNode);
  Serial.println(isTurning);
  Serial.print("Detected: ");
  Serial.println(detectedStrings[detected]);
  Serial.print("Movement: ");
  Serial.println(motionStrings[movement]);
  delay(200);


  switch (movement) {
    case forward:
      // Perform forward motion action
      digitalWrite(LEFT_MOTOR_IN1, HIGH);
      digitalWrite(LEFT_MOTOR_IN2, LOW);
      digitalWrite(RIGHT_MOTOR_IN3, HIGH);
      digitalWrite(RIGHT_MOTOR_IN4, LOW);
      break;

    case left:
      // Perform left motion action
      digitalWrite(LEFT_MOTOR_IN1, LOW);
      digitalWrite(LEFT_MOTOR_IN2, HIGH);
      digitalWrite(RIGHT_MOTOR_IN3, HIGH);
      digitalWrite(RIGHT_MOTOR_IN4, LOW);
      break;

    case right:
      // Perform right motion action
      digitalWrite(LEFT_MOTOR_IN1, HIGH);
      digitalWrite(LEFT_MOTOR_IN2, LOW);
      digitalWrite(RIGHT_MOTOR_IN3, LOW);
      digitalWrite(RIGHT_MOTOR_IN4, HIGH);
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
