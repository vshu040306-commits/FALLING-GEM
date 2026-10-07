#include <Servo.h>

// =========================================================
// 1. 引脚设置
// =========================================================

const int SERVO_PIN = 6;
const int TRIG_PIN = 8;
const int ECHO_PIN = 9;


// =========================================================
// 2. 舵机设置
// =========================================================

Servo stoppingServo;

int currentServoAngle = 0;


// =========================================================
// 3. 超声波参数
// =========================================================

// 小于该距离，认为物体已经放回
const float OBJECT_PLACED_DISTANCE_CM = 2.5;

// 大于该距离，认为前方没有物体
// 使用两个不同阈值，防止距离临界值抖动
const float OBJECT_REMOVED_DISTANCE_CM = 3.0;

// 物体必须持续存在多久才确认
const unsigned long OBJECT_CONFIRM_TIME_MS = 600;


// =========================================================
// 4. 系统状态
// =========================================================

// 只有进入Python的state 6后，才允许传感器触发
bool sensorArmed = false;

// 已确认物体是否存在
bool objectPlaced = false;

// 候选状态
bool candidateObjectState = false;

// 候选状态开始时间
unsigned long candidateStartTime = 0;

// 调试距离输出时间
unsigned long lastDistancePrintTime = 0;


// =========================================================
// 5. 串口命令
// =========================================================

String serialCommand = "";


// =========================================================
// 6. 舵机平滑移动
// =========================================================

void moveServoSmoothly(int targetAngle) {

  targetAngle = constrain(
    targetAngle,
    0,
    180
  );

  if (targetAngle > currentServoAngle) {

    for (
      int angle = currentServoAngle;
      angle <= targetAngle;
      angle++
    ) {
      stoppingServo.write(angle);
      delay(8);
    }

  } else {

    for (
      int angle = currentServoAngle;
      angle >= targetAngle;
      angle--
    ) {
      stoppingServo.write(angle);
      delay(8);
    }
  }

  currentServoAngle = targetAngle;
}


// =========================================================
// 7. 测量距离
// =========================================================

float readDistanceCM() {

  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);

  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(
    ECHO_PIN,
    HIGH,
    30000
  );

  if (duration == 0) {
    return -1.0;
  }

  float distanceCM = (
    duration * 0.0343
  ) / 2.0;

  return distanceCM;
}


// =========================================================
// 8. 启动物体检测
// =========================================================

void armObjectSensor() {

  sensorArmed = true;

  objectPlaced = false;
  candidateObjectState = false;
  candidateStartTime = millis();

  Serial.println("SENSOR_ARMED");
}


// =========================================================
// 9. 停止物体检测
// =========================================================

void disarmObjectSensor() {

  sensorArmed = false;

  objectPlaced = false;
  candidateObjectState = false;
  candidateStartTime = millis();

  Serial.println("SENSOR_DISARMED");
}


// =========================================================
// 10. 更新物体状态
// =========================================================

void updateObjectDetection(float distanceCM) {

  if (!sensorArmed) {
    return;
  }

  if (distanceCM <= 0) {
    return;
  }

  bool measuredState = objectPlaced;

  if (!objectPlaced) {

    if (
      distanceCM <= OBJECT_PLACED_DISTANCE_CM
    ) {
      measuredState = true;
    }

  } else {

    if (
      distanceCM >= OBJECT_REMOVED_DISTANCE_CM
    ) {
      measuredState = false;
    }
  }

  // 当前测量状态与正式状态相同
  if (measuredState == objectPlaced) {

    candidateObjectState = objectPlaced;
    candidateStartTime = millis();

    return;
  }

  // 出现新的候选状态
  if (
    measuredState != candidateObjectState
  ) {

    candidateObjectState = measuredState;
    candidateStartTime = millis();

    return;
  }

  // 候选状态持续足够长
  if (
    millis() - candidateStartTime
    >= OBJECT_CONFIRM_TIME_MS
  ) {

    objectPlaced = candidateObjectState;

    if (objectPlaced) {

      Serial.println("OBJECT_PLACED");

      // 只触发一次，等待Python发送RESET_HOME
      sensorArmed = false;

    } else {

      Serial.println("OBJECT_REMOVED");
    }
  }
}


// =========================================================
// 11. 处理Python命令
// =========================================================

void processCommand(String command) {

  command.trim();

  if (command == "ENTER_STATE_6") {

    Serial.println("ENTERING_STATE_6");

    moveServoSmoothly(180);

    Serial.println("SERVO_AT_180");

    armObjectSensor();

  } else if (command == "RESET_HOME") {

    disarmObjectSensor();

    moveServoSmoothly(0);

    Serial.println("SERVO_AT_0");
    Serial.println("SYSTEM_RESET_DONE");

  } else if (command == "SERVO_180") {

    moveServoSmoothly(180);
    Serial.println("SERVO_AT_180");

  } else if (command == "SERVO_0") {

    moveServoSmoothly(0);
    Serial.println("SERVO_AT_0");

  } else if (command == "PING") {

    Serial.println("ARDUINO_READY");

  } else {

    Serial.print("UNKNOWN_COMMAND:");
    Serial.println(command);
  }
}


// =========================================================
// 12. 初始化
// =========================================================

void setup() {

  Serial.begin(115200);

  stoppingServo.attach(SERVO_PIN);
  stoppingServo.write(0);

  currentServoAngle = 0;

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  digitalWrite(TRIG_PIN, LOW);

  delay(600);

  Serial.println("ARDUINO_READY");
}


// =========================================================
// 13. 主循环
// =========================================================

void loop() {

  // 接收Python命令
  while (Serial.available() > 0) {

    char incomingCharacter = Serial.read();

    if (incomingCharacter == '\n') {

      processCommand(serialCommand);

      serialCommand = "";

    } else if (incomingCharacter != '\r') {

      serialCommand += incomingCharacter;
    }
  }

  // 超声波检测
  float distanceCM = readDistanceCM();

  updateObjectDetection(distanceCM);

  // 每500ms打印一次距离，方便调试
  if (
    millis() - lastDistancePrintTime
    >= 500
  ) {

    lastDistancePrintTime = millis();

    Serial.print("DISTANCE:");

    if (distanceCM > 0) {
      Serial.println(distanceCM, 1);
    } else {
      Serial.println("INVALID");
    }
  }

  delay(40);
}
