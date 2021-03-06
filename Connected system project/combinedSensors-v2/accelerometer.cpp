#include "accelerometer.h"
#include <Energia.h>
#include <Wire.h>

#define BMA_ADDR (0x18)
#define BMA_X    (0x03)
#define BMA_Y    (0x05)
#define BMA_Z    (0x07)

void initializeI2C(uint8_t base_address, uint8_t register_address) {
  Wire.beginTransmission(base_address);
  Wire.write(register_address);
  Wire.endTransmission();
}

uint8_t readI2C(uint8_t base_address, uint8_t register_address) {
  initializeI2C(base_address, register_address);
  Wire.requestFrom(base_address, 1u);

  while (Wire.available() < 1u);

  return Wire.read();
}

int8_t readSingleAxis(uint8_t axis) {
  return readI2C(BMA_ADDR, axis);
}

AccData readAccelerometer() {
  AccData data;
  data.x = readSingleAxis(BMA_X);//in x direction
  data.y = readSingleAxis(BMA_Y);//in y direction
  data.z = readSingleAxis(BMA_Z);//in z direction
  return data;
}
