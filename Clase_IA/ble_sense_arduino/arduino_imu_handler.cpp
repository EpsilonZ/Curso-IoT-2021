/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#ifndef ARDUINO_EXCLUDE_CODE

#include "accelerometer_handler.h"

#include <Arduino.h>
#include <Arduino_LSM9DS1.h>

#include "constants.h"

int sample_every_n;
  // Keep track of whether we stored any new data
// A buffer holding the last 384 sets of 3-channel values
float save_data_acc_x[128] = {0.0};
float save_data_acc_y[128] = {0.0};
float save_data_acc_z[128] = {0.0};
float save_data_gyro_x[128] = {0.0};
float save_data_gyro_y[128] = {0.0};
float save_data_gyro_z[128] = {0.0};
// Most recent position in the save_data buffer
int begin_index_acc = 0;
int begin_index_gyro = 0;
// True if there is not yet enough data to run inference
bool pending_initial_data = true;
// How often we should save a measurement during downsampling

// The number of measurements since we last saved one
int sample_skip_counter = 1;

TfLiteStatus SetupGyroscope(tflite::ErrorReporter* error_reporter) {
  // Switch on the IMU
  if (!IMU.begin()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize IMU");
    return kTfLiteError;
  }

  // Make sure we are pulling measurements into a FIFO.
  // If you see an error on this line, make sure you have at least v1.1.0 of the
  // Arduino_LSM9DS1 library installed.
  IMU.setContinuousMode();

  // Determine how many measurements to keep in order to
  // meet kTargetHz
  float sample_rate = IMU.gyroscopeSampleRate();
  sample_every_n = static_cast<int>(roundf(sample_rate / kTargetHz));

  TF_LITE_REPORT_ERROR(error_reporter, "Magic starts!");

  return kTfLiteOk;
}

TfLiteStatus SetupAccelerometer(tflite::ErrorReporter* error_reporter) {
  // Switch on the IMU
  if (!IMU.begin()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize IMU");
    return kTfLiteError;
  }

  // Make sure we are pulling measurements into a FIFO.
  // If you see an error on this line, make sure you have at least v1.1.0 of the
  // Arduino_LSM9DS1 library installed.
  IMU.setContinuousMode();

  // Determine how many measurements to keep in order to
  // meet kTargetHz
  float sample_rate = IMU.accelerationSampleRate();
  sample_every_n = static_cast<int>(roundf(sample_rate / kTargetHz));

  TF_LITE_REPORT_ERROR(error_reporter, "Magic starts!");

  return kTfLiteOk;
}

bool ReadIMUData(tflite::ErrorReporter* error_reporter, float* input,
                       int length) {
  
  bool new_data = false;
  pending_initial_data = true;
  
  // Loop through new samples and add to buffer
  //TF_LITE_REPORT_ERROR(error_reporter, "READING IMU DATA... index acc %d index gyro %d\n", begin_index_acc, begin_index_gyro);
  while (IMU.gyroscopeAvailable() && IMU.accelerationAvailable() && pending_initial_data) {
    float acc_x, acc_y, acc_z;
    float gyro_x, gyro_y, gyro_z;
    // Read each sample, removing it from the device's FIFO buffer
    if (!IMU.readGyroscope(gyro_x, gyro_y, gyro_z)) {
      TF_LITE_REPORT_ERROR(error_reporter, "Failed to read data");
      break;
    }
    if (!IMU.readAcceleration(acc_x, acc_y, acc_z)) {
      TF_LITE_REPORT_ERROR(error_reporter, "Failed to read data");
      break;
    }

    // Throw away this sample unless it's the nth
    if (sample_skip_counter != sample_every_n) {
      sample_skip_counter += 1;
      continue;
    }
    // Write samples to our buffer, converting to milli-Gs and rotating the axis
    // order for compatibility with model (sensor orientation is different on
    // Arduino Nano BLE Sense compared with SparkFun Edge).
    // The expected orientation of the Arduino on the wand is with the USB port
    // facing down the shaft towards the user's hand, with the reset button
    // pointing at the user's face:
    //
    //                  ____
    //                 |    |<- Arduino board
    //                 |    |
    //                 | () |  <- Reset button
    //                 |    |
    //                  -TT-   <- USB port
    //                   ||
    //                   ||<- Wand
    //                  ....
    //                   ||
    //                   ||
    //                   ()
    //
    save_data_acc_x[begin_index_acc] = acc_x;
    save_data_acc_y[begin_index_acc] = -acc_y;
    save_data_acc_z[begin_index_acc] = acc_z;

    begin_index_acc = begin_index_acc + 1;
    
    save_data_gyro_x[begin_index_gyro] = gyro_x*degrees_to_radians;
    save_data_gyro_y[begin_index_gyro] = -gyro_y*degrees_to_radians;
    save_data_gyro_z[begin_index_gyro] = gyro_z*degrees_to_radians;

    begin_index_gyro = begin_index_gyro + 1;

    //Serial.print(acc_x); Serial.print(" "); Serial.print(-acc_y); Serial.print(" "); Serial.print(acc_z); Serial.print(" ");
    //Serial.print(gyro_x*degrees_to_radians); Serial.print(" "); Serial.print(-gyro_y*degrees_to_radians); Serial.print(" "); Serial.println(gyro_z*degrees_to_radians);
    // Check if we are ready for prediction or still pending more initial data
    if (pending_initial_data && begin_index_acc >= 128 || begin_index_gyro >= 128) {
      pending_initial_data = false;
    }

    // Since we took a sample, reset the skip counter
    sample_skip_counter = 1;

    // If we reached the end of the circle buffer, reset
    if (begin_index_acc >= 128) {
      begin_index_acc = 0;
    } 
    if (begin_index_gyro >= 128) {
      begin_index_gyro = 0;
    }

    new_data = true;
  }

  // Skip this round if data is not ready yet
  if (!new_data) {
    return false;
  }

  // Check if we are ready for prediction or still pending more initial data
  if (pending_initial_data && begin_index_acc >= 128 || begin_index_gyro >= 128) {
    pending_initial_data = false;
  }

  // Return if we don't have enough data
  if (pending_initial_data) {
    return false;
  }

  TF_LITE_REPORT_ERROR(error_reporter, "Copy acc values"); 

  // first we copy acc values
  for (int i = 0; i < 128; ++i) {
    input[i] = save_data_acc_x[i];
  }

  for (int i = 0; i < 128; ++i) {
    input[i+128] = save_data_acc_y[i];
  }

  for (int i = 0; i < 128; ++i) {
    input[i+(128*2)] = save_data_acc_z[i];
  }  

  for (int i = 0; i < 128; ++i) {
    input[i+(128*3)] = save_data_gyro_x[i];
  }

  for (int i = 0; i < 128; ++i) {
    input[i+(128*4)] = save_data_gyro_y[i];
  }

  for (int i = 0; i < 128; ++i) {
    input[i+(128*5)] = save_data_gyro_z[i];
  }    

  TF_LITE_REPORT_ERROR(error_reporter, "Copy gyro values");

  return true;
}

#endif  // ARDUINO_EXCLUDE_CODE
