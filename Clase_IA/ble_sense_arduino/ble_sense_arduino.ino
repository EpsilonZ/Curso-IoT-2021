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

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "accelerometer_handler.h"
#include "constants.h"
#include "gesture_predictor.h"
#include "model_data.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
int input_length;

char activities[6][20] = {"laying","standing","sitting","walking","walkingdownstairs","walkingupstairs"};

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  Serial.begin(9600);
  while(!Serial);
  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(model_quantized_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
 // static tflite::MicroMutableOpResolver<3> micro_op_resolver;  // NOLINT
  static tflite::AllOpsResolver resolver;
  /*micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_FULLY_CONNECTED,
      tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RELU,
                               tflite::ops::micro::Register_RELU());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_QUANTIZE,
                               tflite::ops::micro::Register_QUANTIZE());*/

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor.

  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != 128*6) ||
      (model_input->type != kTfLiteFloat32)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }

  input_length = model_input->bytes / sizeof(float);

  Serial.print("input length ");
  Serial.print(input_length);
  Serial.println();

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Set up of accelerometer failed\n");
  }
  setup_status = SetupGyroscope(error_reporter);
  if (setup_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Set up of gyroscope failed\n");
  }
}

void loop() {
  
  // Attempt to read new data from the accelerometer.
  bool got_data =
      ReadIMUData(error_reporter, model_input->data.f, input_length);
  // If there was no new data, wait until next time.
  if (!got_data) return;

  for(int i = 0; i < input_length; i++){
    Serial.print(model_input->data.f[i]);
    Serial.print(" ");
    //Serial.print(model_input->params.scale);
    //Serial.print("||");
  }
  Serial.println();

  // Run inference, and report any error.  
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on index gyro\n");
    return;
  }
  // Analyze the results to obtain a prediction

  float highest = -100.0;
  int activity = -1;
  for(int i = 0; i < 6; i++){
    //Serial.print("Activity pred %d: %f  %s\n",
     //                   i, interpreter->output(0)->data.f[i], activities[i]);
   // TF_LITE_REPORT_ERROR(error_reporter, "Activity pred %d: %f  %s\n",
   //                     i, interpreter->output(0)->data.f[i], activities[i]);
    if(interpreter->output(0)->data.f[i] > highest){
      highest = interpreter->output(0)->data.f[i];
      activity = i;
    }
  } 

  Serial.print("act "); Serial.print(highest); Serial.print(" "); Serial.println(activities[activity]);

  
  //TF_LITE_REPORT_ERROR(error_reporter, "time: %d ACTIVITY: %s\n",
   //                    millis(), activities[activity]);

  //delay(3000);

  /*
  int gesture_index = PredictGesture(interpreter->output(0)->data.f);

  // Produce an output
  HandleOutput(error_reporter, gesture_index);*/
}
