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

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MAGIC_WAND_CONSTANTS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MAGIC_WAND_CONSTANTS_H_

// The expected accelerometer data sample frequency
const float kTargetHz = 50;
const float degrees_to_radians = 0.0174533;

//activity labels
//1 WALKING
//2 WALKING_UPSTAIRS
//3 WALKING_DOWNSTAIRS
//4 SITTING
//5 STANDING
//6 LAYING
// What gestures are supported.
constexpr int kGestureCount = 4; //??
constexpr int kWalkingActivity = 0;
constexpr int kWalkingUpstairsActivity = 1;
constexpr int kWalkingDownstairsActivity = 2;
constexpr int kSittingActivity = 3;
constexpr int kStandingActivity = 4;
constexpr int kLayingActivity = 5;


// These control the sensitivity of the detection algorithm. If you're seeing
// too many false positives or not enough true positives, you can try tweaking
// these thresholds. Often, increasing the size of the training set will give
// more robust results though, so consider retraining if you are seeing poor
// predictions.
constexpr float kDetectionThreshold = 0.4f;
constexpr int kPredictionHistoryLength = 5;
constexpr int kPredictionSuppressionDuration = 25;

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MAGIC_WAND_CONSTANTS_H_
