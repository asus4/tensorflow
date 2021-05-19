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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_DELEGATE_H_

#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/delegate_options.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a new delegate instance that need to be destroyed with
// TfLiteGpuDelegateV2Delete when delegate is no longer used by TFLite.
//
// This delegate encapsulates multiple GPU-acceleration APIs under the hood to
// make use of the fastest available on a device.
//
// When `options` is set to `nullptr`, then default options are used.
TFL_CAPI_EXPORT TfLiteDelegate* TfLiteGpuDelegateV2Create(
    const TfLiteGpuDelegateOptionsV2* options);

// Destroys a delegate created with `TfLiteGpuDelegateV2Create` call.
TFL_CAPI_EXPORT void TfLiteGpuDelegateV2Delete(TfLiteDelegate* delegate);

typedef uint32_t GLuint;
// Binds GL shader storage object to an input in the initialized
// delegate. Bound buffer should have sufficient storage to accommodate
// all elements of a tensor.
TFL_CAPI_EXPORT TfLiteStatus TfLiteGpuDelegateV2BindInputBuffer(
  TfLiteDelegate* delegate, int index, GLuint buffer);

// Binds GL shader storage object to an output in the initialized
// delegate. Bound buffer should have sufficient storage to accommodate
// all elements of a tensor.
TFL_CAPI_EXPORT TfLiteStatus TfLiteGpuDelegateV2BindOutputBuffer(
  TfLiteDelegate* delegate, int index, GLuint buffer);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_DELEGATE_H_
