/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/

#ifndef ACC_HIP_H
#define ACC_HIP_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas.h>
#include <hip/hiprtc.h>

#define ACC(x) hip##x
#define ACC_DRV(x) ACC(x)
#define ACC_BLAS(x) hipblas##x
#define ACC_RTC(x) hiprtc##x
#define BACKEND "HIP"

/* Macro for HIP error handling
 * Wrap calls to HIP API
 */
#define HIP_API_CALL(func, args) \
  do { \
    hipError_t result = ACC(func) args; \
    if (result != hipSuccess) { \
      printf("\nHIP error: %s failed with error %s\n", #func, hipGetErrorName(result)); \
      exit(1); \
    } \
  } while (0)

/* HIP does not differentiate between "runtime" API and "driver" API */
#define ACC_API_CALL(func, args) HIP_API_CALL(func, args)
#define ACC_DRV_CALL(func, args) HIP_API_CALL(func, args)

/* Wrap calls to HIPRTC API */
#define ACC_RTC_CALL(func, args) \
  do { \
    hiprtcResult result = ACC_RTC(func) args; \
    if (result != HIPRTC_SUCCESS) { \
      printf("\nHIPRTC ERROR: %s failed with error %s\n", #func, hiprtcGetErrorString(result)); \
      exit(1); \
    } \
  } while (0)

/* Wrap calls to HIPBLAS API */
#define ACC_BLAS_CALL(func, args) \
  do { \
    hipblasStatus_t result = ACC_BLAS(func) args; \
    if (result != HIPBLAS_STATUS_SUCCESS) { \
      const char* error_name = "HIPBLAS_ERRROR"; \
      if (result == HIPBLAS_STATUS_NOT_INITIALIZED) { \
        error_name = "HIPBLAS_STATUS_NOT_INITIALIZED "; \
      } \
      else if (result == HIPBLAS_STATUS_ALLOC_FAILED) { \
        error_name = "HIPBLAS_STATUS_ALLOC_FAILED "; \
      } \
      else if (result == HIPBLAS_STATUS_INVALID_VALUE) { \
        error_name = "HIPBLAS_STATUS_INVALID_VALUE "; \
      } \
      else if (result == HIPBLAS_STATUS_MAPPING_ERROR) { \
        error_name = "HIPBLAS_STATUS_MAPPING_ERROR "; \
      } \
      else if (result == HIPBLAS_STATUS_EXECUTION_FAILED) { \
        error_name = "HIPBLAS_STATUS_EXECUTION_FAILED "; \
      } \
      else if (result == HIPBLAS_STATUS_INTERNAL_ERROR) { \
        error_name = "HIPBLAS_STATUS_INTERNAL_ERROR "; \
      } \
      else if (result == HIPBLAS_STATUS_NOT_SUPPORTED) { \
        error_name = "HIPBLAS_STATUS_NOT_SUPPORTED "; \
      } \
      else if (result == HIPBLAS_STATUS_ARCH_MISMATCH) { \
        error_name = "HIPBLAS_STATUS_ARCH_MISMATCH "; \
      } \
      else if (result == HIPBLAS_STATUS_HANDLE_IS_NULLPTR) { \
        error_name = "HIPBLAS_STATUS_HANDLE_IS_NULLPTR "; \
      } \
      printf("\nHIPBLAS ERROR: %s failed with error %s\n", #func, error_name); \
      exit(1); \
    } \
  } while (0)

#ifdef __HIP_PLATFORM_AMD__
hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags) { return hipHostMalloc(ptr, size, flags); }
hipError_t hipFreeHost(void* ptr) { return hipHostFree(ptr); }
#endif

unsigned int hipHostAllocDefault = hipHostMallocDefault;


hiprtcResult hiprtcGetLowLevelCode(hiprtcProgram prog, char* code) { return hiprtcGetCode(prog, code); }

hiprtcResult hiprtcGetLowLevelCodeSize(hiprtcProgram prog, size_t* codeSizeRet) { return hiprtcGetCodeSize(prog, codeSizeRet); }

hipError_t hipEventCreate(hipEvent_t* event, unsigned flags) { return hipEventCreateWithFlags(event, flags); }

hipError_t hipStreamCreate(hipStream_t* stream, unsigned int flags) { return hipStreamCreateWithFlags(stream, flags); }

hipError_t hipLaunchJITKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
  unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream,
  void** kernelParams, void** extra) {
  return hipModuleLaunchKernel(
    f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra);
}

hipblasStatus_t ACC_BLAS_STATUS_SUCCESS = HIPBLAS_STATUS_SUCCESS;
hipblasOperation_t ACC_BLAS_OP_N = HIPBLAS_OP_N;
hipblasOperation_t ACC_BLAS_OP_T = HIPBLAS_OP_T;
hiprtcResult ACC_RTC_SUCCESS = HIPRTC_SUCCESS;


/* HIP API: types
 * In the HIP API, there is no difference between runtime API and driver API
 * we therefore remap what the Driver API types would look like back to runtime API
 */
using hipfunction = hipFunction_t;
using hipstream = hipStream_t;
using hipevent = hipEvent_t;
using hipmodule = hipModule_t;
using hipdevice = hipDevice_t;
using hipDeviceProp = hipDeviceProp_t;
using hipcontext = hipCtx_t;

/* HIPBLAS status and operations */
extern hipblasStatus_t ACC_BLAS_STATUS_SUCCESS;
extern hipblasOperation_t ACC_BLAS_OP_N;
extern hipblasOperation_t ACC_BLAS_OP_T;

/* HIPRTC error status */
extern hiprtcResult ACC_RTC_SUCCESS;

#endif /*ACC_HIP_H*/
