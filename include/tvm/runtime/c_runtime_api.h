/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * \file tvm/runtime/c_runtime_api.h
 * \brief TVM runtime library.
 *
 *  The philosophy of TVM project is to customize the compilation
 *  stage to generate code that can used by other projects transparently.
 *  So this is a minimum runtime code gluing, and some limited
 *  memory management code to enable quick testing.
 *
 *  The runtime API is independent from TVM compilation stack and can
 *  be linked via libtvm_runtime.
 *
 *  The common flow is:
 *   - Use TVMFuncListGlobalNames to get global function name
 *   - Use TVMFuncCall to call these functions.
 *
 *  Possible return values of the API functions:
 *  * 0: success
 *  * -1: the error can be retrieved through TVMGetLastError.
 *  * -2: a frontend error occurred and recorded in the frontend.
 */
#ifndef TVM_RUNTIME_C_RUNTIME_API_H_
#define TVM_RUNTIME_C_RUNTIME_API_H_

// Macros to do weak linking
#ifdef _MSC_VER
#define TVM_WEAK __declspec(selectany)
#else
#define TVM_WEAK __attribute__((weak))
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define TVM_DLL EMSCRIPTEN_KEEPALIVE
#endif

// helper macro to suppress unused warning
#if defined(__GNUC__)
#define TVM_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define TVM_ATTRIBUTE_UNUSED
#endif

#ifndef TVM_DLL
#ifdef _WIN32
#ifdef TVM_EXPORTS
#define TVM_DLL __declspec(dllexport)
#else
#define TVM_DLL __declspec(dllimport)
#endif
#else
#define TVM_DLL __attribute__((visibility("default")))
#endif
#endif

// TVM version
#define TVM_VERSION "0.20.0"

// TVM Runtime is DLPack compatible.
#include <dlpack/dlpack.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*! \brief type of array index. */
typedef int64_t tvm_index_t;

/*! \brief Extension device types in TVM
 *
 * Additional enumerators to supplement those provided by
 * DLPack's `DLDeviceType` enumeration.
 *
 * MAINTAINERS NOTE #1: We need to ensure that the two devices
 * are identified by the same integer.
 * Currently this requires manual verification.
 * Discussed here: https://github.com/dmlc/dlpack/issues/111
 * As of DLPack v0.7, the highest-valued enumerator in
 * `DLDeviceType` is kDLHexagon = 16.
 *
 * MAINTAINERS NOTE #2: As of DLPack v0.7, the definition for
 * `DLDeviceType` specifies an underlying storage type of
 * `int32_t`.  That guarantees a variable of type
 * `DLDeviceType` is capable of holding any integers provided
 * by *either* of these enumerations.
 *
 * However, the `int32_t` specification only applies when the
 * header file is compiled as C++, and this header file is also
 * meant to work as C code.  So the unspecified storage type
 * could be a latent bug when compiled as C.
 */
#ifdef __cplusplus
typedef enum : int32_t {
#else
typedef enum {
#endif
  // To help avoid accidental conflicts between `DLDeviceType`
  // and this enumeration, start numbering the new enumerators
  // a little higher than (currently) seems necessary.
  TVMDeviceExtType_End = 36,  // sentinel value
} TVMDeviceExtType;

#ifdef __cplusplus
// Some other parts of TVM hardcode the integer identifier for
// some DLPack / TVM devices, rather then using the symbolic
// enumerator.   E.g., `2` rather than `kDLCUDA`.
// These asserts should alert us when that mapping breaks.
#define TVM_HARCODED_INTEGER_CHANGED_MSG                                                          \
  "Change in compile-time integer.  Make sure hardcoded uses of this integer throughout TVM are " \
  "updated."
static_assert(kDLCPU == 1, TVM_HARCODED_INTEGER_CHANGED_MSG);
static_assert(kDLCUDA == 2, TVM_HARCODED_INTEGER_CHANGED_MSG);
static_assert(kDLCUDAHost == 3, TVM_HARCODED_INTEGER_CHANGED_MSG);
static_assert(kDLOpenCL == 4, TVM_HARCODED_INTEGER_CHANGED_MSG);
static_assert(kDLVulkan == 7, TVM_HARCODED_INTEGER_CHANGED_MSG);
static_assert(kDLMetal == 8, TVM_HARCODED_INTEGER_CHANGED_MSG);
static_assert(kDLVPI == 9, TVM_HARCODED_INTEGER_CHANGED_MSG);
static_assert(kDLROCM == 10, TVM_HARCODED_INTEGER_CHANGED_MSG);
static_assert(kDLROCMHost == 11, TVM_HARCODED_INTEGER_CHANGED_MSG);
static_assert(kDLExtDev == 12, TVM_HARCODED_INTEGER_CHANGED_MSG);
static_assert(kDLCUDAManaged == 13, TVM_HARCODED_INTEGER_CHANGED_MSG);
static_assert(kDLOneAPI == 14, TVM_HARCODED_INTEGER_CHANGED_MSG);
static_assert(kDLWebGPU == 15, TVM_HARCODED_INTEGER_CHANGED_MSG);
static_assert(kDLHexagon == 16, TVM_HARCODED_INTEGER_CHANGED_MSG);

#undef TVM_HARCODED_INTEGER_CHANGED_MSG
#endif

/*!
 * \brief The type code in used and only used in TVM FFI for argument passing.
 *
 * DLPack consistency:
 * 1) kTVMArgInt is compatible with kDLInt
 * 2) kTVMArgFloat is compatible with kDLFloat
 * 3) kDLUInt is not in ArgTypeCode, but has a spared slot
 *
 * Downstream consistency:
 * The kDLInt, kDLUInt, kDLFloat are kept consistent with the original ArgType code
 *
 * It is only used in argument passing, and should not be confused with
 * DataType::TypeCode, which is DLPack-compatible.
 *
 * \sa tvm::runtime::DataType::TypeCode
 */
typedef enum {
  kTVMArgInt = kDLInt,
  kTVMArgFloat = kDLFloat,
  kTVMOpaqueHandle = 3U,
  kTVMNullptr = 4U,
  kTVMDataType = 5U,
  kDLDevice = 6U,
  kTVMDLTensorHandle = 7U,
  kTVMObjectHandle = 8U,
  kTVMModuleHandle = 9U,
  kTVMPackedFuncHandle = 10U,
  kTVMStr = 11U,
  kTVMBytes = 12U,
  kTVMNDArrayHandle = 13U,
  kTVMObjectRValueRefArg = 14U,
  kTVMArgBool = 15U,
  // Extension codes for other frameworks to integrate TVM PackedFunc.
  // To make sure each framework's id do not conflict, use first and
  // last sections to mark ranges.
  // Open an issue at the repo if you need a section of code.
  kTVMExtBegin = 16U,
  kTVMNNVMFirst = 16U,
  kTVMNNVMLast = 20U,
  // The following section of code is used for non-reserved types.
  kTVMExtReserveEnd = 64U,
  kTVMExtEnd = 128U,
} TVMArgTypeCode;

/*! \brief the array handle */
typedef DLTensor* TVMArrayHandle;

/*!
 * \brief Union type of values
 *  being passed through API and function calls.
 */
typedef union {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  const char* v_str;
  DLDataType v_type;
  DLDevice v_device;
} TVMValue;

/*!
 * \brief Byte array type used to pass in byte array
 *  When kTVMBytes is used as data type.
 */
typedef struct {
  const char* data;
  size_t size;
} TVMByteArray;

/*! \brief Handle to TVM runtime modules. */
typedef void* TVMModuleHandle;
/*! \brief Handle to packed function handle. */
typedef void* TVMFunctionHandle;
/*! \brief Handle to hold return value. */
typedef void* TVMRetValueHandle;
/*!
 * \brief The stream that is specific to device
 * can be NULL, which indicates the default one.
 */
typedef void* TVMStreamHandle;
/*! \brief Handle to Object. */
typedef void* TVMObjectHandle;

/*!
 * \brief Used for implementing C API function.
 *  Set last error message before return.
 * \param msg The error message to be set.
 */
TVM_DLL void TVMAPISetLastError(const char* msg);

/*!
 * \brief Used for implementing C API function.
 *  Set last exception before return.
 * \param py_object The python exception to be set
 */
TVM_DLL void TVMAPISetLastPythonError(void* py_object);

/*! \brief Return the previous python error, if any.
 *
 * Used to propagate the original Python exception to a python
 * try/except, when there are C++ stack frames between the location thro
 *
 * \return The previous argument passed during the most recent call to
 *     TVMAPISetLastPythonError.  If TVMAPISetLastPythonError has not
 *     been called, or if TVMDropLastPythonError has been called since
 *     the most recent to TVMAPISetLastPythonError, returns nullptr.
 */
TVM_DLL void* TVMGetLastPythonError();

/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and nonzero when an error occurred,
 *  TVMGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 *  \return error info
 */
TVM_DLL const char* TVMGetLastError(void);

/*!
 * \brief Return the backtrace of the most recent error
 *
 * Returns the backtrace of the most recent error, if an error exists,
 * and the error contains a backtrace.  If no error exists or the
 * error does not contain a backtrace, returns nullptr.
 *
 *  \return The backtrace of the most recent error
 */
TVM_DLL const char* TVMGetLastBacktrace();

/*!
 * \brief Remove the propagated python error, if any
 *
 * Removes the TVM-held reference to a thrown python exception object.
 * Because these objects contain references to the stack frames from
 * which the exception was thrown, maintaining a reference to an
 * exception object prevents any local python variables from being
 * garbage-collected.  After retrieving the object using
 * TVMGetLastPythonError, the Python FFI interface uses this method to
 * clear the TVM-held reference to the exception, to allow garbage
 * collection to continue.
 */
TVM_DLL void TVMDropLastPythonError();

/*! \brief Re-throw the most recent error.
 *
 * If an error was previously set using TVMAPISetLastError or
 * TVMAPISetLastPythonError, re-throw the error.  This is similar to
 * `LOG(FATAL) << TVMGetLastError()`, but includes handling to
 * propagate a python exception across C++ stack frames, or to append
 * a stack trace to an error message.
 */
TVM_DLL void TVMThrowLastError();

/*!
 * \brief Load module from file.
 * \param file_name The file name to load the module from.
 * \param format The format of the module.
 * \param out The result module
 *
 * \return 0 when success, nonzero when failure happens
 * \note The resulting module do not contain import relation.
 *  It can be reconstructed by TVMModImport.
 */
TVM_DLL int TVMModLoadFromFile(const char* file_name, const char* format, TVMModuleHandle* out);

/*!
 * \brief Add dep to mod's dependency.
 *  This allows functions in this module to use modules.
 *
 * \param mod The module handle.
 * \param dep The dependent module to be imported.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMModImport(TVMModuleHandle mod, TVMModuleHandle dep);

/*!
 * \brief Get function from the module.
 * \param mod The module handle.
 * \param func_name The name of the function.
 * \param query_imports Whether to query imported modules
 * \param out The result function, can be NULL if it is not available.
 * \return 0 when no error is thrown, nonzero when failure happens
 */
TVM_DLL int TVMModGetFunction(TVMModuleHandle mod, const char* func_name, int query_imports,
                              TVMFunctionHandle* out);

/*!
 * \brief Free the Module
 * \param mod The module to be freed.
 *
 * \note This may not free up the module's resources.
 *  If there is active TVMFunctionHandle uses the module
 *  Or if this module is imported by another active module.
 *
 *  The all functions remains valid until TVMFuncFree is called.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMModFree(TVMModuleHandle mod);

/*!
 * \brief Free the function when it is no longer needed.
 * \param func The function handle
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMFuncFree(TVMFunctionHandle func);

/*!
 * \brief Call a Packed TVM Function.
 *
 * \param func node handle of the function.
 * \param arg_values The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 *
 * \param ret_val The return value.
 * \param ret_type_code the type code of return value.
 *
 * \return 0 when success, nonzero when failure happens
 * \note TVM calls always exchanges with type bits=64, lanes=1
 *
 * \note API calls always exchanges with type bits=64, lanes=1
 *   If API call returns container handles (e.g. FunctionHandle)
 *   these handles should be managed by the front-end.
 *   The front-end need to call free function (e.g. TVMFuncFree)
 *   to free these handles.
 */
TVM_DLL int TVMFuncCall(TVMFunctionHandle func, TVMValue* arg_values, int* type_codes, int num_args,
                        TVMValue* ret_val, int* ret_type_code);

/*!
 * \brief Set the return value of TVMPackedCFunc.
 *
 *  This function is called by TVMPackedCFunc to set the return value.
 *  When this function is not called, the function returns null by default.
 *
 * \param ret The return value handle, pass by ret in TVMPackedCFunc
 * \param value The value to be returned.
 * \param type_code The type of the value to be returned.
 * \param num_ret Number of return values, for now only 1 is supported.
 */
TVM_DLL int TVMCFuncSetReturn(TVMRetValueHandle ret, TVMValue* value, int* type_code, int num_ret);

/*!
 * \brief Inplace translate callback argument value to return value.
 *  This is only needed for non-POD arguments.
 *
 * \param value The value to be translated.
 * \param code The type code to be translated.
 * \note This function will do a shallow copy when necessary.
 *
 * \return 0 when success, nonzero when failure happens.
 */
TVM_DLL int TVMCbArgToReturn(TVMValue* value, int* code);

/*!
 * \brief C type of packed function.
 *
 * \param args The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 * \param ret The return value handle.
 * \param resource_handle The handle additional resouce handle from front-end.
 * \return 0 if success, -1 if failure happens, set error via TVMAPISetLastError.
 * \sa TVMCFuncSetReturn
 */
typedef int (*TVMPackedCFunc)(TVMValue* args, int* type_codes, int num_args, TVMRetValueHandle ret,
                              void* resource_handle);

/*!
 * \brief C callback to free the resource handle in C packed function.
 * \param resource_handle The handle additional resouce handle from front-end.
 */
typedef void (*TVMPackedCFuncFinalizer)(void* resource_handle);

/*!
 * \brief Signature for extension function declarer.
 *
 *  TVM call this function to get the extension functions
 *  The declarer will call register_func to register function and their name.
 *
 * \param register_func_handle The register function
 * \return 0 if success, -1 if failure happens
 */
typedef int (*TVMExtensionFuncDeclarer)(TVMFunctionHandle register_func_handle);

/*!
 * \brief Wrap a TVMPackedCFunc to become a FunctionHandle.
 *
 * The resource_handle will be managed by TVM API, until the function is no longer used.
 *
 * \param func The packed C function.
 * \param resource_handle The resource handle from front-end, can be NULL.
 * \param fin The finalizer on resource handle when the FunctionHandle get freed, can be NULL
 * \param out the result function handle.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMFuncCreateFromCFunc(TVMPackedCFunc func, void* resource_handle,
                                   TVMPackedCFuncFinalizer fin, TVMFunctionHandle* out);

/*!
 * \brief Register the function to runtime's global table.
 *
 * The registered function then can be pulled by the backend by the name.
 *
 * \param name The name of the function.
 * \param f The function to be registered.
 * \param override Whether allow override already registered function.
 */
TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override);

/*!
 * \brief Get a global function.
 *
 * \param name The name of the function.
 * \param out the result function pointer, NULL if it does not exist.
 *
 * \note The function handle of global function is managed by TVM runtime,
 *  So TVMFuncFree is should not be called when it get deleted.
 */
TVM_DLL int TVMFuncGetGlobal(const char* name, TVMFunctionHandle* out);

/*!
 * \brief List all the globally registered function name
 * \param out_size The number of functions
 * \param out_array The array of function names.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMFuncListGlobalNames(int* out_size, const char*** out_array);

/*!
 * \brief Remove a global function.
 * \param name The name of the function.
 */
TVM_DLL int TVMFuncRemoveGlobal(const char* name);

// Array related apis for quick proptyping
/*!
 * \brief Allocate a nd-array's memory,
 *  including space of shape, of given spec.
 *
 * \param shape The shape of the array, the data content will be copied to out
 * \param ndim The number of dimension of the array.
 * \param dtype_code The type code of the dtype
 * \param dtype_bits The number of bits of dtype
 * \param dtype_lanes The number of lanes in the dtype.
 * \param device_type The device type.
 * \param device_id The device id.
 * \param out The output handle.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMArrayAlloc(const tvm_index_t* shape, int ndim, int dtype_code, int dtype_bits,
                          int dtype_lanes, int device_type, int device_id, TVMArrayHandle* out);

/*!
 * \brief Free the TVM Array.
 * \param handle The array handle to be freed.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMArrayFree(TVMArrayHandle handle);

/*!
 * \brief Copy array data from CPU byte array.
 * \param handle The array handle.
 * \param data the data pointer
 * \param nbytes The number of bytes to copy.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMArrayCopyFromBytes(TVMArrayHandle handle, void* data, size_t nbytes);

/*!
 * \brief Copy array data to CPU byte array.
 * \param handle The array handle.
 * \param data the data pointer
 * \param nbytes The number of bytes to copy.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMArrayCopyToBytes(TVMArrayHandle handle, void* data, size_t nbytes);

/*!
 * \brief Copy the array, both from and to must be valid during the copy.
 * \param from The array to be copied from.
 * \param to The target space.
 * \param stream The stream where the copy happens, can be NULL.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMArrayCopyFromTo(TVMArrayHandle from, TVMArrayHandle to, TVMStreamHandle stream);

/*!
 * \brief Produce an array from the DLManagedTensor that shares data memory
 * with the DLManagedTensor.
 * \param from The source DLManagedTensor.
 * \param out The output array handle.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMArrayFromDLPack(DLManagedTensor* from, TVMArrayHandle* out);

/*!
 * \brief Produce a DLMangedTensor from the array that shares data memory with
 * the array.
 * \param from The source array.
 * \param out The DLManagedTensor handle.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMArrayToDLPack(TVMArrayHandle from, DLManagedTensor** out);

/*!
 * \brief Delete (free) a DLManagedTensor's data.
 * \param dltensor Pointer to the DLManagedTensor.
 */
TVM_DLL void TVMDLManagedTensorCallDeleter(DLManagedTensor* dltensor);

/*!
 * \brief Create a new runtime stream.
 *
 * \param device_type The device type.
 * \param device_id The device id.
 * \param out The new stream handle.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMStreamCreate(int device_type, int device_id, TVMStreamHandle* out);

/*!
 * \brief Free a created stream handle.
 *
 * \param device_type The device type.
 * \param device_id The device id.
 * \param stream The stream to be freed.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMStreamFree(int device_type, int device_id, TVMStreamHandle stream);

/*!
 * \brief Set the runtime stream of current thread to be stream.
 *  The subsequent calls to the same device_type
 *  will use the setted stream handle.
 *  The specific type of stream is runtime device dependent.
 *
 * \param device_type The device type.
 * \param device_id The device id.
 * \param handle The stream handle.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMSetStream(int device_type, int device_id, TVMStreamHandle handle);

/*!
 * \brief Wait until all computations on stream completes.
 *
 * \param device_type The device type.
 * \param device_id The device id.
 * \param stream The stream to be synchronized.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMSynchronize(int device_type, int device_id, TVMStreamHandle stream);

/*!
 * \brief Synchronize two streams of execution.
 *
 * \param device_type The device type.
 * \param device_id The device id.
 * \param src The source stream to synchronize.
 * \param dst The destination stream to synchronize.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMStreamStreamSynchronize(int device_type, int device_id, TVMStreamHandle src,
                                       TVMStreamHandle dst);

/*!
 * \brief Get the type_index from an object.
 *
 * \param obj The object handle.
 * \param out_tindex the output type index.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMObjectGetTypeIndex(TVMObjectHandle obj, unsigned* out_tindex);

/*!
 * \brief Convert type key to type index.
 * \param type_key The key of the type.
 * \param out_tindex the corresponding type index.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMObjectTypeKey2Index(const char* type_key, unsigned* out_tindex);

/*!
 * \brief Convert type index to type key.
 * \param tindex The type index.
 * \param out_type_key The output type key.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMObjectTypeIndex2Key(unsigned tindex, char** out_type_key);

/*!
 * \brief Increase the reference count of an object.
 *
 * \param obj The object handle.
 * \note Internally we increase the reference counter of the object.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMObjectRetain(TVMObjectHandle obj);

/*!
 * \brief Free the object.
 *
 * \param obj The object handle.
 * \note Internally we decrease the reference counter of the object.
 *       The object will be freed when every reference to the object are removed.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMObjectFree(TVMObjectHandle obj);

/*!
 * \brief Free a TVMByteArray returned from TVMFuncCall, and associated memory.
 * \param arr The TVMByteArray instance.
 * \return 0 on success, -1 on failure.
 */
TVM_DLL int TVMByteArrayFree(TVMByteArray* arr);

/*!
 * \brief Allocate a data space on device.
 * \param dev The device to perform operation.
 * \param nbytes The number of bytes in memory.
 * \param alignment The alignment of the memory.
 * \param type_hint The type of elements. Only needed by certain backends such
 *                   as nbytes & alignment are sufficient for most backends.
 * \param out_data The allocated device pointer.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMDeviceAllocDataSpace(DLDevice dev, size_t nbytes, size_t alignment,
                                    DLDataType type_hint, void** out_data);

/*!
 * \brief Allocate a data space on device with special memory scope.
 * \note The memory could use a special multi-dimensional memory layout.
 *       That is why we pass shape and dtype instead of raw number of bytes.
 * \param dev The device to perform operation.
 * \param ndim The number of dimension of the tensor.
 * \param shape The shape of the tensor.
 * \param dtype The type of elements.
 * \param mem_scope The memory scope of the tensor,
 *        can be nullptr, which indicate the default global DRAM
 * \param out_data The allocated device pointer.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMDeviceAllocDataSpaceWithScope(DLDevice dev, int ndim, const int64_t* shape,
                                             DLDataType dtype, const char* mem_scope,
                                             void** out_data);

/*!
 * \brief Free a data space on device.
 * \param dev The device to perform operation.
 * \param ptr The data space.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMDeviceFreeDataSpace(DLDevice dev, void* ptr);

/*!
 * \brief Copy data from one place to another.
 * \note This API is designed to support special memory with shape dependent layout.
 *       We pass in DLTensor* with shape information to support these cases.
 * \param from The source tensor.
 * \param to The target tensor.
 * \param stream Optional stream object.
 * \return 0 when success, nonzero when failure happens.
 */
TVM_DLL int TVMDeviceCopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream);

/*!
 * \brief Check that an object is derived from another.
 * \param child_type_index The type index of the derived type.
 * \param parent_type_index The type index of the parent type.
 * \param is_derived A boolean representing whether this predicate holds.
 * \return 0 when success, nonzero when failure happens.
 */
TVM_DLL int TVMObjectDerivedFrom(uint32_t child_type_index, uint32_t parent_type_index,
                                 int* is_derived);

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
#endif  // TVM_RUNTIME_C_RUNTIME_API_H_
