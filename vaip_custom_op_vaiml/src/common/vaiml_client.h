/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>

// KEEP THE EMPTY LINE ABOVE SO CLANGFORMAT DOES NOT REORDER THE HEADER BELOW
#  include <errhandlingapi.h>
#else
#  include <dlfcn.h>
#endif

#include <algorithm>
#include <exception>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Temporary, for debugging
#include <iostream>

#ifdef _WIN32
#  ifdef FLEXMLRT_EXPORT
#    define FLEXMLRT_DLL __declspec(dllexport)
#  else
#    define FLEXMLRT_DLL __declspec(dllimport)
#  endif
#  define FLEXMLRT_LIB_NAME "flexmlrt.dll"
#else
#  define FLEXMLRT_DLL
#  define FLEXMLRT_LIB_NAME "libflexmlrt.so"
#endif

// Define this macro to use dlmopen instead of dlopen
// #define FLEXML_LOADER_USE_DLMOPEN 1

// Define this macro to turn on debug messages
// #define FLEXML_LOADER_DEBUG 1

// Define this macro to use in-library destructor
// #define FLEXML_CLIENT_USE_IN_LIBRARY_DESTRUCTOR 1

// Prints debug messages to stdout if FLEXML_LOADER_DEBUG is on.
#ifdef FLEXML_LOADER_DEBUG
#  include <iostream>

#  define FLEXML_LOADER_DEBUG_PRINT(str)                                       \
    do {                                                                       \
      std::cout << "DEBUG: xilinx_apps loader: " << str << std::endl;          \
    } while (false)
#else
#  define FLEXML_LOADER_DEBUG_PRINT(str)
#endif

namespace flexmlrt::client {

/**
 * @brief %Exception class for shared library dynamic loading errors
 *
 * This exception class is derived from `std::exception` and provides the
 * standard @ref what() member function. An object of this class is constructed
 * with an error message string, which is stored internally and retrieved with
 * the @ref what() member function.
 */
class LoaderException : public std::exception {
  std::string message;

public:
  /**
   * Constructs an Exception object.
   *
   * @param msg an error message string, which is copied and stored internal to
   * the object
   */
  LoaderException(std::string msg) : message(std::move(msg)) {}

  /**
   * Returns the error message string passed to the constructor.
   *
   * @return the error message string
   */
  // Lint: not compatible with C++11
  // NOLINTNEXTLINE(modernize-use-nodiscard)
  const char* what() const noexcept override { return message.c_str(); }
};

/**
 * @brief %Exception class for FlexML API run-time errors
 *
 * This exception class is derived from `std::exception` and provides the
 * standard @ref what() member function. An object of this class is constructed
 * with an error message string, which is stored internally and retrieved with
 * the @ref what() member function.
 */
class Exception : public std::exception {
  std::string message;

public:
  /**
   * Constructs an Exception object.
   *
   * @param msg an error message string, which is copied and stored internal to
   * the object
   */
  Exception(std::string msg) : message(std::move(msg)) {}

  /**
   * Returns the error message string passed to the constructor.
   *
   * @return the error message string
   */
  // Lint: not compatible with C++11
  // NOLINTNEXTLINE(modernize-use-nodiscard)
  const char* what() const noexcept override { return message.c_str(); }
};

enum class ErrorCode : int {
  NoError = 0,
  IncompatibleInput = 1,
  IncompatibleOutput = 2,
  WrongNumInputs = 3,
  WrongNumOutputs = 4
};

class BadArgsException : public Exception {

  static std::string getString(int rc) {
    std::string codeStr;
    switch (ErrorCode(rc)) {
    case ErrorCode::NoError:
      codeStr = "No error";
      break;
    case ErrorCode::IncompatibleInput:
      codeStr = "Incompatible input shape";
      break;
    case ErrorCode::IncompatibleOutput:
      codeStr = "Incompatible output shape";
      break;
    case ErrorCode::WrongNumInputs:
      codeStr = "Wrong number of inputs";
      break;
    case ErrorCode::WrongNumOutputs:
      codeStr = "Wrong number of outputs";
      break;
    }

    return std::string("Model argument error: ") + codeStr;
  }

public:
  BadArgsException(int rc) : Exception(getString(rc)) {}
};

/**
 * @brief Struct containing FlexML API Input/Ouput buffer type for
 * general External RunTime (ERT) integration
 */
struct ErtIoType {
  /**
   * Pointer to the buffer data
   */
  void* data;

  /**
   * Buffer name in the Model ADF graph. This is used to verify node names from
   * ORT matches port names in the codegen ADF by Flexml Backend
   */
  std::string name;

  /**
   * Buffer index from external runtime to match against the argument index
   * of forward function
   */
  int idx;

  /**
   * Total size of valid data or allocated space in buffer data pointer
   */
  size_t size;

  /**
   * Data type of buffer elements
   */
  std::string type;
};

/**
 * @brief Struct containing FlexML API Input/Ouput buffer type for
 * general External RunTime (ERT) integration
 */
struct ErtIoTypeNew {
  /**
   * Pointer to the buffer data
   */
  void* data;

  /**
   * Buffer name in the Model ADF graph. This is used to verify node names from
   * ORT matches port names in the codegen ADF by Flexml Backend
   */
  std::string name;

  /**
   * Buffer index from external runtime to match against the argument index
   * of forward function
   */
  int idx;

  /**
   * Total size of valid data or allocated space in buffer data pointer
   */
  size_t size;

  /**
   * Data type of buffer elements
   */
  std::string type;

  /**
   * Shape of buffer element
   */
  std::vector<std::size_t> shape;
};

/**
 * @brief Struct containing FlexML API configuration options
 */
struct Options {
  /**
   * File path to the directory containing the compiled ML model.
   */
  std::string modelPath;

  /**
   * The name of the device to run the model on.
   */
  std::string deviceName;

  /**
   * debug flag to turn on debug messages
   */
  bool debug = false;

  /**
   * flag to generate execution profiling summary
   */
  bool profileSummary = false;

  /**
   * executeMode indicates the mode of execution
   *   0: Pass the data as it is. No quantization or reformatting.
   *   1:
   */
  uint32_t executeMode = 1;

  /**
   * apiVersion sets the version of APIs to used.
   *   1: APIs support a single input
   *   2: Default. Latest API version that supports multiple inputs
   */
  uint32_t apiVersion = 2;

  /**
   * Destroys this Options object.
   */
  ~Options() = default;

  /**
   * Constructs an Options object with default values.
   */
  Options() = default;

  /**
   * Copy constructor, which deep-copies the reference Options object.
   * @param opt the Options object to copy from
   */
  Options(const Options& opt) { copyIn(opt); }

  /**
   * Deep-copies the reference Options object into this Options object.
   *
   * @param opt the Options object to copy from
   * @return this Options object
   */
  Options& operator=(const Options& opt) {
    copyIn(opt);
    return *this;
  }

private:
  void copyIn(const Options& opt) {
    modelPath = opt.modelPath;
    deviceName = opt.deviceName;
    debug = opt.debug;
    executeMode = opt.executeMode;
    profileSummary = opt.profileSummary;
  }
};

/**
 * @brief Tensor to pass as input or output of a model.
 */
struct Tensor {
  using Dimension = std::size_t;

  /**
   * Pointer to tensor values.  The user owns the memory pointed to.
   */
  float* data = nullptr;

  /**
   * The shape of the tensor, one vector element per dimenion.
   */
  std::vector<Dimension> shape;
};

/**
 * @brief Tensor to pass as input or output of a model.
 */
struct OpsTensor {
  using Dimension = std::size_t;

  /**
   * Pointer to tensor values.  The user owns the memory pointed to.
   */
  void* data = nullptr;

  /**
   * The shape of the tensor, one vector element per dimenion.
   */
  std::vector<Dimension> shape;

  /**
   * Datatype of the tensor.
   */
  std::string dtype;
};

class ImplBase;

// #############################################################################

namespace impl {

#ifdef _WIN32

// Name of the model .so inside the .flexml directory
const std::string modelLibName = "libflexml_usermodel.dll";

inline std::string getWinErrorStr(DWORD errorId) {
  LPVOID bufptr = nullptr;
  FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL, errorId, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                (LPTSTR)&bufptr, 0, NULL);

  std::string errStr;
  if (bufptr) {
    errStr = reinterpret_cast<char*>(bufptr);
    LocalFree(bufptr);
  }
  return errStr;
}

template <typename FuncType>
FuncType getDynamicFunction(const std::string& libName,
                            const std::string& funcName, bool silent = false) {
  FLEXML_LOADER_DEBUG_PRINT("libName=" << libName << ", funcName=" << funcName);

  // open the library

  HMODULE handle = nullptr;
  SetLastError(0);
  std::string winLibName = libName;
  std::replace(winLibName.begin(), winLibName.end(), '/', '\\');
  handle = LoadLibraryA(winLibName.c_str());

  // To consider: save the handle for each libName in a map.  Calling dlopen
  // multiple times increases the ref count, requiring an equal number of
  // dlclose, but in the context of an application that runs forever, leaving
  // the .so open doesn't matter.

  FLEXML_LOADER_DEBUG_PRINT("after LoadLibrary, handle=0x" << std::hex << handle
                                                           << std::dec);
  if (handle == nullptr) {
    FLEXML_LOADER_DEBUG_PRINT("inside handle==nullptr");
    std::ostringstream oss;
    oss << "ERROR: Cannot open library " << winLibName << ": "
        << getWinErrorStr(GetLastError()) << "." << std::endl;
    FLEXML_LOADER_DEBUG_PRINT("after oss filling");
    throw LoaderException(oss.str());
  }

  // load the symbol
  FLEXML_LOADER_DEBUG_PRINT("after handle==nullptr check");
  SetLastError(0); // reset errors
  FLEXML_LOADER_DEBUG_PRINT("before GetProcAddress");
  FARPROC pFunc = GetProcAddress(handle, funcName.c_str());
  FLEXML_LOADER_DEBUG_PRINT("after GetProcAddress");
  if (pFunc == nullptr) {
    FLEXML_LOADER_DEBUG_PRINT("inside null pFunc");
    std::ostringstream oss;
    oss << "ERROR: Cannot load symbol '" << funcName
        << "': " << getWinErrorStr(GetLastError())
        << ".  Please recompile the model." << std::endl;
    FLEXML_LOADER_DEBUG_PRINT("after 2nd oss filling");
    throw LoaderException(oss.str());
  }
  FLEXML_LOADER_DEBUG_PRINT("before return");
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return reinterpret_cast<FuncType>(pFunc);
}

#else
// Name of the model .so inside the .flexml directory
const std::string modelLibName = "libflexml_usermodel.so";

template <typename FuncType>
FuncType getDynamicFunction(const std::string& libName,
                            const std::string& funcName, bool silent = false) {
  FLEXML_LOADER_DEBUG_PRINT("libName=" << libName << ", funcName=" << funcName);

  // open the library

  void* handle =
#  ifdef FLEXML_LOADER_USE_DLMOPEN
      dlmopen(LM_ID_NEWLM, libName.c_str(), RTLD_LAZY);
#  else
      // We need RTLD_LOCAL to confine the symbols of the loaded library.
      // Otherwise, when calling this a second time to load another model,
      // it would resolve to the symbols from the first model.
      dlopen(libName.c_str(), RTLD_LAZY | RTLD_LOCAL);
#  endif
  // To consider: save the handle for each libName in a map.  Calling dlopen
  // multiple times increases the ref count, requiring an equal number of
  // dlclose, but in the context of an application that runs forever, leaving
  // the .so open doesn't matter.

  FLEXML_LOADER_DEBUG_PRINT("after dlopen, handle=0x" << std::hex << handle
                                                      << std::dec);
  if (handle == nullptr) {
    if (silent)
      return nullptr;
    FLEXML_LOADER_DEBUG_PRINT("inside handle==nullptr");
    std::ostringstream oss;
    oss << "ERROR: Cannot open library " << libName << ": " << dlerror() << "."
        << std::endl;
    FLEXML_LOADER_DEBUG_PRINT("after oss filling");
    throw LoaderException(oss.str());
  }

  // load the symbol
  FLEXML_LOADER_DEBUG_PRINT("after handle==nullptr check");
  dlerror(); // reset errors
  FLEXML_LOADER_DEBUG_PRINT("before dlsym");
  void* pFunc = dlsym(handle, funcName.c_str());
  FLEXML_LOADER_DEBUG_PRINT("after dlsym");
  const char* dlsymError2 = dlerror();
  if (dlsymError2 != nullptr) {
    if (silent)
      return nullptr;
    FLEXML_LOADER_DEBUG_PRINT("inside dlsym_error2");
    std::ostringstream oss;
    oss << "ERROR: Cannot load symbol '" << funcName << "': " << dlsymError2
        << ".  Please recompile the model." << std::endl;
    FLEXML_LOADER_DEBUG_PRINT("after 2nd oss filling");
    throw LoaderException(oss.str());
  }
  FLEXML_LOADER_DEBUG_PRINT("before return");
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return reinterpret_cast<FuncType>(pFunc);
}

#endif

} // namespace impl

// #############################################################################

/// @cond INTERNAL

class ImplBase {
public:
  virtual ~ImplBase() = default;
  virtual float* forward(const float* ifm) = 0;
  virtual float* forward(unsigned batchSize, const float* ifm) = 0;

  // TODO: Remove this function from here
  virtual void forward(const float* ifm,
                       const std::vector<std::size_t>& inShape, float* ofm,
                       const std::vector<std::size_t>& outShape);

  virtual void forward(const std::vector<Tensor>& ifms,
                       std::vector<Tensor>& ofms) = 0;

  virtual void forward(std::vector<ErtIoType>& ifms,
                       std::vector<ErtIoType>& ofms) = 0;

  virtual void forward(std::vector<ErtIoTypeNew>& ifms,
                       std::vector<ErtIoTypeNew>& ofms,
                       std::vector<ErtIoTypeNew>& wts) = 0;

  [[nodiscard]] virtual std::vector<std::vector<std::size_t>>
  getOfmShapes() const = 0;

  virtual void timestampStart(std::string);
  virtual void timestampEnd();

  virtual void setAIAnalyzerProfiling();
  virtual void unsetAIAnalyzerProfiling();
};
/// @endcond

/**
 * @brief ML Model API class.
 *
 * This class provides the API for the compiled ML model.  Use this class to run
 * the model from a C++ application.
 */
class Model {
public:
  /**
   * Constructs a Model object.
   *
   * @param options the configuration options for the model.  See the Options
   * struct for details.
   */
  Model(const Options& options)
      : options(options), libName(options.modelPath + "/" + impl::modelLibName),
        pImpl(createImpl(FLEXMLRT_LIB_NAME, options)) {}

  /**
   * Destroys this Model object.
   */
  ~Model() { destroyImpl(FLEXMLRT_LIB_NAME, pImpl); }

  /**
   * Runs the compiled ML model.
   *
   * @param ifm the input tensor (input feature map)
   * @return the output tensor (output feature map)
   */
  float* forward(const float* ifm) { return pImpl->forward(ifm); }

  float* forward(unsigned batchSize, const float* ifm) {
    return pImpl->forward(batchSize, ifm);
  }

  void forward(const float* ifm, const std::vector<size_t>& ishape, float* ofm,
               const std::vector<size_t>& oshape) {
    return pImpl->forward(ifm, ishape, ofm, oshape);
  }

  void forward(const std::vector<Tensor>& ifms, std::vector<Tensor>& ofms) {
    pImpl->forward(ifms, ofms);
  }

  void forward(std::vector<ErtIoType>& ifms, std::vector<ErtIoType>& ofms) {
    pImpl->forward(ifms, ofms);
  }

  void forward(std::vector<ErtIoTypeNew>& ifms, std::vector<ErtIoTypeNew>& ofms,
               std::vector<ErtIoTypeNew>& wts) {
    pImpl->forward(ifms, ofms, wts);
  }

  [[nodiscard]] std::vector<std::vector<std::size_t>> getOfmShapes() const {
    return pImpl->getOfmShapes();
  }

  void setAIAnalyzerProfiling() { pImpl->setAIAnalyzerProfiling(); }
  void unsetAIAnalyzerProfiling() { pImpl->unsetAIAnalyzerProfiling(); }

private:
  Options options;
  std::string libName = FLEXMLRT_LIB_NAME;
  ImplBase* pImpl = nullptr;

  static ImplBase* createImpl(const std::string& libName,
                              const flexmlrt::client::Options& options) {
    using CreateFunc =
        flexmlrt::client::ImplBase* (*)(const void*, std::size_t);
    FLEXML_LOADER_DEBUG_PRINT("Loading FlexML API shared library dynamically "
                              "(via dlopen).");
    auto pCreateFunc = impl::getDynamicFunction<CreateFunc>(
        libName, "flexml_client_Model_createImpl");
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return pCreateFunc(reinterpret_cast<const void*>(&options),
                       sizeof(Options));
  }

  static void destroyImpl(const std::string& libName,
                          flexmlrt::client::ImplBase* pImpl) {
    using DestroyFunc = void (*)(ImplBase*);
    try {
      auto pDestroyFunc = impl::getDynamicFunction<DestroyFunc>(
          libName, "flexml_client_Model_destroyImpl");
      pDestroyFunc(pImpl);
    } catch (const LoaderException& ex) {
      (void)ex; // usage is debug only; causes warning in release mode
      FLEXML_LOADER_DEBUG_PRINT(
          "getDynamicFunction(destroyFunc) failed: " << ex.what());
    }
  }
};

} // namespace flexmlrt::client
