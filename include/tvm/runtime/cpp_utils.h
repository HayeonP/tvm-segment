#ifndef TVM_RUNTIME_CPP_UTILS_H_
#define TVM_RUNTIME_CPP_UTILS_H_

#include <tvm/runtime/file_utils.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/relax_vm/executable.h>
#include <tvm/runtime/relax_vm/vm.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/device_api.h>
#include <dmlc/memory_io.h>

#include <vector>
#include <fstream>
#include <memory>

namespace tvm{
namespace runtime {

bool NaturalSortCompare(const std::string& a, const std::string& b);
std::vector<NDArray> LoadParamsAsNDArrayList(std::string& path);
Module LoadExecutableModule(std::string& path);
std::shared_ptr<relax_vm::VirtualMachineImpl> InitVirtualMachine(DLDevice& device, Module& executableModule);
std::vector<NDArray> moveParamsToDevice(std::vector<NDArray>& params, DLDevice& device);
NDArray convertVecToNDArray(std::vector<float>& data, std::vector<int64_t>& shape);

} // namespace runtime
} // namespace tvm


#endif