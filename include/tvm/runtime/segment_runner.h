#ifndef TVM_RUNTIME_SEGMENT_RUNNER_H_
#define TVM_RUNTIME_SEGMENT_RUNNER_H_

#include <tvm/runtime/vm/vm.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/vm/bytecode.h>


namespace tvm {
namespace runtime {

class SegmentRunner : public ffi::Object {
public:
    SegmentRunner(const Module& exec, Device device);

    std::string GetRuntimeSequence();
    int Load(const std::string runtime_sequence_path);
    void SetInput(std::vector<NDArray>& input);
    void SetInputWithParams(std::vector<NDArray>& input, std::vector<NDArray>& params);
    std::vector<NDArray> GetOutput();
    void Execute(const int segment_id);
    size_t GetLength();
private:
    Module vm_module_;
    std::vector<std::vector<int64_t>> segment_list_;
    bool is_initialized_ = false;
};

} // namespace runtime
} // namespace tvm

#endif