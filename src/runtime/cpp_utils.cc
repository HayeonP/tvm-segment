#include <tvm/runtime/cpp_utils.h>

namespace tvm {
namespace runtime {

bool NaturalSortCompare(const std::string& a, const std::string& b) {
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (std::isdigit(a[i]) && std::isdigit(b[j])) {
            // 숫자 부분 추출
            long num_a = 0, num_b = 0;
            while (i < a.size() && std::isdigit(a[i])) {
                num_a = num_a * 10 + (a[i++] - '0');
            }
            while (j < b.size() && std::isdigit(b[j])) {
                num_b = num_b * 10 + (b[j++] - '0');
            }
            
            // compare number
            if (num_a != num_b) {
                return num_a < num_b;
            }
        } else {
            // compare non-number
            if (a[i] != b[j]) {
                return a[i] < b[j];
            }
            i++;
            j++;
        }
    }
    return a.size() < b.size();
}

Module LoadExecutableModule(std::string& path){
    return Module::LoadFromFile(path);    
}

std::vector<NDArray> LoadParamsAsNDArrayList(std::string& path) {
    std::ifstream binary_file(path, std::ios::binary);
    std::string binary_data( (std::istreambuf_iterator<char>(binary_file)), std::istreambuf_iterator<char>() );
    binary_file.close();
    dmlc::MemoryStringStream strm(&binary_data);
    
    tvm::runtime::Map<tvm::runtime::String, tvm::runtime::NDArray> params = tvm::runtime::LoadParams(&strm);
    
    std::vector<std::string> keys;

    for (const auto& kv : params) {
        keys.push_back(kv.first);
    }
    
    // Sorting
    std::sort(keys.begin(), keys.end(), NaturalSortCompare);
    
    // Create sorted parameters
    std::vector<tvm::runtime::NDArray> sorted_params;
    sorted_params.reserve(keys.size());
    
    for (const auto& key : keys) {
        sorted_params.push_back(params[key]);
    }
    
    return sorted_params;
}

// TODO: Support single accelerator only
std::shared_ptr<relax_vm::VirtualMachineImpl> InitVirtualMachine(DLDevice& device, Module& executableModule){
    std::vector<DLDevice> devices;
    devices.push_back(device);

    if(devices.back().device_type % kRPCSessMask != kDLCPU){
        devices.push_back(DLDevice{kDLCPU, 0});
    }

    std::vector<memory::AllocatorType> alloc_types;
    auto default_alloc_type = memory::AllocatorType::kPooled;

    for(auto& dev : devices) {
        dev.device_type = DLDeviceType{dev.device_type % tvm::runtime::kRPCSessMask};
        alloc_types.push_back(default_alloc_type); // TODO: Consider diffent memory cfg - relax_vm.py:_setup_device()
    }

    auto executable = executableModule.as<tvm::runtime::relax_vm::VMExecutable>();
    
    auto vm_module_ptr = std::make_shared<tvm::runtime::Module>(
        executable->VMLoadExecutable()
    );

    auto vm_ptr = const_cast<relax_vm::VirtualMachineImpl *>(
        vm_module_ptr->as<relax_vm::VirtualMachineImpl>()
    );

    vm_ptr->Init(devices, alloc_types);
    
    return std::shared_ptr<relax_vm::VirtualMachineImpl>(
        vm_ptr,
        [vm_module_ptr](auto) {}
    );
}

NDArray convertVecToNDArray(std::vector<float>& data, std::vector<int64_t>& shape){
    tvm::runtime::NDArray ndarray = tvm::runtime::NDArray::Empty(
        tvm::runtime::ShapeTuple(shape),
        DLDataType{kDLFloat, 32, 1},
        DLDevice{kDLCPU, 0}
    );
    
    size_t num_bytes = data.size() * sizeof(float);
    ndarray.CopyFromBytes(data.data(), num_bytes);
    return ndarray;
}

} // namespace runtime
} // namespace tvm

