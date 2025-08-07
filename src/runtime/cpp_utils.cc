#include <tvm/runtime/cpp_utils.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/vm/executable.h>
#include "file_utils.h"

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
    return Module::LoadFromFile(path); // HayeonP: Same with self.module of VM in python
}

std::vector<NDArray> LoadParamsAsNDArrayList(std::string& path) {
    std::ifstream binary_file(path, std::ios::binary);
    std::string binary_data( (std::istreambuf_iterator<char>(binary_file)), std::istreambuf_iterator<char>() );
    binary_file.close();
    dmlc::MemoryStringStream strm(&binary_data);
    
    Map<String, tvm::runtime::NDArray> params = tvm::runtime::LoadParams(&strm);
    
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

NDArray convertVecToNDArray(std::vector<float>& data, std::vector<int64_t>& shape){

    NDArray ndarray = tvm::runtime::NDArray::Empty(
        ffi::Shape(shape),
        DLDataType{kDLFloat, 32, 1},
        DLDevice{kDLCPU, 0}
    );
    
    size_t num_bytes = data.size() * sizeof(float);
    ndarray.CopyFromBytes(data.data(), num_bytes);
    return ndarray;
}

} // namespace runtime
} // namespace tvm

