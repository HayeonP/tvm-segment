#include <tvm/runtime/segment_runner.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/vm/bytecode.h>
#include <tvm/ffi/cast.h>
#include <regex>

#include <iostream>

#include <cassert>

namespace tvm {
namespace runtime {

SegmentRunner::SegmentRunner(const Module& exec, Device device){
  // HayeonP: TODO - Support multiple devices
  std::vector<Device> devices;
  devices.push_back(device);
  if(devices.back().device_type % kRPCSessMask != kDLCPU){
      devices.push_back(Device{kDLCPU, 0});
  }

  // HayeonP: TODO - Support manual memory configuration
  std::vector<memory::AllocatorType> alloc_types;
  auto default_alloc_type = memory::AllocatorType::kPooled;

  std::vector<AnyView> packed_args;
  for(auto dev : devices){
    packed_args.push_back(static_cast<int>(dev.device_type));
    packed_args.push_back(static_cast<int>(dev.device_id));
    packed_args.push_back(static_cast<int>(default_alloc_type));  // HayeonP: TODO - Support manual memory configuration
  }
	
  // (1) Call "vm_initialization"
  auto vm_exec = exec.as<tvm::runtime::vm::VMExecutable>();
  if (vm_exec == nullptr) {
    std::cerr << "Casting failed: exec is not a VMExecutable" << std::endl;
    exit(0);
  }
  
  vm_module_ = vm_exec->VMLoadExecutable();  
  
  /////////////////////

  ffi::Function init_func = vm_module_->GetFunction("vm_initialization");
  ffi::Any rv;
  
  init_func.CallPacked(ffi::PackedArgs(packed_args.data(), packed_args.size()), &rv);

  // (2) Init persistent frame
  ffi::Function init_persistent_frame_func = vm_module_->GetFunction("init_persistent_frame");
  init_persistent_frame_func();

  return;
}

std::string SegmentRunner::GetRuntimeSequence(){  
  ffi::Function get_runtime_sequence_func = vm_module_->GetFunction("get_runtime_sequence", false);

  ffi::Any rv = get_runtime_sequence_func();
  std::string runtime_sequence = rv.cast<std::string>();
  return runtime_sequence;
}

int SegmentRunner::Load(const std::string runtime_sequence){
  if(runtime_sequence.empty()){
    std::cout<<"ParsingError: Runtime sequence is empty"<<std::endl;
    return -1;
  }

  struct SegmentsInfoLine {
    std::string raw;
    std::string trimmed;
  };

  // Preprocessing (trimming, remove empty lines)
  std::istringstream iss(runtime_sequence);
  std::string line;  
  std::vector<SegmentsInfoLine> runtime_sequence_lines;

  while (std::getline(iss, line)) {
    std::string trimmed;
    if (line.empty()) continue;

    size_t trim_start = 0;
    while (trim_start < line.size() && std::isspace(line[trim_start])) {
        trim_start++;
    }
    size_t trim_end = line.size() - 1;
    while (trim_end > trim_start && std::isspace(line[trim_end])) {
        trim_end--;
    }
    trimmed = line.substr(trim_start, trim_end - trim_start + 1);

    if(!trimmed.empty()){
      runtime_sequence_lines.push_back({line, trimmed});
    }
  }  

  // Front-end validation
  if(runtime_sequence_lines.front().trimmed != "@seg"){
    std::cout<<"ParsingError: Does not start with @seg annotator"<<std::endl;
    return -1;
  }

  if(runtime_sequence_lines.back().trimmed != "@seg"){
    std::cout<<"ParsingError: Does not end with @seg annotator"<<std::endl;
    return -1;
  }

  // Parsing
  std::regex pattern(R"(pc\s*=\s*(\d+))");
  for(auto it = runtime_sequence_lines.begin(); it != runtime_sequence_lines.end(); it++){    
    std::string line = (*it).trimmed;
    
    if(line == "@seg"){
      segment_list_.push_back(std::vector<int64_t>());
      continue;
    }

    int pc;
    std::smatch matches;
    auto begin = std::sregex_iterator(line.begin(), line.end(), pattern);
    auto end = std::sregex_iterator();
    int count = std::distance(begin, end);

    if(count == 0){
        std::cout << "ParsingError: No program counter found in a line: \"" << (*it).raw << "\"" << std::endl;
        return -1;
    }

    if(count > 1){
        std::cout << "ParsingError: Multiple program counters in a line: \"" << (*it).raw << "\"" << std::endl;
        return -1;
    } 

    std::regex_search(line, matches, pattern);
    pc = std::stoi(matches[1].str());
    
    segment_list_.back().push_back(pc);
  }

  if(segment_list_.back().empty()){
    segment_list_.pop_back();
  }

  is_initialized_ = true;

  return 0;
}

// NOTE: 내부적으로는 frame에 0~n까지 input과 param들이 차례차례 들어가면 된다
void SegmentRunner::SetInput(std::vector<NDArray>& input){
  std::vector<AnyView> set_input_packed_args;
  for(auto input_v : input){
    set_input_packed_args.push_back(input_v);
  }
  ffi::Function set_input_func = vm_module_->GetFunction("set_input_to_persistent_frame", false);
  ffi::Any set_input_rv;
  set_input_func.CallPacked(ffi::PackedArgs(set_input_packed_args.data(), set_input_packed_args.size()), &set_input_rv);

  return;
}

void SegmentRunner::SetInputWithParams(std::vector<NDArray>& input, std::vector<NDArray>& params){
  std::vector<AnyView> set_input_packed_args;
  for(auto input_v : input){
    set_input_packed_args.push_back(input_v);
  }
  for(auto param_v : params){
    set_input_packed_args.push_back(param_v);
  }

  ffi::Function set_input_func = vm_module_->GetFunction("set_input_to_persistent_frame", false);
  ffi::Any set_input_rv;
  set_input_func.CallPacked(ffi::PackedArgs(set_input_packed_args.data(), set_input_packed_args.size()), &set_input_rv);

  return;
}

void SegmentRunner::Execute(const int segment_id){
  static int prev_segment_id = -1;  

  if(!is_initialized_){
    std::cout<<"SegmentRunnerError: Segments are not initialized"<<std::endl;
    exit(0);
    return;
  }

  if(segment_id > segment_list_.size() - 1){
    std::cout<<"InvalidSegmentIdError: Segment id is bigger than length (segment_id: "<<segment_id<<", length: "<<segment_list_.size()<<")"<<std::endl;
    exit(0);
    return;
  }

  if(segment_id > prev_segment_id + 1){
    std::cout<<"SegmentSkipWarning: Segments are skipped (segment_id: "<<segment_id<<", prev_segment_id: "<<prev_segment_id <<")"<<std::endl;
  }
  
  std::vector<AnyView> segment;
  for(auto v : segment_list_[segment_id]) segment.push_back(v);

  // Invoke segment
  ffi::Function invoke_segment_func = vm_module_->GetFunction("invoke_semgnet", false);
  ffi::Any invoke_segment_rv;
  invoke_segment_func.CallPacked(ffi::PackedArgs(segment.data(), segment.size()), &invoke_segment_rv);

  prev_segment_id = segment_id;

  return;
}

std::vector<NDArray> SegmentRunner::GetOutput(){
  ffi::Function get_output_func = vm_module_->GetFunction("get_output_from_persistent_frame", false);
  ffi::Any get_output_rv = get_output_func();
  
  std::vector<NDArray> output;
  if(get_output_rv.as<ffi::ArrayObj>()){
    auto output_array = Downcast<Array<NDArray>>(get_output_rv);
    for (auto& nd_array : output_array){
      output.push_back(nd_array);
    }
  }
  else{
    output.push_back(Downcast<NDArray>(get_output_rv));
  }

  return output;
}

size_t SegmentRunner::GetLength(){
	return segment_list_.size();
}


} // namespace runtime
} // namespace tvm