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

/*!
 * \file src/runtime/relax_vm/vm.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/nvtx.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/relax_vm/vm.h>

#include <optional>
#include <thread>
#include <sstream>
#include <regex>
#include <iterator>

namespace tvm {
namespace runtime {
namespace relax_vm {

//---------------------------------------------
// VM Closure object
//---------------------------------------------
TVM_REGISTER_OBJECT_TYPE(VMClosureObj);

VMClosure::VMClosure(String func_name, PackedFunc impl) {
  auto ptr = make_object<VMClosureObj>();
  ptr->func_name = func_name;
  ptr->impl = std::move(impl);
  data_ = std::move(ptr);
}

/*!
 * \brief Create another PackedFunc with last arguments already bound to last_args.
 * \param func The input func, can be a VMClosure or PackedFunc.
 * \param last_args The arguments to bound to in the end of the function.
 * \note The new function takes in arguments and append the last_args in the end.
 */
PackedFunc VMClosure::BindLastArgs(PackedFunc func, std::vector<TVMRetValue> last_args) {
  return PackedFunc([func, last_args](TVMArgs args, TVMRetValue* rv) {
    std::vector<TVMValue> values(args.size() + last_args.size());
    std::vector<int> tcodes(args.size() + last_args.size());
    runtime::TVMArgsSetter setter(values.data(), tcodes.data());
    std::copy(args.values, args.values + args.size(), values.data());
    std::copy(args.type_codes, args.type_codes + args.size(), tcodes.data());
    for (size_t i = 0; i < last_args.size(); ++i) {
      setter(i + args.size(), last_args[i]);
    }
    func.CallPacked(TVMArgs(values.data(), tcodes.data(), values.size()), rv);
  });
}

//-----------------------------------------------------------
// Utility functions.
//-----------------------------------------------------------
// Use the args after `starting_arg_idx` as a series of indices into `obj`,
// indexing into nested Array and returning the final indexed object.
ObjectRef IndexIntoNestedObject(ObjectRef obj, TVMArgs args, int starting_arg_idx) {
  for (int i = starting_arg_idx; i < args.size(); i++) {
    // the object must be an Array to be able to index into it
    if (!obj.as<ArrayNode>()) {
      LOG(FATAL) << "ValueError: Attempted to index into an object that is not an Array.";
    }
    int index = args[i];
    auto arr = Downcast<Array<ObjectRef>>(obj);
    // make sure the index is in bounds
    if (index >= static_cast<int>(arr.size())) {
      LOG(FATAL) << "IndexError: Invalid index (" << index << " >= " << arr.size() << ").";
    }
    obj = arr[index];
  }
  return obj;
}

NDArray ConvertNDArrayToDevice(NDArray src, const DLDevice& dev, Allocator* alloc) {
  if (src->device.device_type == dev.device_type && src->device.device_id == dev.device_id) {
    return src;
  } else {
    auto res = alloc->Empty(src.Shape(), src->dtype, dev);
    res.CopyFrom(src);
    return res;
  }
}

ObjectRef ConvertObjectToDevice(ObjectRef src, const Device& dev, Allocator* alloc) {
  if (src->IsInstance<NDArray::ContainerType>()) {
    return ConvertNDArrayToDevice(Downcast<NDArray>(src), dev, alloc);
  } else if (src->IsInstance<ArrayNode>()) {
    std::vector<ObjectRef> ret;
    auto arr = Downcast<Array<ObjectRef>>(src);
    for (size_t i = 0; i < arr.size(); i++) {
      ret.push_back(ConvertObjectToDevice(arr[i], dev, alloc));
    }
    return Array<ObjectRef>(ret.begin(), ret.end());
  } else {
    return src;
  }
}

TVMRetValue ConvertArgToDevice(TVMArgValue input, Device dev, Allocator* alloc) {
  // NOTE: NDArray::FromExternalDLTensor is not safe
  // in terms of memory-behavior.
  // To be extra careful, we copy DLTensor.
  // The developer can still explicitly allocate NDArray
  // in TVM Native API or NDArray::FromDLPack to regain zero copy behavior.
  TVMRetValue ret;

  if (input.type_code() == kTVMDLTensorHandle) {
    DLTensor* tensor = input;
    std::vector<int64_t> shape(tensor->shape, tensor->shape + tensor->ndim);
    auto dst = alloc->Empty(shape, tensor->dtype, dev);
    dst.CopyFrom(tensor);
    ret = dst;
  } else if (input.IsObjectRef<ObjectRef>()) {
    ret = ConvertObjectToDevice(input.operator ObjectRef(), dev, alloc);
  } else {
    ret = input;
  }
  return ret;
}

TVMRetValue ConvertRegToDevice(TVMRetValue input, Device dev, Allocator* alloc) {
  TVMRetValue ret;
  if (input.IsObjectRef<ObjectRef>()) {
    ret = ConvertObjectToDevice(input.operator ObjectRef(), dev, alloc);
  } else {
    ret = input;
  }
  return ret;
}

//-----------------------------------------------------------
// VM implementations.
//-----------------------------------------------------------

void VirtualMachineImpl::LoadExecutable(ObjectPtr<VMExecutable> exec) {
  this->exec_ = exec;
  this->imports_ = exec_->imports();
}

void VirtualMachineImpl::Init(const std::vector<Device>& devices,
                              const std::vector<AllocatorType>& alloc_types) {
  ICHECK_EQ(devices.size(), alloc_types.size());

  this->devices.reserve(devices.size());
  this->allocators.reserve(alloc_types.size());
  for (size_t i = 0; i < devices.size(); i++) {
    auto alloc = MemoryManager::GetOrCreateAllocator(devices[i], alloc_types[i]);
    this->devices.push_back(devices[i]);
    this->allocators.push_back(alloc);
  }
  // Setup constant sections.
  this->const_pool_.reserve(exec_->constants.size());
  for (const auto& constant : exec_->constants) {
    if (constant.type_code() != kTVMNDArrayHandle) {
      this->const_pool_.push_back(constant);
    } else {
      this->const_pool_.push_back(ConvertRegToDevice(constant, devices[0], allocators[0]));
    }
  }
  // Setup function sections.
  this->InitFuncPool();
}

VMFuncInfo VirtualMachineImpl::LookupVMFuncInfo(const std::string& func_name) {
  ICHECK(exec_) << "The executable is not created yet.";
  auto it = this->exec_->func_map.find(func_name);
  CHECK(it != this->exec_->func_map.end()) << "ValueError: Unknown function: " << func_name;

  return exec_->func_table[it->second];
}

RegType VirtualMachineImpl::LookupVMOutput(const std::string& func_name) {
  if (!outputs_.count(func_name)) {
    LOG(FATAL) << "ValueError: No output saved for call of \"" << func_name
               << "\"; use `invoke_stateful` to call it first.";
  }
  return outputs_[func_name];
}

void VirtualMachineImpl::SetInput(std::string func_name, bool with_param_module, TVMArgs args) {
  const auto& m = exec_->func_map;
  if (m.find(func_name) != m.end()) {
    Index gf_idx = m.at(func_name);
    const VMFuncInfo& vm_func = exec_->func_table[gf_idx];
    size_t params_num = vm_func.num_args;
    ICHECK_EQ(args.size(), params_num)
        << "The number of provided parameters doesn't match the number of arguments for";
    std::vector<RegType> func_args(params_num);
    for (int i = 0; i < args.size(); ++i) {
      if (with_param_module && i == args.size() - 1) {
        // call param func to get the arguments(usually corresponds to param pack.)
        func_args[i] = (args[i].operator Module()).GetFunction("get_params")();
      } else {
        func_args[i] = ConvertArgToDevice(args[i], devices[0], allocators[0]);
      }
    }
    inputs_[func_name] = func_args;
  } else {
    LOG(FATAL) << "ValueError: Unknown function: " << func_name;
  }
}

//------------------------------------------
// Closure handling
//------------------------------------------
void VirtualMachineImpl::InvokeClosurePacked(const ObjectRef& closure_or_packedfunc, TVMArgs args,
                                             TVMRetValue* rv) {
  // run packed call if it is a packed func.
  if (auto* packed = closure_or_packedfunc.as<PackedFunc::ContainerType>()) {
    packed->CallPacked(args, rv);
    return;
  }
  // run closure call.
  auto* clo = closure_or_packedfunc.as<VMClosureObj>();
  ICHECK(clo != nullptr) << "Function expects a closure or PackedFunc ";

  std::vector<TVMValue> values(args.size() + 1);
  std::vector<int> tcodes(args.size() + 1);
  runtime::TVMArgsSetter setter(values.data(), tcodes.data());
  // per convention, ctx ptr must be VirtualMachine* casted to void.
  // this and VirtualMachine* may or maynot be the same
  // do first cast to VirtualMachine* then to void*
  setter(0, static_cast<void*>(static_cast<VirtualMachine*>(this)));
  std::copy(args.values, args.values + args.size(), values.begin() + 1);
  std::copy(args.type_codes, args.type_codes + args.size(), tcodes.begin() + 1);
  {
    NVTXScopedRange scope("RelaxVM: " + clo->func_name);
    clo->impl.CallPacked(TVMArgs(values.data(), tcodes.data(), args.size() + 1), rv);
  }
}

// internal variant version of invoke closurepacked
RegType VirtualMachineImpl::InvokeClosureInternal(const ObjectRef& closure_or_packed,
                                                  const std::vector<RegType>& args) {
  RegType ret;
  auto* packed = closure_or_packed.as<PackedFunc::ContainerType>();
  auto* clo = closure_or_packed.as<VMClosureObj>();
  int clo_offset = clo != nullptr ? 1 : 0;
  std::vector<TVMValue> values(args.size() + clo_offset);
  std::vector<int> tcodes(args.size() + clo_offset);
  runtime::TVMArgsSetter setter(values.data(), tcodes.data());

  if (clo != nullptr) {
    setter(0, static_cast<void*>(static_cast<VirtualMachine*>(this)));
  }
  for (size_t i = 0; i < args.size(); ++i) {
    setter(i + clo_offset, args[i]);
  }

  if (packed != nullptr) {
    packed->CallPacked(TVMArgs(values.data(), tcodes.data(), values.size()), &ret);
  } else {
    ICHECK(clo != nullptr);
    clo->impl.CallPacked(TVMArgs(values.data(), tcodes.data(), values.size()), &ret);
  }
  return ret;
}

void VirtualMachineImpl::SaveClosure(const String& func_name, const String& save_name,
                                     bool include_return, TVMArgs args) {
  VMClosure clo = this->GetClosure(func_name);
  std::vector<RegType> inputs(args.size());
  for (int i = 0; i < args.size(); ++i) {
    inputs[i] = ConvertArgToDevice(args[i], this->devices[0], this->allocators[0]);
  }
  PackedFunc impl = VMClosure::BindLastArgs(clo->impl, inputs);
  if (!include_return) {
    impl = PackedFunc([impl](TVMArgs args, TVMRetValue* rv) {
      TVMRetValue temp;
      impl.CallPacked(args, &temp);
    });
  }
  saved_closures_[save_name] = VMClosure(save_name, impl);
}

Optional<VMClosure> VirtualMachineImpl::GetClosureInternal(const String& func_name,
                                                           bool allow_missing) {
  // look up saved closures.
  auto saved_it = saved_closures_.find(func_name);
  if (saved_it != saved_closures_.end()) {
    return saved_it->second;
  }
  auto it = exec_->func_map.find(func_name);
  if (it == exec_->func_map.end()) {
    if (allow_missing) return NullOpt;
    LOG(FATAL) << "ValueError: Unknown function: " << func_name;
  }

  Index gf_idx = it->second;
  const VMFuncInfo& finfo = exec_->func_table[gf_idx];

  if (finfo.kind == VMFuncInfo::FuncKind::kVMFunc) {
    // NOTE: should not capture strong ref to self and avoid cyclic ref.
    auto impl = PackedFunc([gf_idx](TVMArgs args, TVMRetValue* rv) {
      // Per convention, ctx ptr is a VirtualMachine*
      VirtualMachine* ctx_ptr = static_cast<VirtualMachine*>(args[0].operator void*());

      std::vector<RegType> inputs(args.size() - 1);
      for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = args[i + 1];
      }
      *rv = static_cast<VirtualMachineImpl*>(ctx_ptr)->InvokeBytecode(gf_idx, inputs);
    });
    return VMClosure(func_name, impl);
  } else {
    ICHECK(finfo.kind == VMFuncInfo::FuncKind::kVMTIRFunc)
        << "Cannot support closure with function kind " << static_cast<int>(finfo.kind);
    PackedFunc tir_func = GetFuncFromImports("__vmtir__" + finfo.name);
    ICHECK(tir_func != nullptr) << "Cannot find underlying compiled tir function of VMTIRFunc "
                                << finfo.name;
    auto impl = PackedFunc([this, finfo, tir_func](TVMArgs args, TVMRetValue* rv) {
      // Per convention, ctx ptr is a VirtualMachine*
      VirtualMachine* ctx_ptr = static_cast<VirtualMachine*>(args[0].operator void*());
      ICHECK(ctx_ptr == this);
      ICHECK_EQ(args.size() - 1, finfo.num_args)
          << "Function " << finfo.name << " expects " << finfo.num_args << " arguments";
      ICHECK_GE(finfo.register_file_size, finfo.num_args + 1);
      std::vector<TVMRetValue> reg_file(finfo.register_file_size);
      for (int64_t i = 0; i < finfo.num_args; ++i) {
        reg_file[i] = args[i + 1];
      }
      void* reg_anylist_handle = reg_file.data();
      void* const_anylist_handle = this->const_pool_.data();
      void* func_anylist_handle = this->func_pool_.data();
      tir_func(static_cast<void*>(ctx_ptr), reg_anylist_handle, const_anylist_handle,
               func_anylist_handle);
      // Return value always stored after inputs.
      *rv = reg_file[finfo.num_args];
    });
    return VMClosure(func_name, impl);
  }
}

//--------------------------------------------------------------------
// Instruction interpretations.
//--------------------------------------------------------------------
RegType VirtualMachineImpl::InvokeBytecode(Index gf_idx, const std::vector<RegType>& args) {
  const VMFuncInfo& gfunc = exec_->func_table[gf_idx];
  ICHECK(gfunc.kind == VMFuncInfo::FuncKind::kVMFunc);

  // Get the curr instr which might be a potential caller.
  Instruction curr_instr = exec_->GetInstruction(pc_);
  auto guard = PushFrame(this->pc_, gfunc);
  // Get new frame and set the caller info.
  VMFrame* curr_frame = frames_.back().get();
  if (curr_instr.op == Opcode::Call) {
    curr_frame->caller_return_register = curr_instr.dst;
  }

  // load arguments to the register file
  ICHECK_EQ(static_cast<size_t>(gfunc.num_args), args.size()) << "ValueError: Invoking function "
                                                              << gfunc.name << " expects "
                                                              << gfunc.num_args << " arguments" <<
      [&]() {
        std::stringstream ss;
        if (gfunc.param_names.size()) {
          ss << " (";
          for (size_t i = 0; i < gfunc.param_names.size(); i++) {
            if (i) {
              ss << ", ";
            }
            ss << gfunc.param_names[i];
          }
          ss << ")";
        }
        return ss.str();
      }() << ", but " << args.size() << " arguments were provided.";

  for (size_t i = 0; i < args.size(); ++i) {
    WriteRegister(frames_.back().get(), i, args[i]);
  }
  // set program counter
  pc_ = gfunc.start_instr;
  RunLoop();
  return return_value_;
}

void VirtualMachineImpl::InitFuncPool() {
  func_pool_.resize(exec_->func_table.size());

  for (size_t func_index = 0; func_index < exec_->func_table.size(); ++func_index) {
    const VMFuncInfo& info = exec_->func_table[func_index];
    if (info.kind == VMFuncInfo::FuncKind::kPackedFunc) {
      // only look through imports first
      PackedFunc func = GetFuncFromImports(info.name);
      if (!func.defined()) {
        const PackedFunc* p_func = Registry::Get(info.name);
        if (p_func != nullptr) func = *(p_func);
      }
      ICHECK(func.defined())
          << "Error: Cannot find PackedFunc " << info.name
          << " in either Relax VM kernel library, or in TVM runtime PackedFunc registry, or in "
             "global Relax functions of the VM executable";
      func_pool_[func_index] = func;

    } else {
      ICHECK(info.kind == VMFuncInfo::FuncKind::kVMFunc ||
             info.kind == VMFuncInfo::FuncKind::kVMTIRFunc);
      auto clo = this->GetClosure(info.name);
      func_pool_[func_index] = clo;
    }
  }
}

void VirtualMachineImpl::RunInstrCall(VMFrame* curr_frame, Instruction instr) {
  DLOG(INFO) << "\n  pc = " << pc_ << ", execute: " << GetFuncName(instr.func_idx);
  // std::cout<<  "\n  pc = " << pc_ << ", execute: " << GetFuncName(instr.func_idx) << std::endl;
  
  int args_begin_offset = instrument_ != nullptr ? 4 : 0;
  // Use the call arg stack from the current frame to increase reuse
  // and avoid re-allocation
  curr_frame->call_arg_values.resize(args_begin_offset + instr.num_args);
  curr_frame->call_arg_tcodes.resize(args_begin_offset + instr.num_args);

  // NOTE: no changes and resize to those vector ref(otherwise can leads to segfault)
  //       in the remainder part of the function.
  std::vector<TVMValue>& values = curr_frame->call_arg_values;
  std::vector<int>& tcodes = curr_frame->call_arg_tcodes;

  runtime::TVMArgsSetter setter(values.data(), tcodes.data());
  for (Index i = 0; i < instr.num_args; ++i) {
    Instruction::Arg arg = instr.args[i];
    int arg_index = args_begin_offset + i;
    switch (arg.kind()) {
      case Instruction::ArgKind::kRegister: {
        setter(arg_index, ReadRegister(curr_frame, arg.value()));
        break;
      }
      case Instruction::ArgKind::kImmediate: {
        setter(arg_index, arg.value());
        break;
      }
      case Instruction::ArgKind::kConstIdx: {
        setter(arg_index, this->const_pool_[arg.value()]);
        break;
      }
      case Instruction::ArgKind::kFuncIdx: {
        ICHECK_LT(static_cast<size_t>(arg.value()), this->func_pool_.size());
        setter(arg_index, this->func_pool_[arg.value()]);
        break;
      }
      default: {
        LOG(FATAL) << "ValueError: Unknown argument kind: " << int(arg.kind());
      }
    }
  }
  TVMArgs args(values.data() + args_begin_offset, tcodes.data() + args_begin_offset,
               instr.num_args);
  TVMRetValue ret;

  ICHECK_LT(static_cast<size_t>(instr.func_idx), this->func_pool_.size());

  if (instrument_ == nullptr) {
    this->InvokeClosurePacked(func_pool_[instr.func_idx], args, &ret);
  } else {
    // insert light-weight instrument callback
    setter(0, func_pool_[instr.func_idx]);
    setter(1, GetFuncName(instr.func_idx));
    setter(2, true);
    setter(3, nullptr);
    TVMRetValue rv;
    // store dtype to str since py callback cannot handle dtype atm.
    std::vector<std::unique_ptr<std::string>> temp_dtype;
    for (int i = 0; i < instr.num_args; ++i) {
      if (tcodes[i + args_begin_offset] == kTVMDataType) {
        std::string str_dtype = args[i];
        temp_dtype.emplace_back(std::make_unique<std::string>(str_dtype));
        setter(i + args_begin_offset, *temp_dtype.back());
      }
    }
    int ret_kind = static_cast<int>(VMInstrumentReturnKind::kNoOp);
    instrument_.CallPacked(TVMArgs(values.data(), tcodes.data(), values.size()), &rv);
    if (rv.type_code() == kDLInt) {
      ret_kind = rv;
    }

    if (ret_kind != static_cast<int>(VMInstrumentReturnKind::kSkipRun)) {
      this->InvokeClosurePacked(func_pool_[instr.func_idx], args, &ret);
      setter(2, false);
      setter(3, ret);
      instrument_.CallPacked(TVMArgs(values.data(), tcodes.data(), values.size()), &rv);
    }
  }


  // save the return value to the register
  // saving to special register is a NOP
  if (instr.dst < Instruction::kBeginSpecialReg) {
    WriteRegister(curr_frame, instr.dst, ret);
  }

  // increment pc
  pc_++;
}

void VirtualMachineImpl::RunLoop() {
  VMFrame* curr_frame = frames_.back().get();
  
  std::cout<<"Before Call"<<std::endl;
  while (true) {    
    ICHECK_LT(static_cast<size_t>(pc_), exec_->instr_offset.size()) << "run into invalid section";
    Instruction instr = exec_->GetInstruction(pc_);
    switch (instr.op) {
      case Opcode::Call: {
        this->RunInstrCall(curr_frame, instr);
        break;
      }
      case Opcode::Ret: {
        std::cout<<"RETURN"<<std::endl;
        // If we have hit the point from which we started
        // running, we should return to the caller breaking
        // the dispatch loop.
        return_value_ = ReadRegister(curr_frame, instr.result);
        RegName caller_return_register = curr_frame->caller_return_register;
        if (frames_.size() <= 1) {
          // directly return if no other frame in the call stack.
        } else {
          // return from a local call.
          // Update the current frame to be the parent frame.
          VMFrame* parent_frame = frames_.end()[-2].get();
          WriteRegister(parent_frame, caller_return_register, return_value_);
        }
        return;
      }
      case Opcode::Goto: {
        pc_ += instr.pc_offset;
        break;
      }
      case Opcode::If: {
        int64_t cond_val = ReadRegister(curr_frame, instr.cond);
        if (cond_val != 0) {
          pc_++;
        } else {
          ICHECK_GT(instr.false_offset, 1);
          pc_ += instr.false_offset;
        }
        break;
      }
    }
  }
}

ObjectPtr<VirtualMachine> VirtualMachine::Create() { return make_object<VirtualMachineImpl>(); }

//--------------------------------------------------------------------
// FFI related code
//--------------------------------------------------------------------

void VirtualMachineImpl::_Init(TVMArgs args, TVMRetValue* rv) {
  ICHECK_EQ(args.size() % 3, 0);
  std::vector<Device> devices;
  std::vector<AllocatorType> alloc_types;
  for (int i = 0; i < args.size(); i += 3) {
    int device_type = args[i];
    int device_id = args[i + 1];
    int alloc_type = args[i + 2];
    devices.push_back(Device{DLDeviceType(device_type), device_id});
    alloc_types.push_back(AllocatorType(alloc_type));

  }
  this->Init(devices, alloc_types);
}

void VirtualMachineImpl::_SaveClosure(TVMArgs args, TVMRetValue* rv) {
  ICHECK_GE(args.size(), 3);
  std::string func_name = args[0];
  this->SaveClosure(func_name, args[1], args[2],
                    TVMArgs(args.values + 3, args.type_codes + 3, args.size() - 3));
}

void VirtualMachineImpl::_InvokeClosure(TVMArgs args, TVMRetValue* rv) {
  this->InvokeClosurePacked(args[0], TVMArgs(args.values + 1, args.type_codes + 1, args.size() - 1),
                            rv);
}

void VirtualMachineImpl::_InvokeClosureStateful(std::string func_name) {
  const std::unordered_map<std::string, Index>& m = this->exec_->func_map;
  if (m.find(func_name) == m.end()) {
    LOG(FATAL) << "ValueError: Unknown function: " << func_name;
  }
  if (!inputs_.count(func_name)) {
    LOG(FATAL) << "ValueError: No inputs set for stateful call of " << func_name
               << "; use `set_input` first.";
    return;
  }
  outputs_[func_name] =
      this->InvokeClosureInternal(func_pool_[m.at(func_name)], inputs_[func_name]);
}

void VirtualMachineImpl::_SetInstrument(TVMArgs args, TVMRetValue* rv) {
  if (args[0].type_code() == kTVMPackedFuncHandle) {
    this->SetInstrument(args[0]);
  } else {
    String func_name = args[0];
    const PackedFunc* factory = Registry::Get(func_name);
    CHECK(factory) << "Cannot find factory " << func_name;
    TVMRetValue rv;
    factory->CallPacked(TVMArgs(args.values + 1, args.type_codes + 1, args.num_args - 1), &rv);
    this->SetInstrument(rv);
  }
}

void VirtualMachineImpl::_GetOutputArity(TVMArgs args, TVMRetValue* rv) {
  std::string func_name = args[0];
  RegType out = LookupVMOutput(func_name);
  ObjectRef obj = IndexIntoNestedObject(out.AsObjectRef<ObjectRef>(), args, 1);
  if (const auto* arr = obj.as<ArrayNode>()) {
    *rv = static_cast<int>(arr->size());
  } else {
    *rv = -1;
  }
}

void VirtualMachineImpl::_GetOutput(TVMArgs args, TVMRetValue* rv) {
  std::string func_name = args[0];
  RegType out = LookupVMOutput(func_name);
  ObjectRef obj = IndexIntoNestedObject(out.AsObjectRef<ObjectRef>(), args, 1);
  if (obj.as<ArrayNode>()) {
    LOG(FATAL) << "ValueError: `get_output` cannot return a tuple for RPC compatibility. "
                  "Please specify another index argument.";
    return;
  }
  *rv = obj;
}

void VirtualMachineImpl::_SetInputWithoutParamModule(TVMArgs args, TVMRetValue* rv) {
  std::string func_name = args[0];
  this->SetInput(func_name, false,
                 TVMArgs(args.values + 1, args.type_codes + 1, args.num_args - 1));
}

void VirtualMachineImpl::_SetInputWithParamModule(TVMArgs args, TVMRetValue* rv) {
  std::string func_name = args[0];
  this->SetInput(func_name, true, TVMArgs(args.values + 1, args.type_codes + 1, args.num_args - 1));
}

int VirtualMachineImpl::_GetFunctionArity(std::string func_name) {
  const VMFuncInfo& vm_func = LookupVMFuncInfo(func_name);
  return vm_func.param_names.size();
}

std::string VirtualMachineImpl::_GetFunctionParamName(std::string func_name, int index) {
  const VMFuncInfo& vm_func = LookupVMFuncInfo(func_name);
  if (static_cast<size_t>(index) >= vm_func.param_names.size()) {
    LOG(FATAL) << "ValueError: Invalid index for " << func_name << " (" << index << " out of "
               << vm_func.param_names.size() << ")";
  }
  return vm_func.param_names[index];
}

// HayeonP
String VirtualMachineImpl::_SegmentRunnerGetSkeleton(){
  return this->SegmentRunnerGetSkeleton();
}

String VirtualMachineImpl::SegmentRunnerGetSkeleton(){
  std::string output_str;
  
  auto it = exec_->func_map.find("main");
  if (it == exec_->func_map.end()) {
    LOG(FATAL) << "ValueError: Cannot find main function";    
    return output_str;
  }

  Index gf_idx = it->second;
  const VMFuncInfo& gfunc = exec_->func_table[gf_idx];
  auto guard = PushFrame(this->pc_, gfunc);
  VMFrame* curr_frame = frames_.back().get();

  pc_ = gfunc.start_instr;

  bool is_finished = false;
  while (!is_finished) {
    Instruction instr = exec_->GetInstruction(pc_);
    switch (instr.op) {
      case Opcode::Call: {
        std::ostringstream oss;
        oss << "pc = " << pc_ << ", execute: " << GetFuncName(instr.func_idx) << std::endl;
        output_str += oss.str();
        pc_++;
        break;
      }
      case Opcode::Ret: {
        is_finished = true;
        break;
      }
      case Opcode::Goto: {
        pc_ += instr.pc_offset;
        break;
      }
      case Opcode::If: {
        int64_t cond_val = ReadRegister(curr_frame, instr.cond);
        if (cond_val != 0) {
          pc_++;
        } else {
          ICHECK_GT(instr.false_offset, 1);
          pc_ += instr.false_offset;
        }
        break;
      }
    }
  }

  return output_str;
}

// HayeonP
int VirtualMachineImpl::_SegmentRunnerLoad(std::string segments_info){
  return this->SegmentRunnerLoad(segments_info);
}

int VirtualMachineImpl::SegmentRunnerLoad(std::string segments_info){
  if(segments_info.empty()){
    std::cout<<"SegmentsInfoParsingError: segments_info is empty"<<std::endl;
    return -1;
  }

  struct SegmentsInfoLine {
    std::string raw;
    std::string trimmed;
  };

  // Preprocessing (trimming, remove empty lines)
  std::istringstream iss(segments_info);
  std::string line;  
  std::vector<SegmentsInfoLine> segements_info_lines;
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
      segements_info_lines.push_back({line, trimmed});
    }
  }

  // Front-end validation
  if(segements_info_lines.front().trimmed != "@seg"){
    std::cout<<"SegmentsInfoParsingError: Does not start with @seg annotator"<<std::endl;
    return -1;
  }

  if(segements_info_lines.back().trimmed != "@seg"){
    std::cout<<"SegmentsInfoParsingError: Does not end with @seg annotator"<<std::endl;
    return -1;
  }

  // Parsing
  std::regex pattern(R"(pc\s*=\s*(\d+))");
  for(auto it = segements_info_lines.begin(); it != segements_info_lines.end(); it++){    
    std::string line = (*it).trimmed;
    if(line == "@seg"){
      per_segment_pc_list_.push_back(std::vector<int>());
      continue;
    }

    int pc;
    std::smatch matches;
    auto begin = std::sregex_iterator(line.begin(), line.end(), pattern);
    auto end = std::sregex_iterator();
    int count = std::distance(begin, end);

    if(count == 0){
        std::cout << "SegmentsInfoParsingError: No program counter found in a line: \"" << (*it).raw << "\"" << std::endl;
        return -1;
    }

    if(count > 1){
        std::cout << "SegmentsInfoParsingError: Multiple program counters in a line: \"" << (*it).raw << "\"" << std::endl;
        return -1;
    } 

    std::regex_search(line, matches, pattern);
    pc = std::stoi(matches[1].str());
    
    per_segment_pc_list_.back().push_back(pc);
  }

  if(per_segment_pc_list_.back().empty()){
    per_segment_pc_list_.pop_back();
  }

  // for(auto it1 = per_segment_pc_list_.begin(); it1 != per_segment_pc_list_.end(); it1++){
  //   auto segment_pc_list = *it1;
  //   std::cout<<"SEGMENT"<<std::endl;
  //   for(auto it2 = segment_pc_list.begin(); it2 != segment_pc_list.end(); it2++){
  //     std::cout<<*it2<<std::endl;
  //   }
  // }

  
  are_segments_initialized_ = true;

  auto func_main_it = exec_->func_map.find("main");
  if (func_main_it == exec_->func_map.end()) {
    LOG(FATAL) << "ValueError: Cannot find main function";    
    return -1;
  }

  auto main_it = exec_->func_map.find("main");
  Index main_func_idx = main_it->second;
  const VMFuncInfo& main_func = exec_->func_table[main_func_idx];
  pc_ = main_func.start_instr;

  segments_frame_ = std::make_unique<VMFrame>(main_func_idx, main_func.register_file_size);
  VMFrame* curr_frame = segments_frame_.get();

  return per_segment_pc_list_.size();
}


void VirtualMachineImpl::_SegmentRunnerSetInput(TVMArgs args, TVMRetValue* rv){
  if(!segments_frame_){
    std::cout<<"InvalidSegmentsFrame: segments_frame doesn't exist"<<std::endl;
    *rv = -1;
  }
  std::vector<RegType> input(args.size() - 1);
  for (size_t i = 0; i < input.size(); ++i) {
        input[i] = args[i + 1];
  }

  VMFrame* curr_frame = segments_frame_.get();
  for(size_t i = 0; i < input.size(); ++i) {
    WriteRegister(curr_frame, i, input[i]);
  }

  *rv = 0;

  return;
}

// HayeonP
int VirtualMachineImpl::SegmentRunnerSetInput(NDArray& input, std::vector<NDArray>& params){

  if(!segments_frame_){
    std::cout<<"InvalidSegmentsFrame: segments_frame doesn't exist"<<std::endl;
    return -1;
  }
  
  VMFrame* curr_frame = segments_frame_.get();
  // Input
  RegType input_reg;
  input_reg = input;
  WriteRegister(curr_frame, 0, input_reg);

  // Params
  for(size_t i = 0; i < params.size(); ++i) {
    RegType param_reg;
    param_reg = params[i];
    WriteRegister(curr_frame, i+1, param_reg);
  }

  return 0;
}

// HayeonP
void VirtualMachineImpl::_SegmentRunnerGetOutput(TVMArgs args, TVMRetValue* rv){
  Instruction instr = exec_->GetInstruction(pc_);

  if(instr.op != Opcode::Ret) {
    std::cout<<"OutputError: Inference isn't finished"<<std::endl;
  }
  
  // If we have hit the point from which we started
  // running, we should return to the caller breaking
  // the dispatch loop.
  VMFrame* curr_frame = segments_frame_.get();
  return_value_ = ReadRegister(curr_frame, instr.result);

  RegName caller_return_register = curr_frame->caller_return_register;
  if (frames_.size() <= 1) {
    // directly return if no other frame in the call stack.
  } else {
    // return from a local call.
    // Update the current frame to be the parent frame.
    VMFrame* parent_frame = frames_.end()[-2].get();
    WriteRegister(parent_frame, caller_return_register, return_value_);
  }

  *rv = return_value_;

  return;
}

std::vector<NDArray> VirtualMachineImpl::SegmentRunnerGetOutput(){
  Instruction instr = exec_->GetInstruction(pc_);

  if(instr.op != Opcode::Ret) {
    std::cout<<"OutputError: Inference isn't finished"<<std::endl;
  }
  
  // If we have hit the point from which we started
  // running, we should return to the caller breaking
  // the dispatch loop.
  VMFrame* curr_frame = segments_frame_.get();
  return_value_ = ReadRegister(curr_frame, instr.result);

  RegName caller_return_register = curr_frame->caller_return_register;
  if (frames_.size() <= 1) {
    // directly return if no other frame in the call stack.
  } else {
    std::cout<<"Debug: Write the output to a register"<<std::endl;
    // return from a local call.
    // Update the current frame to be the parent frame.
    VMFrame* parent_frame = frames_.end()[-2].get();
    WriteRegister(parent_frame, caller_return_register, return_value_);
  }

  tvm::runtime::ObjectRef obj_ref = return_value_;
  std::vector<NDArray> output_list;

  if(auto array = obj_ref.as<ArrayNode>()){
    for(size_t i = 0; i < array->size(); ++i){
      if(auto node = array->at(i).as<NDArray>()){
        output_list.push_back(node.value());
      }\      
    }
  }

  if(auto node = obj_ref.as<NDArray>()){
    output_list.push_back(node.value());
  }
  
  return output_list;
}

int VirtualMachineImpl::_SegmentRunnerRun(const int segment_id){
  return this->SegmentRunnerRun(segment_id);
}

// HayeonP
int VirtualMachineImpl::SegmentRunnerRun(const int segment_id) {
  if(!are_segments_initialized_){
    std::cout<<"RunSegmentError: Segments are not initialized"<<std::endl;
    return -1;
  }

  VMFrame* curr_frame = segments_frame_.get();

  static int prev_segment_id = -1;
  int segment_length = per_segment_pc_list_.size();
  
  if(segment_id > segment_length - 1){
    std::cout<<"InvalidSegmentIdError: Segment id is bigger than length (segment_id: "<<segment_id<<", length: "<<segment_length<<")"<<std::endl;
    return -1;
  }

  if(segment_id > prev_segment_id + 1){
    std::cout<<"SegmentSkipWarning: Segment is skipped (segment_id: "<<segment_id<<", prev_segment_id: "<<prev_segment_id <<")"<<std::endl;
  }

  
  for(auto it = per_segment_pc_list_[segment_id].begin(); it != per_segment_pc_list_[segment_id].end(); it++){
    pc_ = *it;
    ICHECK_LT(static_cast<size_t>(pc_), exec_->instr_offset.size()) << "run into invalid section";
    Instruction instr = exec_->GetInstruction(pc_);

    switch (instr.op) {
      case Opcode::Call: {
        this->RunInstrCall(curr_frame, instr);
        break;
      }
      case Opcode::Ret: {
        std::cout<<"RunSegmentError: Reached a return before execution was completed"<<std::endl;
        
        return -1;
      }
      case Opcode::Goto: {
        pc_ += instr.pc_offset;
        break;
      }
      case Opcode::If: {
        int64_t cond_val = ReadRegister(curr_frame, instr.cond);
        if (cond_val != 0) {
          pc_++;
        } else {
          ICHECK_GT(instr.false_offset, 1);
          pc_ += instr.false_offset;
        }
        break;
      }
    }
  }

  if(segment_id == segment_length - 1){
    prev_segment_id = -1;
  }

  prev_segment_id = segment_id;

  return segment_id;
}

PackedFunc VirtualMachineImpl::_LookupFunction(const String& name) {
  if (Optional<VMClosure> opt = this->GetClosureInternal(name, true)) {
    return PackedFunc(
        [clo = opt.value(), _self = GetRef<Module>(this)](TVMArgs args, TVMRetValue* rv) -> void {
          auto* self = const_cast<VirtualMachineImpl*>(_self.as<VirtualMachineImpl>());
          ICHECK(self);
          self->InvokeClosurePacked(clo, args, rv);
        });
  }
  return PackedFunc(nullptr);
}

//----------------------------------------------------------------
// Profiler can be optionally disabled via a macro to reduce dep.
//----------------------------------------------------------------
#if TVM_RELAX_VM_ENABLE_PROFILER

/*!
 * \brief An extension of VirtualMachineImpl to support per-op profiling
 * It overrides RunInstrCall to add instrumentations around it.
 */
class VirtualMachineProfiler : public VirtualMachineImpl {
 public:
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) override {
    if (name == "profile") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        std::string f_name = args[0];
        VMClosure clo = this->GetClosure(f_name);

        std::vector<Device> devices;
        for (auto dev : this->devices) {
          if (dev.device_type > 0) {
            devices.push_back(dev);
          }
        }

        prof_ = profiling::Profiler(devices, {}, {{String("Executor"), String("VM")}});

        auto inputs = GetInputsFor(f_name);

        bool clear_inputs = false;
        if (inputs.size() == 0) {
          ICHECK(args.num_args > 1) << "No input is provided";
          TVMArgs f_args(args.values + 1, args.type_codes + 1, args.num_args - 1);
          SetInput(f_name, false, TVMArgs(args.values + 1, args.type_codes + 1, args.num_args - 1));
          inputs = GetInputsFor(f_name);
          clear_inputs = true;
        } else {
          ICHECK_EQ(args.num_args, 1) << "Inputs are already provided by set_input.";
        }

        // warmup
        this->InvokeClosureInternal(clo, inputs);

        prof_->Start();
        this->InvokeClosureInternal(clo, inputs);
        prof_->Stop();

        // Return the report as json, since profiling::Report object is not supported by RPC
        std::string report_json = prof_->Report()->AsJSON();
        *rv = report_json;

        prof_ = std::nullopt;  // releases hardware counters
        if (clear_inputs) {
          // SetInput modifies the internal states of VM. Undo the change after profiling.
          ClearInputsFor(f_name);
        }
      });
    } else {
      return VirtualMachineImpl::GetFunction(name, sptr_to_self);
    }
  }

 protected:
  void RunInstrCall(VMFrame* curr_frame, Instruction inst) override {
    bool profiling = false;
    if (prof_ && prof_->IsRunning()) {
      auto f_name = GetFuncName(inst.func_idx);
      std::optional<Device> dev;
      std::vector<NDArray> arrs;

      auto f_check_ndarray_arg = [&dev, &arrs](const RegType& arg) {
        if (arg.type_code() == kTVMNDArrayHandle) {
          NDArray arr = arg;
          dev = arr->device;
          arrs.push_back(arr);
        }
      };

      for (Index i = 0; i < inst.num_args; ++i) {
        Instruction::Arg arg = inst.args[i];
        if (arg.kind() == Instruction::ArgKind::kRegister) {
          auto reg = ReadRegister(curr_frame, arg.value());
          f_check_ndarray_arg(reg);
        } else if (arg.kind() == Instruction::ArgKind::kConstIdx) {
          const auto& const_val = this->const_pool_[arg.value()];
          f_check_ndarray_arg(const_val);
        }
      }

      std::unordered_map<std::string, ObjectRef> metrics;
      metrics["Argument Shapes"] = profiling::ShapeString(arrs);

      // If a suitable device is found, enable profiling.
      if (dev) {
        profiling = true;
        prof_->StartCall(f_name, *dev, metrics);
      }
    }

    VirtualMachineImpl::RunInstrCall(curr_frame, inst);

    if (profiling) {
      prof_->StopCall();
    }
  }

 private:
  std::optional<profiling::Profiler> prof_;
};

ObjectPtr<VirtualMachine> VirtualMachine::CreateProfiler() {
  return make_object<VirtualMachineProfiler>();
}

#else
ObjectPtr<VirtualMachine> VirtualMachine::CreateProfiler() {
  LOG(FATAL) << "Profiler support is disabled";
  return nullptr;
}
#endif  // TVM_RELAX_VM_ENABLE_PROFILER
}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
