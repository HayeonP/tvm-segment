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
 * \file tvm/runtime/relax_vm/vm.h
 */
#ifndef TVM_RUNTIME_RELAX_VM_VM_H_
#define TVM_RUNTIME_RELAX_VM_VM_H_

#ifndef TVM_RELAX_VM_ENABLE_PROFILER
#define TVM_RELAX_VM_ENABLE_PROFILER 1
#endif

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../memory/memory_manager.h"
#include "./bytecode.h"
#include "./executable.h"

namespace tvm {
namespace runtime {

using memory::Allocator;
using memory::AllocatorType;
using memory::MemoryManager;
using memory::Storage;
using memory::StorageObj;

namespace relax_vm {

/*!
 * \brief Possible instrument actions.
 */
enum class VMInstrumentReturnKind : int {
  /*! \brief Running as normal. */
  kNoOp = 0,
  /*! \brief Skip the following run, only valid in before. */
  kSkipRun = 1,
};

/*!
 * \brief An object representing a vm closure.
 */
class VMClosureObj : public Object {
 public:
  /*!
   * \brief The function name. The function could be any
   * function object that is compatible to the VM runtime.
   */
  String func_name;

  /*!
   * \brief The implementation of the Closure.
   * \note This function takes context pointer(VirtualMachine*)
   *       as the first argument. The rest of arguments follows
   *       the same arguments as the normal function call.
   */
  PackedFunc impl;

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.vm.Closure";
  TVM_DECLARE_FINAL_OBJECT_INFO(VMClosureObj, Object);
};

/*! \brief reference to closure. */
class VMClosure : public ObjectRef {
 public:
  VMClosure(String func_name, PackedFunc impl);
  TVM_DEFINE_OBJECT_REF_METHODS(VMClosure, ObjectRef, VMClosureObj);

  /*!
   * \brief Create another PackedFunc with last arguments already bound to last_args.
   *
   * This is a helper function to create captured closures.
   * \param func The input func, can be a VMClosure or PackedFunc.
   * \param last_args The arguments to bound to in the end of the function.
   * \note The new function takes in arguments and append the last_args in the end.
   */
  static PackedFunc BindLastArgs(PackedFunc func, std::vector<TVMRetValue> last_args);
};

/*!
 * \brief Represent a VM extension.
 * A VM extension allows the user to extend the VM with target specific functionalities.
 * The VM holds the reference of the extensions to ensure the extensions have the same lifetime
 * as the VM.
 *
 * This is the base class for all VM extensions and should not be used directly.
 */
class VMExtensionNode : public Object {
 protected:
  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "runtime.VMExtension";
  TVM_DECLARE_BASE_OBJECT_INFO(VMExtensionNode, Object);
};

/*! \brief Managed reference to VM extension. */
class VMExtension : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(VMExtension, ObjectRef, VMExtensionNode);
};

/*!
 * \brief The virtual machine.
 *
 * The virtual machine contains all the current execution state,
 * as well as the executable.
 *
 * The goal is to have a single self-contained object,
 * enabling one to easily pass around VMs, execute them on
 * multiple threads, or serialize them to disk or over the
 * wire.
 */
class VirtualMachine : public runtime::ModuleNode {
 public:
  /*!
   * \brief Initialize the virtual machine for a set of devices.
   * \param devices The set of TVM devices.
   * \param alloc_types The allocator types for each device.
   */
  virtual void Init(const std::vector<Device>& devices,
                    const std::vector<AllocatorType>& alloc_types) = 0;
  /*!
   * \brief Load the executable for the virtual machine.
   * \param exec The executable.
   */
  virtual void LoadExecutable(ObjectPtr<VMExecutable> exec) = 0;
  /*!
   * \brief Get global function in the VM.
   * \param func_name The name of the function.
   * \return The closure
   */
  virtual VMClosure GetClosure(const String& func_name) = 0;
  /*!
   * \brief Invoke closure or packed function using PackedFunc convention.
   * \param closure_or_packedfunc A VM closure or a packed_func.
   * \param args The input arguments.
   * \param rv The return value.
   */
  virtual void InvokeClosurePacked(const ObjectRef& closure_or_packedfunc, TVMArgs args,
                                   TVMRetValue* rv) = 0;
  /*!
   * \brief Set an instrumentation function.
   *
   * If instrument is present, the function will be called
   * before/after each Call instruction.
   *
   * bool instrument(func, func_symbol, before_run, args...)
   *
   * - func: Union[VMClosure, PackedFunc], the function object.
   * - func_symbol: string, the symbol of the function.
   * - before_run: bool, whether it is before or after call.
   * - ret_value: Only valid in after run, otherwise it is null.
   * - args: the arguments being passed to call.
   *
   * instrument can return an int which corresponds to the action value.
   * \sa VMInstrumentAction
   *
   * \param instrument The instrument function.
   */
  virtual void SetInstrument(PackedFunc instrument) = 0;

  /*!
   * \brief Get or create a VM extension. Once created, the extension will be stored in the VM
   * and held until the VM is destructed.
   *
   * \tparam T The type of the extension
   * \return The extension instance
   */
  template <typename T, typename = std::enable_if_t<std::is_base_of<VMExtension, T>::value>>
  T GetOrCreateExtension() {
    using ContainerType = typename T::ContainerType;
    uint32_t key = ContainerType::RuntimeTypeIndex();
    if (auto it = extensions.find(key); it != extensions.end()) {
      return Downcast<T>((*it).second);
    }
    auto [it, _] = extensions.emplace(key, T::Create());
    return Downcast<T>((*it).second);
  }

  /*!
   * \brief Create a specific instance of VM.
   * \return Created VM
   */
  static ObjectPtr<VirtualMachine> Create();
  /*!
   * \brief Create an instance of VM with the profiling feature enabled.
   * \return Created VM
   */
  static ObjectPtr<VirtualMachine> CreateProfiler();
  /*!
   * \brief Helper function for vm closure functions to get the context ptr
   * \param arg The argument value.
   */
  static VirtualMachine* GetContextPtr(TVMArgValue arg) {
    return static_cast<VirtualMachine*>(arg.operator void*());
  }

  ~VirtualMachine() {}

  //--------------------------------------------------------------------------
  // The following section contains states that other builtin can depend on
  //--------------------------------------------------------------------------
  /*! \brief The memory allocators. */
  std::vector<Allocator*> allocators;
  /*! \brief Runtime physical device list. */
  std::vector<Device> devices;
  /*! \brief The VM extensions. Mapping from the type index of the extension to the extension
   * instance. */
  std::unordered_map<uint32_t, VMExtension> extensions;
};


//-----------------------------------------------------------
// VM implementations.
//-----------------------------------------------------------
/*!
 * \brief The register type.
 */
using RegType = TVMRetValue;

/*!
 * \brief A representation of a stack frame.
 *
 * A stack frame is a record containing the information needed
 * to restore the caller's virtual machine state after returning
 * from a function call.
 */
struct VMFrame {
  /*! \brief The return program counter. */
  Index return_pc;
  /*! \brief Statically allocated space for objects */
  std::vector<RegType> register_file;
  /*! \brief Register in caller's frame to put return value */
  RegName caller_return_register;
  // The following fields are used for PackedFunc call within
  // a single function scope. The space is reused across multiple
  // packed func calls to increase cache locality and avoid re-allocation
  /*! \brief Temporary argument value stack for packed func call. */
  std::vector<TVMValue> call_arg_values;
  /*! \brief Temporary argument tcode stack for packed func call. */
  std::vector<int> call_arg_tcodes;

  VMFrame(Index pc, Index register_file_size)
      : return_pc(pc), register_file(register_file_size), caller_return_register(0) {}

  void Clear() {
    this->caller_return_register = 0;
    this->call_arg_values.clear();
    this->call_arg_tcodes.clear();
    for (RegType& reg : register_file) {
      reg = nullptr;
    }
  }

  void ResetForRecycle(Index pc, Index register_file_size) {
    this->return_pc = pc;
    this->register_file.resize(register_file_size);
  }
};


class VirtualMachineImpl : public VirtualMachine {
 public:
  //---------------------------------------------------
  // Public facing functions overloading
  //---------------------------------------------------
  void LoadExecutable(ObjectPtr<VMExecutable> exec) final;
  void Init(const std::vector<Device>& devices,
            const std::vector<AllocatorType>& alloc_types) final;
  VMClosure GetClosure(const String& func_name) final {
    return this->GetClosureInternal(func_name, false).value();
  }
  void InvokeClosurePacked(const ObjectRef& closure_or_packedfunc, TVMArgs args,
                           TVMRetValue* rv) final;
  void SetInstrument(PackedFunc instrument) final { this->instrument_ = instrument; }

  //---------------------------------------------------
  // Functions in the vtable of Module
  //---------------------------------------------------
  void _Init(TVMArgs args, TVMRetValue* rv);
  void _SaveClosure(TVMArgs args, TVMRetValue* rv);
  void _InvokeClosure(TVMArgs args, TVMRetValue* rv);
  void _InvokeClosureStateful(std::string func_name);
  void _SetInstrument(TVMArgs args, TVMRetValue* rv);
  void _GetOutputArity(TVMArgs args, TVMRetValue* rv);
  void _GetOutput(TVMArgs args, TVMRetValue* rv);
  void _SetInputWithoutParamModule(TVMArgs args, TVMRetValue* rv);
  void _SetInputWithParamModule(TVMArgs args, TVMRetValue* rv);
  int _GetFunctionArity(std::string func_name);
  std::string _GetFunctionParamName(std::string func_name, int index);
  
  PackedFunc _LookupFunction(const String& name);
  
  String _SegmentRunnerGetSkeleton(); // HayeonP
  String SegmentRunnerGetSkeleton();
  int _SegmentRunnerLoad(std::string segments_info_str); // HayeonP
  int SegmentRunnerLoad(std::string segments_info_str);
  void _SegmentRunnerSetInput(TVMArgs args, TVMRetValue* rv); // HayeonP
  int SegmentRunnerSetInput(NDArray& input, std::vector<NDArray>& params);
  int _SegmentRunnerRun(const int segment_id); // HayeonP
  int SegmentRunnerRun(const int segment_id); // HayeonP
  void _SegmentRunnerGetOutput(TVMArgs args, TVMRetValue* rv); // HayeonP
  std::vector<NDArray> SegmentRunnerGetOutput(); // HayeonP

  TVM_MODULE_VTABLE_BEGIN("relax.VirtualMachine");
  TVM_MODULE_VTABLE_ENTRY_PACKED("vm_initialization", &VirtualMachineImpl::_Init);
  TVM_MODULE_VTABLE_ENTRY_PACKED("save_function", &VirtualMachineImpl::_SaveClosure);
  TVM_MODULE_VTABLE_ENTRY_PACKED("invoke_closure", &VirtualMachineImpl::_InvokeClosure);
  TVM_MODULE_VTABLE_ENTRY("invoke_stateful", &VirtualMachineImpl::_InvokeClosureStateful);
  TVM_MODULE_VTABLE_ENTRY_PACKED("set_instrument", &VirtualMachineImpl::_SetInstrument);
  TVM_MODULE_VTABLE_ENTRY_PACKED("get_output_arity", &VirtualMachineImpl::_GetOutputArity);
  TVM_MODULE_VTABLE_ENTRY_PACKED("get_output", &VirtualMachineImpl::_GetOutput);
  TVM_MODULE_VTABLE_ENTRY_PACKED("set_input", &VirtualMachineImpl::_SetInputWithoutParamModule);
  TVM_MODULE_VTABLE_ENTRY_PACKED("set_input_with_param_module",
                                 &VirtualMachineImpl::_SetInputWithParamModule);
  TVM_MODULE_VTABLE_ENTRY("get_function_arity", &VirtualMachineImpl::_GetFunctionArity);
  TVM_MODULE_VTABLE_ENTRY("get_function_param_name", &VirtualMachineImpl::_GetFunctionParamName);
  TVM_MODULE_VTABLE_ENTRY("segment_runner.get_skeleton", &VirtualMachineImpl::_SegmentRunnerGetSkeleton); // HayeonP
  TVM_MODULE_VTABLE_ENTRY("segment_runner.load", &VirtualMachineImpl::_SegmentRunnerLoad); // HayeonP
  TVM_MODULE_VTABLE_ENTRY_PACKED("segment_runner.set_input", &VirtualMachineImpl::_SegmentRunnerSetInput); // HayeonP
  TVM_MODULE_VTABLE_ENTRY("segment_runner.run", &VirtualMachineImpl::_SegmentRunnerRun); // HayeonP
  TVM_MODULE_VTABLE_ENTRY_PACKED("segment_runner.get_output", &VirtualMachineImpl::_SegmentRunnerGetOutput); // HayeonP
  TVM_MODULE_VTABLE_END_WITH_DEFAULT(&VirtualMachineImpl::_LookupFunction);

  //--------------------------------------------------
  // Additional support arguments functions for VM
  //--------------------------------------------------
  /*!
   * \brief Internal implementation of GetClosure which also allow none.
   * \param func_name The name of the function.
   * \param allow_missing Whether none is allowed.
   * \return The result
   */
  Optional<VMClosure> GetClosureInternal(const String& func_name, bool allow_missing);

  /*!
   * \brief Set inputs to a function.
   * \param func_name The function name.
   * \param args args[offset:] are arguments to the function. If the arguments are not of the
   * correct device for the function, they will be copied to the device.
   * \param with_param_module If set to true, the last argument will be a module and can be invoked
   *        to get the argument, this is mainly used for debugging purposes and setting composite
   * objects. \note This interface works when using VM over RPC by internally converting NDArray in
   * the arguments to DLTensor, which is supported in RPC where remote could only have a minimal C
   * runtime.
   */
  void SetInput(std::string func_name, bool with_param_module, TVMArgs args);

  /*!
   * \brief Look up whether the VM has a function by the given name.
   * \param func_name the function's name
   * \return The function, if it exists. Logs a fatal error if not.
   */
  VMFuncInfo LookupVMFuncInfo(const std::string& func_name);

  /*!
   * \brief Look up whether the VM has outputs for the given function.
   * \param func_name the function's name
   * \return The output, if it exists. Logs a fatal error if not.
   */
  RegType LookupVMOutput(const std::string& func_name);

  /*!
   * \brief Fully bind the argument of a global function and save it in the env.
   * \param func_name The global function name to be saved.
   * \param save_name The saved name of the function.
   * \param include_return Whether forward the return value, set it to false allows
   *        us to ignore forwarding return value, which can be helpful to do benchmarking
   *        in RPC environment when return value is complicated Array.
   *
   * \param args The arguments to bound to the function.
   * \note This function is used by RPC server to help benchmarking.
   */
  void SaveClosure(const String& func_name, const String& save_name, bool include_return,
                   TVMArgs args);
  /*!
   * \brief Internal function to invoke a closure.
   * \param closure_or_packed The closure to be invoked.
   * \param args The arguments to the function.
   * \return The result value.
   */
  RegType InvokeClosureInternal(const ObjectRef& closure_or_packed,
                                const std::vector<RegType>& args);
  /*!
   * \brief Invoke a VM function by interpreting bytecode.
   * \param fidx The function index.
   * \param args The arguments to the function.
   * \return The object representing the result.
   */
  RegType InvokeBytecode(Index fidx, const std::vector<RegType>& args);

 protected:
  /*!
   * \brief Get function by querying all of the current module's imports.
   * \param name The name of the function.
   * \return The result function, can return PackedFunc(nullptr) if nothing is found.
   */
  PackedFunc GetFuncFromImports(const String& name) {
    for (auto& lib : this->imports_) {
      PackedFunc func = lib->GetFunction(name, true);
      if (func.defined()) return func;
    }
    return PackedFunc(nullptr);
  }
  /*!
   * \brief Initialize function pool.
   */
  void InitFuncPool();

  /*!
   * \brief A RAII wrapper that pushes and pops VM frames.
   */
  class FrameGuard {
   public:
    VirtualMachineImpl* vm;
    explicit FrameGuard(VirtualMachineImpl* vm, std::unique_ptr<VMFrame> frame) : vm(vm) {
      vm->frames_.emplace_back(std::move(frame));
    }
    ~FrameGuard() {
      ICHECK_GT(vm->frames_.size(), 0);
      vm->pc_ = vm->frames_.back()->return_pc;
      vm->frames_.back()->Clear();
      vm->frame_free_list_.emplace_back(std::move(vm->frames_.back()));
      vm->frames_.pop_back();
    }
  };
  //-------------------------------------------------
  // Instruction interpretations.
  //-------------------------------------------------
  /*!
   * \brief Push a call frame onto the call stack.
   * \param ret_pc The program counter to return to.
   * \param vm_func The function to be pushed to the call stack.
   * \return A RAII wrapper that pops the frame when going out of scope.
   */
  FrameGuard PushFrame(Index ret_pc, const VMFuncInfo& vm_func) {
    std::unique_ptr<VMFrame> new_frame;
    if (!frame_free_list_.empty()) {
      new_frame = std::move(frame_free_list_.back());
      frame_free_list_.pop_back();
      new_frame->ResetForRecycle(ret_pc, vm_func.register_file_size);
    } else {
      new_frame = std::make_unique<VMFrame>(ret_pc, vm_func.register_file_size);
    }
    return FrameGuard(this, std::move(new_frame));
  }
  /*!
   * \brief Write to a VM register.
   * \param frame current vm frame.
   * \param reg The register to write to.
   * \param obj The object to write to.
   */
  TVM_ALWAYS_INLINE void WriteRegister(VMFrame* frame, RegName reg, const RegType& obj) {
    ICHECK_LT(reg, frame->register_file.size());
    frame->register_file[reg] = obj;
  }
  /*!
   * \brief Read a VM register.
   * \param frame current vm frame.
   * \param reg The register to read from.
   * \return The value of the register.
   */
  TVM_ALWAYS_INLINE RegType ReadRegister(VMFrame* frame, RegName reg) {
    if (reg < Instruction::kBeginSpecialReg) {
      return frame->register_file[reg];
    }
    RegType ret;
    if (reg == Instruction::kVoidRegister) {
      ret = nullptr;
    } else {
      ICHECK_EQ(reg, Instruction::kVMRegister);
      // per convention, ctx ptr must be VirtualMachine* casted to void.
      // this and VirtualMachine* may or may not be the same
      // do first cast to VirtualMachine* then to void*
      ret = static_cast<void*>(static_cast<VirtualMachine*>(this));
    }
    return ret;
  }
  /*!
   * \brief Run call instruction.
   * \param curr_frame The current frame.
   * \param inst The call instruction.
   */
  virtual void RunInstrCall(VMFrame* curr_frame, Instruction inst);

  /*! \brief Run VM dispatch loop. */
  void RunLoop();

  /*!
   * \brief Retrieve the name of the function identified by the given index.
   * \param idx The index into the VM executable function table.
   * \return The name of the function.
   */
  const std::string& GetFuncName(int idx) { return exec_->func_table[idx].name; }

  /*!
   * \brief Retrieve the inputs for a function.
   * \param func_name The name of the function.
   * \return The function inputs.
   */
  const std::vector<RegType>& GetInputsFor(const std::string& func_name) {
    return inputs_[func_name];
  }

  void ClearInputsFor(const std::string& func_name) { inputs_.erase(func_name); }

  //--------------------------------------------------------
  // Internal states for execution.
  //--------------------------------------------------------
  /*! \brief The loaded executable. */
  ObjectPtr<VMExecutable> exec_;
  /*! \brief The global constant pool */
  std::vector<TVMRetValue> const_pool_;
  /*!
   * \brief Function pool to cache functions in func_table
   */
  std::vector<TVMRetValue> func_pool_;
  //--------------------------------------------------------
  // Executor interface support
  //--------------------------------------------------------
  /*! \brief The function name to input register mapping. */
  std::unordered_map<std::string, std::vector<RegType>> inputs_;
  /*! \brief The function name to output register. */
  std::unordered_map<std::string, RegType> outputs_;
  /*! \brief A store of closures created by `save_function`. */
  std::unordered_map<std::string, VMClosure> saved_closures_;
  //------------------------------------------------------------
  // VM Instruction execution.
  //------------------------------------------------------------
  /*!
   * \brief The current stack of call frames.
   * \note: Use unique ptr to avoid re-allocation and copy when frames_ get resized.
   */
  std::vector<std::unique_ptr<VMFrame>> frames_;
  /*!
   * \brief A free list of frame
   */
  std::vector<std::unique_ptr<VMFrame>> frame_free_list_;

  /*! \brief The virtual machine PC. */
  Index pc_{0};
  /*! \brief The special return register. */
  RegType return_value_;
  /*!\ brief instrument function. */
  PackedFunc instrument_ = nullptr;

  // HayeonP
  /*! \brief List whose entry is program counters for a segment */ 
  std::vector< std::vector<int> > per_segment_pc_list_;
  bool are_segments_initialized_ = false;
  std::unique_ptr<VMFrame> segments_frame_ = NULL;
};

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_RELAX_VM_VM_H_
