from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from numbers import Number, Integral

import numpy as np  # type: ignore

import tvm
from tvm.ffi import register_func
from tvm.runtime import Device, Object, PackedFunc
from tvm.runtime.profiling import Report

from ..rpc.base import RPC_SESS_MASK

import re


class SegmentRunner:
    
    NAIVE_ALLOCATOR = 1
    POOLED_ALLOCATOR = 2
    
    def __init__(
            self, 
            rt_mod: Union[tvm.runtime.Module, tvm.runtime.Executable],
            device: Union[Device, List[Device]],
            memory_cfg: Optional[Union[str, Dict[Device, str]]] = None,
            profile: bool = False,
        ) -> None:
        
        if not isinstance(rt_mod, tvm.runtime.Module):
            if isinstance(rt_mod, tvm.runtime.Executable):
                rt_mod = rt_mod.jit()
            else:
                raise ValueError("Expect the rt_mod to be an runtime.Module")

        load_exec = "vm_profiler_load_executable" if profile else "vm_load_executable"
        
        # Default VM functions
        self.module = rt_mod[load_exec]()
        self._invoke_closure = self.module["invoke_closure"]
        self._save_function = self.module["save_function"]
        self._set_input = self.module["set_input"]
        self._invoke_stateful = self.module["invoke_stateful"]
        self._get_output = self.module["get_output"]
        self._get_output_arity = self.module["get_output_arity"]
        self._get_function_arity = self.module["get_function_arity"]
        self._get_function_param_name = self.module["get_function_param_name"]
        self._set_instrument = self.module["set_instrument"]
        
        # VM functions for segment runner
        self._init_persistent_frame = self.module['init_persistent_frame']
        self._get_runtime_sequence = self.module['get_runtime_sequence']
        self._set_input_to_persistent_frame = self.module['set_input_to_persistent_frame']
        self._invoke_segment = self.module['invoke_semgnet']
        self._get_output_from_persistent_frame = self.module['get_output_from_persistent_frame']
        
        # Initialization
        self._setup_device(device, memory_cfg)
        self._init_persistent_frame()
        
        # Variables
        self.segment_list = []
        self._is_initialized = False
        self._prev_segment_id = -1
        
        pass
    
    def _setup_device(self, dev: Device, memory_cfg: Union[str, Dict[Device, str]]) -> None:
        """init devices and allocators."""
        devs = dev
        if not isinstance(dev, (list, tuple)):
            if not isinstance(dev, tvm.runtime.Device):
                raise TypeError("dev is expected to be Device or List[Device]")
            devs = [dev]

        # CPU is required for executing shape functions
        if devs[-1].device_type % RPC_SESS_MASK != tvm.cpu().device_type:
            devs.append(tvm.cpu())

        default_alloc_type = SegmentRunner.POOLED_ALLOCATOR
        if memory_cfg is None:
            memory_cfg = {}
        elif isinstance(memory_cfg, str):
            assert memory_cfg in ["naive", "pooled"]
            if memory_cfg == "naive":
                default_alloc_type = SegmentRunner.NAIVE_ALLOCATOR
            memory_cfg = {}
        elif not isinstance(memory_cfg, dict):
            raise TypeError(
                "memory_cfg is expected be string or dictionary, "
                + "but received {}".format(type(memory_cfg))
            )
        init_args = []
        for device in devs:
            init_args.append(device.device_type % RPC_SESS_MASK)
            init_args.append(device.device_id)
            alloc_type = memory_cfg[device] if device in memory_cfg else default_alloc_type
            init_args.append(alloc_type)
        self.module["vm_initialization"](*init_args)

    
    def get_runtime_sequence(self) -> str:
        return self._get_runtime_sequence()
    
    def load(self, runtime_sequence: str):
        if not runtime_sequence.strip():
            print("ParsingError: Runtime sequence is empty")
            return -1

       # Step 1: Preprocessing (trimming, remove empty lines)
        runtime_sequence_lines = []
        for raw_line in runtime_sequence.splitlines():
            trimmed = raw_line.strip()
            if trimmed:
                runtime_sequence_lines.append({
                    "raw": raw_line,
                    "trimmed": trimmed
                })

        # Step 2: Front-end validation
        if runtime_sequence_lines[0]["trimmed"] != "@seg":
            print("ParsingError: Does not start with @seg annotator")
            return -1

        if runtime_sequence_lines[-1]["trimmed"] != "@seg":
            print("ParsingError: Does not end with @seg annotator")
            return -1

        # Step 3: Parsing
        pc_pattern = re.compile(r"pc\s*=\s*(\d+)")
        for line_info in runtime_sequence_lines:
            trimmed = line_info["trimmed"]

            if trimmed == "@seg":
                self.segment_list.append([])
                continue

            matches = pc_pattern.findall(trimmed)
            if len(matches) == 0:
                print(f'ParsingError: No program counter found in a line: "{line_info["raw"]}"')
                return -1

            if len(matches) > 1:
                print(f'ParsingError: Multiple program counters in a line: "{line_info["raw"]}"')
                return -1

            pc = int(matches[0])
            self.segment_list[-1].append(pc)

        # Step 4: Remove trailing empty segment if needed
        if not self.segment_list[-1]:
            self.segment_list.pop()

        self._is_initialized = True
        
        return
    
    def set_input(self, input) -> None:
        def ensure_iterable(obj):            
            if isinstance(obj, (list, tuple)):
                return obj
            else:
                return [obj]
            
        input = ensure_iterable(input)
        
        self._set_input_to_persistent_frame(*input)
        return
    
    def set_input_with_params(self, input, params) -> None:
        def ensure_iterable(obj):            
            if isinstance(obj, (list, tuple)):
                return obj
            else:
                return [obj]

        input = ensure_iterable(input)
        params = ensure_iterable(params)

        self._set_input_to_persistent_frame(*input, *params)
        return
    
    
    def execute(self, segment_id: int) -> None:
        
        if(not self._is_initialized):
            print("SegmentRunnerError: Segments are not initialized")
            exit()            
        
        if segment_id > len(self.segment_list):
            print(f"SegmentRunnerError: Segment id is bigger than the length (segmet_id: {segment_id}, length: {len(self.segment_list)})")
            exit()
        
        if segment_id > self._prev_segment_id + 1:
            print(f"SegmentSkipWarning: Segments are skipped: (segment_id: {segment_id}, prev_segment_id: {self._prev_segment_id})")
        
        self._invoke_segment(*self.segment_list[segment_id])
        
        self._prev_segment_id = segment_id
            
        return

    def get_output(self) -> List[tvm.runtime.NDArray]: # Return NDArray list
        return self._get_output_from_persistent_frame()