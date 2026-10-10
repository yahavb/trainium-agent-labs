# Native stack probe

Generated 2026-10-10 18:46:54 UTC on seat-233

## libtorch-neuronx-lite contents

```
$ pip show -f libtorch-neuronx-lite | head -60
Name: libtorch-neuronx-lite
Version: 2.11.0.1.0.1284+f49d8626
Summary: LibtorchNeuronXLite — Custom torch-neuronx runtime for vLLM inference with DI on Neuron
Home-page: 
Author: 
Author-email: 
License: 
Location: /opt/conda/lib/python3.13/site-packages
Requires: protobuf, psutil, torch, torch-xla
Required-by: vllm-neuron
Files:
  libtorch_neuronx_lite-2.11.0.1.0.1284+f49d8626.dist-info/INSTALLER
  libtorch_neuronx_lite-2.11.0.1.0.1284+f49d8626.dist-info/METADATA
  libtorch_neuronx_lite-2.11.0.1.0.1284+f49d8626.dist-info/RECORD
  libtorch_neuronx_lite-2.11.0.1.0.1284+f49d8626.dist-info/REQUESTED
  libtorch_neuronx_lite-2.11.0.1.0.1284+f49d8626.dist-info/WHEEL
  libtorch_neuronx_lite-2.11.0.1.0.1284+f49d8626.dist-info/top_level.txt
  libtorch_neuronx_lite/LICENSE.txt
  libtorch_neuronx_lite/__init__.py
  libtorch_neuronx_lite/__pycache__/__init__.cpython-313.pyc
  libtorch_neuronx_lite/__pycache__/_config.cpython-313.pyc
  libtorch_neuronx_lite/__pycache__/_version.cpython-313.pyc
  libtorch_neuronx_lite/__pycache__/contexts.cpython-313.pyc
  libtorch_neuronx_lite/__pycache__/envs.cpython-313.pyc
  libtorch_neuronx_lite/__pycache__/libtorchneuron.cpython-313.pyc
  libtorch_neuronx_lite/_config.py
  libtorch_neuronx_lite/_version.py
  libtorch_neuronx_lite/compile/__init__.py
  libtorch_neuronx_lite/compile/__pycache__/__init__.cpython-313.pyc
  libtorch_neuronx_lite/compile/__pycache__/artifacts.cpython-313.pyc
  libtorch_neuronx_lite/compile/__pycache__/backend.cpython-313.pyc
  libtorch_neuronx_lite/compile/__pycache__/cache.cpython-313.pyc
  libtorch_neuronx_lite/compile/__pycache__/capture_backend.cpython-313.pyc
  libtorch_neuronx_lite/compile/__pycache__/execute_context.cpython-313.pyc
  libtorch_neuronx_lite/compile/__pycache__/hlo.cpython-313.pyc
  libtorch_neuronx_lite/compile/__pycache__/parallel_compile.cpython-313.pyc
  libtorch_neuronx_lite/compile/__pycache__/parallel_trace.cpython-313.pyc
  libtorch_neuronx_lite/compile/__pycache__/platform.cpython-313.pyc
  libtorch_neuronx_lite/compile/__pycache__/schema.cpython-313.pyc
  libtorch_neuronx_lite/compile/artifacts.py
  libtorch_neuronx_lite/compile/backend.py
  libtorch_neuronx_lite/compile/cache.py
  libtorch_neuronx_lite/compile/capture_backend.py
  libtorch_neuronx_lite/compile/execute_context.py
  libtorch_neuronx_lite/compile/hlo.py
  libtorch_neuronx_lite/compile/parallel_compile.py
  libtorch_neuronx_lite/compile/parallel_trace.py
  libtorch_neuronx_lite/compile/platform.py
  libtorch_neuronx_lite/compile/schema.py
  libtorch_neuronx_lite/contexts.py
  libtorch_neuronx_lite/envs.py
  libtorch_neuronx_lite/fx_passes/__init__.py
  libtorch_neuronx_lite/fx_passes/__pycache__/__init__.cpython-313.pyc
  libtorch_neuronx_lite/fx_passes/__pycache__/aliasing_pass.cpython-313.pyc
  libtorch_neuronx_lite/fx_passes/__pycache__/backend_config_pass.cpython-313.pyc
  libtorch_neuronx_lite/fx_passes/__pycache__/base.cpython-313.pyc
  libtorch_neuronx_lite/fx_passes/__pycache__/collective_replica_groups_pass.cpython-313.pyc
  libtorch_neuronx_lite/fx_passes/__pycache__/device_rewriter.cpython-313.pyc
  libtorch_neuronx_lite/fx_passes/__pycache__/inplace_rewrite_pass.cpython-313.pyc
  libtorch_neuronx_lite/fx_passes/__pycache__/pass_manager.cpython-313.pyc
ERROR: Pipe to stdout was broken
```

## torch neuron attributes

```
$ python3 -c "import torch; print(torch.__version__, [d for d in dir(torch) if 'neuron' in d.lower()])"
2.11.0+cu130 []
```

## 'neuron' device

```
$ python3 -c "import torch; t=torch.ones(2,2).to('neuron'); print(t.device, (t@t).cpu())"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import torch; t=torch.ones(2,2).to('neuron'); print(t.device, (t@t).cpu())
                    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^
RuntimeError: Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: neuron
[exit code: 1]
```

## torch_xla device

```
$ PJRT_DEVICE=NEURON python3 -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"
<string>:1: DeprecationWarning: Use torch_xla.device instead
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import torch_xla.core.xla_model as xm; print(xm.xla_device())
                                                 ~~~~~~~~~~~~~^^
  File "/opt/conda/lib/python3.13/site-packages/typing_extensions.py", line 3140, in wrapper
    return arg(*args, **kwargs)
  File "/opt/conda/lib/python3.13/site-packages/torch_xla/core/xla_model.py", line 137, in xla_device
    return torch_xla.device(n)
           ~~~~~~~~~~~~~~~~^^^
  File "/opt/conda/lib/python3.13/site-packages/torch_xla/torch_xla.py", line 40, in device
    return torch.device(torch_xla._XLAC._xla_get_default_device())
                        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/opt/conda/lib/python3.13/site-packages/torch_xla/_internal/neuron.py", line 118, in library_path
    from libneuronxla.libneuronpjrt_path import libneuronpjrt_path
ModuleNotFoundError: No module named 'libneuronxla'
[exit code: 1]
```

## vllm_neuron location

```
$ python3 -c "import vllm_neuron, os; print(os.path.dirname(vllm_neuron.__file__))"
2026-10-10 18:47:02,968 - INFO - xla_collectives.py:214 - ✓ Custom XLA implementations registered via torch.library.impl (all_reduce, all_gather, reduce_scatter, all_to_all)
INFO 10-10 18:47:02 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 10-10 18:47:02 [__init__.py:46] - neuron -> vllm_neuron:register
INFO 10-10 18:47:02 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 10-10 18:47:03 [__init__.py:237] Platform plugin neuron is activated
2026-10-10 18:47:03,066 - INFO - port_hold_patch.py:204 - Port-hold patch applied (EADDRINUSE file-rendezvous)
2026-10-10 18:47:03,066 - INFO - port_hold_patch.py:204 - Port-hold patch applied (EADDRINUSE file-rendezvous)
/opt/conda/lib/python3.13/site-packages/vllm_neuron
```

## libtorch_neuronx_lite file tree

```
$ cd '/opt/conda/lib/python3.13/site-packages/libtorch_neuronx_lite' && find . -name '*.py' -o -name '*.so' | grep -v __pycache__ | sort
./__init__.py
./_config.py
./_version.py
./compile/__init__.py
./compile/artifacts.py
./compile/backend.py
./compile/cache.py
./compile/capture_backend.py
./compile/execute_context.py
./compile/hlo.py
./compile/parallel_compile.py
./compile/parallel_trace.py
./compile/platform.py
./compile/schema.py
./contexts.py
./envs.py
./fx_passes/__init__.py
./fx_passes/aliasing_pass.py
./fx_passes/backend_config_pass.py
./fx_passes/base.py
./fx_passes/collective_replica_groups_pass.py
./fx_passes/device_rewriter.py
./fx_passes/inplace_rewrite_pass.py
./fx_passes/pass_manager.py
./lib/libjemalloc.so
./lib/libtorchneuron.so
./libtorchneuron.py
./nki/__init__.py
./nki/nki_cache.py
./nki/nki_compile.py
./nki/nki_cpu_sim.py
./nki/nki_dtype.py
./nki/nki_hop.py
./overrides/__init__.py
./overrides/neuron_collectives.py
./overrides/xla_collectives.py
./pyhlo/constant/serialize_numpy.py
./pyhlo/constant/serialize_numpy_bfloat16.py
./pyhlo/constant/serialize_tf.py
./pyhlo/constant/serialize_torch.py
./pyhlo/decomposed_hlo_snapshot.py
./pyhlo/decomposed_hlo_snapshot_pb2.py
./pyhlo/hlo_pb2.py
./pyhlo/scribe.py
./pyhlo/service/hlo_pb2.py
./pyhlo/xla_data_pb2.py
./pyhlo/xla_pb2.py
./pyhlo/xplane_pb2.py
./utils/__init__.py
./utils/hardware_config.py
./utils/timer.py
./utils/utils.py
./xla_impl/base.py
./xla_impl/custom_call_targets.py
./xla_impl/hlo_conversion.py
./xla_impl/ops.py
./xla_impl/structure.py
./xla_impl/xla_hlo_tools/__init__.py
./xla_impl/xla_hlo_tools/literal_proto_utils.py
./xla_impl/xla_hlo_tools/xla_primitive_enum_utils.py
```

## libtorch_neuronx_lite public API

```
$ python3 -c "import libtorch_neuronx_lite as m; print([n for n in dir(m) if not n.startswith('_')])" 2>&1 | tail -5
INFO 10-10 18:47:06 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 10-10 18:47:06 [__init__.py:237] Platform plugin neuron is activated
2026-10-10 18:47:06,258 - INFO - port_hold_patch.py:204 - Port-hold patch applied (EADDRINUSE file-rendezvous)
2026-10-10 18:47:06,258 - INFO - port_hold_patch.py:204 - Port-hold patch applied (EADDRINUSE file-rendezvous)
['base', 'compile', 'contexts', 'envs', 'fx_passes', 'libtorchneuron', 'nki', 'ops', 'os', 'overrides', 'owns_privateuse1', 'pyhlo', 'torch', 'types', 'unsupported_dtype', 'utils', 'xla_impl']
```

## libtorch_neuronx_lite registers 'neuron' device?

```
$ python3 -c "import torch, libtorch_neuronx_lite; print(torch._C._get_privateuse1_backend_name()); t=torch.ones(2,2).to('neuron'); print(t.device, (t@t).cpu())" 2>&1 | tail -8
2026-10-10 18:47:09,456 - INFO - port_hold_patch.py:204 - Port-hold patch applied (EADDRINUSE file-rendezvous)
2026-10-10 18:47:09,456 - INFO - port_hold_patch.py:204 - Port-hold patch applied (EADDRINUSE file-rendezvous)
neuron
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import torch, libtorch_neuronx_lite; print(torch._C._get_privateuse1_backend_name()); t=torch.ones(2,2).to('neuron'); print(t.device, (t@t).cpu())
                                                                                            ~~~~~~~~~~~~~~~~~~^^^^^^^^^^
RuntimeError: Invalid device index. Device index must be between 0 and 3. Check if NEURON_RT_NUM_CORES is set to the total number of NeuronCores.
```

## libtorch_neuronx_lite: compile backend registration

```
$ cd '/opt/conda/lib/python3.13/site-packages/libtorch_neuronx_lite' && grep -rnE 'register_backend|rename_privateuse1|privateuse1|def trace|torch\.compile|backend' --include='*.py' . | head -30
./__init__.py:9:from ._config import owns_privateuse1
./__init__.py:12:# 1. Register the Neuron backend FIRST (no XLA dependency)
./__init__.py:15:if owns_privateuse1():
./__init__.py:17:        torch.utils.rename_privateuse1_backend("neuron")
./__init__.py:28:        torch.utils.generate_methods_for_privateuse1_backend(
./__init__.py:84:#    purely as a runtime behind torch_neuronx's dynamo backend and torch-xla
./__init__.py:110:# 6. Register collective overrides and torch.compile backends
./__init__.py:114:def _register_compile_backends():
./__init__.py:117:    import torch._dynamo.backends.registry as registry
./__init__.py:122:    from .compile.capture_backend import capture
./__init__.py:124:    if "neuron_libtorch_graph_capture" not in registry.list_backends():
./__init__.py:125:        registry.register_backend(compiler_fn=capture, name="neuron_libtorch_graph_capture")
./__init__.py:128:    if "vllm_neuron_graph_capture" not in registry.list_backends():
./__init__.py:129:        registry.register_backend(compiler_fn=capture, name="vllm_neuron_graph_capture")
./__init__.py:131:    # neuron_libtorch backend works in CPU mode — only skip if no hw and not CPU mode
./__init__.py:139:    from .compile.backend import compile
./__init__.py:141:    if "neuron_libtorch" not in registry.list_backends():
./__init__.py:142:        registry.register_backend(compiler_fn=compile, name="neuron_libtorch")
./__init__.py:145:    if "vllm_neuron" not in registry.list_backends():
./__init__.py:146:        registry.register_backend(compiler_fn=compile, name="vllm_neuron")
./__init__.py:154:    _register_compile_backends()
./__init__.py:161:if owns_privateuse1():
./__init__.py:195:# 9. (TORCH_NEURONX_DYNAMO_BACKEND_ONLY=1) Reuse the torch.compile dynamo
./__init__.py:196:#    backend from torch-neuronx.
./__init__.py:199:# torch.compile(backend="neuron") to torch-neuronx's neuron_dynamo_backend.
./__init__.py:208:        import torch_neuronx.neuron_dynamo_backend  # registers torch.compile backend="neuron"
./__init__.py:210:        # torch-neuronx isn't installed; lite still works for its own backends.
./_config.py:24:_resolved_backend: str | None = None
./_config.py:28:    """Resolve and validate the execution backend. Returns 'lite' or 'native'."""
./_config.py:29:    global _resolved_backend
```

## vllm_neuron file tree

```
$ cd '/opt/conda/lib/python3.13/site-packages/vllm_neuron' && find . -name '*.py' | sort | head -80
./__init__.py
./accuracy/__init__.py
./accuracy/accuracy_debugger/__init__.py
./accuracy/accuracy_debugger/api.py
./accuracy/accuracy_debugger/prompt_plugins/__init__.py
./accuracy/accuracy_debugger/prompt_plugins/base.py
./accuracy/accuracy_debugger/prompt_plugins/kv_cache.py
./accuracy/accuracy_debugger/prompt_plugins/logit_val.py
./accuracy/accuracy_debugger/prompt_plugins/tensor_compare.py
./accuracy/accuracy_debugger/report_plugins/__init__.py
./accuracy/accuracy_debugger/report_plugins/base.py
./accuracy/accuracy_debugger/report_plugins/kv_analysis.py
./accuracy/accuracy_debugger/report_plugins/logit_validation.py
./accuracy/accuracy_debugger/report_plugins/task_analysis.py
./accuracy/accuracy_debugger/report_plugins/tensor_compare.py
./accuracy/accuracy_debugger/report_plugins/utils.py
./accuracy/accuracy_debugger/task_plugins/__init__.py
./accuracy/accuracy_debugger/task_plugins/lm_eval_analyzer.py
./accuracy/accuracy_debugger/utils/__init__.py
./accuracy/accuracy_debugger/utils/api_utils.py
./accuracy/accuracy_debugger/utils/report_utils.py
./accuracy/constants.py
./accuracy/encoder_cache_analysis.py
./accuracy/goldens/__init__.py
./accuracy/goldens/fp8_kv_golden.py
./accuracy/goldens/reference_logits.py
./accuracy/goldens/reference_model.py
./accuracy/kv_cache_analysis.py
./accuracy/kv_cache_visualize.py
./accuracy/lm_eval.py
./accuracy/logit_validation.py
./accuracy/logit_visualization.py
./accuracy/plotting.py
./accuracy/tensor_alignment_utils.py
./accuracy/tensor_capture.py
./accuracy/tensor_compare.py
./accuracy/tensor_histogram.py
./accuracy/tensor_io.py
./accuracy/tensor_replacement.py
./accuracy/testing.py
./accuracy/types.py
./accuracy/utils.py
./backend.py
./envs.py
./exceptions.py
./functional/__init__.py
./functional/argmax.py
./functional/argsort_unstable.py
./functional/attention/__init__.py
./functional/attention/attention_cte.py
./functional/attention/attention_decode.py
./functional/attention/attention_decode_mask.py
./functional/attention/attention_segmented_cte.py
./functional/attention/o_proj.py
./functional/attention/qkv.py
./functional/attention/swa_fused.py
./functional/collectives/__init__.py
./functional/collectives/all_gather_v.py
./functional/collectives/all_to_all.py
./functional/collectives/all_to_all_v.py
./functional/collectives/reduce_scatter_v.py
./functional/cumsum.py
./functional/embedding.py
./functional/expert_parallel.py
./functional/mlp.py
./functional/moe/__init__.py
./functional/moe/build_all2all_combine_metadata.py
./functional/moe/build_all2all_dispatch_metadata.py
./functional/moe/build_all_gatherv_metadata.py
./functional/moe/hierarchical_all2all_combine_reduce.py
./functional/moe/hierarchical_all2all_dispatch_permute.py
./functional/moe/moe_block_tkg.py
./functional/moe/moe_block_tkg_wrapper.py
./functional/moe/moe_blockwise.py
./functional/moe/moe_cte.py
./functional/moe/moe_tkg.py
./functional/moe/moe_tkg_wrapper.py
./functional/moe/pack_tokens.py
./functional/moe/permute_routed_tokens.py
./functional/moe/rmsnorm_router_topk_tkg.py
```

## vllm_neuron imports of Neuron libraries

```
$ cd '/opt/conda/lib/python3.13/site-packages/vllm_neuron' && grep -rhoE '^\s*(from|import) [A-Za-z_.]+' --include='*.py' . | sed -E 's/^\s+//' | sort | uniq -c | sort -rn | grep -iE 'neuron|xla|nxd|nki|torch' | head -40
    118 import torch
     38 from vllm_neuron.parallel.neuron_parallel_state
     28 from vllm_neuron.utils.neuron_utils
     28 from torch
     28 from libtorch_neuronx_lite.nki.nki_hop
     26 import nki
     26 from vllm_neuron.utils.weight_loader
     26 from vllm_neuron
     24 import nki.language
     22 from vllm_neuron.model.neuron_config
     19 from nkilib.core.utils.common_types
     18 import torch.distributed
     16 import torch.nn
     16 from vllm_neuron.utils.dtype_utils
     14 import nki.isa
     12 from nkilib.core.utils.kernel_assert
     10 import vllm_neuron.functional
     10 from torch.distributed
     10 from libtorch_neuronx_lite.compile.platform
      9 from vllm_neuron.utils.checkpoints
      8 from vllm_neuron.model.kv_cache
      7 import torch.nn.functional
      7 from vllm_neuron.model.interfaces
      7 from torch.distributed._functional_collectives
      7 from nkilib.core.utils.kernel_helpers
      6 import vllm_neuron.nn
      6 from vllm_neuron.utils.vision_utils
      6 from vllm_neuron.utils.bucket_utils
      6 from vllm_neuron.nn.sampler
      6 from vllm_neuron.nn.embedding
      6 from nki.isa
      5 from vllm_neuron.snapshot.context
      5 from vllm_neuron.snapshot.config
      5 from vllm_neuron.nn.rejection_sampler
      5 from vllm_neuron.metrics
      5 from vllm_neuron.functional.attention.attention_decode
      5 from vllm_neuron.envs
      5 from vllm_neuron.accuracy.tensor_io
      5 from vllm_neuron.accuracy.tensor_capture
      5 from vllm_neuron.accuracy.logit_validation
```

## vllm_neuron: torch.compile / backend

```
$ cd '/opt/conda/lib/python3.13/site-packages/vllm_neuron' && grep -rnE 'torch\.compile|backend=' --include='*.py' . | head -30
./accuracy/lm_eval.py:154:        f"tokenizer_backend=None,"
./accuracy/tensor_capture.py:11:    compiled = torch.compile(model, backend="vllm_neuron", fullgraph=True)
./accuracy/tensor_capture.py:155:    """Registry for captured tensors during torch.compile tracing.
./accuracy/tensor_replacement.py:17:    # Warmup: zero tensors with correct shape for torch.compile tracing
./envs.py:354:    """Return the torch.compile backend name.
./functional/attention/attention_decode.py:594:    # of one packed slot). Eager-only: under torch.compile ``num_rows % 2``
./functional/attention/attention_decode.py:597:    # torch.compiler.is_compiling() is unreliable on the Neuron backend). The
./functional/attention/attention_segmented_cte.py:143:    This implementation is dynamo-traceable under ``torch.compile(..., fullgraph=True)``:
./functional/rmsnorm_quant.py:60:    cannot be traced through ``torch.compile`` / FX cleanly. Mirroring the
./functional/spec_decode_correction.py:83:    # slot_mapping; torch.compile's CSE should elide the redundant work.
./functional/topk.py:48:# vLLM-Neuron runs top-k under torch.compile and cannot call nkilib's host-side
./model/llama3/eagle3_model.py:825:        recurrent decode steps. The recurrent loop is unrolled by torch.compile into a
./model/llama3/eagle3_model.py:933:        # Recurrent decode loop (torch.compile unrolls this Python for-loop
./model/llama3/eagle3_model.py:984:        # of stacked_tokens; under torch.compile, CSE elides the
./model/neuron_config.py:236:            all2all_backend=config_dict.get("all2all_backend"),
./model/qwen3_vl/utils/merge_vision_embeds.py:38:    sentinel values inside torch.compile-traced graphs.
./model/qwen3_vl/vision_encoder_bf16.py:7:libtorch_neuronx_lite patches F.gelu/nn.GELU with a C extension that torch.compile
./model/qwen3_vl/vision_encoder_bf16.py:9:decorated with @torch.compiler.allow_in_graph.
./model/qwen3_vl/vision_encoder_bf16.py:54:@torch.compiler.allow_in_graph
./model/qwen3_vl/vision_encoder_bf16.py:56:    """GELU activation via erf, traceable by torch.compile on Neuron."""
./nn/cpl.py:18:# all_gather_tensor with gather_dim != 0 under torch.compile(fullgraph=True).
./nn/sampler.py:126:                # Build indices as tensors so torch.compile doesn't extract scalars
./parallel/neuron_parallel_state.py:656:                backend=backend,
./parallel/neuron_parallel_state.py:935:        backend=backend,
./parallel/neuron_parallel_state.py:945:        backend=backend,
./parallel/neuron_parallel_state.py:1064:            backend=backend,
./utils/dtype_utils.py:40:# Resolved once at import time so it's a constant during torch.compile tracing.
./utils/executor.py:463:        ...     return torch.compile(model, backend="vllm_neuron")
./utils/neuron_utils.py:32:    return torch.compiler.set_stance("fail_on_recompile")
./vllm/patches/port_hold_patch.py:43:    backend=None, init_method=None, store=None, rank=-1, world_size=-1, **kwargs
```

## vllm_neuron: trace / NEFF / nrt

```
$ cd '/opt/conda/lib/python3.13/site-packages/vllm_neuron' && grep -rniE 'torch_neuronx|\.trace\(|neff|nrt|parallel_model_trace|ModelBuilder' --include='*.py' . | head -40
./__init__.py:67:        import libtorch_neuronx_lite  # noqa: F401
./__init__.py:72:        from libtorch_neuronx_lite.compile.capture_backend import capture
./__init__.py:112:    import libtorch_neuronx_lite
./__init__.py:114:    from libtorch_neuronx_lite.compile.backend import compile
./__init__.py:115:    from libtorch_neuronx_lite.compile.capture_backend import capture
./__init__.py:127:            from libtorch_neuronx_lite.overrides import neuron_collectives  # noqa: F401
./__init__.py:128:            from libtorch_neuronx_lite.overrides import xla_collectives  # noqa: F401
./__init__.py:132:        sys.modules["torch.neuron"] = libtorch_neuronx_lite
./__init__.py:133:        torch.neuron = libtorch_neuronx_lite
./accuracy/logit_validation.py:316:            uses neuron_allclose from libtorch_neuronx_lite. Defaults to "cpu".
./accuracy/logit_validation.py:1807:            uses neuron_allclose from libtorch_neuronx_lite. Defaults to "cpu".
./accuracy/testing.py:10:  ``libtorch_neuronx_lite.testing.assert_close`` semantics (rtol normalized by abs_max).
./accuracy/testing.py:20:    # Two-way (drop-in replacement for libtorch_neuronx_lite.testing.assert_close)
./accuracy/testing.py:266:    """Two-way assertion compatible with libtorch_neuronx_lite.testing.assert_close.
./envs.py:223:    # Master switch for NRT-boundary input snapshot capture; off by default.
./envs.py:277:    Falls through to libtorch_neuronx_lite.envs for NEURON_LIBTORCH_* names
./envs.py:298:        import libtorch_neuronx_lite.envs as libtorch_envs
./envs.py:358:    torch_neuronx package so a separate install keeps upstream semantics.
./envs.py:362:        from libtorch_neuronx_lite.compile.native_backend import register
./envs.py:455:    Snapshots reference the HLO/NEFF that live under the compile cache, so they
./functional/argmax.py:19:from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
./functional/argsort_unstable.py:10:from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
./functional/attention/attention_cte.py:9:from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
./functional/attention/attention_decode.py:28:from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
./functional/attention/attention_decode.py:640:    per compiled decode NEFF, whether to read the packed cache directly or
./functional/attention/attention_decode_mask.py:88:from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
./functional/attention/attention_segmented_cte.py:8:from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
./functional/attention/o_proj.py:10:from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
./functional/attention/qkv.py:20:from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
./functional/attention/swa_fused.py:40:from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
./functional/collectives/all_gather_v.py:13:from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
./functional/collectives/all_gather_v.py:69:            cases, since NEFF I/O cannot also be collective I/O.
./functional/collectives/all_gather_v.py:172:    # NEFF I/O cannot be collective I/O; when kernel I/O is NEFF I/O, copy to intermediate buffers before calling collective
./functional/collectives/all_gather_v.py:197:    # When kernel I/O is NEFF I/O, copy back from intermediate buffers after calling collective
./functional/collectives/all_to_all.py:19:    The platform helper lives in ``libtorch_neuronx_lite`` and is imported
./functional/collectives/all_to_all.py:26:        from libtorch_neuronx_lite.compile.platform import get_platform_target
./functional/collectives/all_to_all_v.py:18:from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
./functional/collectives/all_to_all_v.py:49:            which comes with a small performance penalty. Necessary in some cases, since NEFF I/O cannot also be collective I/O.
./functional/collectives/all_to_all_v.py:154:    # NEFF I/O cannot be collective I/O; when kernel I/O is NEFF I/O, copy to intermediate buffers before calling collective
./functional/collectives/all_to_all_v.py:226:    # When kernel I/O is NEFF I/O, copy back from intermediate buffers after calling collective
```

## vllm_neuron: device strings

```
$ cd '/opt/conda/lib/python3.13/site-packages/vllm_neuron' && grep -rnE "device\(['\"](neuron|xla)|'neuron'|\"neuron\"" --include='*.py' . | head -30
./__init__.py:65:        # Import Lite so it names the PrivateUse1 "neuron" device before vLLM
./__init__.py:142:        torch.utils.rename_privateuse1_backend("neuron")
./__init__.py:170:                return torch.device("neuron", torch.neuron.current_device())
./__init__.py:171:            elif device.type == "neuron" and device.index is None:
./__init__.py:172:                return torch.device("neuron", torch.neuron.current_device())
./accuracy/accuracy_debugger/prompt_plugins/tensor_compare.py:161:        neuron_dir = os.path.join(base_dir, "neuron")
./accuracy/logit_validation.py:314:        test_device: Device to run validation on. Either "cpu" or "neuron".
./accuracy/logit_validation.py:315:            When "cpu", uses a CPU-based allclose implementation. When "neuron",
./accuracy/logit_validation.py:1805:        test_device: Device to run validation on. Either "cpu" or "neuron".
./accuracy/logit_validation.py:1806:            When "cpu", uses a CPU-based allclose implementation. When "neuron",
./envs.py:357:    That name is intentionally not "neuron": "neuron" stays reserved for the
./functional/moe/moe_tkg.py:148:        ...     rank_id=torch.zeros((1, 1), dtype=torch.int32, device="neuron"),
./model/gpt_oss/model_mxfp4.py:231:        and config.neuron_config.all2all_backend == "neuron"
./model/neuron_config.py:145:    # Literal does not include 'neuron'.
./model/neuron_config.py:146:    all2all_backend: Literal[None, "neuron"] = None
./parallel/neuron_communicator.py:24:        # not include 'neuron'. NeuronAll2AllManager should be mounted to Neuron's
./parallel/neuron_communicator.py:29:            self.is_neuron_ep_communicator and self.all2all_backend == "neuron"
./parallel/neuron_parallel_state.py:915:        backend: Distributed backend (e.g. "gloo", "xla", "neuron").
./utils/checkpoints.py:164:        ...     mappings={}, device=torch.device("neuron")
./utils/checkpoints.py:324:            ...     mappings={}, device=torch.device("neuron")
./utils/executor.py:867:    os.environ.setdefault("VLLM_TARGET_DEVICE", "neuron")
./vllm/kv_connector/neuron_nixl_connector.py:68:    attn_backend_name: str = "neuron"
./vllm/kv_connector/neuron_nixl_connector.py:137:                attn_backend_name="neuron",
./vllm/platform.py:125:    device_name: str = "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron"
./vllm/platform.py:126:    device_type: str = "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron"
./vllm/worker/neuron_model_runner.py:442:        self.device = device if device is not None else torch.device("neuron:0")
```

## Neuron python packages installed

```
$ pip list 2>/dev/null | grep -iE 'neuron|nxd|xla|nki'
libtorch-neuronx-lite                    2.11.0.1.0.1284+f49d8626
neuron-agentic-development               1.3
neuronx-cc                               2.27.5334.0+f702b353
nki                                      0.6.0+31049202112.g85070674
torch-xla                                2.11.0
vllm-neuron                              0.24.0.1.1.0
```

## torch backend / privateuse1 name

```
$ python3 -c "import torch; print(torch._C._get_privateuse1_backend_name())"
privateuseone
```

## Follow-up: eager `neuron:0` and native torch.compile (seat-233, NEURON_RT_VISIBLE_CORES=0)

```
$ python3 -c 'import torch, libtorch_neuronx_lite; t=torch.ones(2,2).to("neuron:0"); print(t.device, (t@t).cpu().tolist())'
eager: neuron:0 [[2.0, 2.0], [2.0, 2.0]]

$ torch.compile(Linear(64,64) -> GELU, backend="neuron_libtorch", fullgraph=True), bf16
ERROR - cache.py:362 - Compilation failed - terminating process for cleanup: apply() takes no keyword arguments
FATAL: apply() takes no keyword arguments

$ torch.compile(Linear(64,64) -> ReLU -> Linear(64,64), backend="neuron_libtorch", fullgraph=True), bf16
compile ok 2.0 s
cos 0.999998152256012
```
