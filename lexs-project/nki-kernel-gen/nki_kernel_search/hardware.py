"""SDK 2.27 standalone adapter; isolates version-specific compiler APIs here."""
import inspect
import os

import numpy as np


class HardwareKernel:
    def __init__(self, fn):
        self.fn = fn
        self.compiled = {}
        self.signature = inspect.signature(fn)

    def key(self, args):
        return tuple((a.shape, str(a.dtype)) if isinstance(a, np.ndarray) else repr(a) for a in args)

    def __call__(self, *args):
        from nki.framework.compiled import StandaloneKernel
        from nki.compiler.driver import _build_nki_return_value
        key = self.key(args)
        if key not in self.compiled:
            def capture(compiled, inputs, outputs):
                self.compiled[key] = compiled
                compiled.execute(inputs, outputs)
            return StandaloneKernel(func=self.fn, lnc=int(os.getenv("NEURON_LOGICAL_NC_CONFIG", "1")),
                                    _executor=capture)(*args)
        compiled = self.compiled[key]
        inputs = compiled.prepare_inputs(self.signature.bind(*args).arguments)
        outputs = compiled.prepare_outputs()
        compiled.execute(inputs, outputs)
        return _build_nki_return_value(compiled, outputs)

    def benchmark(self, *args):
        from nrtpy.spike_tensor import SpikeTensor
        compiled = self.compiled[self.key(args)]
        inputs = compiled.prepare_inputs(self.signature.bind(*args).arguments)
        inputs = {k: SpikeTensor.from_numpy(v, name=k) for k, v in inputs.items()}
        outputs = {k: SpikeTensor.from_numpy(v, name=k) for k, v in compiled.prepare_outputs().items()}
        model = compiled._ensure_loaded(0, 1)
        result = model.benchmark(inputs, outputs=outputs, warmup_iter=10,
                                 benchmark_iter=100, mode="device")
        return float(np.median(result.durations_ms) * 1000)
