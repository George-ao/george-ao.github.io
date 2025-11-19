---
title: "vLLM quantization workflow"
date: 2025-01-18
---

This post is to discuss how quantization works in vLLM.

### **initialize llm_engine**

```python
INFO 12-23 02:54:03 llm_engine.py:234] Initializing an LLM engine (v0.6.6.dev31+gb880ffb8) with config: model='TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ', speculative_config=None, tokenizer='TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=gptq_marlin, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=False, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output"],"candidate_compile_sizes":[],"compile_sizes":[],"capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=False,
INFO 12-23 02:54:11 [selector.py:120](http://selector.py:120/)] Using Flash Attention backend.
INFO 12-23 02:54:12 model_runner.py:1094] Starting to load model TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ...
```

### **create model_config**

vllm/config.py: *__init__():* store relevant infomation of the model in __model_config__.

### **create model: Initialize model + load weight**

model_runner.py: *self.model = get_model(vllm_config=self.vllm_config)* 

&nbsp;&nbsp;&nbsp;&nbsp;**1.initialize model:** Initialize a model with the given configurations → __init__() of each model

vllml/model_executor/model_loader/loader.py

```python
            with target_device:
                model = _initialize_model(vllm_config=vllm_config)
```

vLLM will create layers(weights) according to quant_method when they initialize the model
```python
    self.quant_method.create_weights(
        layer=self,
        input_size_per_partition=self.input_size,
        output_partition_sizes=self.output_partition_sizes,
        input_size=self.input_size,
        output_size=self.output_size,
        params_dtype=self.params_dtype,
        weight_loader=(
            self.weight_loader_v2 if self.quant_method.__class__.__name__
            in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
```

&nbsp;&nbsp;&nbsp;&nbsp;**2.load weight:** loader.py: call *load_model()* 

1. load weights
```
            loaded_weights = model.load_weights(
                self._get_all_weights(model_config, model))
```

2. Call *process_weights_after_loading()* of quantization to repack the weights for kernel

```python
            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    # When quant methods need to process weights after loading
                    # (for repacking, quantizing, etc), they expect parameters
                    # to be on the global target device. This scope is for the
                    # case where cpu offloading is used, where we will move the
                    # parameters onto device for processing and back off after.
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
```

### Inference

If there is a quantization method, it will call its *apply()* in the __forward__ path.
Below is an example for __gptq_marlin__ method.
```python
def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        workspace = MarlinWorkspace(self.output_size_per_partition,
                                    GPTQ_MARLIN_MIN_THREAD_N,
                                    GPTQ_MARLIN_MAX_PARALLEL)

        scales = layer.marlin_scales
        zeros = layer.marlin_zeros
        orig_type = x.dtype

        if orig_type != torch.float16:
            x = x.to(torch.float16)
            scales = scales.to(torch.float16)
            zeros = zeros.to(torch.float16)

        marlin_out = ops.gptq_marlin_gemm(
            x,
            layer.marlin_qweight,
            scales,
            zeros,
            layer.g_idx,
            layer.g_idx_sort_indices,
            workspace.scratch,
            scalar_types.uint4,
            x.shape[0],
            self.output_size_per_partition,
            self.input_size_per_partition,
            True,  # is_k_full
            True,  # has_zp
            True,  # use 32-bit reduce
            True,  # use float zp
        )

        if orig_type != torch.float16:
            marlin_out = marlin_out.to(orig_type)

        if bias is not None:
            marlin_out.add_(bias)

        return marlin_out
```