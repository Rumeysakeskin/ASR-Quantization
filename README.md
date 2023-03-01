## Speed Up Speech Recognition Inference on CPU Devices with Post-Training Quantization to Nvidia NeMo ASR Model

Model quantization is performance optimization technique that allows speeding up inference and decreasing memory requirements 
by performing computations and storing tensors at lower bitwidths (such as INT8 or FLOAT16) than floating-point precision. 
This is particularly beneficial during model deployment.

There are two model quantization methods, 
- Quantization Aware Training (QAT)
- Post-training Quantization (PTQ)

QAT mimics the effects of quantization during training: 
The computations are carried-out in floating-point precision but the subsequent quantization effect is taken into account. 
The weights and activations are quantized into lower precision only for inference, when training is completed. 

PTQ focuses on quantize the fine-tuned model without retraining. 
The weights and activations of ops are converted into lower precision for saving the memory and computation losses.

In this project, Post Training Static Quantization quantizes the weights and activations of the model. 
It fuses activations into preceding layers where possible. 
It requires calibration with a representative dataset to determine optimal quantization parameters for activations. 
Post Training Static Quantization is typically used when both memory bandwidth and compute savings are important with CNNs being a typical use case. 

### Choose a Quantization Backend for Hardware
PyTorch currently has two quantization backends that support quantization.
- FBGEMM is specific to x86 CPUs and is intended for deployments of quantized models on server CPUs.
- QNNPACK has a range of targets that includes ARM CPUs (typically found in mobile/embedded devices).

### Apply torch.quantization.QuantStub() and torch.quantization.QuantStub() to the inputs and outputs
```python
# Insert the QuantStub() before the first layer of the model
first_asr_model.quant = torch.quantization.QuantStub()
first_asr_model.encoder.quant = torch.quantization.QuantStub()

# Insert a DeQuantStub() at the end of the model
first_asr_model.dequant = torch.quantization.DeQuantStub()
first_asr_model.decoder.dequant = torch.quantization.DeQuantStub()
```

