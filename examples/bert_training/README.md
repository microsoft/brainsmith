# BERT Training Example

This example shows how to train and quantize a small BERT model on the SST2 dataset and export to QONNX,
which can be consumed by Brainsmith.

## Prerequisites

In order to install the prerequisites for this example,
simply install the the packages in the requirements file in the root directory of this repository.
This can be done as follows:

```bash
pip install -f requirements.txt
```

It's recommended that you have a PyTorch compatible GPU on your system to reduce the runtime of the training / quantization process.

## Running the Example

The example runs in 2 phases:
1. train (finetune) the floating-point model; and
2. quantize the trained floating-point model and continue in that direction.

### Train the Floating-Point Model

The floating-point model can be trained as follows:

```bash
python train_fp32_model.py
```

At time of writing, the expected validation accuracy is: 80.50%

### Quantize the Floating-Point Model Int8

After training the floating-point model, it can be quantized to 8-bits as follows:

```bash
python quantize_to_int8.py [--validate]
```

At time of writing, the expected validation accuracy is: 80.73%

#### (Optional): Quantize the Floating-Point Model Int4

Optionally, if you want to heavily quantize the model to Int4 you can do it as follows:

```bash
python quantize_to_int8.py --validate --qronos --qat --bitwidth 4
```

At time of writing, the expected validation accuracy is: 80.50%

### Evaluating the QONNX Model

After quantization step, you will have an artefact `quantized_int8_model.onnx`.
The accuracy of this model can be evaluated as follows:

```bash
evaluate_onnx_accuracy.py
```

Once your QONNX model is validated, you can use this as an input to the BERT Brainsmith example.



## Building the example design

To build a 6-Layer bert model with brainsmith for the V80 use the following command

```bash
python bert_demo.py -m <onnx model filename> -o bert_demo_build --blueprint blueprints/l6_bert_demo.yaml
```

