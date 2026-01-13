"""
ONNX Model Quantization Script

Applies INT8 quantization to reduce model size and improve inference speed.

Quantization types:
- Dynamic: Weights are quantized, activations quantized at runtime
- Static: Both weights and activations pre-quantized (requires calibration data)

Usage:
    python training/src/export/quantize_onnx.py \
        --model-path serving/models/autoint_emb64_att3x4_dnn256128_neg150_best.onnx \
        --output-path serving/models/autoint_int8.onnx \
        --mode dynamic
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import quantize_dynamic, quantize_static
from onnxruntime.quantization import QuantType, QuantFormat


def quantize_model_dynamic(model_path: str, output_path: str):
    """Apply dynamic INT8 quantization"""
    print(f"Input model: {model_path}")
    print(f"Output model: {output_path}")
    print(f"Quantization: Dynamic INT8")

    # Dynamic quantization - no calibration data needed
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        extra_options={
            'ActivationSymmetric': False,
            'WeightSymmetric': True,
        }
    )

    print("Dynamic quantization complete!")


def quantize_model_static(model_path: str, output_path: str, calibration_data: np.ndarray):
    """Apply static INT8 quantization with calibration"""
    from onnxruntime.quantization import CalibrationDataReader
    import onnxruntime.quantization.shape_inference as shape_inf
    import tempfile
    import os

    class DataReader(CalibrationDataReader):
        def __init__(self, data):
            self.data = data
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.data):
                return None
            result = {'input': self.data[self.idx:self.idx+1]}
            self.idx += 1
            return result

        def rewind(self):
            self.idx = 0

    print(f"Input model: {model_path}")
    print(f"Output model: {output_path}")
    print(f"Quantization: Static INT8")
    print(f"Calibration samples: {len(calibration_data)}")

    # Preprocess model for static quantization (required step)
    print("Preprocessing model for static quantization...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        preprocessed_path = os.path.join(tmp_dir, "preprocessed.onnx")

        from onnxruntime.quantization import preprocess
        preprocess.quant_pre_process(
            model_path,
            preprocessed_path,
            skip_symbolic_shape=True
        )

        data_reader = DataReader(calibration_data)

        quantize_static(
            model_input=preprocessed_path,
            model_output=output_path,
            calibration_data_reader=data_reader,
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,  # Use UInt8 for activations (common practice)
            extra_options={
                'ActivationSymmetric': False,
                'WeightSymmetric': True,
            }
        )

    print("Static quantization complete!")


def compare_models(original_path: str, quantized_path: str, test_input: np.ndarray):
    """Compare original and quantized model outputs"""
    import onnxruntime as ort

    print("\n=== Model Comparison ===")

    # Original model
    orig_session = ort.InferenceSession(original_path)
    orig_output = orig_session.run(None, {'input': test_input})[0]

    # Quantized model
    quant_session = ort.InferenceSession(quantized_path)
    quant_output = quant_session.run(None, {'input': test_input})[0]

    # Compare
    diff = np.abs(orig_output - quant_output)
    print(f"Original output: {orig_output.flatten()[:5]}...")
    print(f"Quantized output: {quant_output.flatten()[:5]}...")
    print(f"Max difference: {diff.max():.6f}")
    print(f"Mean difference: {diff.mean():.6f}")
    print(f"Correlation: {np.corrcoef(orig_output.flatten(), quant_output.flatten())[0,1]:.6f}")

    # File sizes
    orig_size = Path(original_path).stat().st_size / 1024 / 1024
    quant_size = Path(quantized_path).stat().st_size / 1024 / 1024
    reduction = (1 - quant_size / orig_size) * 100

    print(f"\n=== Size Comparison ===")
    print(f"Original: {orig_size:.2f} MB")
    print(f"Quantized: {quant_size:.2f} MB")
    print(f"Reduction: {reduction:.1f}%")

    return diff.max(), diff.mean()


def main():
    parser = argparse.ArgumentParser(description='Quantize ONNX model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to input ONNX model')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Path to output quantized model')
    parser.add_argument('--mode', type=str, default='dynamic',
                       choices=['dynamic', 'static'],
                       help='Quantization mode')
    parser.add_argument('--calibration-data', type=str, default=None,
                       help='Path to calibration data (numpy .npy file) for static mode')
    parser.add_argument('--compare', action='store_true',
                       help='Compare original and quantized outputs')

    args = parser.parse_args()

    # Default output path
    if args.output_path is None:
        model_path = Path(args.model_path)
        suffix = '_int8_dynamic' if args.mode == 'dynamic' else '_int8_static'
        args.output_path = str(model_path.parent / f"{model_path.stem}{suffix}.onnx")

    print("=" * 60)
    print("ONNX Model Quantization")
    print("=" * 60)

    if args.mode == 'dynamic':
        quantize_model_dynamic(args.model_path, args.output_path)
    else:
        if args.calibration_data is None:
            print("Static quantization requires --calibration-data")
            return
        calibration_data = np.load(args.calibration_data)
        quantize_model_static(args.model_path, args.output_path, calibration_data)

    # Verify quantized model
    print("\n=== Verifying Quantized Model ===")
    quant_model = onnx.load(args.output_path)
    onnx.checker.check_model(quant_model)
    print("Quantized model verification passed!")

    # Compare if requested
    if args.compare:
        # Create dummy input for comparison
        orig_model = onnx.load(args.model_path)
        input_shape = [d.dim_value for d in orig_model.graph.input[0].type.tensor_type.shape.dim]
        if input_shape[0] == 0:
            input_shape[0] = 1  # batch size
        test_input = np.random.randn(*input_shape).astype(np.float32)
        compare_models(args.model_path, args.output_path, test_input)

    print(f"\nQuantized model saved to: {args.output_path}")


if __name__ == '__main__':
    main()
