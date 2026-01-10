"""
Compare ONNX Runtime outputs between Python and Rust implementations.

This script runs inference with the same dummy input on both Python ONNX Runtime
and the Rust serving API to verify output consistency.
"""

import numpy as np
import onnxruntime as ort
import requests
import json

# Model path (relative to serving directory)
MODEL_PATH = "models/deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx"

# Dummy input: 15 features, all zeros
dummy_input = np.zeros((1, 15), dtype=np.float32)

print("=" * 60)
print("ONNX Runtime Output Comparison: Python vs Rust")
print("=" * 60)
print()

# 1. Python ONNX Runtime inference
print("1. Python ONNX Runtime Inference")
print("-" * 60)

sess = ort.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

print(f"   Model input name: {input_name}")
print(f"   Model output name: {output_name}")
print(f"   Input shape: {dummy_input.shape}")
print(f"   Input values: {dummy_input[0][:5]}... (showing first 5)")

python_output = sess.run([output_name], {input_name: dummy_input})[0]
python_ctr = float(python_output[0][0])

print(f"   ✓ Python CTR prediction: {python_ctr:.10f}")
print()

# 2. Rust ONNX Runtime inference (via HTTP API)
print("2. Rust ONNX Runtime Inference (via HTTP)")
print("-" * 60)

# Create a dummy AdRequest with all features mapped to 0 (same as Python)
ad_request = {
    "weekday": 0,
    "hour": 0,
    "region": 0,
    "city": 0,
    "adexchange": 0,
    "domain": "0",
    "slotid": "0",
    "slotwidth": 0,
    "slotheight": 0,
    "slotvisibility": "0",
    "slotformat": "0",
    "creative": "0",
    "user_tag": "0"
}

try:
    response = requests.post(
        "http://localhost:8080/predict",
        json=ad_request,
        timeout=5
    )
    response.raise_for_status()
    result = response.json()

    rust_ctr = result["ctr"]
    latency_ms = result["latency_ms"]

    print(f"   ✓ Rust CTR prediction: {rust_ctr:.10f}")
    print(f"   ✓ Latency: {latency_ms:.3f} ms")
    print()

    # 3. Comparison
    print("3. Comparison")
    print("-" * 60)

    diff = abs(python_ctr - rust_ctr)
    relative_error = (diff / python_ctr * 100) if python_ctr != 0 else 0

    print(f"   Python CTR:  {python_ctr:.10f}")
    print(f"   Rust CTR:    {rust_ctr:.10f}")
    print(f"   Difference:  {diff:.2e}")
    print(f"   Relative Error: {relative_error:.6f}%")
    print()

    # Tolerance check (allow small floating point differences)
    TOLERANCE = 1e-6
    if diff < TOLERANCE:
        print("   ✅ PASSED: Outputs match within tolerance!")
    else:
        print(f"   ⚠️  WARNING: Difference exceeds tolerance ({TOLERANCE})")

except requests.exceptions.ConnectionError:
    print("   ❌ ERROR: Cannot connect to Rust server at localhost:8080")
    print("   Please ensure the server is running with: cargo run")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

print()
print("=" * 60)
