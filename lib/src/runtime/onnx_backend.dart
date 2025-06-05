import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:onnxruntime/onnxruntime.dart';
import 'tensor_utils.dart';

/// A backend for loading ONNX models and running inference with Ort.
class OnnxBackend {
  final OrtEnv _env = OrtEnv.instance;
  late OrtSession _session;
  bool _initialized = false;
  final sessionOptions = OrtSessionOptions();

  /// Initializes the ONNX session from an asset file.
  ///
  /// Loads the model bytes from [assetPath] (e.g., "assets/model.onnx")
  /// and creates a single OrtSession. Throws [OrtException] on failure.
  Future<void> init(String assetPath) async {
    if (_initialized) return;
    _env.init();
    final raw = await rootBundle.load(assetPath);
    final bytes = raw.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, sessionOptions);;
    _initialized = true;
  }

  /// Builds an OrtValue tensor for input data [mel] with shape [1, channels, frames].
  ///
  /// [mel] must be a Float32List of length channels*frames.
  OrtValue buildInput(Float32List mel, int channels, int frames) {
    final shape = shape1xCxT(channels, frames);
    return OrtValueTensor.createTensorWithDataList(mel, shape);
  }

  /// Runs inference on [input] OrtValue.
  ///
  /// [inputName] is the name of the input node (default: "input").
  /// Returns a Float32List of raw logits or probabilities.
  /// Throws [OrtException] on inference errors.
  List<List<List<double>>> infer(OrtValue inputORT, {String inputName = 'audio_signal'}) {
    if (!_initialized) {
      throw StateError('OnnxBackend not initialized. Call init() first.');
    }
    final input = {inputName: inputORT};
    final runOptions = OrtRunOptions();
    final List<OrtValue?> outputs = _session.run(runOptions, input);

    if (outputs.isEmpty || outputs[0] == null) {
      throw Exception(
        'ONNX inference returned no outputs (expected at least one tensor).',
      );
    }

    // The first OrtValue in the list is our output tensor
    final OrtValueTensor outTensor = outputs[0]! as OrtValueTensor;
    return outTensor.value as List<List<List<double>>>;
  }

  /// Releases the OrtSession and OrtEnv.
  void dispose() {
    if (_initialized) {
      _session.release();
      _env.release();
      _initialized = false;
    }
  }
}

