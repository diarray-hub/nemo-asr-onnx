import 'package:flutter_test/flutter_test.dart';
import 'package:nemo_asr_onnx/src/audio/wav_loader.dart';
import 'package:nemo_asr_onnx/src/audio/mel_spectrogram.dart';
import 'package:nemo_asr_onnx/src/runtime/onnx_backend.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('OnnxBackend integration', () {
    const audioPath = 'tests/data/mono_16k.wav';

    test('end-to-end: wav -> mel -> onnx inference', () async {
      const modelAsset = 'assets/stt-bm-quartznet15x5-V0.onnx';
      // Step 1: Load raw audio
      final (rawAudio, numSamples) = await WavLoader.load(audioPath);
      expect(rawAudio.length, equals(numSamples));

      // Step 2: Compute mel spectrogram
      final melSpec = MelSpectrogram(
        sampleRate: 16000,
        nMels: 64,
        windowSize: 0.02,
        windowStride: 0.01,
        nFft: 512,
        normalisePerFeature: true,
      );
      final (melFlat, nFrames) = melSpec.process(rawAudio);
      expect(melFlat.isNotEmpty, isTrue);
      expect(nFrames, greaterThan(0));

      // Step 3: Initialize OnnxBackend and run inference
      final backend = OnnxBackend();
      await backend.init(modelAsset);

      final input = backend.buildInput(
        melFlat,
        melSpec.nMels,
        nFrames,
      );

      final List<List<List<double>>> logits = backend.infer(input, inputName: 'audio_signal');

      // Step 4: Basic sanity checks on output
      print((logits[0].length));

      expect(logits.isNotEmpty, isTrue);

      backend.dispose();
    });

    /*test('end-to-end: wav -> mel -> onnx inference', () async {
      const modelAsset = 'assets/soloni-114m-tdt-ctc-V0.onnx';
      // Step 1: Load raw audio
      final (rawAudio, numSamples) = await WavLoader.load(audioPath);
      expect(rawAudio.length, equals(numSamples));

      // Step 2: Compute mel spectrogram
      final melSpec = MelSpectrogram(
        sampleRate: 16000,
        nMels: 80,
        windowSize: 0.025,
        windowStride: 0.01,
        nFft: 512,
        normalisePerFeature: true,
      );
      final (melFlat, nFrames) = melSpec.process(rawAudio);
      expect(melFlat.isNotEmpty, isTrue);
      expect(nFrames, greaterThan(0));

      // Step 3: Initialize OnnxBackend and run inference
      final backend = OnnxBackend();
      await backend.init(modelAsset);

      final input = backend.buildInput(
        melFlat,
        melSpec.nMels,
        nFrames,
      );

      final List<List<List<double>>> logits = backend.infer(input, inputName: 'audio_signal');

      // Step 4: Basic sanity checks on output
      print((logits, logits.length));

      expect(logits.isNotEmpty, isTrue);

      backend.dispose();
    });*/
  });
}
