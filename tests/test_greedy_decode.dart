import 'package:collection/collection.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:nemo_asr_onnx/src/audio/wav_loader.dart';
import 'package:nemo_asr_onnx/src/audio/mel_spectrogram.dart';
import 'package:nemo_asr_onnx/src/runtime/onnx_backend.dart';
import 'package:nemo_asr_onnx/src/decoding/ctc_greedy_decode.dart';
import 'const.dart';

bool areListsClose(List<double> list1, List<double> list2, double tolerance) {
  if (list1.length != list2.length) {
    throw "List must have the same length";
  }

  for (int i = 0; i < list1.length; i++) {
    if (!isCloseTo(list1[i], list2[i], tolerance)) {
      return false; // If any element pair is not close enough
    }
  }

  return true; // All elements are close enough
}

// Helper function for comparing two numbers for closeness
bool isCloseTo(double a, double b, double tolerance) {
  return (a - b).abs() <= tolerance;
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('OnnxBackend integration', () {
    const audioPath = 'tests/data/mono_44k.wav';

    test('end-to-end: wav -> mel -> onnx inference -> ctc decode', () async {
      const modelAsset = 'assets/stt-bm-quartznet15x5-V0.onnx';
      // Step 1: Load raw audio
      final (rawAudio, numSamples) = await WavLoader.load(audioPath);
      expect(rawAudio.length, equals(numSamples));
      print('Loaded audio with $numSamples samples');

      // Step 2: Compute mel spectrogram
      final melSpec = MelSpectrogram(
        sampleRate: 16000,
        nMels: 64,
        windowSize: 0.02,
        windowStride: 0.01,
        padTo: 16,
        nFft: 512,
        normalisePerFeature: true,
      );
      final (melFlat, nFrames) = melSpec.process(rawAudio);
      expect(melFlat.isNotEmpty, isTrue);
      expect(nFrames, greaterThan(0));

      print((melFlat));

      expect(true, areListsClose(melFlat.toList(), flattenedMelSpec, 1e-5));

      print(
          'Computed mel spectrogram with $nFrames frames and 64 features per frame (${melFlat.length} total values). No yet Matching quartznet.preprocessor');
      

      // Step 3: Initialize OnnxBackend and run inference
      final backend = OnnxBackend();
      await backend.init(modelAsset);

      final input = backend.buildInput(
        melFlat,
        melSpec.nMels,
        nFrames,
      );
      final List<List<List<double>>> logits =
          backend.infer(input, inputName: 'audio_signal');

      // Step 4: Basic sanity checks on output
      print((logits[0].length, logits[0][0].length));

      expect(logits.isNotEmpty, isTrue);

      expect(true, areListsClose(logits.flattenedToList.flattenedToList, logProbs, 1e-5));

      final List<String> vocab = await loadQuartznetVocab();
      print(vocab.length);

      expect(
          vocab.length, equals(logits[0][0].length - 1)); // +1 for blank symbol

      final String transcript = ctcGreedyDecode(logits, vocab);

      print('Transcript: $transcript');

      backend.dispose();
    });
  });
}
