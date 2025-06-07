import 'package:nemo_asr_onnx/src/audio/wav_loader.dart';
import 'package:nemo_asr_onnx/src/audio/mel_spectrogram.dart';
import 'package:test/test.dart';
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

  group('OnnxBackend integration', () {
    const audioPath = 'tests/data/mono_44k.wav';

    test('Audio Preprocessing: wav -> mel', () async {
      // Step 1: Load raw audio
      final (rawAudio, numSamples) = await WavLoader.load(audioPath);
      print(rawAudio);
      expect(rawAudio.length, equals(numSamples));
      expect(true, areListsClose(rawAudio.toList(), LoadedWav, 1e-6));
      print('Loaded audio with $numSamples samples, close to expected.');

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
      //print((melFlat));

      expect(true, areListsClose(melFlat.toList(), flattenedMelSpec, 1e-5));

      print(
          'Computed mel spectrogram with $nFrames frames and 64 features per frame (${melFlat.length} total values). No yet Matching quartznet.preprocessor');
    });
  });
}
