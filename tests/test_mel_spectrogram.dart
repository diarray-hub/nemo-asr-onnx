import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:nemo_asr_onnx/src/audio/wav_loader.dart';
import 'package:nemo_asr_onnx/src/audio/mel_spectrogram.dart';

void main() {
  group('MelSpectrogram', () {
    test('produces correct shape for a 16kHz mono WAV', () async {
      // Load a known 16 kHz mono WAV
      final (audio, numSamples) = await WavLoader.load('tests/data/mono_44k.wav');
      final melSpec = MelSpectrogram();
      final (mel, nFrames) = melSpec.process(audio);
      print((mel, nFrames));

      // With center=True padding, expected frames = 1 + (numSamples ~/ hopLen)
      final expectedFrames = 1 + (numSamples ~/ melSpec.hopLen);
      expect(nFrames, equals(expectedFrames));

      // The mel array should have length = nFrames * nMels
      expect(mel.length, equals(nFrames * melSpec.nMels));

      // All values should be finite (no NaN or infinite)
      for (final v in mel) {
        expect(v.isFinite, isTrue);
      }
    });

    test('handles shorter-than-window audio by padding to one frame', () {
      // Create an artificially short signal (< winLen)
      final melSpec = MelSpectrogram();
      final shortSignal = Float32List(melSpec.winLen ~/ 2);

      final (mel, nFrames) = melSpec.process(shortSignal);
      // Expect exactly 1 frame
      expect(nFrames, equals(1));
      // Length should be equal to nMels
      expect(mel.length, equals(melSpec.nMels));
      // Values should be finite
      for (final v in mel) {
        expect(v.isFinite, isTrue);
      }
    });

    test('handles empty audio array by padding and returning one frame', () {
      final melSpec = MelSpectrogram();
      final empty = Float32List(0);
      final (mel, nFrames) = melSpec.process(empty);
      expect(nFrames, equals(1));
      expect(mel.length, equals(melSpec.nMels));
      for (final v in mel) {
        expect(v.isFinite, isTrue);
      }
    });
  });
}
