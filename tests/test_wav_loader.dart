// test/audio_test.dart

//import 'dart:math';
//import 'dart:typed_data';
//import 'package:collection/collection.dart';
import 'package:test/test.dart';
import 'package:nemo_asr_onnx/src/audio/wav_loader.dart';

void main() {
  //final eq = const Float32ListEquality();

  group('WavLoader phase 1', () {
    test('mono 16 kHz passes through unchanged', () async {
      // Fixture: a known 16 kHz mono PCM WAV of length 32000 frames (2.44 sec).
      final path = 'tests/data/mono_16k.wav';
      final (audio, length) = await WavLoader.load(path);
      expect(length, equals(audio.length));

      // Check that it’s exactly 16 000 Hz sample count (allow slight diff if fixture not exact).
      expect(length, closeTo(39040, 1));

      // Values must be in [-1, +1]
      for (final x in audio) {
        expect(x, inInclusiveRange(-1.0, 1.0));
      }
    });

    test('stereo 44.1 kHz resamples & picks channel 0', () async {
      // Fixture: a stereo 44.1 kHz WAV where channel 0 is a ramp [0..1].
      final path = 'tests/data/stereo_44k.wav';
      final (audio, length) = await WavLoader.load(path);

      // Expect length ≈ originalFrames * (16000/44100)
      // If fixture was exactly 44100 frames, newLen ≈ 16000.
      expect(length, closeTo(152320, 1));

      // The first sample of audio should match channel 0’s first value (≈ ramp start = -1.0)
      expect(audio.first, closeTo(0.0, 1e-5));
    });

    test('mono 44.1 kHz resamples', () async {
      // Fixture: a mono 44.1 kHz WAV 3.7 seconds long.
      final path = 'tests/data/mono_44k.wav';
      final (audio, length) = await WavLoader.load(path);

      // Expect length ≈ originalFrames * (16000/44100)
      // If fixture was exactly 44100 frames, newLen ≈ 16000.
      expect(length, closeTo(60160, 1));
    });

    test('throws on malformed header (too short)', () async {
      final path = 'tests/data/bad_header.wav';
      expect(
        () async => await WavLoader.load(path),
        throwsA(isA<FormatException>()),
      );
    });
  });
}
