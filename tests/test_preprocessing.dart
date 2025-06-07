import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:nemo_asr_onnx/src/audio/wav_loader.dart';
import 'loaded_wav.dart';

bool areListsClose(List<double> list1, List<double> list2, double tol) {
  if (list1.length != list2.length) return false;
  for (int i = 0; i < list1.length; i++) {
    if ((list1[i] - list2[i]).abs() > tol) return false;
  }
  return true;
}

void main() {
  test('WavLoader matches librosa.load', () async {
    final (audio, numSamples) = await WavLoader.load('tests/data/mono_44k.wav');
    expect(audio.length, equals(numSamples));
    expect(audio.length, equals(loadedWav.length));
    expect(areListsClose(audio.toList(), loadedWav, 1e-4), isTrue);
  });
}
