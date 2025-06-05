import 'dart:async';
import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;

/// Loads `assets/quartznet.vocab` once and caches the token list.
/// Each line is one token; trailing newline ignored.
/// Throws [StateError] if the file is empty or missing.
Future<List<String>> loadQuartznetVocab({String asset = 'assets/quartznet.vocab'}) async {
  if (_cachedVocab != null) return _cachedVocab!;
  final contents = await rootBundle.loadString(asset);
  final lines = const LineSplitter().convert(contents);
  if (lines.isEmpty) {
    throw StateError('Vocab file is empty: $asset');
  }
  _cachedVocab = lines;
  return _cachedVocab!;
}

List<String>? _cachedVocab;

/// Greedy-decodes ONNX QuartzNet logits.
///
/// [logits] must have shape [1, T, V].
/// Blank symbol index == vocab.length.
/// Repeated symbols and blanks are collapsed.
/// Returns the final UTF-8 transcript.
String ctcGreedyDecode(List<List<List<double>>> logits, List<String> vocab) {
  if (logits.isEmpty) {
    throw StateError('Logits input is empty');
  }
  if (logits[0].isEmpty) {
    return '';
  }
  // Shape check: logits.length == 1
  if (logits.length != 1) {
    throw StateError('Expected shape [1, T, V], got outer dimension ${logits.length}');
  }
  final int T = logits[0].length;
  final int V = logits[0][0].length;
  final int blankIndex = vocab.length;
  if (V != vocab.length + 1) {
    throw StateError('Logits vocab dimension $V does not match vocab length ${vocab.length}+1');
  }

  final StringBuffer decoded = StringBuffer();
  int? prev;
  for (int t = 0; t < T; t++) {
    final List<double> frame = logits[0][t];
    // Argmax over V entries
    int maxIdx = 0;
    double maxVal = frame[0];
    for (int k = 1; k < frame.length; k++) {
      if (frame[k] > maxVal) {
        maxVal = frame[k];
        maxIdx = k;
      }
    }
    // Collapse repeats and blanks
    if (maxIdx != prev && maxIdx != blankIndex) {
      decoded.write(vocab[maxIdx]);
    }
    prev = maxIdx;
  }
  return decoded.toString();
}
