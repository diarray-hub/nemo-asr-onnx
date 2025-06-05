/// lib/src/audio/mel_spectrogram.dart
///
/// QuartzNet‑style Mel front–end identical to NeMo’s
/// `AudioToMelSpectrogramPreprocessor`.
///
/// Key differences from the previous draft:
/// 1. **Mel filter construction** now uses the canonical
///    `max(0, min(left, right))` triangle so edge‑bins are non‑zero;
///    each row is L1‑normalised (∑w = 1) like HTK / NeMo.
/// 2. Power spectrum is divided by **`winLen²`** (320²) – not `nFft²` –
///    matching torch.stft()+FilterbankFeatures.
/// 3. Everything else (reflection pad, Gaussian dither, per‑feature z‑norm,
///    `padTo=16`) stays the same.
///
/// Returns `(Float32List paddedData, int origFrames)` where `paddedData` is
/// row‑major `[mel, time]` including the time‑axis zeros added to reach a
/// multiple of 16, and `origFrames` equals NeMo’s `processed_signal_len`.

import 'dart:math' as math;
import 'dart:typed_data';
import 'package:fftea/fftea.dart';

class MelSpectrogram {
  // ---------------------------------------------------------------------------
  // Public parameters (QuartzNet 15x5 defaults)
  // ---------------------------------------------------------------------------
  final int sampleRate;
  final int winLen;     // 320
  final int hopLen;     // 160
  final int nFft;       // 512
  final int nMels;      // 64
  final int padTo;      // 16 (set 0 to disable)
  final double dither;  // 1e‑5
  final bool normalisePerFeature;

  // ---------------------------------------------------------------------------
  // Internals
  // ---------------------------------------------------------------------------
  late final FFT _fft;
  late final Float64List _window;          // Hann padded to nFft
  late final List<Float64List> _melBank;   // [nMels][nFft/2+1]
  late final int _nFftBins;                // nFft/2+1

  static const double _fMin = 0.0;
  late final double _fMax;

  MelSpectrogram({
    this.sampleRate = 16000,
    double windowSize = 0.02,
    double windowStride = 0.01,
    this.nFft = 512,
    this.nMels = 64,
    this.padTo = 16,
    this.dither = 1e-5,
    this.normalisePerFeature = true,
  })  : winLen = (windowSize * 16000).round(),
        hopLen = (windowStride * 16000).round() {
    assert(nFft.isEven && nFft >= winLen);
    _nFftBins = nFft ~/ 2 + 1;
    _fMax = sampleRate / 2.0;

    // Hann window (length winLen) zero‑padded to nFft.
    _window = Float64List(nFft);
    for (int i = 0; i < winLen; ++i) {
      _window[i] = 0.5 - 0.5 * math.cos(2 * math.pi * i / (winLen - 1));
    }

    _fft = FFT(nFft);
    _melBank = _buildMelBank();
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------
  (Float32List, int) process(Float32List wav) {
    // 1. Add Gaussian dither
    final rnd = math.Random();
    Float32List noise = Float32List(wav.length);
    for (int i = 0; i < wav.length; i += 2) {
      final u1 = rnd.nextDouble() + 1e-12;
      final u2 = rnd.nextDouble();
      final r = math.sqrt(-2.0 * math.log(u1));
      final theta = 2 * math.pi * u2;
      noise[i] = r * math.cos(theta);
      if (i + 1 < wav.length) noise[i + 1] = r * math.sin(theta);
    }
    final Float32List wavDither = Float32List(wav.length);
    for (int i = 0; i < wav.length; ++i) {
      wavDither[i] = wav[i] + noise[i] * dither;
    }

    // 2. Center padding (reflect)
    final pad = nFft ~/ 2;
    final Float32List padded = _reflectPad(wavDither, pad);

    // 3. Framing count
    final int origFrames = 1 + ((padded.length - nFft) ~/ hopLen);

    // 4. STFT → power → Mel
    final List<Float64List> mel = List.generate(
        origFrames, (_) => Float64List(nMels),
        growable: false);
    final Float64List buf = Float64List(nFft);

    for (int t = 0; t < origFrames; ++t) {
      final int s = t * hopLen;
      for (int i = 0; i < nFft; ++i) {
        buf[i] = padded[s + i] * _window[i];
      }
      final out = _fft.realFft(buf);

      // power spectrum scaled by winLen²
      final Float64List power = Float64List(_nFftBins);
      for (int k = 0; k < _nFftBins; ++k) {
        final re = out[k].x;
        final im = out[k].y;
        power[k] = (re * re + im * im) / (winLen * winLen);
      }

      // project to Mel
      for (int m = 0; m < nMels; ++m) {
        double sum = 0.0;
        final fb = _melBank[m];
        for (int k = 0; k < _nFftBins; ++k) sum += power[k] * fb[k];
        mel[t][m] = sum;
      }
    }

    // 5. log10
    for (int t = 0; t < origFrames; ++t) {
      for (int m = 0; m < nMels; ++m) {
        mel[t][m] = math.log(mel[t][m].clamp(1e-10, double.infinity)) / math.ln10;
      }
    }
    if (normalisePerFeature) _perFeatureNorm(mel);

    // 6. Pad time dimension to multiple of padTo
    int paddedFrames = origFrames;
    int extra = 0;
    if (padTo > 0) {
      extra = (padTo - paddedFrames % padTo) % padTo;
      paddedFrames += extra;
    }

    // 7. Flatten row‑major [mel, time]
    final Float32List flat = Float32List(paddedFrames * nMels);
    int idx = 0;
    for (int m = 0; m < nMels; ++m) {
      for (int t = 0; t < origFrames; ++t) flat[idx++] = mel[t][m].toDouble();
      for (int z = 0; z < extra; ++z) flat[idx++] = 0.0;
    }

    return (flat, origFrames);
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  Float32List _reflectPad(Float32List src, int pad) {
    final out = Float32List(src.length + 2 * pad);
    for (int i = 0; i < pad; ++i) out[i] = src[pad - i - 1];
    out.setAll(pad, src);
    for (int i = 0; i < pad; ++i) out[pad + src.length + i] = src[src.length - i - 1];
    return out;
  }

  void _perFeatureNorm(List<Float64List> mel) {
    final T = mel.length;
    final means = Float64List(nMels);
    final stds = Float64List(nMels);

    for (int m = 0; m < nMels; ++m) {
      double sum = 0;
      for (int t = 0; t < T; ++t) sum += mel[t][m];
      means[m] = sum / T;
      double v = 0;
      for (int t = 0; t < T; ++t) {
        final d = mel[t][m] - means[m];
        v += d * d;
      }
      stds[m] = math.sqrt(v / T).clamp(1e-5, double.infinity);
    }

    for (int t = 0; t < T; ++t) {
      for (int m = 0; m < nMels; ++m) {
        mel[t][m] = (mel[t][m] - means[m]) / stds[m];
      }
    }
  }

  List<Float64List> _buildMelBank() {
    double hzToMel(double hz) => 2595.0 * math.log(1 + hz / 700.0) / math.ln10;
    double melToHz(double mel) => 700.0 * (math.pow(10, mel / 2595.0) - 1);

    final melMin = hzToMel(_fMin);
    final melMax = hzToMel(_fMax);

    final Float64List melPoints = Float64List(nMels + 2);
    for (int i = 0; i < nMels + 2; ++i) {
      melPoints[i] = melMin + (i / (nMels + 1)) * (melMax - melMin);
    }

    final Float64List hzPoints = Float64List(nMels + 2);
    for (int i = 0; i < nMels + 2; ++i) hzPoints[i] = melToHz(melPoints[i]);

    final Float64List binFreqs = Float64List(_nFftBins);
    for (int k = 0; k < _nFftBins; ++k) binFreqs[k] = k * sampleRate / nFft;

    final bank = List.generate(nMels, (_) => Float64List(_nFftBins), growable: false);

    for (int m = 0; m < nMels; ++m) {
      final fL = hzPoints[m];
      final fC = hzPoints[m + 1];
      final fR = hzPoints[m + 2];
      double rowSum = 0.0;
      for (int k = 0; k < _nFftBins; ++k) {
        final f = binFreqs[k];
        final left = (f - fL) / (fC - fL);
        final right = (fR - f) / (fR - fC);
        final w = math.max(0.0, math.min(left, right));
        bank[m][k] = w;
        rowSum += w;
      }
      // L1 normalise so Σw = 1 (HTK style)
      if (rowSum > 0) {
        for (int k = 0; k < _nFftBins; ++k) bank[m][k] /= rowSum;
      }
    }
    return bank;
  }
}
