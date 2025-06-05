// lib/src/audio/wav_loader.dart

import 'dart:io';
import 'dart:typed_data';
// import 'dart:math';

/// Return type for wav loader: (audio samples, number of samples).
typedef AudioLoadResult = (Float32List audio, int numSamples);

/// Loads a WAV file from disk, picks channel 0, resamples to 16 kHz,
/// converts PCM to float32 in [−1,+1], and returns (audio, length).
///
/// Supports PCM WAV 8/16/24/32-bit (little-endian), any numChannels, any sampleRate.
/// Throws [FormatException] on malformed header, non-PCM, or unsupported bit-depth.
class WavLoader {
  /// Target sample rate (16 000 Hz).
  static const int targetRate = 16000;

  /// Loads the WAV file at [path] and returns `(audio, numSamples)`.
  ///
  /// * Reads RIFF/WAVE header; finds the first `fmt ` and `data` chunks.
  /// * Verifies `audioFormat == 1` (PCM). Supports bitDepth ∈ {8,16,24,32}.
  /// * Picks channel 0 if `numChannels > 1`.
  /// * If `origRate != 16000`, upsamples/downsamples via linear interpolation.
  /// * Converts PCM → float32 in [−1.0, +1.0]:
  ///   - 8-bit unsigned: `(u8 − 128)/128`
  ///   - 16-bit signed: `i16/32768`
  ///   - 24-bit signed: `i24/ 8388608`
  ///   - 32-bit signed: `i32/2147483648`
  /// * Returns a `Float32List` of length `newNumSamples`, plus the integer `newNumSamples`.
  ///
  /// Throws [FormatException] if:
  ///  • File is <44 bytes, or
  ///  • Missing `"RIFF"`/`"WAVE"`, or
  ///  • No `fmt ` chunk, or
  ///  • No `data` chunk, or
  ///  • `audioFormat != 1`, or
  ///  • Bit-depth not in {8,16,24,32}.
  static Future<AudioLoadResult> load(String path) async {
    final bytes = await File(path).readAsBytes();
    if (bytes.length < 44) {
      throw FormatException('Invalid WAV: file too short (<44 bytes)');
    }
    final data = bytes.buffer.asByteData();

    // --- Check RIFF / WAVE ---
    final riff = String.fromCharCodes(bytes.sublist(0, 4));
    if (riff != 'RIFF') {
      throw FormatException('Invalid WAV: missing "RIFF" header');
    }
    final wave = String.fromCharCodes(bytes.sublist(8, 12));
    if (wave != 'WAVE') {
      throw FormatException('Invalid WAV: missing "WAVE" header');
    }

    // --- Iterate chunks to find fmt and data ---
    int offset = 12;
    int? audioFormat;
    int? numChannels;
    int? sampleRate;
    int? bitsPerSample;
    int dataOffset = -1;
    int dataSize = 0;

    while (offset + 8 <= bytes.length) {
      final chunkId = String.fromCharCodes(bytes.sublist(offset, offset + 4));
      final chunkSize = data.getUint32(offset + 4, Endian.little);

      if (chunkId == 'fmt ') {
        // Parse fmt chunk
        audioFormat = data.getUint16(offset + 8, Endian.little);
        numChannels = data.getUint16(offset + 10, Endian.little);
        sampleRate = data.getUint32(offset + 12, Endian.little);
        bitsPerSample = data.getUint16(offset + 22, Endian.little);
      } else if (chunkId == 'data') {
        dataOffset = offset + 8;
        dataSize = chunkSize;
        break;
      }
      // Move to next chunk (account for padding if chunkSize is odd)
      offset += 8 + chunkSize + (chunkSize % 2);
    }

    if (audioFormat == null ||
        numChannels == null ||
        sampleRate == null ||
        bitsPerSample == null) {
      throw FormatException('Invalid WAV: missing "fmt " chunk');
    }
    if (audioFormat != 1) {
      throw FormatException(
          'Unsupported WAV: audioFormat=$audioFormat (only PCM=1 allowed)');
    }
    if (dataOffset < 0 || dataSize <= 0) {
      throw FormatException('Invalid WAV: missing "data" chunk');
    }

    final int origChannels = numChannels;
    final int origRate = sampleRate;
    final int bitDepth = bitsPerSample;

    // --- Read raw PCM samples into appropriate TypedList ---
    final int totalSamples = dataSize ~/ (bitDepth ~/ 8);
    // We'll pick channel 0 only, so number of frames = totalSamples / origChannels
    final int numFrames = totalSamples ~/ origChannels;

    // Extract only channel 0 into a Float32List (unscaled for now).
    final Float32List rawMono = Float32List(numFrames);

    for (int frame = 0; frame < numFrames; frame++) {
      // index of channel 0 sample = (frameIndex * origChannels) + 0
      final int sampleIndex = frame * origChannels;

      late num pcmValue;
      switch (bitDepth) {
        case 8:
          // 8-bit is unsigned
          final int u8 = data.getUint8(dataOffset + sampleIndex);
          pcmValue = (u8 - 128) / 128.0;
          break;
        case 16:
          final int i16 = data.getInt16(
              dataOffset + sampleIndex * 2, Endian.little);
          pcmValue = i16 / 32768.0;
          break;
        case 24:
          // 3 bytes: little-endian signed
          final int b0 = data.getUint8(dataOffset + sampleIndex * 3);
          final int b1 = data.getUint8(dataOffset + sampleIndex * 3 + 1);
          final int b2 = data.getUint8(dataOffset + sampleIndex * 3 + 2);
          // assemble signed 24-bit
          int i24 = (b2 << 16) | (b1 << 8) | b0;
          // sign extension
          if (i24 & 0x800000 != 0) i24 |= 0xFF000000;
          pcmValue = i24 / 8388608.0;
          break;
        case 32:
          final int i32 = data.getInt32(
              dataOffset + sampleIndex * 4, Endian.little);
          pcmValue = i32 / 2147483648.0;
          break;
        default:
          throw FormatException(
              'Unsupported WAV bit-depth: $bitDepth (only 8/16/24/32)');
      }
      // clamp to [-1.0, +1.0] just in case
      rawMono[frame] = pcmValue.clamp(-1.0, 1.0).toDouble();
    }

    // --- Resample if origRate != targetRate ---
    if (origRate != targetRate) {
      final int oldLen = rawMono.length;
      final int newLen =
          ((oldLen * targetRate) / origRate).round();
      final Float32List resampled = Float32List(newLen);

      if (newLen == 0) {
        // Edge case: too short to resample meaningfully
        return (Float32List(0), 0);
      }

      for (int i = 0; i < newLen; i++) {
        final double pos = (i * (oldLen - 1)) / (newLen - 1);
        final int idx0 = pos.floor();
        final int idx1 = pos.ceil();
        final double t = pos - idx0;
        final double s0 = rawMono[idx0];
        final double s1 = rawMono[idx1];
        resampled[i] = ((1 - t) * s0 + t * s1).clamp(-1.0, 1.0).toDouble();
      }
      return (resampled, newLen);
    }

    // No resampling needed
    return (rawMono, rawMono.length);
  }
}
