#include "FastPitchDetector.h"
#include <cmath>
#include <limits>

/*=============================================================================
   Fast Pitch Detection using YIN Algorithm
   
   Based on "YIN, a fundamental frequency estimator for speech and music"
   by Alain de Cheveigne and Hideki Kawahara
   
   The YIN algorithm provides highly accurate and stable pitch detection
   for musical notes through:
   - Cumulative Mean Normalized Difference Function (CMNDF)
   - Parabolic interpolation for sub-sample accuracy
   - Adaptive thresholding for robust period estimation
   
   Superior to autocorrelation for guitar tuning applications.
=============================================================================*/

// Helper function
template <typename T>
constexpr T FastPitchDetector::smallest_pow2(T n, T m)
{
	return (m < n) ? smallest_pow2(n, m << 1) : m;
}

// FastPitchDetector implementation
FastPitchDetector::FastPitchDetector(float minFreq, float maxFreq)
	: minFreq(minFreq)
	, maxFreq(maxFreq)
	, bufIndex(0)
	, bufferSize(0)
	, currentPitch(0.0f)
	, muteOutput(1.0f)
	, minPeriod(0)
	, maxPeriod(0)
	, samplesProcessed(0)
	, updateInterval(4096)
	, currentNote(-1)
	, targetFreq(0.0f)
	, centsOff(0.0f)
	, noteConfidence(0.0f)
	, pitchHistoryIndex(0)
	, smoothedPitch(0.0f)
{
	Name = "FastPitchDetector";
	Description = "Fast guitar tuner using YIN pitch detection";

	auto& muteParam = AddParameter();
	muteParam.Name = "Mute";
	muteParam.SourceVariable = &muteOutput;
	muteParam.IsAdvanced = true;
	muteParam.ParameterType = PARAMETER_TYPE_BOOL;
	muteParam.MinValue = 0;
	muteParam.MaxValue = 1;
	muteParam.DefaultValue = 1;
	muteParam.Description = "Mute audio output";

	auto& pitchParam = AddParameter();
	pitchParam.Name = "Pitch";
	pitchParam.SourceVariable = &currentPitch;
	pitchParam.IsOutput = true;
	pitchParam.MinValue = 0;
	pitchParam.MaxValue = 10000;
	pitchParam.DefaultValue = 0;
	pitchParam.Description = "Detected pitch frequency";

	auto& centsParam = AddParameter();
	centsParam.Name = "Cents";
	centsParam.SourceVariable = &centsOff;
	centsParam.IsOutput = true;
	centsParam.MinValue = -50;
	centsParam.MaxValue = 50;
	centsParam.DefaultValue = 0;
	centsParam.Description = "Cents deviation from target note";

	auto& noteParam = AddParameter();
	noteParam.Name = "Note";
	noteParam.SourceVariable = &targetFreq;
	noteParam.IsOutput = true;
	noteParam.MinValue = 0;
	noteParam.MaxValue = 10000;
	noteParam.DefaultValue = 0;
	noteParam.Description = "Target note frequency";

	auto& confidenceParam = AddParameter();
	confidenceParam.Name = "Confidence";
	confidenceParam.SourceVariable = &noteConfidence;
	confidenceParam.IsOutput = true;
	confidenceParam.MinValue = 0;
	confidenceParam.MaxValue = 1;
	confidenceParam.DefaultValue = 0;
	confidenceParam.Description = "Note lock confidence (0-1)";
}

void FastPitchDetector::init(int newSamplingFreq)
{
	StompBox::init(newSamplingFreq);

	// Calculate period ranges based on frequency limits
	minPeriod = static_cast<std::size_t>(samplingFreq / maxFreq);
	maxPeriod = static_cast<std::size_t>(samplingFreq / minFreq);

	// Buffer needs to be at least 2x the maximum period (power of 2)
	bufferSize = smallest_pow2(maxPeriod * 2);
	buffer.resize(bufferSize, 0.0f);
	
	bufIndex = 0;
	currentPitch = 0.0f;
	samplesProcessed = 0;
	
	// Initialize pitch tracking
	pitchHistory.resize(5, 0.0f);  // Keep last 5 readings for faster response
	pitchHistoryIndex = 0;
	smoothedPitch = 0.0f;
	currentNote = -1;
	targetFreq = 0.0f;
	centsOff = 0.0f;
	noteConfidence = 0.0f;
	
	// Update pitch detection ~30 times per second for responsive tuning
	updateInterval = samplingFreq / 30;
}

float FastPitchDetector::detectPitch()
{
	// YIN algorithm - superior pitch detection for musical notes
	// Based on "YIN, a fundamental frequency estimator for speech and music" by Cheveigne & Kawahara
	
	std::vector<float> difference(bufferSize / 2, 0.0f);
	std::vector<float> cumulativeMean(bufferSize / 2, 0.0f);
	
	// Step 1: Calculate difference function (autocorrelation-based)
	for (std::size_t tau = 1; tau < bufferSize / 2; ++tau)
	{
		float sum = 0.0f;
		for (std::size_t i = 0; i < bufferSize / 2; ++i)
		{
			float delta = buffer[i] - buffer[i + tau];
			sum += delta * delta;
		}
		difference[tau] = sum;
	}
	
	// Step 2: Calculate cumulative mean normalized difference function (CMNDF)
	cumulativeMean[0] = 1.0f;
	float runningSum = 0.0f;
	
	for (std::size_t tau = 1; tau < bufferSize / 2; ++tau)
	{
		runningSum += difference[tau];
		cumulativeMean[tau] = (runningSum > 0.0f) ? (difference[tau] * tau / runningSum) : 1.0f;
	}
	
	// Step 3: Absolute threshold - find first minimum below threshold
	const float YIN_THRESHOLD = 0.10f;  // Lower = more strict/accurate, 0.10-0.15 good for guitar
	std::size_t tau = minPeriod;
	
	// Find first valley below threshold
	while (tau < maxPeriod)
	{
		if (cumulativeMean[tau] < YIN_THRESHOLD)
		{
			// Found a candidate, but check if it's a local minimum
			while (tau + 1 < maxPeriod && cumulativeMean[tau + 1] < cumulativeMean[tau])
			{
				tau++;
			}
			break;
		}
		tau++;
	}
	
	// If no good candidate found, find the global minimum in the valid range
	if (tau >= maxPeriod)
	{
		tau = minPeriod;
		float minVal = cumulativeMean[tau];
		
		for (std::size_t t = minPeriod + 1; t < maxPeriod; ++t)
		{
			if (cumulativeMean[t] < minVal)
			{
				minVal = cumulativeMean[t];
				tau = t;
			}
		}
		
		// If global minimum is still too high, signal is likely not periodic
		if (minVal > 0.5f)
			return 0.0f;
	}
	
	// Step 4: Parabolic interpolation for sub-sample accuracy
	float betterTau = static_cast<float>(tau);
	
	if (tau > 0 && tau < bufferSize / 2 - 1)
	{
		float s0 = cumulativeMean[tau - 1];
		float s1 = cumulativeMean[tau];
		float s2 = cumulativeMean[tau + 1];
		
		// Parabolic interpolation formula
		float adjustment = 0.5f * (s0 - s2) / (s0 - 2.0f * s1 + s2);
		if (std::isfinite(adjustment) && std::abs(adjustment) < 1.0f)
		{
			betterTau = tau + adjustment;
		}
	}
	
	// Convert period to frequency
	if (betterTau <= 0.0f)
		return 0.0f;
		
	float est_freq = samplingFreq / betterTau;
	
	// Validate frequency is in expected range
	if (est_freq < minFreq || est_freq > maxFreq)
		return 0.0f;
	
	// Calculate confidence from CMNDF value (lower CMNDF = higher confidence)
	float periodicity = 1.0f - cumulativeMean[tau];
	
	// Only return pitch if confidence is high enough
	if (periodicity < 0.5f)
		return 0.0f;

	return est_freq;
}

void FastPitchDetector::compute(int count, float* input0, float* output0)
{
	// Process input samples
	for (int i = 0; i < count; i++)
	{
		float val = input0[i];
		
		// Add to circular buffer
		buffer[bufIndex] = val;
		
		bufIndex++;
		if (bufIndex >= bufferSize)
			bufIndex = 0;

		// Output (muted or pass-through)
		output0[i] = (muteOutput == 1.0f) ? 0.0f : input0[i];
	}

	// Update pitch detection at regular intervals (~30 times per second)
	samplesProcessed += count;
	if (samplesProcessed >= updateInterval)
	{
		samplesProcessed = 0;
		float detectedPitch = detectPitch();
		if (detectedPitch > 0.0f)
		{
			updateNoteTracking(detectedPitch);
		}
	}
}

int FastPitchDetector::frequencyToMidiNote(float freq)
{
	if (freq <= 0.0f)
		return -1;
	
	// MIDI note = 69 + 12 * log2(freq / 440)
	float noteFloat = 69.0f + 12.0f * std::log2(freq / 440.0f);
	return static_cast<int>(std::round(noteFloat));
}

float FastPitchDetector::midiNoteToFrequency(int midiNote)
{
	if (midiNote < 0)
		return 0.0f;
	
	// freq = 440 * 2^((note - 69) / 12)
	return 440.0f * std::pow(2.0f, (midiNote - 69) / 12.0f);
}

float FastPitchDetector::calculateCents(float freq, float targetFreq)
{
	if (freq <= 0.0f || targetFreq <= 0.0f)
		return 0.0f;
	
	// cents = 1200 * log2(freq / targetFreq)
	return 1200.0f * std::log2(freq / targetFreq);
}

void FastPitchDetector::updateNoteTracking(float detectedPitch)
{
	// Add to pitch history for smoothing
	pitchHistory[pitchHistoryIndex] = detectedPitch;
	pitchHistoryIndex = (pitchHistoryIndex + 1) % pitchHistory.size();
	
	// Calculate median of recent pitches (more robust than average)
	std::vector<float> sortedHistory = pitchHistory;
	std::sort(sortedHistory.begin(), sortedHistory.end());
	float medianPitch = sortedHistory[sortedHistory.size() / 2];
	
	// Calculate pitch variation to adjust smoothing
	float minPitch = sortedHistory.front();
	float maxPitch = sortedHistory.back();
	float variation = (maxPitch - minPitch) / ((maxPitch + minPitch) * 0.5f);
	
	// Adaptive smoothing - smoother when stable, more responsive when changing
	float smoothing = (variation < 0.01f) ? 0.1f : 0.25f;
	
	if (smoothedPitch == 0.0f)
		smoothedPitch = medianPitch;
	else
		smoothedPitch = smoothing * medianPitch + (1.0f - smoothing) * smoothedPitch;
	
	// Always update current pitch for responsive display
	currentPitch = smoothedPitch;
	
	// Determine MIDI note from smoothed pitch
	int detectedNote = frequencyToMidiNote(smoothedPitch);
	
	if (detectedNote < 0)
	{
		targetFreq = 0.0f;
		centsOff = 0.0f;
		return;
	}
	
	// Note locking with hysteresis
	const float lockThresholdCents = 20.0f;   // Need to be within 20 cents to lock
	const float unlockThresholdCents = 40.0f; // Must drift 40 cents away to unlock
	
	float detectedFreq = midiNoteToFrequency(detectedNote);
	float cents = calculateCents(smoothedPitch, detectedFreq);
	
	if (currentNote == -1)
	{
		// No note locked yet - lock to detected note if confident
		if (std::abs(cents) < lockThresholdCents)
		{
			currentNote = detectedNote;
			targetFreq = detectedFreq;
			noteConfidence = 1.0f;
		}
		else
		{
			// Not locked yet - show the closest note anyway
			targetFreq = detectedFreq;
		}
	}
	else if (currentNote == detectedNote)
	{
		// Same note - increase confidence
		noteConfidence = std::min(1.0f, noteConfidence + 0.15f);
		targetFreq = detectedFreq;
	}
	else
	{
		// Different note detected
		float centsFromCurrent = calculateCents(smoothedPitch, targetFreq);
		
		if (std::abs(centsFromCurrent) > unlockThresholdCents)
		{
			// Drifted too far - check if new note is better
			if (std::abs(cents) < lockThresholdCents)
			{
				// Switch to new note
				currentNote = detectedNote;
				targetFreq = detectedFreq;
				noteConfidence = 0.5f;
			}
			else
			{
				// Reduce confidence but keep current note
				noteConfidence = std::max(0.0f, noteConfidence - 0.3f);
				if (noteConfidence < 0.2f)
				{
					// Lost confidence - unlock and show new note
					currentNote = -1;
					targetFreq = detectedFreq;
				}
			}
		}
		else
		{
			// Still close enough to current note - reduce confidence slightly
			noteConfidence = std::max(0.2f, noteConfidence - 0.1f);
		}
	}
	
	// Always calculate cents from target frequency
	if (targetFreq > 0.0f)
	{
		centsOff = calculateCents(smoothedPitch, targetFreq);
		// Clamp cents to reasonable display range
		centsOff = std::max(-50.0f, std::min(50.0f, centsOff));
	}
	else
	{
		centsOff = 0.0f;
	}
}
