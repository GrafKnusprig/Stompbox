#pragma once

#include "StompBox.h"
#include <vector>
#include <cstdint>
#include <algorithm>

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

class FastPitchDetector : public StompBox
{
private:
	std::size_t bufferSize;
	std::vector<float> buffer;
	std::size_t bufIndex;
	
	float currentPitch;
	float minFreq;
	float maxFreq;
	std::size_t minPeriod;
	std::size_t maxPeriod;
	
	float muteOutput;
	int samplesProcessed;
	int updateInterval;
	
	// Note tracking and smoothing
	int currentNote;       // MIDI note number of locked note
	float targetFreq;      // Frequency of the locked note
	float centsOff;        // Cents deviation from target note
	float noteConfidence;  // Confidence in current note lock
	std::vector<float> pitchHistory;  // Recent pitch readings
	int pitchHistoryIndex;
	float smoothedPitch;   // Exponentially smoothed pitch
	
	// Helper functions
	template <typename T>
	static constexpr T smallest_pow2(T n, T m = 1);
	
	float detectPitch();
	int frequencyToMidiNote(float freq);
	float midiNoteToFrequency(int midiNote);
	float calculateCents(float freq, float targetFreq);
	void updateNoteTracking(float detectedPitch);

public:
	FastPitchDetector(float minFreq = 50.0f, float maxFreq = 500.0f);
	virtual ~FastPitchDetector() {}

	virtual void init(int samplingFreq);
	virtual void compute(int count, float* input, float* output);

	// Interface methods for TunerInterface.cs
	float GetCurrentPitch() { return currentPitch; }
	int GetCurrentNote() { return currentNote; }
	float GetCentsOff() { return centsOff; }
	float GetTargetFreq() { return targetFreq; }
	float GetNoteConfidence() { return noteConfidence; }
};
