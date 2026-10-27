package gosdm

import (
	"fmt"
	"math"
	"math/bits"
)

var BIPOLAR_OVERLAP_SPARSITY = 50.0

type BipolarOverlapHardLocation struct {
	Address []uint64
	Values  []float64
}

type BipolarOverlapSDM struct {
	BipolarOverlapHardLocations []BipolarOverlapHardLocation
	AddressSizeBits             int
	DataSizeBits                int
	Beta                        float64
}

func NewBipolarOverlapSDM(bitSize int, numHardLocations int, beta float64) *BipolarOverlapSDM {
	hl := make([]BipolarOverlapHardLocation, numHardLocations)
	for i := 0; i < numHardLocations; i++ {
		addr, err := NewRandomVector(bitSize, BIPOLAR_OVERLAP_SPARSITY)
		if err != nil {
			return nil
		}
		hl[i] = BipolarOverlapHardLocation{
			Address: addr,
			Values:  make([]float64, bitSize),
		}
	}
	fmt.Printf("NewBipolarOverlapSDM(BitSize: %d, HardLocations: %d, Beta: %f)\n", bitSize, numHardLocations, beta)
	return &BipolarOverlapSDM{
		BipolarOverlapHardLocations: hl,
		AddressSizeBits:             bitSize,
		DataSizeBits:                bitSize,
		Beta:                        beta,
	}
}

func CalculateCosineOverlap(a, b []uint64, totalBits int) float64 {
	hammingDist := 0
	for i := 0; i < len(a); i++ {
		hammingDist += bits.OnesCount64(a[i] ^ b[i])
	}
	overlap := float64(totalBits - (2 * hammingDist))
	return overlap / float64(totalBits)
}

// Write scans the random locations, determines the closest local match,
// and writes exclusively to that local neighborhood cluster.
func (sdm *BipolarOverlapSDM) Write(address []uint64, data []uint64) {
	cosines := make([]float64, len(sdm.BipolarOverlapHardLocations))
	maxCosine := -1.0

	// Pass 1: Find the absolute closest unseeded location available in the space
	for i := 0; i < len(sdm.BipolarOverlapHardLocations); i++ {
		cosines[i] = CalculateCosineOverlap(address, sdm.BipolarOverlapHardLocations[i].Address, sdm.AddressSizeBits)
		if cosines[i] > maxCosine {
			maxCosine = cosines[i]
		}
	}

	// Threshold is relative to the best local peak found, not an arbitrary 1.0
	threshold := maxCosine - (23.0 / sdm.Beta)

	// Pass 2: Update the counters for the relative neighborhood cluster
	for i := 0; i < len(sdm.BipolarOverlapHardLocations); i++ {
		if cosines[i] < threshold {
			continue
		}

		// Subtracting maxCosine prevents floating-point exponential overflow
		w := math.Exp(sdm.Beta * (cosines[i] - maxCosine))

		for j := 0; j < sdm.DataSizeBits; j++ {
			wordIdx := j / 64
			bitIdx := uint(j % 64)

			bit := (data[wordIdx] >> bitIdx) & 1
			var bipolarVal float64
			if bit == 1 {
				bipolarVal = 1.0
			} else {
				bipolarVal = -1.0
			}

			sdm.BipolarOverlapHardLocations[i].Values[j] += w * bipolarVal
		}
	}
}

// Read extracts data by pooling weights relative to the maximum local neighborhood match.
func (sdm *BipolarOverlapSDM) Read(address []uint64) []uint64 {
	cosines := make([]float64, len(sdm.BipolarOverlapHardLocations))
	maxCosine := -1.0

	for i := 0; i < len(sdm.BipolarOverlapHardLocations); i++ {
		cosines[i] = CalculateCosineOverlap(address, sdm.BipolarOverlapHardLocations[i].Address, sdm.AddressSizeBits)
		if cosines[i] > maxCosine {
			maxCosine = cosines[i]
		}
	}

	threshold := maxCosine - (23.0 / sdm.Beta)
	weights := make([]float64, len(sdm.BipolarOverlapHardLocations))
	sumWeights := 0.0

	for i := 0; i < len(sdm.BipolarOverlapHardLocations); i++ {
		if cosines[i] < threshold {
			continue
		}
		weights[i] = math.Exp(sdm.Beta * (cosines[i] - maxCosine))
		sumWeights += weights[i]
	}

	numWords := (sdm.DataSizeBits + 63) / 64
	recoveredBinary := make([]uint64, numWords)

	if sumWeights == 0 {
		return recoveredBinary
	}

	accumulators := make([]float64, sdm.DataSizeBits)
	for i, location := range sdm.BipolarOverlapHardLocations {
		if weights[i] == 0.0 {
			continue
		}

		normalizedWeight := weights[i] / sumWeights
		for j := 0; j < sdm.DataSizeBits; j++ {
			accumulators[j] += normalizedWeight * location.Values[j]
		}
	}

	for j := 0; j < sdm.DataSizeBits; j++ {
		if accumulators[j] > 0 {
			wordIdx := j / 64
			bitIdx := uint(j % 64)
			recoveredBinary[wordIdx] |= (1 << bitIdx)
		}
	}

	return recoveredBinary
}

func EqualBinaryWords(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (sdm *BipolarOverlapSDM) ReadIterative(initialNoisyAddress []uint64, maxIterations int) []uint64 {
	if maxIterations <= 0 {
		maxIterations = 20
	}
	currentAddress := make([]uint64, len(initialNoisyAddress))
	copy(currentAddress, initialNoisyAddress)

	for i := 0; i < maxIterations; i++ {
		nextAddress := sdm.Read(currentAddress)
		if EqualBinaryWords(currentAddress, nextAddress) {
			return currentAddress
		}
		currentAddress = nextAddress
	}
	return currentAddress
}
