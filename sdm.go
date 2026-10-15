package gosdm

import (
	"errors"
	"fmt"
	"math"
)

var SPARSITY = 50.0

// HardLocation coordinates separate counter registers to protect data signals from cross-talk.
type HardLocation struct {
	Address         Vector
	AddressCounters []int16 // Dedicated entirely to address self-convergence (auto-association)
	DataCounters    []int16 // Dedicated entirely to storing arbitrary payloads (hetero-association)
}

// SDM implements the Sparse Distributed Memory system.
type SDM struct {
	BitSize        int
	ActivationDist int
	HardLocations  []HardLocation
}

// CalculateOptimalRadius calculates the max acceptable Hamming distance
// for an optimal target selection density of ~1.5% using the Z-score equation.
func CalculateOptimalRadius(bitSize int) int {
	n := float64(bitSize)

	// Mean (μ) and Standard Deviation (σ) of a random high-dimensional binary space
	mean := n / 2.0
	stdDev := math.Sqrt(n) / 2.0

	// A Z-score of -2.12 targets the optimal ~1.67% lower-tail firing probability
	zScore := 2.12

	radius := mean - (zScore * stdDev)

	// Round to the nearest whole integer bit coordinate
	return int(math.Round(radius))
}

// NewSDM initializes a new Sparse Distributed Memory instance with split counter allocations.
func NewSDM(bitSize int, numHardLocations int) (*SDM, error) {

	if bitSize%64 != 0 {
		return nil, errors.New("bit size must be a multiple of 64")
	}

	hl := make([]HardLocation, numHardLocations)
	for i := 0; i < numHardLocations; i++ {
		addr, err := NewRandomVector(bitSize, SPARSITY)
		if err != nil {
			return nil, err
		}
		hl[i] = HardLocation{
			Address:         addr,
			AddressCounters: make([]int16, bitSize),
			DataCounters:    make([]int16, bitSize),
		}
	}

	return &SDM{
		BitSize:        bitSize,
		ActivationDist: CalculateOptimalRadius(bitSize),
		HardLocations:  hl,
	}, nil
}

// Write trains both topologies simultaneously inside their respective channels without bleeding.
func (s *SDM) Write(address, data Vector) {
	counter := 0

	for i := range s.HardLocations {
		if s.HardLocations[i].Address.HammingDistance(address) <= s.ActivationDist {
			for bitIdx := 0; bitIdx < s.BitSize; bitIdx++ {
				wordIdx := bitIdx / 64
				bitOffset := bitIdx % 64

				// 1. Train the Address Track auto-associatively (storing address at address)
				addrBit := (address[wordIdx] >> bitOffset) & 1
				if addrBit == 1 {
					if s.HardLocations[i].AddressCounters[bitIdx] < 32767 {
						s.HardLocations[i].AddressCounters[bitIdx]++
					}
				} else {
					if s.HardLocations[i].AddressCounters[bitIdx] > -32768 {
						s.HardLocations[i].AddressCounters[bitIdx]--
					}
				}

				// 2. Train the Data Track hetero-associatively (storing data payload at address)
				dataBit := (data[wordIdx] >> bitOffset) & 1
				if dataBit == 1 {
					if s.HardLocations[i].DataCounters[bitIdx] < 32767 {
						s.HardLocations[i].DataCounters[bitIdx]++
					}
				} else {
					if s.HardLocations[i].DataCounters[bitIdx] > -32768 {
						s.HardLocations[i].DataCounters[bitIdx]--
					}
				}
			}
			counter++
		}
	}

	fmt.Printf("Write() to %d\n", counter)
}

// ReadAddress retrieves the next convergence address coordinate from the Address track.
func (s *SDM) ReadAddress(address Vector) Vector {
	counter := 0
	sumCounters := make([]int, s.BitSize)
	activatedCount := 0

	for i := range s.HardLocations {
		if s.HardLocations[i].Address.HammingDistance(address) <= s.ActivationDist {
			activatedCount++
			for bitIdx := 0; bitIdx < s.BitSize; bitIdx++ {
				sumCounters[bitIdx] += int(s.HardLocations[i].AddressCounters[bitIdx])
			}
			counter++
		}
	}

	result := make(Vector, s.BitSize/64)
	if activatedCount == 0 {
		return result
	}

	for bitIdx := 0; bitIdx < s.BitSize; bitIdx++ {
		wordIdx := bitIdx / 64
		bitOffset := bitIdx % 64
		if sumCounters[bitIdx] > 0 {
			result[wordIdx] |= (1 << bitOffset)
		}
	}

	fmt.Printf("ReadAddress() from %d\n", counter)

	return result
}

// ReadData retrieves the structural payload token from the independent Data track.
func (s *SDM) ReadData(address Vector) Vector {
	counter := 0
	sumCounters := make([]int, s.BitSize)
	activatedCount := 0

	for i := range s.HardLocations {
		if s.HardLocations[i].Address.HammingDistance(address) <= s.ActivationDist {
			activatedCount++
			for bitIdx := 0; bitIdx < s.BitSize; bitIdx++ {
				sumCounters[bitIdx] += int(s.HardLocations[i].DataCounters[bitIdx])
			}
			counter++
		}
	}

	result := make(Vector, s.BitSize/64)
	if activatedCount == 0 {
		return result
	}

	for bitIdx := 0; bitIdx < s.BitSize; bitIdx++ {
		wordIdx := bitIdx / 64
		bitOffset := bitIdx % 64
		if sumCounters[bitIdx] > 0 {
			result[wordIdx] |= (1 << bitOffset)
		}
	}

	fmt.Printf("ReadData() from %d\n", counter)

	return result
}

// ReadConverge drives loop stabilization across the address track, then reads out the data payload.
// func (s *SDM) Read(address Vector) Vector {
// 	x0 := make(Vector, len(address)) // Address state 2 cycles ago
// 	x1 := make(Vector, len(address)) // Address state 1 cycle ago
// 	x2 := make(Vector, len(address)) // Current address state

// 	copy(x1, address)
// 	maxSteps := 20

// 	for steps := 0; steps < maxSteps; steps++ {
// 		// Evaluate the address track to extract a cleaner address vector
// 		x2 = s.ReadAddress(x1)

// 		// Loop detection: Stop when the address tracks match a fixed point or period-2 oscillation
// 		if x1.HammingDistance(x2) == 0 || (steps > 0 && x0.HammingDistance(x2) == 0) {
// 			return s.ReadData(x2)
// 		}

// 		copy(x0, x1)
// 		copy(x1, x2)
// 	}

// 	return s.ReadData(x2)
// }
