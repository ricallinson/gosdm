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

// InverseErf computes the inverse error function using Winitzki's highly accurate approximation.
func InverseErf(x float64) float64 {
	const a = 0.147
	if x == 0 {
		return 0
	}

	logTerm := math.Log(1.0 - x*x)
	term1 := 2.0/(math.Pi*a) + logTerm/2.0
	term2 := logTerm / a

	innerSqrt := math.Sqrt(term1*term1 - term2)
	val := math.Sqrt(innerSqrt - term1)

	if x < 0 {
		return -val
	}
	return val
}

// InverseNormalCDF (Probit function) converts a probability p into its corresponding Z-score.
func InverseNormalCDF(p float64) float64 {
	// Standard normal distribution translation to inverse error function
	return math.Sqrt(2.0) * InverseErf(2.0*p-1.0)
}

// CalculateRadiusDynamic scales the Z-score and radius automatically based on the bit size (n).
// It sets the target activation probability to p = 1 / n.
func CalculateOptimalRadius(bitSize int) int {
	// 1. Calculate target probability dynamically based on bit size
	targetP := 0.8 / float64(bitSize)

	// 2. Derive the exact Z-score for this probability (one-tailed)
	// We pass targetP to find the Z-score mapping the lower tail boundary
	z := -InverseNormalCDF(targetP)

	// 3. Apply the Gaussian approximation formula for the radius
	mean := float64(bitSize) / 2.0
	stdDev := math.Sqrt(float64(bitSize)) / 2.0

	radius := mean - (z * stdDev)
	return int(math.Floor(radius))
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

	radius := CalculateOptimalRadius(bitSize)
	fmt.Printf("NewSDM(BitSize: %d, Radius: %d, HardLocations: %d)\n", bitSize, radius, numHardLocations)

	return &SDM{
		BitSize:        bitSize,
		ActivationDist: radius,
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

	// fmt.Printf("Write() to %d HardLocations\n", counter)
}

func (s *SDM) sumCounters(counters []int) Vector {
	result := make(Vector, s.BitSize/64)

	for bitIdx := 0; bitIdx < s.BitSize; bitIdx++ {
		wordIdx := bitIdx / 64
		bitOffset := bitIdx % 64
		if counters[bitIdx] > 0 {
			result[wordIdx] |= (1 << bitOffset)
		}
	}

	return result
}

func (s *SDM) read(address Vector, readData bool) Vector {
	sumCounters := make([]int, s.BitSize)
	activatedCount := 0

	for i := range s.HardLocations {
		if s.HardLocations[i].Address.HammingDistance(address) <= s.ActivationDist {
			activatedCount++
			for bitIdx := 0; bitIdx < s.BitSize; bitIdx++ {
				if readData {
					sumCounters[bitIdx] += int(s.HardLocations[i].DataCounters[bitIdx])
				} else {
					sumCounters[bitIdx] += int(s.HardLocations[i].AddressCounters[bitIdx])
				}
			}
		}
	}

	if activatedCount == 0 {
		return make(Vector, s.BitSize/64)
	}

	return s.sumCounters(sumCounters)
}

// ReadAddress retrieves the next convergence address coordinate from the Address track.
func (s *SDM) ReadAddress(address Vector) Vector {
	return s.read(address, false)
}

// ReadData retrieves the structural payload token from the independent Data track.
func (s *SDM) ReadData(address Vector) Vector {
	return s.read(address, true)
}
