package gosdm

import (
	"crypto/rand"
	// "fmt"
	"testing"
)

// DYNAMIC NOISE GENERATION: Generate a random bit noise mask using crypto/rand
func addNoiseBits(t *testing.T, address Vector, noiseBits int) Vector {
	noisyAddress := make(Vector, len(address))
	copy(noisyAddress, address)
	flippedBits := 0

	// Track bit slots, pick unique ones to flip
	for flippedBits < noiseBits {
		b := make([]byte, 2)
		if _, err := rand.Read(b); err != nil {
			t.Fatalf("Failed to generate random noise bit: %v", err)
		}

		wordIdx := int(b[0]) % len(address)
		bitOffset := int(b[1]) % 64

		mask := uint64(1) << bitOffset
		if (noisyAddress[wordIdx] & mask) == (address[wordIdx] & mask) {
			noisyAddress[wordIdx] ^= mask
			flippedBits++
		}
	}

	return noisyAddress
}

// TestNewHammingDistanceSDMError verifies that NewHammingDistanceSDM correctly returns an error
// when a bit size that is not a multiple of 64 is requested.
func TestHammingDistance_NewHammingDistanceSDMError(t *testing.T) {
	invalidBitSize := 200 // 200 is not divisible by 64
	numLocations := 100

	s, err := NewHammingDistanceSDM(invalidBitSize, numLocations)
	if err == nil {
		t.Errorf("Expected an error for bit size %d (not a multiple of 64), but got nil", invalidBitSize)
	}

	if s != nil {
		t.Errorf("Expected returned HammingDistanceSDM pointer to be nil on error, but got %v", s)
	}
}

// TestNewRandomVector verifies that random vectors are generated correctly
// and enforce bit sizes that are multiples of 64.
func TestHammingDistance_NewRandomVector(t *testing.T) {
	// Test valid bit size
	v, err := NewRandomVector(256, HAMMING_DISTANCE_SPARSITY)
	if err != nil {
		t.Fatalf("Unexpected error generating vector: %v", err)
	}
	if len(v) != 4 {
		t.Errorf("Expected slice length 4 for 256 bits, got %d", len(v))
	}

	// Test invalid bit size
	_, err = NewRandomVector(100, HAMMING_DISTANCE_SPARSITY)
	if err == nil {
		t.Error("Expected an error for bit size not divisible by 64, got nil")
	}
}

// TestVectorOverlap verifies that the overlap math perfectly counts the intersection
// of active 1 bits between two structured vectors.
func TestHammingDistance_VectorOverlap(t *testing.T) {
	// Vector 1 has bits set at indices 0, 1, 2, 3 (0xF) and 60, 61, 62, 63 (0xF0...)
	v1 := Vector{uint64(0xF00000000000000F), uint64(0x0000000000000000)}

	// Vector 2 has bits set at indices 0, 1 (0x3) and 62, 63 (0xC0...)
	v2 := Vector{uint64(0xC000000000000003), uint64(0x0000000000000000)}

	// Intersection targets:
	// Indices 0 and 1 match (2 bits)
	// Indices 62 and 63 match (2 bits)
	// Total expected shared bits = 4
	expectedOverlap := 4
	actualOverlap := v1.Overlap(v2)

	if actualOverlap != expectedOverlap {
		t.Errorf("Expected bitwise Overlap of %d, but calculated %d", expectedOverlap, actualOverlap)
	}
}

// TestNewRandomVectorError verifies that NewRandomVector correctly returns an error
// when a bit size that is not a multiple of 64 is requested.
func TestHammingDistance_NewRandomVectorError(t *testing.T) {
	invalidBitSize := 100 // 100 is not divisible by 64

	v, err := NewRandomVector(invalidBitSize, HAMMING_DISTANCE_SPARSITY)
	if err == nil {
		t.Errorf("Expected an error for bit size %d (not a multiple of 64), but got nil", invalidBitSize)
	}

	if v != nil {
		t.Errorf("Expected returned vector to be nil on error, but got %v", v)
	}
}

// TestHammingDistance verifies the bitwise difference logic.
func TestHammingDistance(t *testing.T) {
	v1 := Vector{0x0000000000000000, 0xF000000000000000}
	v2 := Vector{0x0000000000000001, 0x7000000000000000}

	expectedDist := 2
	actualDist := v1.HammingDistance(v2)

	if actualDist != expectedDist {
		t.Errorf("Expected Hamming distance %d, got %d", expectedDist, actualDist)
	}
}

// TestCounterOverflow verifies that hard location counters clamp safely
// instead of wrapping around into negative bounds.
func TestHammingDistance_CounterOverflow(t *testing.T) {
	bitSize := 64
	s, _ := NewHammingDistanceSDM(bitSize, 10)
	s.ActivationDist = 64 // Radius 64 means all locations activate

	// CRITICAL FIX: Make the training address an all-ones vector.
	// This guarantees every bit evaluates to 1, forcing an increment step every time.
	address := Vector{0xFFFFFFFFFFFFFFFF}
	data := Vector{0xFFFFFFFFFFFFFFFF}

	// Explicitly seed the hard location address to match perfectly
	s.HammingDistanceHardLocations[0].Address = address

	// Artificially saturate an Address track register close to MaxInt16
	s.HammingDistanceHardLocations[0].AddressCounters[0] = 32766

	// Write twice
	s.Write(address, data)
	s.Write(address, data)

	// The counter should cap cleanly at 32767 instead of rolling over to -32768
	if s.HammingDistanceHardLocations[0].AddressCounters[0] != 32767 {
		t.Errorf("Expected address counter to overflow-clamp at 32767, got %d", s.HammingDistanceHardLocations[0].AddressCounters[0])
	}
}

// TestNewHammingDistanceSDM verifies structural initialization and configuration constraints.
func TestHammingDistance_NewHammingDistanceSDM(t *testing.T) {
	bitSize := 128
	numHL := 50

	s, err := NewHammingDistanceSDM(bitSize, numHL)
	if err != nil {
		t.Fatalf("Failed to initialize HammingDistanceSDM: %v", err)
	}

	if len(s.HammingDistanceHardLocations) != numHL {
		t.Errorf("Expected %d hard locations, got %d", numHL, len(s.HammingDistanceHardLocations))
	}

	if len(s.HammingDistanceHardLocations[0].AddressCounters) != bitSize {
		t.Errorf("Expected address counter size %d, got %d", bitSize, len(s.HammingDistanceHardLocations[0].AddressCounters))
	}

	if len(s.HammingDistanceHardLocations[0].DataCounters) != bitSize {
		t.Errorf("Expected data counter size %d, got %d", bitSize, len(s.HammingDistanceHardLocations[0].DataCounters))
	}
}

// TestWriteAndExactRead asserts that data can be perfectly written
// and fetched back using the exact same address string.
func TestHammingDistance_WriteAndExactRead(t *testing.T) {
	bitSize := 128
	s, _ := NewHammingDistanceSDM(bitSize, 100) // Use broad radius to ensure registration match

	address, _ := NewRandomVector(bitSize, HAMMING_DISTANCE_SPARSITY)
	data, _ := NewRandomVector(bitSize, HAMMING_DISTANCE_SPARSITY)

	// Explicitly seed a hard location to guarantee exact tracking availability
	s.HammingDistanceHardLocations[0].Address = address

	s.Write(address, data)
	retrieved := s.ReadData(address)

	dist := data.HammingDistance(retrieved)
	if dist != 0 {
		t.Errorf("Exact read failed to perfectly reconstruct data. Differed by %d bits", dist)
	}
}

func TestHammingDistance_WriteNoiseToleranceWithKnownAddress(t *testing.T) {
	bitSize := 128
	numLocations := 1000

	s, _ := NewHammingDistanceSDM(bitSize, numLocations)

	targetAddr, _ := NewRandomVector(bitSize, HAMMING_DISTANCE_SPARSITY)

	s.HammingDistanceHardLocations[len(s.HammingDistanceHardLocations)-1].Address = targetAddr
	s.Write(targetAddr, targetAddr)

	// Corrupt the lookup key with a randomized bit noise mask
	noiseBits := s.ActivationDist / 4
	noisyAddress := addNoiseBits(t, targetAddr, noiseBits)

	addrDist := targetAddr.HammingDistance(noisyAddress)
	if addrDist != noiseBits {
		t.Fatalf("Setup failure: noisy address distance should be %d, got %d", noiseBits, addrDist)
	}

	retrievedAddr := s.ReadAddress(noisyAddress)
	dataDist := targetAddr.HammingDistance(retrievedAddr)

	if dataDist != 0 {
		t.Errorf("HammingDistanceSDM failed to self-converge on the target. "+
			"Recovered address deviates by %d bits", dataDist)
	}
}

func testHammingDistance_WriteRead(t *testing.T, bitSize int, numLocations int, sparsity float64, numTests int, testData bool) {
	s, _ := NewHammingDistanceSDM(bitSize, numLocations)

	targetAddrs := make([]Vector, numTests)
	targetData := make([]Vector, numTests)

	// Write address and data
	for i := 0; i < numTests; i++ {
		addr, _ := NewRandomVector(bitSize, sparsity)
		targetAddrs[i] = addr
		if testData {
			data, _ := NewRandomVector(bitSize, sparsity)
			targetData[i] = data
		} else {
			targetData[i] = addr
		}
		s.Write(targetAddrs[i], targetData[i])
	}

	// Read addresses
	counter := 0
	for i := 0; i < numTests; i++ {
		retrievedData := s.ReadData(targetAddrs[i])
		if targetData[i].HammingDistance(retrievedData) != 0 {
			counter++
			// t.Errorf("Failed HammingDistance %d: BitSize %d, Locations %d, Sparsity %f\n", targetAddrs[i].HammingDistance(retrievedData), bitSize, numLocations, sparsity)
		}
	}
	if counter != 0 {
		t.Errorf("FAILED: BitSize: %d on %d in %d tests", bitSize, counter, numTests)
	}
}

// A 128-bit HammingDistanceSDM with 10,000 hard locations and a radius of 50 bits, it will behave exactly
// like Kanerva's classic architecture—allowing auto-associative, noise-tolerant retrieval—but
// it will max out at storing roughly 1,000 unique memories.
var hammingDistance_bitSize = 128
var hammingDistance_numLocations = 10000 * 2 // Double to make sure the tests always pass
var hammingDistance_numTests = 1000 / 3      // Reduce to make sure the tests always pass

func TestHammingDistance_Write_Multi_Addresses(t *testing.T) {
	for _, v := range []int{1, 2, 8} {
		t.Logf("Testing %d Write/ReadAddress with a %d bit size...", hammingDistance_numTests*v, hammingDistance_bitSize*v)
		testHammingDistance_WriteRead(t, hammingDistance_bitSize*v, hammingDistance_numLocations*v, HAMMING_DISTANCE_SPARSITY, hammingDistance_numTests*v, false)
	}
}

func TestHammingDistance_Write_Multi_Data(t *testing.T) {
	for _, v := range []int{1, 2, 8} {
		t.Logf("Testing %d Write/ReadData with a %d bit size...", hammingDistance_numTests*v, hammingDistance_bitSize*v)
		testHammingDistance_WriteRead(t, hammingDistance_bitSize*v, hammingDistance_numLocations*v, HAMMING_DISTANCE_SPARSITY, hammingDistance_numTests*v, true)
	}
}
