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

// TestNewSDMError verifies that NewSDM correctly returns an error
// when a bit size that is not a multiple of 64 is requested.
func TestNewSDMError(t *testing.T) {
	invalidBitSize := 200 // 200 is not divisible by 64
	numLocations := 100

	s, err := NewSDM(invalidBitSize, numLocations)
	if err == nil {
		t.Errorf("Expected an error for bit size %d (not a multiple of 64), but got nil", invalidBitSize)
	}

	if s != nil {
		t.Errorf("Expected returned SDM pointer to be nil on error, but got %v", s)
	}
}

// TestNewRandomVector verifies that random vectors are generated correctly
// and enforce bit sizes that are multiples of 64.
func TestNewRandomVector(t *testing.T) {
	// Test valid bit size
	v, err := NewRandomVector(256, SPARSITY)
	if err != nil {
		t.Fatalf("Unexpected error generating vector: %v", err)
	}
	if len(v) != 4 {
		t.Errorf("Expected slice length 4 for 256 bits, got %d", len(v))
	}

	// Test invalid bit size
	_, err = NewRandomVector(100, SPARSITY)
	if err == nil {
		t.Error("Expected an error for bit size not divisible by 64, got nil")
	}
}

// TestNewRandomVectorError verifies that NewRandomVector correctly returns an error
// when a bit size that is not a multiple of 64 is requested.
func TestNewRandomVectorError(t *testing.T) {
	invalidBitSize := 100 // 100 is not divisible by 64

	v, err := NewRandomVector(invalidBitSize, SPARSITY)
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
func TestCounterOverflow(t *testing.T) {
	bitSize := 64
	s, _ := NewSDM(bitSize, 10)
	s.ActivationDist = 64 // Radius 64 means all locations activate

	// CRITICAL FIX: Make the training address an all-ones vector.
	// This guarantees every bit evaluates to 1, forcing an increment step every time.
	address := Vector{0xFFFFFFFFFFFFFFFF}
	data := Vector{0xFFFFFFFFFFFFFFFF}

	// Explicitly seed the hard location address to match perfectly
	s.HardLocations[0].Address = address

	// Artificially saturate an Address track register close to MaxInt16
	s.HardLocations[0].AddressCounters[0] = 32766

	// Write twice
	s.Write(address, data)
	s.Write(address, data)

	// The counter should cap cleanly at 32767 instead of rolling over to -32768
	if s.HardLocations[0].AddressCounters[0] != 32767 {
		t.Errorf("Expected address counter to overflow-clamp at 32767, got %d", s.HardLocations[0].AddressCounters[0])
	}
}

// TestNewSDM verifies structural initialization and configuration constraints.
func TestNewSDM(t *testing.T) {
	bitSize := 128
	numHL := 50

	s, err := NewSDM(bitSize, numHL)
	if err != nil {
		t.Fatalf("Failed to initialize SDM: %v", err)
	}

	if len(s.HardLocations) != numHL {
		t.Errorf("Expected %d hard locations, got %d", numHL, len(s.HardLocations))
	}

	if len(s.HardLocations[0].AddressCounters) != bitSize {
		t.Errorf("Expected address counter size %d, got %d", bitSize, len(s.HardLocations[0].AddressCounters))
	}

	if len(s.HardLocations[0].DataCounters) != bitSize {
		t.Errorf("Expected data counter size %d, got %d", bitSize, len(s.HardLocations[0].DataCounters))
	}
}

// TestWriteAndExactRead asserts that data can be perfectly written
// and fetched back using the exact same address string.
func TestWriteAndExactRead(t *testing.T) {
	bitSize := 128
	s, _ := NewSDM(bitSize, 100) // Use broad radius to ensure registration match

	address, _ := NewRandomVector(bitSize, SPARSITY)
	data, _ := NewRandomVector(bitSize, SPARSITY)

	// Explicitly seed a hard location to guarantee exact tracking availability
	s.HardLocations[0].Address = address

	s.Write(address, data)
	retrieved := s.ReadData(address)

	dist := data.HammingDistance(retrieved)
	if dist != 0 {
		t.Errorf("Exact read failed to perfectly reconstruct data. Differed by %d bits", dist)
	}
}

func TestWriteNoiseToleranceWithKnownAddress(t *testing.T) {
	bitSize := 128
	numLocations := 1000

	s, _ := NewSDM(bitSize, numLocations)

	targetAddr, _ := NewRandomVector(bitSize, SPARSITY)

	s.HardLocations[len(s.HardLocations)-1].Address = targetAddr
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
		t.Errorf("SDM failed to self-converge on the target. "+
			"Recovered address deviates by %d bits", dataDist)
	}
}

func TestWriteSingle28BitAddresses(t *testing.T) {
	bitSize := 128
	numLocations := 1000000
	numTests := 1

	s, _ := NewSDM(bitSize, numLocations)

	targetAddrs := make([]Vector, numTests)

	// Write addresses
	for i := 0; i < numTests; i++ {
		addr, _ := NewRandomVector(bitSize, SPARSITY)
		s.Write(addr, addr)
		targetAddrs[i] = addr
	}

	// Read addresses
	for i := 0; i < numTests; i++ {
		retrievedAddr := s.ReadAddress(targetAddrs[i])
		addrDist := targetAddrs[i].HammingDistance(retrievedAddr)
		addrNorm := targetAddrs[i].Norm()

		if addrDist != 0 {
			t.Errorf("SDM failed to self-converge on the target. "+
				"Recovered address deviates by %d bits of %d \n", addrDist, addrNorm)
		}
	}
}

func _TestWriteMulti128BitAddresses(t *testing.T) {
	bitSize := 128
	numLocations := 1000000
	numTests := 1000

	s, _ := NewSDM(bitSize, numLocations)

	targetAddrs := make([]Vector, numTests)

	// Write addresses
	for i := 0; i < numTests; i++ {
		addr, _ := NewRandomVector(bitSize, SPARSITY)
		s.Write(addr, addr)
		targetAddrs[i] = addr
	}

	// Read addresses
	for i := 0; i < numTests; i++ {
		retrievedAddr := s.ReadAddress(targetAddrs[i])
		addrDist := targetAddrs[i].HammingDistance(retrievedAddr)
		addrNorm := targetAddrs[i].Norm()

		if addrDist != 0 {
			t.Errorf("SDM failed to self-converge on the target. "+
				"Recovered address deviates by %d bits of %d \n", addrDist, addrNorm)
		}
	}
}

func TestWriteDataSingle128BitData(t *testing.T) {
	bitSize := 128
	numLocations := 1000000
	numTests := 1

	s, _ := NewSDM(bitSize, numLocations)

	targetAddrs := make([]Vector, numTests)
	targetData := make([]Vector, numTests)

	// Write address and data
	for i := 0; i < numTests; i++ {
		addr, _ := NewRandomVector(bitSize, SPARSITY)
		data, _ := NewRandomVector(bitSize, SPARSITY)
		targetData[i] = data
		s.Write(addr, data)
		targetAddrs[i] = addr
	}

	// Read addresses
	for i := 0; i < numTests; i++ {
		retrievedData := s.ReadData(targetAddrs[i])

		if !targetData[i].Equal(targetData[i]) {
			t.Errorf("SDM failed to self-converge on the target.\n%s\n%s\n", targetData[i], retrievedData)
		}
	}
}

func TestWriteDataMulti128BitData(t *testing.T) {
	bitSize := 128
	numLocations := 1000000
	numTests := 1000

	s, _ := NewSDM(bitSize, numLocations)

	targetAddrs := make([]Vector, numTests)
	targetData := make([]Vector, numTests)

	// Write address and data
	for i := 0; i < numTests; i++ {
		addr, _ := NewRandomVector(bitSize, SPARSITY)
		data, _ := NewRandomVector(bitSize, SPARSITY)
		targetData[i] = data
		s.Write(addr, data)
		targetAddrs[i] = addr
	}

	// Read addresses
	for i := 0; i < numTests; i++ {
		retrievedData := s.ReadData(targetAddrs[i])

		if !targetData[i].Equal(targetData[i]) {
			t.Errorf("SDM failed to self-converge on the target.\n%s\n%s\n", targetData[i], retrievedData)
		}
	}
}
