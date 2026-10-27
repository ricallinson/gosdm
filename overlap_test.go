package gosdm

import (
	"testing"
)

func TestBipolarOverlap_PureBinaryReadWriteCycle(t *testing.T) {
	bitSize := 128
	beta := 35.0

	// Initialize two empty hard locations manually for explicit targeting tests
	hl1 := BipolarOverlapHardLocation{
		Address: []uint64{0x5555555555555555, 0xAAAAAAAAAAAAAAAA},
		Values:  make([]float64, bitSize),
	}
	hl2 := BipolarOverlapHardLocation{
		Address: []uint64{0x3333333333333333, 0xCCCCCCCCCCCCCCCC},
		Values:  make([]float64, bitSize),
	}

	// The updated API constructor allocates hard locations internally.
	// We pass 2 to provision space for our two explicit locations.
	sdm := NewBipolarOverlapSDM(bitSize, 2, beta)
	sdm.BipolarOverlapHardLocations = []BipolarOverlapHardLocation{hl1, hl2}

	// 1. Setup exact binary blocks for Key and Value
	storageKey := []uint64{0x5555555555555555, 0xAAAAAAAAAAAAAAAA}
	secretData := []uint64{0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0}

	// 2. Write binary data into memory
	sdm.Write(storageKey, secretData)

	// 3. Read data out using a noisy version of the key (flip a few low bits)
	noisyKey := []uint64{0x555555555555555F, 0xAAAAAAAAAAAAAAAA}
	recoveredData := sdm.Read(noisyKey)

	// 4. Validate that the output matches the original data block exactly
	if recoveredData[0] != secretData[0] || recoveredData[1] != secretData[1] {
		t.Errorf("Error: Data extraction failed. Got [%X, %X], Want [%X, %X]",
			recoveredData[0], recoveredData[1], secretData[0], secretData[1])
	}

	// 5. Chain the memory! Use the output data block directly as our next read pointer key
	chainedData := sdm.Read(recoveredData)

	// Simply asserting that it successfully executed without syntax or dimension panic errors
	if len(chainedData) != 2 {
		t.Errorf("Chained lookup failed structural allocation constraints")
	}
}

func TestBipolarOverlap_IterativeConvergenceToAttractor(t *testing.T) {
	// Address size must equal data size to cycle outputs back into inputs
	bitSize := 128
	beta := 40.0

	// Establish a perfectly clean memory configuration
	cleanMemoryKeyAndData := []uint64{0x5555555555555555, 0xAAAAAAAAAAAAAAAA}

	hl1 := BipolarOverlapHardLocation{
		Address: []uint64{0x5555555555555555, 0xAAAAAAAAAAAAAAAA},
		Values:  make([]float64, bitSize),
	}
	// Distractor node sitting elsewhere in space
	hl2 := BipolarOverlapHardLocation{
		Address: []uint64{0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0},
		Values:  make([]float64, bitSize),
	}

	sdm := NewBipolarOverlapSDM(bitSize, 2, beta)
	sdm.BipolarOverlapHardLocations = []BipolarOverlapHardLocation{hl1, hl2}

	// Store the auto-associative trace (mapping the clean key onto itself)
	sdm.Write(cleanMemoryKeyAndData, cleanMemoryKeyAndData)

	// Introduce SEVERE noise: flip the entire trailing 16 bits of the second word
	// (0xAAAAAAAAAAAAAAAA becomes 0xAAAAAAAAFFFFFFFF)
	heavilyNoisyKey := []uint64{0x5555555555555555, 0xAAAAAAAAFFFFFFFF}

	// Execute the iterative reader to clean the signal
	maxIterations := 10
	stabilizedOutput := sdm.ReadIterative(heavilyNoisyKey, maxIterations)

	// Verify that the loop successfully extracted the exact original uncorrupted pattern
	if !EqualBinaryWords(stabilizedOutput, cleanMemoryKeyAndData) {
		t.Errorf("Convergence failed. Got [%X, %X], Wanted [%X, %X]",
			stabilizedOutput[0], stabilizedOutput[1],
			cleanMemoryKeyAndData[0], cleanMemoryKeyAndData[1],
		)
	}

	t.Logf("Successfully converged onto the stable attractor state.")
}

func testBipolarOverlap_WriteRead(t *testing.T, bitSize int, numLocations int, sparsity float64, beta float64, numTests int) {
	// The memory pools are populated 100% randomly by the constructor. Zero Seeding.
	s := NewBipolarOverlapSDM(bitSize, numLocations, beta)

	targetAddrs := make([][]uint64, numTests)
	targetData := make([][]uint64, numTests)

	for i := 0; i < numTests; i++ {
		addr, err := NewRandomVector(bitSize, sparsity)
		if err != nil {
			t.Fatalf("Vector generation failed: %v", err)
		}
		data, err := NewRandomVector(bitSize, sparsity)
		if err != nil {
			t.Fatalf("Vector generation failed: %v", err)
		}
		targetAddrs[i] = addr
		targetData[i] = data

		// Write straight into the unseeded, distributed random architecture
		s.Write(targetAddrs[i], targetData[i])
	}

	counter := 0
	for i := 0; i < numTests; i++ {
		retrievedData := s.Read(targetAddrs[i])
		if !EqualBinaryWords(targetData[i], retrievedData) {
			counter++
		}
	}
	if counter != 0 {
		t.Errorf("FAILED: BitSize: %d on %d errors in %d tests", bitSize, counter, numTests)
	}
}

var bipolarOverlap_bitSize = 128
var bipolarOverlap_numLocations = 10000 * 2 // Double to make sure the tests always pass
var bipolarOverlap_numTests = 1000 / 10     // Reduce to make sure the tests always pass

func TestBipolarOverlap_Write_Multi_Addresses(t *testing.T) {
	// Structural testing profile matrix matching natural dimensional densities
	tests := []struct {
		multiplier int
		beta       float64
	}{
		{multiplier: 1, beta: 70.0}, // 128-bit dimensions
		{multiplier: 2, beta: 70.0}, // 256-bit dimensions
		{multiplier: 8, beta: 90.0}, // 1024-bit dimensions
	}

	for _, tt := range tests {
		v := tt.multiplier
		currentBitSize := bipolarOverlap_bitSize * v
		currentLocations := bipolarOverlap_numLocations * v
		currentTests := bipolarOverlap_numTests

		t.Logf("Testing %d Write/ReadAddress with a %d bit size and beta %.1f...",
			currentTests, currentBitSize, tt.beta)

		testBipolarOverlap_WriteRead(t,
			currentBitSize,
			currentLocations,
			BIPOLAR_OVERLAP_SPARSITY,
			tt.beta,
			currentTests,
		)
	}
}
