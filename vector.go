package gosdm

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"math/rand/v2"
)

type Vector []uint64

// String converts the high-dimensional Vector into a continuous string of '1's and '0's.
func (v Vector) String() string {
	var result string

	// Process words in reverse order if you want the highest significant word
	// at the left of the string, or forward order (0 to len-1) for sequential indexing.
	// Here we read sequentially from the first word to the last word.
	for _, word := range v {
		// Print each uint64 word as a 64-character wide zero-padded binary string
		result += fmt.Sprintf("%064b", word)
	}

	return result
}

// HammingDistance calculates the number of differing bits between two vectors.
func (v Vector) HammingDistance(other Vector) int {
	dist := 0
	for i := range v {
		dist += bits.OnesCount64(v[i] ^ other[i])
	}
	return dist
}

// Returns the count of on bits.
func (v Vector) Norm() int {
	count := 0
	for i := range v {
		count += bits.OnesCount64(v[i])
	}
	return count
}

// Returns 'true' if the given Vector is a bit match.
func (v Vector) Equal(other Vector) bool {
	if len(v) != len(other) {
		fmt.Println(len(v), len(other))
		return false
	}
	for i := range v {
		if v[i] != other[i] {
			fmt.Println(v[i], other[i])
			return false
		}
	}
	return true
}

// NewRandomVector generates a binary Vector where each 64-bit word contains
// a specific percentage of bits randomly set to 1.
func NewRandomVector(bitSize int, percentage float64) (Vector, error) {
	if bitSize%64 != 0 {
		return nil, errors.New("bit size must be a multiple of 64")
	}

	numWords := bitSize / 64
	v := make(Vector, numWords)

	// Clamp percentage to valid bounds
	if percentage <= 0 {
		return v, nil // Returns a clean all-zeros vector
	}
	if percentage >= 100 {
		for i := 0; i < numWords; i++ {
			v[i] = math.MaxUint64 // All bits set to 1
		}
		return v, nil
	}

	// Calculate the exact number of bits to set to 1 out of 64
	numBitsToSet := int(math.Round(64.0 * (percentage / 100.0)))

	// Process each 64-bit word block independently
	for w := 0; w < numWords; w++ {
		// Create a slice of all possible bit positions (0 to 63)
		positions := make([]int, 64)
		for i := 0; i < 64; i++ {
			positions[i] = i
		}

		// Shuffle the positions randomly
		rand.Shuffle(len(positions), func(i, j int) {
			positions[i], positions[j] = positions[j], positions[i]
		})

		// Set the first 'numBitsToSet' randomly chosen positions to 1
		var result uint64
		for i := 0; i < numBitsToSet; i++ {
			result |= (uint64(1) << positions[i])
		}

		v[w] = result
	}

	return v, nil
}
