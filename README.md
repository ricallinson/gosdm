# Sparse Distributed Memory

## Test

    go test -covermode=atomic -coverprofile=coverage.out ./...
    go tool cover -html=coverage.out
