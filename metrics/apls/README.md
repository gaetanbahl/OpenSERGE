# APLS (Average Path Length Similarity) Metric

> **Note**: This code is adapted from the [Sat2Graph](https://github.com/songtaohe/Sat2Graph) repository with fixes for compatibility with modern Go versions.

## Overview

The APLS metric evaluates road graph extraction quality by measuring the similarity of path lengths between ground truth and predicted graphs. It is implemented in Go for performance.

## Requirements

### Go (2026)

- Go 1.16 or later (tested with Go 1.21+)
- The code uses Go modules for dependency management

### Python

- Python 3.7+
- See `requirements.txt` for Python dependencies

## Installation

1. **Install Go dependencies:**
```bash
cd metrics/apls
go mod download
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Convert graph files to JSON format

If your graphs are in pickle format (Python dictionary), first convert them to JSON:

```bash
python convert.py example/gt.p example/gt.json
python convert.py example/prop.p example/prop.json
```

### Step 2: Run APLS metric

```bash
go run main.go example/gt.json example/prop.json apls_result.txt
```

Or build and run:
```bash
go build -o apls main.go
./apls example/gt.json example/prop.json apls_result.txt
```

## Configuration

The APLS implementation is configured for 2048×2048 meter tiles by default. To modify parameters for different tile sizes, edit lines 15-25 in `main.go`:

```go
const (
    lineStringMatchingDist = 15.0  // meters
    graphMatchingDist = 100.0      // meters
    // ... other parameters
)
```

## Output Format

The output file contains three scores:
```
score1 score2 final_apls_score
```

Where `final_apls_score` is the primary APLS metric (third value).

## Graph Format

Input JSON files should contain a graph in the following format:
```json
{
  "x1,y1": [["x2", "y2"], ["x3", "y3"]],
  "x2,y2": [["x1", "y1"]],
  ...
}
```

Where each key is a vertex coordinate (as a string "x,y") and the value is a list of neighboring vertices.

## Dependencies

- `github.com/dhconnelly/rtreego` - Spatial indexing (automatically installed via go.mod)

## Troubleshooting

**Module errors**: If you encounter module-related errors, ensure you're using Go 1.16+ and run:
```bash
go mod tidy
```

**Import path issues**: The code has been updated to use the correct import path for `rtreego`.