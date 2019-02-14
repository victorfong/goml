package ml

import "math"
// import "fmt"

type Loss interface {
  Loss([]float64, []float64) float64
  Dloss([]float64, []float64) []float64
}

type SquareError struct{}

func (s SquareError) Loss(expected []float64, input []float64) float64{
  n := len(expected)

  output := 0.0

  for i:=0; i<n; i++ {
    output += math.Pow((expected[i] - input[i]), 2.0) * 0.5
  }

  return output
}

func (s SquareError) Dloss(expected []float64, input []float64) []float64{
  n := len(expected)

  output := make([]float64, n)

  return output
}



type CrossEntropy struct{}

func (c CrossEntropy) Loss(expected []float64, input []float64) float64{
  n := len(expected)

  output := 0.0

  for i:=0; i<n; i++ {
    output += expected[i] * math.Log(input[i])
  }

  output *= -1
  return output
}

func (c CrossEntropy) Dloss(expected []float64, input []float64) []float64{
  n := len(expected)

  output := make([]float64, n)

  for i:=0; i<n; i++ {
    output[i] = -1.0 * (expected[i] * (1 / input[i]) +
      (1 - expected[i]) * (1 / (1 - input[i])))
  }

  return output
}
