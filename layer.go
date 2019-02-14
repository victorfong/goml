package main

import "math/rand"
import "fmt"
import "time"

// type Layer interface{
//   Train([]float64) []float64
//   Init(inputUnits int)
//   GetOutputUnits() int
// }

type Layer struct {
  inputUnits int
  outputUnits int
  weights [][]float64
  activation Activation
}

func (l *Layer) Init(inputUnits int) {
  l.inputUnits = inputUnits

  l.weights = make([][]float64, inputUnits)
  s1 := rand.NewSource(time.Now().UnixNano())
  r1 := rand.New(s1)

  for i:=0; i<inputUnits; i++ {
    l.weights[i] = make([]float64, l.outputUnits)
    for j:=0; j<l.outputUnits; j++ {
      l.weights[i][j] = r1.Float64()
    }
  }

  // fmt.Println("Init Weight: ", l.weights)
}

func (l *Layer) Train(input []float64) []float64{
  a := [][]float64{
    input,
  }
  fmt.Println("Input: ", input)
  output := MatrixMultiply(a, l.weights)
  result := l.activation.Activate(output[0])

  return result
}

func NewLayer(unit int, activation Activation) *Layer{
  return &Layer{
    outputUnits: unit,
    activation: activation,
  }
}

func (l *Layer) GetOutputUnits() int {
  return l.outputUnits
}
