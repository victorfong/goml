package main

import "math/rand"

type Layer interface{
  // Train()
  Init(inputUnits int)
  GetOutputUnits() int
}

type layer struct {
  inputUnits int
  outputUnits int
  weights [][]float64
  activation Activation
}

func (l layer) Init(inputUnits int) {
  l.inputUnits = inputUnits

  l.weights = make([][]float64, inputUnits)

  for i:=0; i<inputUnits; i++ {
    l.weights[i] = make([]float64, l.outputUnits)
    for j:=0; j<l.outputUnits; j++ {
      l.weights[i][j] = rand.Float64()
    }
  }
}

func NewLayer(unit int, activation Activation) Layer{
  return &layer{
    outputUnits: unit,
    activation: activation,
  }
}

func (l layer) GetOutputUnits() int {
  return l.outputUnits
}
