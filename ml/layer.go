package ml

import "math/rand"
import "fmt"
import "time"

// type Layer interface{
//   Train([]float64) []float64
//   Init(inputUnits int)
//   GetOutputUnits() int
// }

const LEARN_RATE float64 = 0.5

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

func (l *Layer) SetWeights(weights [][]float64) {
  l.weights = weights
}

func (l *Layer) GetWeights() [][]float64 {
  return l.weights
}

func (l *Layer) UpdateWeights(input[] float64, output []float64, expected []float64, lossFunction Loss) {
  delta := l.CalcDelta(input, output, expected, lossFunction)

  for i:=0; i<len(l.weights); i++ {
    for j:=0; j<len(l.weights[i]); j++ {
      fmt.Println("Original Weight: ", l.weights[i][j])
      fmt.Println("Learn Rate:", LEARN_RATE)
      fmt.Println("Delta:", delta[j])
      l.weights[i][j] = l.weights[i][j] - LEARN_RATE * delta[i]
      fmt.Println("New Weight:", l.weights[i][j])
    }
  }
}

func (l *Layer) CalcDelta(input[] float64, output []float64, expected []float64, lossFunction Loss) []float64{
  dLoss := lossFunction.Dloss(expected, output)
  dActivation := l.activation.DeActivate(output)

  delta := make([]float64, len(input))
  for i:=0; i<len(input); i++ {
    delta[i] = dLoss[i] * dActivation[i] * input[i]
  }

  return delta
}

// func (l *Layer) Backward(output []float64) []float64{
//
// }

func (l *Layer) Forward(input []float64) []float64{
  a := [][]float64{
    input,
  }

  product := MatrixMultiply(a, l.weights)

  row := product[0]
  for i:=0; i<len(row); i++{
    row[i] += 1.0
  }
  // fmt.Println("Before Activation:", row)
  result := l.activation.Activate(row)

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
