package main

import "math"
import "fmt"

type Activation interface{
  Activate([]float64) []float64
}

type ReLU struct {}

func (r ReLU) Activate(input []float64) []float64{
  fmt.Println("ReLU Input: ", input)
  output := make([]float64, len(input))
  for i:=0; i<len(input); i++ {
    if input[i] > 0.0 {
      output[i] = input[i]
    } else {
      output[i] = 0.0
    }
  }
  return output
}

type Softmax struct{}

func (s Softmax) Activate(input []float64) []float64{
  fmt.Println("Softmax Input: ", input)
  output := make([]float64, len(input))
  sum := 0.0

  for i:=0; i<len(input); i++ {
    normalized := input[i] * 0.000001
    output[i] = math.Exp(normalized)
    sum += output[i]
  }

  fmt.Println("Sum:",sum)

  for i:=0; i<len(input); i++ {
    output[i] /= sum
  }

  return output
}
