package ml

import "math"
import "fmt"

type Activation interface{
  Activate([]float64) []float64
  DeActivate([]float64) []float64
}

type ReLU struct {}

func (r ReLU) Activate(input []float64) []float64{
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

func (r ReLU) DeActivate(input []float64) []float64{
  output := make([]float64, len(input))

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

func (s Softmax) DeActivate(input []float64) []float64{
  n := len(input)
  output := make([]float64, n)

  e := make([]float64, n)
  sum := 0.0
  for i:=0; i<n; i++ {
    e[i] = math.Exp(input[i])
    sum += e[i]
  }

  bottom := sum * sum

  for i:=0; i<n ;i++ {
    top := 0.0
    for j:=0;j<n; j++ {
      if i != j {
        top += e[i]
      }
    }
    output[i] = top / bottom
  }

  fmt.Println("Softmax DeActivate:", output)
  return output
}
