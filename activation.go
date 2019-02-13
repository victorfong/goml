package main

type Activation interface{
  Activate(float64) float64
}

type ReLU struct {}

func (r ReLU) Activate(input float64) float64{
  if input > 0.0 {
    return input
  } else {
    return 0.0
  }
}
