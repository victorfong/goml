package ml_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
  . "github.com/victorfong/goml/ml"
)

var _ = Describe("Layer Operations", func() {
	Describe("Layer", func() {
		Context("When following 3 by 3 example", func() {
			It("should calculate first layer correctly", func(){
				layer := NewLayer(3, ReLU{})
				layer.Init(3)
				layer.SetWeights([][]float64{
					{0.1, 0.2, 0.3},
					{0.3, 0.2, 0.7},
					{0.4, 0.3, 0.9},
				})

				input := []float64{
					0.1, 0.2, 0.7,
				}

				result := layer.Forward(input)

				expectedResult := []float64{
          1.35, 1.27, 1.8,
        }

        Expect(result).To(Equal(expectedResult))
			})

			It("should calculate second layer correctly", func(){
				layer := NewLayer(3, Sigmoid{})
				layer.Init(3)
				layer.SetWeights([][]float64{
					{0.2, 0.3, 0.5},
					{0.3, 0.5, 0.7},
					{0.6, 0.4, 0.8},
				})

				input := []float64{
					1.35, 1.27, 1.8,
				}

				result := layer.Forward(input)

				expectedResult := []float64{
          0.9388312894865416, 0.9404756340234984, 0.9820843048123673,
        }

        Expect(result).To(Equal(expectedResult))
			})

			It("should calculate third layer correctly", func(){
				layer := NewLayer(3, Softmax{})
				layer.Init(3)
				layer.SetWeights([][]float64{
					{0.1, 0.4, 0.8},
					{0.3, 0.7, 0.2},
					{0.5, 0.2, 0.9},
				})

				input := []float64{
					0.938, 0.94, 0.98,
				}

				result := layer.Forward(input)

				expectedResult := []float64{
          0.19857651019773828, 0.28559492698949385, 0.5158285628127679,
        }

        Expect(result).To(Equal(expectedResult))
			})

		})


		Context("When doing back-prop", func() {
			var (
				layer 				*Layer
				input					[]float64
				output				[]float64
				expected			[]float64
				lossFunction	Loss
			)

			BeforeEach(func() {
				layer = NewLayer(2, Sigmoid{})
        layer.Init(2)
        layer.SetWeights([][]float64{
          {0.4, 0.5},
          {0.45, 0.55},
        })

        input = []float64{
          0.5932699921071872, 0.596884378259767,
        }

        expected = []float64{
          0.01, 0.99,
        }

        output = []float64{
          0.75136507, 0.772928465,
        }

        lossFunction = SquareError{}
			})

			It("should calculate delta correctly", func() {

        delta := layer.CalcDelta(input, output, expected, lossFunction)

        expectedDelta := []float64{
          0.08216704051485858, -0.02274024227238976,
        }

        Expect(delta).To(Equal(expectedDelta))
			})

			It("should update weights correctly", func() {

        layer.UpdateWeights(input, output, expected, lossFunction)

        expectedWeights := [][]float64{
          {0.35891647974257074, 0.5113701211361948},
					{0.4089164797425707, 0.5613701211361949},
        }

				weights := layer.GetWeights()

        Expect(weights).To(Equal(expectedWeights))
			})
		})


	})
})
