package ml_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
  . "github.com/victorfong/goml/ml"
)

var _ = Describe("Loss Operations", func() {

	BeforeEach(func() {
		// do nothing yet
	})

	Describe("Loss functions", func() {
		Context("When using Square Error loss function", func() {

			It("should calculate correctly", func() {
          expected := []float64{
            0.01, 0.99,
          }

          actual := []float64{
            0.75136507, 0.772928465,
          }

          squareError := SquareError{}
          result := squareError.Loss(expected, actual)

          expectedResult := 0.2983711091616805

          Expect(result).To(Equal(expectedResult))
			})
		})

    Context("When using derivative of Square Error loss function", func() {

			It("should calculate correctly", func() {
          expected := []float64{
            0.01, 0.99,
          }

          actual := []float64{
            0.75136507, 0.772928465,
          }

          squareError := SquareError{}
          result := squareError.Dloss(expected, actual)

          expectedResult := []float64{
            0.74136507, -0.21707153499999998,
          }

          Expect(result).To(Equal(expectedResult))
			})
		})

	})
})
