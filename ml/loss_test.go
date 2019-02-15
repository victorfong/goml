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
		Context("When using Cross Entropy loss function", func() {

			It("should calculate correctly", func() {
          expected := []float64{
            1.0, 0.0, 0.0,
          }

          actual := []float64{
            0.2698, 0.3223, 0.4078,
          }

          crossEntropy := CrossEntropy{}
          result := crossEntropy.Loss(expected, actual)

          expectedResult := 1.3100743352084816

          Expect(result).To(Equal(expectedResult))
			})

			It("should calculate derivative of loss correctly", func() {
          expected := []float64{
            1.0, 0.0, 0.0,
          }

          actual := []float64{
            0.2698, 0.3223, 0.4078,
          }

          crossEntropy := CrossEntropy{}
          result := crossEntropy.Dloss(expected, actual)

					expectedResult := []float64{
            -3.7064492216456637, -1.4755791648221928, -1.6886187098953054,
          }

          Expect(result).To(Equal(expectedResult))
			})
		})

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
