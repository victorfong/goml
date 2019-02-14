package ml_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
  . "github.com/victorfong/goml/ml"
)

var _ = Describe("Matrix Operations", func() {

	BeforeEach(func() {
		// do nothing yet
	})

	Describe("Multiplication", func() {
		Context("when 2 by 3 and 3 by 2 matrices multiplies", func() {

			It("should return 2 by 2 product", func() {
          a := [][]float64{
            {1, 2, 3},
            {4, 5, 6},
          }

          b := [][]float64{
            {7, 8},
            {9, 10},
            {11, 12},
          }

          result := MatrixMultiply(a, b)

          expectedResult := [][]float64{
            {58, 64},
            {139, 154},
          }

          Expect(result).To(Equal(expectedResult))
			})
		})

		Context("when 1 by 3 and 2 by 3 matrices multiplies", func() {

			It("should return 1 by 2 product", func() {
          a := [][]float64{
            {0.05, .1, 1},
          }

          b := [][]float64{
            {.15, .25},
            {.2, .3},
						{.35, .35},
          }

          result := MatrixMultiply(a, b)

          expectedResult := [][]float64{
            {0.3775, 0.39249999999999996},
          }

          Expect(result).To(Equal(expectedResult))
			})
		})
	})
})
