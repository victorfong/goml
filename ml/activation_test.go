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

	Describe("Activation", func() {
		Context("When using Sigmoid activation function", func() {

			It("should activate correctly", func() {
          a := []float64{
            0.3775, 0.39249999999999996,
          }

          sigmoid := Sigmoid{}
          result := sigmoid.Activate(a)

          expectedResult := []float64{
            0.5932699921071872, 0.596884378259767,
          }

          Expect(result).To(Equal(expectedResult))
			})
		})

	})
})
