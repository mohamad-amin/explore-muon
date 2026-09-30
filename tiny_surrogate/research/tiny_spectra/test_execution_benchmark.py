"""CPU-only fidelity guard contracts; no models, datasets, or CUDA execution."""
import copy
import math
import unittest

import torch

from .execution_benchmark import check_loss, compare


class ExecutionFidelityContracts(unittest.TestCase):
    def setUp(self):
        self.plan = dict(floating_tensor_atol=1e-6, floating_tensor_rtol=1e-5,
                         loss_absolute_tolerance=1e-5)

    def test_identical_nested_state_accepts_all_expected_value_types(self):
        state = dict(parameters={"matrix": torch.tensor([[1., -2.], [0., 3.]], device="cpu")},
                     optimizer=dict(step=torch.tensor(21, dtype=torch.int64, device="cpu"),
                                    enabled=torch.tensor([True, False], device="cpu")),
                     groups=[dict(lr=.005, names=("first", "second"), enabled=True, absent=None)],
                     counter=2688)
        result = compare(state, copy.deepcopy(state), self.plan)
        self.assertEqual(result["tensors"], 3)
        self.assertEqual(result["elements"], 7)
        self.assertEqual(result["maximum_absolute_error"], 0.)
        self.assertEqual(result["maximum_tolerance_ratio"], 0.)

    def test_dictionary_order_is_irrelevant_and_empty_tensors_are_valid(self):
        first = dict(a=torch.empty(0, device="cpu"), b={"counter": 1})
        second = dict(b={"counter": 1}, a=torch.empty(0, device="cpu"))
        result = compare(first, second, self.plan)
        self.assertEqual(result["tensors"], 1)
        self.assertEqual(result["elements"], 0)

    def test_small_absolute_and_relative_float_differences_are_allowed(self):
        first = torch.tensor([0., 1., 1000., -1000.], dtype=torch.float64, device="cpu")
        second = first + torch.tensor([5e-7, 5e-6, 5e-3, -5e-3], dtype=torch.float64, device="cpu")
        result = compare({"gradient": first}, {"gradient": second}, self.plan)
        self.assertGreater(result["maximum_absolute_error"], 0.)
        self.assertGreater(result["maximum_tolerance_ratio"], 0.)
        self.assertLessEqual(result["maximum_tolerance_ratio"], 1.)

    def test_tensor_tolerance_boundary_is_inclusive_but_exceeding_it_fails(self):
        plan = dict(self.plan, floating_tensor_atol=.125, floating_tensor_rtol=0.)
        zero = torch.tensor([0.], dtype=torch.float64, device="cpu")
        boundary = torch.tensor([.125], dtype=torch.float64, device="cpu")
        self.assertEqual(compare(zero, boundary, plan)["maximum_tolerance_ratio"], 1.)
        with self.assertRaisesRegex(ValueError, "Tensor fidelity failed: raw_gradients"):
            compare(zero, boundary + .001, plan, path="raw_gradients")

    def test_integer_and_boolean_tensors_must_match_even_with_loose_tolerances(self):
        plan = dict(self.plan, floating_tensor_atol=100., floating_tensor_rtol=100.)
        for first, second in ((torch.tensor([21], dtype=torch.int64, device="cpu"),
                               torch.tensor([22], dtype=torch.int64, device="cpu")),
                              (torch.tensor([True], device="cpu"), torch.tensor([False], device="cpu"))):
            with self.subTest(dtype=first.dtype):
                with self.assertRaisesRegex(ValueError, "Integer state differs: state/counter"):
                    compare({"counter": first}, {"counter": second}, plan)

    def test_python_integer_boolean_and_string_state_remain_exact(self):
        plan = dict(self.plan, floating_tensor_atol=100., floating_tensor_rtol=100.)
        for first, second in ((21, 22), (21, 21.), (True, 1), ("step", "microforward"), (None, 0)):
            with self.subTest(first=first, second=second):
                with self.assertRaisesRegex(ValueError, "State value differs: state/metadata"):
                    compare({"metadata": first}, {"metadata": second}, plan)

    def test_missing_extra_and_replaced_dictionary_keys_are_rejected(self):
        first = {"optimizer": {"step": 21, "momentum": .9}}
        candidates = [{"optimizer": {"step": 21}},
                      {"optimizer": {"step": 21, "momentum": .9, "extra": 0}},
                      {"optimizer": {"step": 21, "second_moment": .9}},
                      {"optimizer": []}]
        for second in candidates:
            with self.subTest(second=second):
                with self.assertRaisesRegex(ValueError, "State keys differ: state/optimizer"):
                    compare(first, second, self.plan)

    def test_sequence_type_length_and_contents_are_checked(self):
        for first, second in (([1, 2], (1, 2)), ((1, 2), [1, 2]), ([1, 2], [1]), ([1, 2], [2, 1])):
            with self.subTest(first=first, second=second):
                with self.assertRaisesRegex(ValueError, "State (sequence|value) differs"):
                    compare({"groups": first}, {"groups": second}, self.plan)

    def test_tensor_shape_dtype_and_tensor_replacement_are_checked(self):
        first = torch.zeros((2, 3), dtype=torch.float32, device="cpu")
        candidates = [torch.zeros((3, 2), dtype=torch.float32, device="cpu"),
                      first.to(dtype=torch.float64), first.to(dtype=torch.int64), first.tolist()]
        for second in candidates:
            with self.subTest(replacement=type(second), dtype=getattr(second, "dtype", None)):
                with self.assertRaisesRegex(ValueError, "Tensor shape/dtype differs: state/parameters"):
                    compare({"parameters": first}, {"parameters": second}, self.plan)

    def test_nonfinite_tensors_fail_on_either_side_even_when_values_match(self):
        finite = torch.tensor([1., 2.], device="cpu")
        for bad in (math.nan, math.inf, -math.inf):
            invalid = torch.tensor([1., bad], device="cpu")
            for first, second in ((invalid, finite), (finite, invalid), (invalid, invalid.clone())):
                with self.subTest(bad=bad, invalid_left=first is invalid):
                    with self.assertRaisesRegex(ValueError, "Nonfinite tensor: state/parameters"):
                        compare({"parameters": first}, {"parameters": second}, self.plan)

    def test_scalar_nonfinite_values_fail_on_either_side(self):
        for bad in (math.nan, math.inf, -math.inf):
            for first, second in ((bad, 1.), (1., bad), (bad, bad)):
                with self.subTest(first=first, second=second):
                    with self.assertRaisesRegex(ValueError, "Invalid scalar: state/scalar"):
                        compare({"scalar": first}, {"scalar": second}, self.plan)

    def test_float_scalar_tolerance_boundary_and_invalid_replacement(self):
        plan = dict(self.plan, floating_tensor_atol=.125, floating_tensor_rtol=0.)
        compare({"scalar": 0.}, {"scalar": .125}, plan)
        with self.assertRaisesRegex(ValueError, "Scalar state differs: state/scalar"):
            compare({"scalar": 0.}, {"scalar": .126}, plan)
        with self.assertRaisesRegex(ValueError, "Invalid scalar: state/scalar"):
            compare({"scalar": 0.}, {"scalar": None}, plan)

    def test_loss_tolerance_boundary_and_reported_absolute_error(self):
        plan = dict(self.plan, loss_absolute_tolerance=.125)
        self.assertEqual(check_loss(2., 2., plan), 0.)
        self.assertEqual(check_loss(2., 2.125, plan), .125)
        self.assertEqual(check_loss(2.125, 2., plan), .125)
        for first, second in ((2., 2.126), (2.126, 2.)):
            with self.subTest(first=first, second=second):
                with self.assertRaisesRegex(ValueError, "Loss fidelity failed"):
                    check_loss(first, second, plan)

    def test_loss_nan_and_infinity_are_never_accepted(self):
        for bad in (math.nan, math.inf, -math.inf):
            for first, second in ((bad, 1.), (1., bad), (bad, bad)):
                with self.subTest(first=first, second=second):
                    with self.assertRaisesRegex(ValueError, "Loss fidelity failed"):
                        check_loss(first, second, self.plan)


if __name__ == "__main__":
    unittest.main()
