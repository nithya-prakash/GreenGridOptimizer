from src.evaluation.backtest import _fold_bounds


def test_fold_bounds_are_contiguous_non_overlapping_and_expanding():
    n_rows, n_folds, test_size = 1511, 4, 168
    bounds = list(_fold_bounds(n_rows, n_folds, test_size))

    assert len(bounds) == n_folds
    for train_end, test_end in bounds:
        assert test_end - train_end == test_size

    # Each fold's train window strictly grows, and test windows are back-to-back.
    for (prev_train_end, prev_test_end), (train_end, test_end) in zip(bounds, bounds[1:]):
        assert train_end == prev_test_end
        assert train_end > prev_train_end

    # Last fold's test window ends exactly at the end of the dataset.
    assert bounds[-1][1] == n_rows
