def estimate_food_weight(food_name, pixel_percentage):
    """
    Estimate food weight from segmentation-mask coverage.

    This is a heuristic, not a physical measurement.
    Pixel coverage cannot directly determine food weight.
    """

    portion_ranges = {
        "steak": (100, 200, 300),
        "chicken duck": (100, 200, 300),
        "chicken": (100, 200, 300),
        "rice": (100, 180, 250),
        "potato": (100, 180, 250),
        "bread": (30, 60, 100),
        "sausage": (50, 100, 150),
        "banana": (80, 120, 180),
        "strawberry": (50, 100, 150),
        "ice cream": (75, 150, 200),
    }

    if not 0 <= pixel_percentage <= 100:
        raise ValueError(
            "pixel_percentage must be between 0 and 100."
        )

    # Explicitly label the fallback as a rough default.
    small, medium, large = portion_ranges.get(
        food_name,
        (50, 100, 150)
    )

    if pixel_percentage < 5:
        return small
    elif pixel_percentage < 15:
        return medium
    else:
        return large


if __name__ == "__main__":
    test_food = "steak"
    test_percentage = 46.90

    estimated_weight = estimate_food_weight(
        test_food,
        test_percentage
    )

    print(f"Food: {test_food}")
    print(f"Pixel coverage: {test_percentage:.2f}%")
    print(f"Estimated weight: {estimated_weight} g")
    print("Note: This is a rough heuristic.")