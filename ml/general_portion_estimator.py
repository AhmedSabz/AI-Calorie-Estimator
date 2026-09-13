def estimate_food_weight(food_name, pixel_percentage):
    """
    Estimate food weight from image pixel coverage.

    IMPORTANT:
    This is a rough heuristic.
    Pixel coverage does not directly measure physical weight.
    """

    portion_ranges = {
        "steak": {
            "small": 100,
            "medium": 200,
            "large": 300,
        },

        "chicken": {
            "small": 100,
            "medium": 200,
            "large": 300,
        },

        "rice": {
            "small": 100,
            "medium": 180,
            "large": 250,
        },

        "potato": {
            "small": 100,
            "medium": 180,
            "large": 250,
        },

        "bread": {
            "small": 30,
            "medium": 60,
            "large": 100,
        },

        "sausage": {
            "small": 50,
            "medium": 100,
            "large": 150,
        },

        "banana": {
            "small": 80,
            "medium": 120,
            "large": 180,
        },

        "strawberry": {
            "small": 50,
            "medium": 100,
            "large": 150,
        },

        "ice cream": {
            "small": 75,
            "medium": 150,
            "large": 200,
        },
    }

    if food_name not in portion_ranges:
        return 100

    portions = portion_ranges[food_name]

    if pixel_percentage < 5:
        return portions["small"]

    elif pixel_percentage < 15:
        return portions["medium"]

    else:
        return portions["large"]


if __name__ == "__main__":
    test_food = "steak"
    test_percentage = 46.90

    estimated_weight = estimate_food_weight(
        test_food,
        test_percentage
    )

    print("==============================")
    print("GENERAL PORTION ESTIMATION")
    print("==============================")
    print(f"Food: {test_food}")
    print(f"Pixel coverage: {test_percentage:.2f}%")
    print(f"Estimated weight: {estimated_weight} g")