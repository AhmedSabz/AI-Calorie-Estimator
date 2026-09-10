# ==============================
# Steak Portion Estimator
# ==============================

def estimate_steak_weight(steak_percentage):
    """
    Estimate steak weight from image coverage.

    This is a heuristic estimate.
    Pixel coverage does NOT directly measure
    physical weight.

    Parameters:
        steak_percentage (float):
            Percentage of the image classified as steak.

    Returns:
        estimated_weight (float):
            Estimated steak weight in grams.
    """

    if steak_percentage < 10:
        estimated_weight = 100

    elif steak_percentage < 25:
        estimated_weight = 200

    elif steak_percentage < 40:
        estimated_weight = 300

    elif steak_percentage < 55:
        estimated_weight = 400

    else:
        estimated_weight = 500

    return estimated_weight


# ==============================
# Test
# ==============================

if __name__ == "__main__":

    test_percentage = 46.89

    weight = estimate_steak_weight(
        test_percentage
    )

    print("==============================")
    print("PORTION ESTIMATION")
    print("==============================")

    print(
        f"Steak coverage: "
        f"{test_percentage:.2f}%"
    )

    print(
        f"Estimated weight: "
        f"{weight} g"
    )