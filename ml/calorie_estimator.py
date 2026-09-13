def estimate_steak_calories(weight_grams, calories_per_100g=250):
    """
    Estimate calories based on steak weight.

    Default assumption:
    250 calories per 100 grams of steak.

    This is an estimate. Actual calories depend on:
    - Cut of steak
    - Fat content
    - Cooking method
    - Added oil or butter
    """

    calories = (weight_grams / 100) * calories_per_100g

    return round(calories)


if __name__ == "__main__":
    test_weight = 400

    estimated_calories = estimate_steak_calories(test_weight)

    print("==============================")
    print("CALORIE ESTIMATION")
    print("==============================")
    print(f"Estimated steak weight: {test_weight} g")
    print(f"Estimated calories: {estimated_calories} kcal")
    print()
    print("Note: This is an approximate estimate.")