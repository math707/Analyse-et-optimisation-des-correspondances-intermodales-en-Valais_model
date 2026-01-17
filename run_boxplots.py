# run_boxplots.py
import os
from BOXPlotMaker import run_boxplots

# ---- Tu règles ça une fois ----
INPUT_DIR = "outputs_boxplots_case_3"
OUTPUT_DIR = "plots_boxplots_case_3"


def run(choice: int):
    """
    1 = 1 figure par correspondance, ALL (recommandé)
    2 = 1 figure par case, ALL
    3 = 1 figure par correspondance, UP only
    4 = 1 figure par correspondance, DOWN only
    5 = 1 figure par case, UP only
    6 = 1 figure par case, DOWN only
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if choice == 1:
        run_boxplots(INPUT_DIR, OUTPUT_DIR, mode="by_transfer", field="all")

    elif choice == 2:
        run_boxplots(INPUT_DIR, OUTPUT_DIR, mode="by_case", field="all")

    elif choice == 3:
        run_boxplots(INPUT_DIR, OUTPUT_DIR, mode="by_transfer", field="up")

    elif choice == 4:
        run_boxplots(INPUT_DIR, OUTPUT_DIR, mode="by_transfer", field="down")

    elif choice == 5:
        run_boxplots(INPUT_DIR, OUTPUT_DIR, mode="by_case", field="up")

    elif choice == 6:
        run_boxplots(INPUT_DIR, OUTPUT_DIR, mode="by_case", field="down")

    else:
        raise ValueError("choice must be in {1,2,3,4,5,6}")


if __name__ == "__main__":
    # ---- Tu peux juste changer ce chiffre ----
    run(1)

    # ou si tu préfères choisir au lancement:
    # choice = int(input("Choose mode (1-6): "))
    # run(choice)
