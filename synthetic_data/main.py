from .config import RATIOS
from .train import train_one_ratio

def main():
    for r in RATIOS:
        train_one_ratio(ratio=r, resume=True)

if __name__ == "__main__":
    main()
