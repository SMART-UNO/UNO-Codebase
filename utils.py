import numpy as np
import torch
from collections import Counter
from icecream import ic

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_payoffs(payoff_lst, verbose=False):
    # Assume that there are only two players
    # Can be extended to more players cases easily later
    payoff_lst = np.array(payoff_lst)
    p0_stats, p1_stats = payoff_lst[:,
                                    :1].squeeze(), payoff_lst[:, 1:].squeeze()
    assert p0_stats.shape == p1_stats.shape
    ic(p0_stats.shape)
    p0_stats, p1_stats = Counter(list(p0_stats)), Counter(list(p1_stats))

    p0_wins, p0_lose, p0_draw = [p0_stats[r] for r in [1, -1, 0]]
    p1_wins, p1_lose, p1_draw = [p1_stats[r] for r in [1, -1, 0]]
    # Assertion check (not necessary but useful)
    n = len(payoff_lst)
    assert p0_draw == p1_draw
    assert p0_lose == p1_wins
    assert p1_lose == p0_wins
    assert p0_wins + p1_wins + p0_draw == n

    if verbose:
        print(f"Total Number of Games: {n}")
        print(
            f"P0 wins {p0_wins} games (P0 win rate: {np.round(100 * p0_wins/n, 3)}%)")
        print(
            f"P1 wins {p1_wins} games (P1 win rate: {np.round(100 * p1_wins/n, 3)}%)")
        print(
            f"Draws {p0_draw} games (Draw rate: {np.round(100 * p0_draw/n, 3)}%)")
    return p1_wins, p0_wins, p1_draw
