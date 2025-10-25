from collections import deque

from scipy import spatial
from vizdoom import ScreenResolution


def get_screen_resolution(resolution: str) -> ScreenResolution:
    try:
        return getattr(ScreenResolution, f"RES_{resolution}")
    except AttributeError as e:
        raise ValueError(f"Invalid resolution: {resolution}")


def distance_traversed(game_var_buf: deque, x_index: int, y_index: int) -> float:
    coordinates_curr = [game_var_buf[-1][x_index],
                        game_var_buf[-1][y_index]]
    coordinates_past = [game_var_buf[0][x_index],
                        game_var_buf[0][y_index]]
    return spatial.distance.euclidean(coordinates_curr, coordinates_past)
