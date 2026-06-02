import rawpy
from pinn_starlight_core.utils.Rasterize import rasterize


class RAWLoader():
    def __init__(self) -> None:
        self.path = None
        self.data = None
        self.a_l = None
        self.coords = None
        self.data = None

    def load(self, path: str) -> None:
        self.path = path
        with rawpy.imread(path) as raw:
            self.data = raw.raw_image.copy()

    def get_raw_data(self):
        self.coords, self.data = rasterize(self.data)

        return self.coords, self.data