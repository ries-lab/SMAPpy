class MapListTar:
    def __init__(self, n_phot: int, _n_z: int = 1):
        """
        Target channel map for list target.
        Note that for n_phot > 1 the photons in the EmitterSet must be 2d.

        Args:
            n_phot: number of photon channels
            _n_z: future use
        """
        self._n_phot = n_phot
        self._n_z = _n_z

        self._ch_map = list(range(n_phot + 2 + _n_z))
        
    @property
    def n(self) -> int:
        return len(self._ch_map)

    @property
    def ix_phot(self) -> list[int]:
        return self._ch_map[: self._n_phot]

    @property
    def ix_xyz(self) -> list[int]:
        return self._ch_map[self._n_phot :]

    @property
    def ix_z(self) -> list[int]:
        return self._ch_map[-self._n_z :]
