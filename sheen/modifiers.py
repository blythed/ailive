import numpy
from scipy.ndimage import gaussian_filter


class Modifier:
    def __init__(self, mods: dict):
        """
        Object containing a store of modifier components (Component) which change the appearance
        of an image.

        :param mods: dictionary of modifiers
        """
        self.mods = mods
        self.active = []

    def __call__(self, i):
        active = []
        for m in self.active:
            i = self.mods[m](i)
            if not self.mods[m].done:
                active.append(m)
        self.active = active
        return i


class Component:
    def __init__(self, n_steps):
        """
        Component class calling transitions to an image over a period of steps.

        :param n_steps: the number of steps the transitions should run.
        """
        self._n_steps = n_steps
        self.it = 0
        self.done = True
        self._reverse = False

    @property
    def n_steps(self):
        return self._n_steps

    @n_steps.setter
    def n_steps(self, value):
        old_steps = self._n_steps
        self._n_steps = value
        self.it = int(self.it * value / old_steps)

    @property
    def reverse(self):
        return self._reverse

    @reverse.setter
    def reverse(self, value):
        self._reverse = value
        self.done = False

    def _call_with_proportion(self, i, alpha):
        raise NotImplementedError

    def __call__(self, i):
        self.it = max(self.it, 0)

        if self.it != 0:
            i = self._call_with_proportion(i, self.it / self.n_steps)
        if not self.done and not self.reverse:
            self.it += 1
        elif not self.done and self.reverse:
            self.it -= 1
        if self.it >= self.n_steps or self.it <= 0:
            self.done = True

        return i


class Greyscale(Component):
    """ Convert to grey and back. """
    def _call_with_proportion(self, i, alpha):
        grey = numpy.tile(numpy.mean(i, 2)[:, :, None], (1, 1, 3))
        return alpha * grey + (1 - alpha) * i


class White(Component):
    """ Convert to white image and back. """
    def _call_with_proportion(self, i, alpha):
        white = 255 * numpy.ones(i.shape)
        return alpha * white + (1 - alpha) * i


class Black(Component):
    """ Convert to black image and back. """
    def _call_with_proportion(self, i, alpha):
        black = numpy.zeros(i.shape)
        return alpha * black + (1 - alpha) * i


class Filter(Component):
    """ Make blurry/ sharp. """
    def _call_with_proportion(self, i, alpha):
        if alpha:
            for j in range(3):
                i[:, :, j] = gaussian_filter(i[:, :, j], sigma=4 * alpha)
        return i
