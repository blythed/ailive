import queue
import sys
import matplotlib.animation
import matplotlib.pyplot
import numpy
import sounddevice
import typing


def stream(
    interval: int = 30,
    device: int = 7,
    window: int = 200,
    channels: typing.List[int] = (1,),
    samplerate: typing.Optional[int] = None,
    downsample: int = 10,
):
    mapping = [c - 1 for c in channels]  # Channel numbers start with 1
    q = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
        q.put(indata[::downsample, mapping])

    def update_plot(frame):
        global plot_data
        while True:
            try:
                data = q.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            plot_data = numpy.roll(plot_data, -shift, axis=0)
            plot_data[-shift:, :] = data
        for column, line in enumerate(lines):
            line.set_ydata(plot_data[:, column])
        return lines

    if samplerate is None:
        device_info = sounddevice.query_devices(device, 'input')
        samplerate = device_info['default_samplerate']

    length = int(window * samplerate / (1000 * downsample))
    plot_data = numpy.zeros((length, len(channels)))

    figure, axis = matplotlib.pyplot.subplots()
    lines = axis.plot(plot_data)
    if len(channels) > 1:
        axis.legend(
            ['channel {}'.format(c) for c in channels],
            loc='lower left', ncol=len(channels),
        )
    axis.axis((0, len(plot_data), -1, 1))
    axis.set_yticks([0])
    axis.yaxis.grid(True)
    axis.tick_params(
        bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off'
    )
    figure.tight_layout(pad=0)

    stream = sounddevice.InputStream(
        device=device, channels=max(channels),
        samplerate=samplerate, callback=audio_callback)
    matplotlib.animation.FuncAnimation(figure, update_plot, interval=interval, blit=True)
    with stream:
        matplotlib.pyplot.show()