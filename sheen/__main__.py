import os
import click
import sounddevice

from sheen.stream import stream as _stream
from aiconfig import build


@click.group()
def cli():
    ...


@cli.command()
@click.argument('experiment')
@click.option('--device', '-d', default=7)
@click.option('--server', '-s', default='server.yaml')
@click.option('--host', '-h', default='localhost')
def live(experiment, device, server, host):
    allowed_models = os.listdir('models')
    build(server, experiment=experiment, device=device, host=host, allowed_models=allowed_models)


@cli.command()
@click.argument('experiment')
def train(experiment):
    build(
        f'models/{experiment}/config.yaml',
        experiment=experiment,
    )


@cli.command()
@click.argument('experiment')
@click.argument('audio_path')
def generate(experiment, audio_path):
    raise NotImplementedError


@cli.command()
@click.option('--interval', '-i', default=30)
@click.option('--device', '-d', default=7)
@click.option('--window', '-w', default=200)
@click.option('--channels', '-c', default=[1])
@click.option('--samplerate', '-s', default=None)
@click.option('--downsample', '-ds', default=10)
def stream(
    interval,
    device,
    window,
    channels,
    samplerate,
    downsample,
):
    _stream(interval, device, window, channels, samplerate, downsample)


@cli.command()
def list_devices():
    print(sounddevice.query_devices())


if __name__ == '__main__':
    cli()