import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import moviepy.editor as mp
import re
import scipy.io.wavfile
import tqdm
import numpy

from ailive.animate import Animator

matplotlib.use("Agg")


def generate(cf_model, cf_audio, checkpoint=None, a=None):

    if not hasattr(cf_model, 'path'):
        cf_model.path = 'checkpoints/' + checkpoint + '/model.pt'
    else:
        checkpoint = re.search('checkpoints/([a-z_]+)/model.pt', cf_model.path).groups()[0]

    if a is None:
        a = Animator(cf_model, cf_audio)
    a.normalizer = 1
    a.sensitivity = 10

    x = scipy.io.wavfile.read('data/audio/sample.wav')[1]
    audio_samples = []
    for i in tqdm.tqdm(range(x.shape[0] // 2205 - 1)):
        audio_samples.append(x[i * 2205: i * 2205 + 8820, :])

    images = []
    for sample in tqdm.tqdm(audio_samples):
        image = a.create_sample(sample)
        image = (image * 255).astype(numpy.uint8)
        images.append(image)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

    fig = plt.figure(frameon=False)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    plt.axis('off')
    plt.box(False)


    ims = []
    for i in tqdm.tqdm(range(len(images))):
        ims.append((plt.imshow(images[i]),))

    im_ani = animation.ArtistAnimation(fig,
                                       ims,
                                       interval=50,
                                       repeat_delay=3000,
                                       blit=True)
    im_ani.save('test.mp4', writer=writer)

    video = mp.VideoFileClip("test.mp4")
    audio = mp.AudioFileClip("data/audio/sample.wav")
    video = video.set_audio(audio)
    video.write_videofile('glances/' + checkpoint + "/example.mp4")

    os.system('rm test.mp4')
