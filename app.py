from flask import Flask, render_template, Response
from flask_cors import CORS
import io
import numpy
from PIL import Image
import time

from ailive.config import cf
from ailive.animate import Animator
from ailive.modifiers import *


app = Flask(__name__)
CORS(app)

animator = Animator(cf.model, cf.sensitivity, cf.audio)

modifiers = {
    'white': White(100),
    'grey': Greyscale(100),
    'black': Black(100),
}
modifiers['black'].it = 100
modifiers['black'].reverse = True


def gen():
    for i in animator:
        i = i * 255
        shape = i.shape

        for k in modifiers:
            i = modifiers[k](i)

        multiplier = round(
            (1 - cf.flask.padding) * animator.screen_width / i.shape[1]
        )

        i = i.astype(numpy.uint8)
        i = Image.fromarray(i)
        i = i.resize((shape[1] * multiplier, shape[0] * multiplier))
        buf = io.BytesIO()
        i.save(buf, format='JPEG')
        frame = buf.getvalue()

        yield (b'--frame\r  \n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/modify/<string:key>')
def modify(key):
    modifiers[key].reverse = False
    return 'ok'


@app.route('/set-transition-len/<string:steps>')
def set_transition_len(steps):
    steps = int(steps)
    for k in modifiers:
        modifiers[k].n_steps = steps
    return 'ok'


@app.route('/reset/<string:key>')
def reset(key):
    modifiers[key].reverse = True
    return 'ok'


@app.route('/pause/<string:key>')
def pause(key):
    modifiers[key].done = True
    return 'ok'


@app.route('/press/<string:key>')
def press(key):
    animator.press(key)
    return 'ok'


@app.route('/set-sensitivity/<string:level>')
def set_sensitivity(level):
    animator.sensitivity = float(level)
    return 'ok'


@app.route('/set-walk-speed/<string:speed>')
def set_walk_speed(speed):
    speed = float(speed)
    if speed == 0:
        animator.random_walk = False
    else:
        window_size = int(10. / speed)
        print('window_size is ' + str(window_size))
        animator.random_walk = True
        animator.n_steps = (window_size / animator.walk_steps) * animator.n_steps
        animator.walk_steps = window_size
    return 'ok'


@app.route('/controls')
def controls():
    return render_template('index.html', model_names=list(cf.flask.model_cfs.keys()))


@app.route('/change-model/<string:name>')
def change_model(name):
    animator.model_cf = cf.flask.model_cfs[name]
    return 'ok'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=False)
