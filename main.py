import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("TkAgg")
from utils.getColorMap import getColorMap
from utils.GenreInference import getGenre
from utils.getChromagram import getChromagram
from matplotlib.widgets import Slider, Button
import imageio
import json
import pyaudio, wave
from pydub import AudioSegment
import os
import argparse

chunk = 4
sides_mn = 1
sides_mx = 5
seq_notes = np.zeros(shape=[100, 1], dtype=np.int32)
seq_len = np.zeros(shape=[100, 1], dtype=np.int32)

last_ptr = 0
cnt = 0
it = 0

def createVisualization(filename):
    axcolor = 'lightgoldenrodyellow'

    chromagram = getChromagram(filename)

    ind = np.argmax(chromagram, axis=0)
    ind = np.reshape(ind, newshape=[ind.shape[0],1])

    with open("json/x.json", 'r') as f:
        x_dict = json.load(f)
    with open("json/y.json", 'r') as f:
        y_dict = json.load(f)

    def koch_snowflake(order, scale=10):
        if (int(order)*101 + int(scale[0])) in x_dict.keys():
            return np.array(x_dict[int(order)*101 + int(scale[0])]), np.array(y_dict[int(order)*101 + int(scale[0])])
        def _koch_snowflake_complex(order):
            if order == 0:
                angles = np.array([0, 120, 240]) + 90
                return scale / np.sqrt(3) * np.exp(np.deg2rad(angles) * 1j)
            else:
                ZR = 0.5 - 0.5j * np.sqrt(3) / 3

                p1 = _koch_snowflake_complex(order - 1)  # start points
                p2 = np.roll(p1, shift=-1)  # end points
                dp = p2 - p1  # connection vectors

                new_points = np.empty(len(p1) * 4, dtype=np.complex128)
                new_points[::4] = p1
                new_points[1::4] = p1 + dp / 3
                new_points[2::4] = p1 + dp * ZR
                new_points[3::4] = p1 + dp / 3 * 2
                return new_points

        points = _koch_snowflake_complex(order)
        x, y = points.real, points.imag
        x_dict[int(order)*101 + int(scale[0])] = x.tolist()
        y_dict[int(order)*101 + int(scale[0])] = y.tolist()
        return x, y

    genre = getGenre(filename)
    # print("Genre: ",genre)
    global colors
    colors = getColorMap(genre)

    plt.close('all')
    plt.ion()

    fig, ax = plt.subplots(1,1)
    ax.title.set_text(filename[:filename.find(".mp3")])

    r = np.zeros(shape=[100,1], dtype=np.int32)

    r[0] = np.random.randint(sides_mn,sides_mx+1,1)
    seq_notes[0] = 0
    seq_len[0] = seq_len.shape[0]

    sound = AudioSegment.from_mp3(filename)
    sound = sound.set_channels(1)
    sound.export("converted.wav", format="wav")
    f = wave.open("converted.wav","rb")

    def callback(in_data, frame_count, time_info, status):
        data = f.readframes(frame_count)
        return (data, pyaudio.paContinue)

    p = pyaudio.PyAudio()
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
                    channels = f.getnchannels(),
                    rate = f.getframerate(),
                    output = True, stream_callback=callback)

    def blues(event):
        global colors
        colors = getColorMap("blues")
    blues_axes = plt.axes([0.03, 0.90, 0.08, 0.08])
    blues_button = Button(blues_axes, '', image=imageio.imread("img/blues.png"))
    blues_button.on_clicked(blues)


    def classical(event):
        global colors
        colors = getColorMap("classical")
    classical_axes = plt.axes([0.03, 0.80, 0.08, 0.08])
    classical_button = Button(classical_axes, '', image=imageio.imread("img/classical.png"))
    classical_button.on_clicked(classical)


    def country(event):
        global colors
        colors = getColorMap("country")
    country_axes = plt.axes([0.03, 0.70, 0.08, 0.08])
    country_button = Button(country_axes, '', image=imageio.imread("img/country.png"))
    country_button.on_clicked(country)


    def disco(event):
        global colors
        colors = getColorMap("disco")
    disco_axes = plt.axes([0.03, 0.60, 0.08, 0.08])
    disco_button = Button(disco_axes, '', image=imageio.imread("img/disco.png"))
    disco_button.on_clicked(disco)


    def hiphop(event):
        global colors
        colors = getColorMap("hiphop")
    hiphop_axes = plt.axes([0.03, 0.50, 0.08, 0.08])
    hiphop_button = Button(hiphop_axes, '', image=imageio.imread("img/hiphop.png"))
    hiphop_button.on_clicked(hiphop)


    def jazz(event):
        global colors
        colors = getColorMap("jazz")
    jazz_axes = plt.axes([0.03, 0.40, 0.08, 0.08])
    jazz_button = Button(jazz_axes, '', image=imageio.imread("img/jazz.png"))
    jazz_button.on_clicked(jazz)


    def metal(event):
        global colors
        colors = getColorMap("metal")
    metal_axes = plt.axes([0.03, 0.30, 0.08, 0.08])
    metal_button = Button(metal_axes, '', image=imageio.imread("img/metal.png"))
    metal_button.on_clicked(metal)


    def pop(event):
        global colors
        colors = getColorMap("pop")
    pop_axes = plt.axes([0.03, 0.20, 0.08, 0.08])
    pop_button = Button(pop_axes, '', image=imageio.imread("img/pop.png"))
    pop_button.on_clicked(pop)


    def reggae(event):
        global colors
        colors = getColorMap("reggae")
    reggae_axes = plt.axes([0.03, 0.10, 0.08, 0.08])
    reggae_button = Button(reggae_axes, '', image=imageio.imread("img/reggae.png"))
    reggae_button.on_clicked(reggae)


    def rock(event):
        global colors
        colors = getColorMap("rock")
    rock_axes = plt.axes([0.03, 0.0, 0.08, 0.08])
    rock_button = Button(rock_axes, '', image=imageio.imread("img/rock.png"))
    rock_button.on_clicked(rock)


    axmin = plt.axes([0.35, 0.15, 0.55, 0.03], facecolor=axcolor)
    axmax = plt.axes([0.35, 0.1, 0.55, 0.03], facecolor=axcolor)

    min0 = sides_mn
    max0 = sides_mx
    smin = Slider(axmin, 'Minimum Value', 0, 6, valinit=min0, valstep=1)
    smax = Slider(axmax, 'Maximum Value', 0, 6, valinit=max0, valstep=1)

    def update(val):
        global sides_mn, sides_mx
        sides_mn = smin.val
        sides_mx = smax.val
        fig.canvas.draw_idle()

    smin.on_changed(update)
    smax.on_changed(update)

    os.remove("spec.png")
    os.remove("inference.png")

    def plotFrame():
        global it
        ax.cla()
        global last_ptr, seq_len, seq_notes, chunk
        for i in range(chunk):
            seq_len[0] -= 1
            if seq_len[0] == 0:
                seq_len[:-1] = seq_len[1:]
                seq_notes[:-1] = seq_notes[1:]
                r[:-1] = r[1:]
                last_ptr -= 1

        for i in range(it - chunk, it):
            if i < 0:
                ind2 = 0
            else:
                ind2 = ind[i]
            if seq_notes[last_ptr] == ind2:
                seq_len[last_ptr] += 1
            else:
                last_ptr += 1
                seq_notes[last_ptr] = ind2
                seq_len[last_ptr] = 1
                r[last_ptr] = np.random.randint(sides_mn, sides_mx + 1, 1)

        scales = np.cumsum(seq_len[:last_ptr + 1][::-1], axis=0)[::-1]
        plt.subplots_adjust(left=0.25, bottom=0.25)
        for i in range(last_ptr + 1):
            x1, y1 = koch_snowflake(order=r[i], scale=scales[i])
            color = colors[int(seq_notes[i])]
            color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
            ax.fill(x1, y1, facecolor=color)
        ax.set_xlim([-27, 27])
        ax.set_ylim([-27, 27])
        ax.axis('off')
        plt.pause(0.001)

    global it
    stream.start_stream()
    while stream.is_active():
        if it >= ind.shape[0]:
            break
        plotFrame()
        it += chunk

    stream.stop_stream()
    stream.close()
    f.close()
    p.terminate()
    os.remove("converted.wav")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audiofile", type=str, help="Audio File name")
    args = parser.parse_args()
    filename = args.audiofile
    createVisualization(filename)
