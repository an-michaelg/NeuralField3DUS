# -*- coding: utf-8 -*-
"""
Create a gif using saved images of certain format
"""
import imageio as iio

n = 151
gif_path = "test_meta_37.gif"
frames_path = "{i}.jpg"

with iio.get_writer(gif_path, mode='I') as writer:
    for i in range(n):
        writer.append_data(iio.imread(frames_path.format(i=i)))

