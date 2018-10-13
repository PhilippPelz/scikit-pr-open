from collections import deque

import matplotlib.pyplot as plt
import numpy as np

class ThresholdMaskPrep(object):
    def __init__(self, data, start_mask, threshold):
        fig, ax = plt.subplots(figsize = (10,10))
        self.start_mask = start_mask
        self.fig = fig
        self.data = data
        fig.canvas.mpl_connect('key_press_event', self.key_press_event)
        self.pos = [0, 0]
        self.ax = ax
        self.holding_button1 = False
        self.holding_button3 = False
        self.threshold = threshold
        self.step = threshold/200
        print 'threshold = ', self.threshold

        self.threshold_mask = self.data < threshold

        self.m = self.threshold_mask * self.start_mask
        ax.grid(False)

        self.imax = ax.imshow(data * self.m, interpolation='nearest', cmap=plt.cm.get_cmap('hot'))
        plt.show()

    def key_press_event(self, event):
        if not event.inaxes:
            return
#        print event.button
#        print event.key

        x, y = int(event.xdata), int(event.ydata)
        self.pos = [y, x]

        if event.key == 'left':
            self.threshold -= self.step
            self.threshold_mask = self.data < self.threshold
            self.m = self.threshold_mask * self.start_mask
            self.imax.set_data(self.data * self.m)
            plt.draw()
        elif event.key == 'right':
            self.threshold += self.step
            self.threshold_mask = self.data < self.threshold
            self.m = self.threshold_mask * self.start_mask
            self.imax.set_data(self.data * self.m)
            plt.draw()

        print 'threshold = %d,  #masked = %d' % (self.threshold, (1-self.threshold_mask).sum())

class MaskPrep(object):
    def __init__(self, data, start_mask):
        fig, (ax,ax1) = plt.subplots(1,2,figsize = (10,10))
        self.start_mask = start_mask
        self.fig = fig
        self.data = data
        fig.canvas.mpl_connect('button_press_event', self.button_press_event)
        fig.canvas.mpl_connect('button_release_event', self.button_release_event)
        fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
#        fig.canvas.mpl_connect('key_press_event', self.key_press_event)
        self.pos = [0, 0]
        self.ax = ax
        self.holding_button1 = False
        self.holding_button3 = False
#        self.txt = ax.text(1.1, 1.1, '', transform=ax.transAxes)
#        self.threshold_mask = self.data < self.threshold
        self.current_mask = start_mask

        ax.grid(False)
        ax1.grid(False)

        self.imax = ax.imshow(data * self.current_mask, interpolation='nearest', cmap=plt.cm.get_cmap('viridis'))
        self.imax1 = ax1.imshow(self.current_mask, interpolation='nearest', cmap=plt.cm.get_cmap('hot'))
        # cbar = fig.colorbar(self.imax)
        plt.show()

    def button_release_event(self, event):
        x, y = int(event.xdata), int(event.ydata)
        self.pos = [y, x]
#        print self.pos
        print event.button
        if self.holding_button1 and event.button == 1:
            self.holding_button1 = False
        elif self.holding_button3 and event.button == 3:
            self.holding_button3 = False

    def button_press_event(self, event):
        x, y = int(event.xdata), int(event.ydata)
        self.pos = [y, x]
#        print self.pos
#         print event.button
        print('data: ', self.data[self.pos[0], self.pos[1]])
        if event.button == 1:
            self.current_mask[self.pos[0],self.pos[1]] = 1
            self.holding_button1 = True
        elif event.button == 3:
            self.current_mask[self.pos[0],self.pos[1]] = 0
            self.holding_button3 = True

        im = self.data * self.current_mask
        self.imax.set_data(im)
        self.imax1.set_data(self.current_mask)
        self.imax.set_clim(vmin=im.min(), vmax=im.max())
        plt.draw()
        dp = np.logical_not(self.current_mask).sum()
        print('dead pixels: ', dp)

    def mouse_move(self, event):
        if not event.inaxes:
            return
        x, y = int(event.xdata), int(event.ydata)
        self.pos = [y, x]
#        print self.pos
#         print('data: ', self.data[self.pos[0],self.pos[1]])

        if self.holding_button1:
            self.current_mask[self.pos[0],self.pos[1]] = 1
        elif self.holding_button3:
            self.current_mask[self.pos[0],self.pos[1]] = 0

        self.imax.set_data(self.data * self.current_mask)
        self.imax1.set_data(self.current_mask)
        plt.draw()
        dp = np.logical_not(self.current_mask).sum()
        # print('dead pixels: ', dp)

class InteractiveDataPrep(object):
    def __init__(self, data, r, action_sequence=None):
        if action_sequence is None:
            action_sequence = [
                (
                'Now move with the arrow keys and select the position of the same feature again. ENTER', 'enter', 'pos',
                np.ones(3)),
                ('Closing', 'close', 'pos', np.ones(3))
            ]
        fig, ax = plt.subplots(figsize=(12, 12))
        self.current_action = None
        self.fig = fig
        self.data = data
        self.actions = deque(action_sequence)
        fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        fig.canvas.mpl_connect('scroll_event', self.scroll)
        fig.canvas.mpl_connect('key_press_event', self.key_press_event)
        self.pos = [0, 0]
        self.ax = ax
        self.r = r
        self.circle1 = plt.Circle((0, 0), self.r, color='r', fill=None)
        ax.add_artist(self.circle1)
        # text location in axes coords
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)
        self.data_index = 0

        self.imax = ax.imshow(data[self.data_index], cmap=plt.cm.get_cmap('viridis'))
        self.next_action()
        figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        plt.grid(False)
        plt.show()

    def next_action(self):
        self.current_action = self.actions.popleft()
        print(self.current_action[0])
        if self.current_action[1] == 'close':
            pass
            # plt.close()
            # plt.clf()

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        self.pos = [y, x]
        # update the line positions
        self.circle1.center = (x, y)
        self.txt.set_text('x=%1.2f, y=%1.2f index=%d' % (self.pos[0], self.pos[1], self.data_index))
        plt.draw()

    def scroll(self, event):
        if not event.inaxes:
            return
        if event.button == 'up':
            self.r += 1
        else:
            self.r -= 1
        x, y = event.xdata, event.ydata
        # update the line positions
        self.circle1.radius = self.r
        plt.draw()

    def key_press_event(self, event):
        #        print(event.key)
        if event.key == 'enter' and self.current_action[1] == 'enter':
            self.current_action[3][:] = [self.data_index, self.pos[0], self.pos[1]]
            self.next_action()
        elif event.key == 'control' and self.current_action[1] == 'control':
            self.current_action[3][:] = [self.r]
            self.next_action()
        elif event.key == 'left':
            self.data_index -= 1
            self.imax.set_data(self.data[self.data_index])
            self.txt.set_text('x=%1.2f, y=%1.2f index=%d' % (self.pos[0], self.pos[1], self.data_index))
            plt.draw()
        elif event.key == 'right':
            self.data_index += 1
            self.imax.set_data(self.data[self.data_index])
            self.txt.set_text('x=%1.2f, y=%1.2f index=%d' % (self.pos[0], self.pos[1], self.data_index))
            plt.draw()
