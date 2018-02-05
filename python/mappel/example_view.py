import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import mappel

M = mappel.Gauss1DMLE(8,1.0)
ims = M.simulate_image(M.sample_prior(100))

win = pg.plot()
win.setWindowTitle('1D Data Set')

bg1 = pg.BarGraphItem(x=range(8), height=ims[0], width=0.3, brush='r')

win.addItem(bg1)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

