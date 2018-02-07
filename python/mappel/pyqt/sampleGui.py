# -*- coding: utf-8 -*-
"""
sampleGUI.py
File to generate a pyqtgraph gui from the tutorials
Created on Thu Jan 11 10:39:25 2018

@author: prelich
"""
# Start with the pyqtgraph example and hack from there
## hacking the pyqtgraph examples
## Add path to library (just for examples; you do not need this)
#import initExample

import sys
import numpy as np
import mappel
from PyQt5.QtWidgets import QPushButton
import pyqtgraph.console
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import h5py

#app = QtGui.QApplication([])

class Example(QtGui.QMainWindow):
    # Import a movie (raw data enhancement for later)
    tree = r''
    h5file = tree+r''
     
    def __init__(self):
        super().__init__()
        
        self.initUI()
        
    def initUI(self):

         self.resize(800,800)
         self.setWindowTitle('GUI Alpha: SubRegion Viewer')
         self.cw = QtGui.QWidget()
         self.setCentralWidget(self.cw)
         self.l = QtGui.QGridLayout()
         self.cw.setLayout(self.l)
         
         # The image window
         self.imv1 = pg.ImageView()
         self.l.addWidget(self.imv1, 0, 0)
         
         # console
         namespace = {'pg': pg, 'np': np, 'self':self, 'mappel':mappel}
         text = """ Console!!! """
         
         c = pyqtgraph.console.ConsoleWidget(namespace=namespace, text=text)
         self.l.addWidget(c,1,0)
         #c.show()
         c.setWindowTitle('pyqtgraph example: ConsoleWidget')
         
         # buttons
         self.fileButton = QPushButton('load File', self)
         self.fileButton.move(200,200)
         
         self.closeButton = QPushButton('close', self)
         self.closeButton.move(300,200)

         self.show()
	 M = mappel.Gauss1DMLE(8,1.0)
         self.data = M.simulate_image(M.sample_prior(1))
         # Display the image data
         self.imv1.setImage(self.data)

#update()
    def loadh5Data(self, h5file):
         h5f = h5py.File(h5file,'r')
         dispMovie = h5f.get('Movie')
         return np.array(dispMovie)
#

# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
