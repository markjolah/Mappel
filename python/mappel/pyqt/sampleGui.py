# -*- coding: utf-8 -*-
"""
sampleGUI.py
File to generate a pyqtgraph gui from the tutorials
Created on Thu Jan 11 10:39:25 2018

@author: prelich
"""
# Modifying PyQt example

import sys
import numpy as np
# import mappel
from PyQt5.QtWidgets import QMainWindow, QPushButton, QWidget, QTabWidget, QVBoxLayout, QLineEdit
import pyqtgraph.console
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSlot

class App(QMainWindow):
     
    def __init__(self):
        super().__init__()
        self.title = 'MAPPEL EVAL GUI - ALPHA'
        self.setWindowTitle(self.title)
        self.setGeometry(100,100,800,800)

        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)

        self.show()

class MyTableWidget(QWidget):
        
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab0 = QWidget()
        self.tab1 = QWidget()
        self.tabs.resize(300,200)

        # Add tabs
        self.tabs.addTab(self.tab0,"Image Viewer")
        self.tabs.addTab(self.tab1,"Interactive Console")

        # image tab
        self.tab0.layout = QVBoxLayout(self)
        self.intensitybox = QLineEdit(self)
        self.tab0.layout.addWidget(self.intensitybox)
        self.image = pg.ImageView()
        self.tab0.layout.addWidget(self.image)
        self.tab0.setLayout(self.tab0.layout)

        # console tab
        self.tab1.layout = QVBoxLayout(self)
        namespace = {'pg': pg, 'np': np, 'self':self}
        text = """ Change the figure by calling self.setImPanel(data) """ 
        self.cons = pyqtgraph.console.ConsoleWidget(namespace=namespace, text=text)
        self.tab1.layout.addWidget(self.cons)
        self.tab1.setLayout(self.tab1.layout)
         
        # buttons (later...)

        self.show()
        
        # Set the image
        #M = mappel.Gauss1DMLE(8,1.0)
        #data = M.simulate_image(M.sample_prior(1))
        data = np.random.randn(8,8)
        self.setImPanel(data)

        # mouse hover event
        self.image.scene.sigMouseMoved.connect(self.mouseMoved)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def mouseMoved(self, pos):
        x = self.image.getImageItem().mapFromScene(pos).x()
        y = self.image.getImageItem().mapFromScene(pos).y()
        pxx = np.floor(x)
        pxy = np.floor(y)
        if 0 < pxx < self.data.shape[0] and 0 < pxy < self.data.shape[1]:
            i = self.data[pxx.astype(int),pxy.astype(int)]
        else:
            i = 0
        self.intensitybox.setText("x pos:{0:.2f}, y pos:{1:.2f}, intensity:{2:.2f}".format(x, y, i))

    @pyqtSlot()
    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())

    def setImPanel(self,data):
        self.data = data
        # Display the image data
        self.image.setImage(self.data)

# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    
    app = QtGui.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
