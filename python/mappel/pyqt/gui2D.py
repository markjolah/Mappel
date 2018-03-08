# -*- coding: utf-8 -*-
"""
gui2D.py
Gui for 2D sub-region data
Created on Thu Jan 11 10:39:25 2018

@author: prelich
"""
# Modifying PyQt example

import sys
import numpy as np
import mappel
from PyQt5.QtWidgets import QMainWindow, QPushButton, QWidget, QTabWidget, QVBoxLayout, QLineEdit, QAction, qApp
import pyqtgraph.console
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSlot

class App(QMainWindow):
     
    def __init__(self):
        super().__init__()
        self.title = 'MAPPEL EVAL GUI 2D- ALPHA'
        self.setWindowTitle(self.title)
        self.setGeometry(100,100,800,800)

        self._setMenu()
        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)

        self.show()

    def _setMenu(self):
        # set buttons to add to the menu
        exitAct = QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)

        genModelAct = QAction('&Generate Model',self)

        genSimAct = QAction('&Generate Simulation',self)

        priorAct = QAction('&Model Parameters',self)

        modelAct = QAction('&Estimator',self)
        modelAct.setShortcut('Ctrl+M')
        modelAct.setStatusTip('Estimator Selection')
        #modelAct.triggered.connect()
        
        helpAct = QAction('&Documentation',self)

        # add the menu
        mainMenu = self.menuBar()
        dataMenu = mainMenu.addMenu('&Data')
        algoMenu = mainMenu.addMenu('&Algorithm')
        helpMenu = mainMenu.addMenu('&Help')

        # Add buttons to the menu
        dataMenu.addAction(exitAct)
        dataMenu.addAction(genModelAct)
        dataMenu.addAction(genSimAct)
        dataMenu.addAction(priorAct)

        algoMenu.addAction(modelAct)

        helpMenu.addAction(helpAct)

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
        M = mappel.Gauss2DMLE([8,8],[1.0,1.0])
        data = M.simulate_image(M.sample_prior())
        # data = np.random.randn(8,8)
        self.setImPanel(data)

        # mouse hover event
        self.image.scene.sigMouseMoved.connect(self.mouseMoved)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def mouseMoved(self, pos):
        x = self.image.getImageItem().mapFromScene(pos).x()
        y = self.image.getImageItem().mapFromScene(pos).y()
        pxx = np.floor(x)+0.5
        pxy = np.floor(y)+0.5
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
