
# from __future__ import print_function

import sys
import os
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from PyQt4 import QtGui
from PyQt4.QtCore import QTimer, QEvent, Qt

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import cv2


class Window(QtGui.QDialog):
	def __init__(self, parent=None):
		super(Window, self).__init__(parent)
		self.setWindowTitle("Multi-mp4-Sync")
		w = 1280; h = 720
		self.resize(w, h)
		self.setAcceptDrops(True)

		self.figure = Figure()
		self.canvas = FigureCanvas(self.figure)
		self.toolbar = NavigationToolbar(self.canvas, self)

		# # Just some button connected to `plot` method
		# self.btnSync = QtGui.QPushButton('Sync')
		# self.btnSync.clicked.connect(self.sync)
		
		# self.btnPlay = QtGui.QPushButton('play')
		# self.btnPlay.clicked.connect(self.play)
		
		# self.btnStop = QtGui.QPushButton('stop')
		# self.btnStop.clicked.connect(self.stop)
	
		# self.btnLoad = QtGui.QPushButton('Load')
		# self.btnLoad.clicked.connect(self.load)

		self.btnCorner = QtGui.QPushButton('Corner')
		self.btnCorner.setFixedWidth(100)
		self.btnCorner.clicked.connect(self.GetCorner)

		self.btnClick = QtGui.QPushButton('Click')
		self.btnClick.setFixedWidth(100)
		self.btnClick.clicked.connect(self.click)

		# self.btnGenerate = QtGui.QPushButton('Generate')
		# self.btnGenerate.clicked.connect(self.generate)
		
		self.edt = QtGui.QPlainTextEdit()
		self.edt.setDisabled(True)
		self.edt.setMaximumBlockCount(10)
		
		# self.listFile = QtGui.QListWidget()
		# self.listFile.installEventFilter(self)
		# self.listFile.setFixedWidth(100)
		

		layout = QtGui.QGridLayout()

		layout.addWidget(self.toolbar,0,0,1,4)
		layout.addWidget(self.canvas,1,1,3,3)
		layout.addWidget(self.btnCorner,1,0,2,1)
		layout.addWidget(self.btnClick,2,0,2,1)
		
		# layout.addWidget(self.edt,2,0,1,4)
		# layout.addWidget(self.btnSync,2,0,1,1)
		# layout.addWidget(self.btnFuse,3,0,1,1)
		
		# layout.addWidget(self.btnGenerate,5,0,1,1)
		# layout.addWidget(self.listFile,2,1,4,1)
		

		self.setLayout(layout)
		# self.lsMp4 = []
		# self.dictWav = {}
		# self.bClick = False
		# self.lsSplitPosition = []
		self.ax = self.figure.add_subplot(111)
		self.figure.tight_layout()

	def eventFilter(self, obj, event):
		if event.type() == QEvent.KeyPress and obj == self.listFile:
			if event.key() == Qt.Key_Delete:
				pass
				# listItems=self.listFile.selectedItems()
				# if not listItems: return        
				# for item in listItems:
					# self.listFile.takeItem(self.listFile.row(item))
					# for mp4 in self.lsMp4:
					# 	if mp4['name'] == item.text():
					# 		self.lsMp4.remove(mp4)
					# 		break
				# self.plot()			
			return super(Window, self).eventFilter(obj, event)
		else:
			return super(Window, self).eventFilter(obj, event)

	def dragEnterEvent(self, event):
		if event.mimeData().hasUrls():
			event.accept()
		else:
			event.ignore()

	def dropEvent(self, event):
		if len(event.mimeData().urls()) != 1:
			return
		file = unicode(event.mimeData().urls()[0].toLocalFile()) 
		_, strExtension = os.path.splitext(file)
		if strExtension.lower() == ".mp4":
			cap = cv2.VideoCapture(file)
			w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
			h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
			if cap.isOpened():
				ret, frame = cap.read()
				if ret == True:
					self.img = frame
		elif strExtension.lower() == ".jpg":
			pass
		else:
			return
		self.ax.clear()
		self.ax.imshow(self.img)
		self.ax.set_xlabel(file)
		self.canvas.draw()


	def GetCorner(self):
		gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
		grayHarris = self.GetCornersHarris(gray)
		print grayHarris
		grayHough = self.GetLinesHough(gray)
		print grayHough

		self.ax.clear()
		# self.ax.imshow((grayHough+grayHarris)/2.0, cmap='gray', vmin = 0, vmax = 255)
		self.ax.imshow(gray + grayHough, cmap='gray', vmin = 0, vmax = 255)
		self.ax.set_xlabel(file)
		self.canvas.draw()
		
		pass

	def GetCornersHarris(self, gray):
		ret = np.zeros(gray.shape)
		gray = np.float32(gray)
		dst = cv2.cornerHarris(gray,2,3,0.04)
		dst = cv2.dilate(dst,None)
		# ret[dst>0.01*dst.max()] = 1
		ret[dst>0.001*dst.max()] = 255.0
		return ret

	def GetLinesHough(self, gray):
		ret = np.zeros(gray.shape)
		edges = cv2.Canny(gray, 0, 200, apertureSize=3)
		lines = cv2.HoughLines(edges,1,np.pi/720,150)

		for line in lines:
		    for rho,theta in line:
		    	grayLine = np.zeros(gray.shape)
		        a = np.cos(theta)
		        b = np.sin(theta)
		        x0 = a*rho
		        y0 = b*rho
		        x1 = int(x0 + 3000*(-b))
		        y1 = int(y0 + 3000*(a))
		        x2 = int(x0 - 3000*(-b))
		        y2 = int(y0 - 3000*(a))
		        cv2.line(grayLine,(x1,y1),(x2,y2),255.0)
		        ret = ret + grayLine
		ret = ret/ret.max()*255
		return ret


	def GetNonMaxSup(self, gray):
		ret = np.zeros(gray.shape)
		gray[:,:]
		np.zeros(gray.shape)[:-1,-1] = gray[:-1,:-1]
		np.pad(gray[:-1,:-1], ((1, 1), (0, 0)), 0)
		np.pad(gray[:-1,:], ((1, 0), (0, 0)), 0)
		np.pad(gray[:-1,1:], ((1, 0), (0, 1)), 0)
		np.pad(gray[:,:-1], ((0, 1), (0, 0)), 0)
		gray[:,:]
		np.pad(gray[:,1:], ((0, 0), (0, 1)), 0)


		return ret

	def click(self):
		if self.bClick:
			x,y = self.getClickedPoint()
			self.ax.plot(x,y,'g.')
			self.canvas.draw()

			if QtGui.QMessageBox.question(self,'', "Is it the corner?", 
				QtGui.QMessageBox.Yes | QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes:
				pass

			# 	self.edt.appendPlainText(" ".join(str(x) for x in self.ls...))

	def getClickedPoint(self):
		self.ax.set_xlim(self.ax.get_xlim()) 
		self.ax.set_ylim(self.ax.get_ylim()) 

		self.edt.appendPlainText("Click point")
		X = self.figure.ginput(1)[0]
		self.edt.appendPlainText(str(X))
		return X
		


if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)

	main = Window()
	main.show()

	sys.exit(app.exec_())
















