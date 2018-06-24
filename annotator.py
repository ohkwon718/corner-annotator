
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

		self.cbGray = QtGui.QCheckBox("Gray")
		self.cbGray.stateChanged.connect(lambda:self.evCheckBox(self.cbGray))

		self.slGray = QtGui.QSlider(Qt.Horizontal)
		self.slGray.setMinimum(0)
		self.slGray.setMaximum(10)
		self.slGray.setValue(0)
		self.slGray.setTickPosition(QtGui.QSlider.TicksBelow)
		self.slGray.setTickInterval(1)
		self.slGray.setFixedWidth(100)
		self.slGray.valueChanged.connect(lambda:self.evSlider(self.slGray))

		self.cbHarris = QtGui.QCheckBox("Harris")
		self.cbHarris.stateChanged.connect(lambda:self.evCheckBox(self.cbHarris))

		self.slHarris = QtGui.QSlider(Qt.Horizontal)
		self.slHarris.setMinimum(0)
		self.slHarris.setMaximum(10)
		self.slHarris.setValue(0)
		self.slHarris.setTickPosition(QtGui.QSlider.TicksBelow)
		self.slHarris.setTickInterval(1)
		self.slHarris.setFixedWidth(100)
		self.slHarris.valueChanged.connect(lambda:self.evSlider(self.slHarris))

		self.cbHough = QtGui.QCheckBox("Hough Line")
		self.cbHough.stateChanged.connect(lambda:self.evCheckBox(self.cbHough))

		self.slHough = QtGui.QSlider(Qt.Horizontal)
		self.slHough.setMinimum(0)
		self.slHough.setMaximum(10)
		self.slHough.setValue(0)
		self.slHough.setTickPosition(QtGui.QSlider.TicksBelow)
		self.slHough.setTickInterval(1)
		self.slHough.setFixedWidth(100)
		self.slHough.valueChanged.connect(lambda:self.evSlider(self.slHough))

		layoutControl = QtGui.QGridLayout()
		layoutControl.addWidget(self.cbGray,0,0,1,1)
		layoutControl.addWidget(self.slGray,1,0,1,1)
		layoutControl.addWidget(self.cbHarris,2,0,1,1)
		layoutControl.addWidget(self.slHarris,3,0,1,1)
		layoutControl.addWidget(self.cbHough,4,0,1,1)
		layoutControl.addWidget(self.slHough,5,0,1,1)

		layout = QtGui.QGridLayout()
		layout.addWidget(self.toolbar,0,0,1,4)
		layout.addWidget(self.canvas,1,1,4,3)
		layout.addWidget(self.btnCorner,1,0,1,1)
		layout.addLayout(layoutControl,2,0,1,1)
		

		self.setLayout(layout)
		self.ax = self.figure.add_subplot(111)
		self.figure.tight_layout()

		self.bGray = False
		self.bHarris = False
		self.bHough = False
		self.imgHarris = False
		self.imgHough = False

		self.testInit()

	def evCheckBox(self, cb):
		if cb.text() == "Gray":
			self.bGray = cb.isChecked()
		if cb.text() == "Harris":
			self.bHarris = cb.isChecked()
		if cb.text() == "Hough Line":
			self.bHough = cb.isChecked()		
		self.getFusedImg()
	
	def evSlider(self, sl):
		self.getFusedImg()	


	def getFusedImg(self):
		ret = np.zeros(self.gray.shape)
		
		if self.bGray:
			ret = ret + float(self.slGray.value())/10 * self.gray

		if self.bHarris:
			ret = ret + float(self.slHarris.value())/10 * self.imgHarris

		if self.bHough:
			ret = ret + float(self.slHough.value())/10 * self.imgHough

		
		xlim = self.ax.get_xlim()
		ylim = self.ax.get_ylim()
		self.ax.clear()
		self.ax.imshow(ret, cmap='gray', vmin = 0, vmax = 255)
		self.ax.set_xlabel(self.file)
		self.ax.set_xlim(xlim)
		self.ax.set_ylim(ylim)
		self.canvas.draw()


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
		self.file = unicode(event.mimeData().urls()[0].toLocalFile()) 
		_, strExtension = os.path.splitext(file)

		if strExtension.lower() == ".mp4":
			cap = cv2.VideoCapture(self.file)
			w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
			h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
			if cap.isOpened():
				ret, frame = cap.read()
				if ret == True:
					self.img = frame
					self.gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
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
		
		grayHough = self.GetLinesHough(gray)
		
		grayConner = gray + grayHough
		LocalMax = self.GetNonMaxSup(grayHough+grayHarris)
		self.ax.clear()
		# self.ax.imshow((grayHough+grayHarris)/2.0 * LocalMax, cmap='gray', vmin = 0, vmax = 255)
		# self.ax.imshow(grayConner, cmap='gray', vmin = 0, vmax = 255)
		self.ax.imshow(gray + (grayHough+grayHarris)/2.0*LocalMax, cmap='gray', vmin = 0, vmax = 255)
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
		lines = cv2.HoughLines(edges,1,np.pi/720,130)

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
		        cv2.line(grayLine,(x1,y1),(x2,y2),1.0)
		        ret = ret + grayLine
		ret = ret/ret.max() * 255
		return ret


	def GetNonMaxSup(self, gray):
		ret = np.zeros(gray.shape)

		lsShift = []
		lsShift.append(np.pad(gray[:-1,:-1], ((1, 0), (1, 0)), 'constant'))
		lsShift.append(np.pad(gray[:-1,:], ((1, 0), (0, 0)), 'constant'))
		lsShift.append(np.pad(gray[:-1,1:], ((1, 0), (0, 1)), 'constant'))
		lsShift.append(np.pad(gray[:,:-1], ((0, 0), (1, 0)), 'constant'))
		lsShift.append(np.pad(gray[:,1:], ((0, 0), (0, 1)), 'constant'))
		lsShift.append(np.pad(gray[1:,:-1], ((0, 1), (1, 0)), 'constant'))
		lsShift.append(np.pad(gray[1:,:], ((0, 1), (0, 0)), 'constant'))
		lsShift.append(np.pad(gray[1:,1:], ((0, 1), (0, 1)), 'constant'))
		grayMax = np.array(lsShift).max(axis=0)

		ret[np.where(gray >= grayMax)] = 1.0

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

	def testInit(self):
		self.file = 'mp4/test1_cam1.MP4'
		cap = cv2.VideoCapture(self.file)
		w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
		h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
		if cap.isOpened():
			ret, frame = cap.read()
			if ret == True:
				self.img = frame
				self.gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
				self.imgHarris = self.GetCornersHarris(self.gray)
				self.imgHough = self.GetLinesHough(self.gray)

		self.ax.clear()
		self.ax.imshow(self.img)
		self.ax.set_xlabel(self.file)
		self.canvas.draw()

if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)

	main = Window()
	main.show()

	sys.exit(app.exec_())
















