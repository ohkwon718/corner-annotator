
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

		self.initUI()
		
		self.imgHarris = False
		self.imgHough = False

		self.lsPoint = []
		self.lsName = []


	def evCheckBox(self, cb):	
		if cb == self.cbAutoCorrection:
			return
		self.plotFusedImg()
	
	def evSlider(self, sl):
		self.plotFusedImg()	


	def plotFusedImg(self):
		disp = np.zeros_like(self.gray)

		disp = disp + float(self.slHarris.value())/10 * self.imgHarris
		disp = disp + float(self.slHough.value())/10 * self.imgHough

		if self.cbNonMaxSup.isChecked():
			disp = self.GetNonMaxSup(disp)

		self.detected = disp

		if self.cbGray.isChecked():
			disp = disp + float(self.slGray.value())/10 * self.gray


		# disp = cv2.Canny(self.gray, 0, 200, apertureSize=3)
		xlim = self.ax.get_xlim()
		ylim = self.ax.get_ylim()
		self.ax.clear()
		self.ax.imshow(disp, cmap='gray', vmin = 0, vmax = 255)
		self.ax.set_xlabel(self.file)
		self.ax.set_xlim(xlim)
		self.ax.set_ylim(ylim)
		self.canvas.draw()


	def eventFilter(self, obj, event):
		if event.type() == QEvent.KeyPress and obj == self.listPoint:
			if event.key() == Qt.Key_Delete:
				listItems=self.listPoint.selectedItems()
				if not listItems: return        
				for item in listItems:
					idx = self.listPoint.row(item)
					del self.lsPoint[idx]
					del self.lsName[idx]
					self.listPoint.takeItem(idx)
					
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
		strFile, strExtension = os.path.splitext(file)

		if strExtension.lower() == ".mp4":
			self.file = file
			cap = cv2.VideoCapture(self.file)
			w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
			h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
			if cap.isOpened():
				ret, frame = cap.read()
				if ret == True:
					cv2.imwrite(strFile + ".jpg", frame)
					self.img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		elif strExtension.lower() == ".jpg":
			self.file = file
			self.img = cv2.cvtColor(cv2.imread(self.file), cv2.COLOR_BGR2RGB)
			h, w, _ = self.img.shape
		elif strExtension.lower() == ".txt":
			with open(file) as f:
				for l in f:
					l2d_split = l.split(' ')
					name = l2d_split[0]
					x,y = int(l2d_split[1]), int(l2d_split[2])
					self.listPoint.addItem(name + str((x,y)))
					self.lsPoint.append((x,y))
					self.lsName.append(name)
		else:
			return
		
		self.gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
		self.imgHarris = self.GetCornersHarris(self.gray)
		self.imgHough = np.zeros_like(self.gray)
		
		self.ax.clear()
		self.ax.imshow(self.img)
		self.ax.set_xlabel(file)
		self.canvas.draw()

	def generate(self):
		strFile, _ = os.path.splitext(self.file)
		
		with open(strFile + ".txt", "w") as file:
			for n, p in zip(self.lsName, self.lsPoint):
				file.write(n + " %d %d\n" % (p[0],p[1]))

		img_new = self.img
		for n, p in zip(self.lsName, self.lsPoint):
			cv2.circle(img_new, center = p, radius = 3, color = (255,128,128), thickness = 2)
			cv2.putText(img_new, n, (p[0]-20, p[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128))
		cv2.imwrite(strFile + "_result.jpg", cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))


	def plotRawImg(self):
		self.ax.clear()
		self.ax.imshow(self.img)
		self.ax.set_xlabel(file)
		self.canvas.draw()


	def GetCornersHarris(self, gray):
		ret = np.zeros(gray.shape)
		gray = np.float32(gray)
		dst = cv2.cornerHarris(gray,2,3,0.04)
		dst = cv2.dilate(dst,None)
		ret[dst>0.001*dst.max()] = 255.0
		
		return ret


	def drawLine(self):
		X1, X2 = self.getClickedBox()	

		if self.cbHough.isChecked():
			l = min(X1[0],X2[0])-2
			r = max(X1[0],X2[0])+2
			t = min(X1[1],X2[1])-2
			b = max(X1[1],X2[1])+2

			box = self.gray[ t:b, l:r ]
			theta = -np.arctan2(X1[0]-X2[0], X1[1]-X2[1])
			dist = np.sqrt((X1[0]-X2[0])**2 + (X1[1]-X2[1])**2)
			self.imgHough[ t:b, l:r ] = self.imgHough[ t:b, l:r ] + self.GetLinesHough(box, dist = dist, theta_target=theta, theta_offset = 0.01)
		else:
			grayLine = np.zeros_like(self.imgHough)
			cv2.line(grayLine,X1,X2,100.0, )
			self.imgHough = self.imgHough + grayLine
		
		self.plotFusedImg()	

	def clearLines(self):
		self.imgHough = np.zeros_like(self.gray)
		self.plotFusedImg()	

	def getClickedBox(self):
		self.ax.set_xlim(self.ax.get_xlim()) 
		self.ax.set_ylim(self.ax.get_ylim()) 

		self.edt.appendPlainText("Click 2 points")
		# X = self.figure.ginput(2)[0]
		X1, X2 = self.figure.ginput(2)
		X1 = (int(round(X1[0])),int(round(X1[1])))
		X2 = (int(round(X2[0])),int(round(X2[1])))
		return X1, X2


	def GetLinesHough(self, gray, theta_target = None, theta_offset = 0.1, dist = 70):
		ret = np.zeros(gray.shape)
		edges = cv2.Canny(gray, 130, 200, apertureSize=3)
		# edges = cv2.Canny(gray, 30, 100, apertureSize=3)
		# edges = cv2.Canny(gray, 30, 80, apertureSize=3)
		dist = int(min(dist,70))
		lines = cv2.HoughLines(edges,1,np.pi/720,int(dist))
		
		org = 0
		if theta_target is None:
			theta_range = (0, np.pi)
		else:
			if theta_target < 0:
				theta_target += np.pi
			theta_range = (theta_target - theta_offset, theta_target + theta_offset)
			if theta_target < np.pi/2:
				org = gray.shape[0]
			
		
		
		for line in lines:
		    for rho,theta in line:
		    	
		    	if np.abs(rho - org) > 3:
		    		continue
		    	if theta < theta_range[0] or theta > theta_range[1]:
		    		continue

		    	grayLine = np.zeros(gray.shape)
		        a = np.cos(theta)
		        b = np.sin(theta)
		        x0 = a*rho
		        y0 = b*rho
		        x1 = int(x0 + 3000*(-b))
		        y1 = int(y0 + 3000*(a))
		        x2 = int(x0 - 3000*(-b))
		        y2 = int(y0 - 3000*(a))
		        cv2.line(grayLine,(x1,y1),(x2,y2),1.0, )
		        ret = ret + grayLine
		# ret = ret/ret.max() * 255
		ret = ret * 100
		return ret


	def GetNonMaxSup(self, gray):
		ret = gray
		gray = cv2.GaussianBlur(gray,(7,7),0)

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

		ret[np.where(gray <= grayMax)] = 0
		ret[np.where(ret > 0)] = 255

		return ret

	def click(self):
		x,y = self.getClickedPoint()
		if self.cbAutoCorrection.isChecked():
			w = 10
			y1 = max(y-w, 0)
			y2 = min(y+w, self.detected.shape[0])
			x1 = max(x-w, 0)
			x2 = min(x+w, self.detected.shape[1])
			fMax = np.max(self.detected[y1:y2, x1:x2])
			indices = np.array(np.where(self.detected == fMax))
			diff = indices - np.array([y,x])[:,None]
			dist = diff[0]*diff[0] + diff[1]*diff[1]
			idxMin = np.argmin(dist)
			y, x = indices[:,idxMin]

		self.ax.plot(x,y,'g.')
		self.canvas.draw()
		
		if QtGui.QMessageBox.question(self,'', "Is it the corner?", 
			QtGui.QMessageBox.Yes | QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes:
			self.listPoint.addItem(self.tbName.text() + str((x,y)))
			self.lsPoint.append((x,y))
			self.lsName.append(self.tbName.text())
		
		# 	self.edt.appendPlainText(" ".join(str(x) for x in self.ls...))

	def test(self):
		self.getClickedBox()

	def getClickedPoint(self):
		self.ax.set_xlim(self.ax.get_xlim()) 
		self.ax.set_ylim(self.ax.get_ylim()) 

		self.edt.appendPlainText("Click point")
		X = self.figure.ginput(1)[0]
		x, y = X
		x = int(round(x))
		y = int(round(y))
		self.edt.appendPlainText(str((x,y)))
		return x,y

	def initUI(self):
		self.figure = Figure()
		self.canvas = FigureCanvas(self.figure)
		self.toolbar = NavigationToolbar(self.canvas, self)

		self.btnRawImg = QtGui.QPushButton('Raw Image')
		self.btnRawImg.setFixedWidth(100)
		self.btnRawImg.clicked.connect(self.plotRawImg)

		self.listPoint = QtGui.QListWidget()
		self.listPoint.installEventFilter(self)
		self.listPoint.setFixedWidth(120)

		self.edt = QtGui.QPlainTextEdit()
		self.edt.setDisabled(True)
		self.edt.setMaximumBlockCount(10)
		self.edt.setFixedWidth(120)
		
		self.cbGray = QtGui.QCheckBox("Gray")
		self.cbGray.stateChanged.connect(lambda:self.evCheckBox(self.cbGray))
		

		self.slGray = QtGui.QSlider(Qt.Horizontal)
		self.slGray.setMinimum(0)
		self.slGray.setMaximum(10)
		self.slGray.setValue(5)
		self.slGray.setTickPosition(QtGui.QSlider.TicksBelow)
		self.slGray.setTickInterval(1)
		self.slGray.setFixedWidth(80)
		self.slGray.valueChanged.connect(lambda:self.evSlider(self.slGray))

		self.slHarris = QtGui.QSlider(Qt.Horizontal)
		self.slHarris.setMinimum(0)
		self.slHarris.setMaximum(10)
		self.slHarris.setValue(0)
		self.slHarris.setTickPosition(QtGui.QSlider.TicksBelow)
		self.slHarris.setTickInterval(1)
		self.slHarris.setFixedWidth(80)
		self.slHarris.valueChanged.connect(lambda:self.evSlider(self.slHarris))

		self.cbHough = QtGui.QCheckBox("Use Hough")

		self.btnDrawLine = QtGui.QPushButton('Draw Line')
		self.btnDrawLine.setFixedWidth(100)
		self.btnDrawLine.clicked.connect(self.drawLine)

		self.btnClearLines = QtGui.QPushButton('Clear Lines')
		self.btnClearLines.setFixedWidth(100)
		self.btnClearLines.clicked.connect(self.clearLines)


		self.slHough = QtGui.QSlider(Qt.Horizontal)
		self.slHough.setMinimum(0)
		self.slHough.setMaximum(10)
		self.slHough.setValue(5)
		self.slHough.setTickPosition(QtGui.QSlider.TicksBelow)
		self.slHough.setTickInterval(1)
		self.slHough.setFixedWidth(80)
		self.slHough.valueChanged.connect(lambda:self.evSlider(self.slHough))

		self.cbNonMaxSup = QtGui.QCheckBox("Non-Max-Sup")
		self.cbNonMaxSup.stateChanged.connect(lambda:self.evCheckBox(self.cbNonMaxSup))

		self.btnClick = QtGui.QPushButton('Click')
		self.btnClick.setFixedWidth(100)
		self.btnClick.clicked.connect(self.click)

		self.cbAutoCorrection = QtGui.QCheckBox("Auto Correction")
		self.cbAutoCorrection.stateChanged.connect(lambda:self.evCheckBox(self.cbAutoCorrection))

		self.btnGenerate = QtGui.QPushButton('Generate')
		self.btnGenerate.setFixedWidth(100)
		self.btnGenerate.clicked.connect(self.generate)

		self.lbName = QtGui.QLabel("Name :")
		self.lbName.setFixedWidth(100)
		self.tbName = QtGui.QLineEdit("")
		self.tbName.setFixedWidth(100)
		
		self.btnTest = QtGui.QPushButton('Test')
		self.btnTest.setFixedWidth(100)
		self.btnTest.clicked.connect(self.test)

		layoutControl = QtGui.QGridLayout()
		lsControl = [self.btnRawImg, self.cbGray, self.slGray, self.slHarris, self.cbHough,
					self.btnDrawLine, self.btnClearLines, self.slHough, self.cbNonMaxSup, 
					self.cbAutoCorrection, 
					self.lbName, self.tbName, self.btnClick, self.btnTest]
		gridW = 1
		for i in range(len(lsControl)):
			if isinstance(lsControl[i], list):
				gridW = max(gridW, len(lsControl[i]))
				
		for i in range(len(lsControl)):
			if isinstance(lsControl[i], list):
				for j in range(len(lsControl[i])):
					layoutControl.addWidget(lsControl[i][j],i,j,1,1)
			else:
				layoutControl.addWidget(lsControl[i],i,0,1,gridW)

		layout = QtGui.QGridLayout()
		layout.addWidget(self.toolbar,0,0,1,4)
		layout.addWidget(self.canvas,1,1,4,3)
		layout.addLayout(layoutControl,1,0,1,1)
		layout.addWidget(self.listPoint,2,0,1,1)
		layout.addWidget(self.edt,3,0,1,1)
		layout.addWidget(self.btnGenerate,4,0,1,1)

		self.setLayout(layout)
		self.ax = self.figure.add_subplot(111)
		self.figure.tight_layout()


if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)

	main = Window()
	main.show()

	sys.exit(app.exec_())
















