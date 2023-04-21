# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui',
# licensing of 'mainwindow.ui' applies.
#
# Created: Tue Feb 16 13:19:26 2021
#      by: pyside2-uic  running on PySide2 5.13.2
#
# WARNING! All changes made in this file will be lost!
import time
from PySide2 import QtCore, QtGui, QtWidgets

class HoverLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super(HoverLabel, self).__init__(parent)

    def enterEvent(self, event):
        # Update the text when the mouse enters the label
        text = self.text()
        self.setText(text + " ")
        time.sleep(.15)
        self.setText(text)

    def leaveEvent(self, event):
        # Restore the original text when the mouse leaves the label
        self.setText(self.text())


       
class DynamicTextLabel(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(DynamicTextLabel, self).__init__(parent)
        self.original_text=""

        # Set up the QGraphicsView
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setRenderHint(QtGui.QPainter.TextAntialiasing)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setFrameStyle(QtWidgets.QFrame.NoFrame)
        self.setStyleSheet("background: transparent;")

        # Set up the QGraphicsScene
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)

        # Add a QGraphicsTextItem to the scene
        self.text_item = QtWidgets.QGraphicsTextItem()
        self.scene.addItem(self.text_item)

        # Set the initial text
        self.text_item.setPlainText("Dynamic text on top!")

    def setText(self, text):
        # Update the text in the QGraphicsTextItem
        self.text_item.setPlainText(text)

    def enterEvent(self, event):
        self.original_text = self.text_item.toPlainText()
        # Update the text when the mouse enters the label
        self.setText(self.original_text[-20:])

    def leaveEvent(self, event):
        # Restore the original text when the mouse leaves the label
        self.setText(self.original_text)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1810, 975)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setMaximumSize(QtCore.QSize(250, 16777215))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.videoBox = QtWidgets.QGroupBox(self.widget)
        self.videoBox.setObjectName("videoBox")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.videoBox)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.name_constant = QtWidgets.QLabel(self.videoBox)
        self.name_constant.setObjectName("name_constant")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.name_constant)
        self.nameLabel = HoverLabel(self.videoBox)
        self.nameLabel.setText("")
        self.nameLabel.setObjectName("nameLabel")
        

        # Add a DynamicTextLabel on top of the widget
        self.dynamic_label = DynamicTextLabel(MainWindow)
        self.dynamic_label.setGeometry(self.nameLabel.geometry())
        self.dynamic_label.setText("")
        self.verticalLayout.addWidget(self.dynamic_label)
        
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.nameLabel)
        self.label = QtWidgets.QLabel(self.videoBox)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.label_3 = QtWidgets.QLabel(self.videoBox)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.label_6 = QtWidgets.QLabel(self.videoBox)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.label_7 = QtWidgets.QLabel(self.videoBox)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.label_8 = QtWidgets.QLabel(self.videoBox)
        self.label_8.setObjectName("label_8")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.nframesLabel = QtWidgets.QLabel(self.videoBox)
        self.nframesLabel.setText("")
        self.nframesLabel.setObjectName("nframesLabel")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.nframesLabel)
        self.nlabeledLabel = QtWidgets.QLabel(self.videoBox)
        self.nlabeledLabel.setText("")
        self.nlabeledLabel.setObjectName("nlabeledLabel")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.nlabeledLabel)
        self.nunlabeledLabel = QtWidgets.QLabel(self.videoBox)
        self.nunlabeledLabel.setText("")
        self.nunlabeledLabel.setObjectName("nunlabeledLabel")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.nunlabeledLabel)
        self.fpsLabel = QtWidgets.QLabel(self.videoBox)
        self.fpsLabel.setText("")
        self.fpsLabel.setObjectName("fpsLabel")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.fpsLabel)
        self.durationLabel = QtWidgets.QLabel(self.videoBox)
        self.durationLabel.setText("")
        self.durationLabel.setObjectName("durationLabel")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.durationLabel)
        self.horizontalLayout_2.addLayout(self.formLayout)
        self.verticalLayout.addWidget(self.videoBox)
        self.groupBox = QtWidgets.QGroupBox(self.widget)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.flow_train = QtWidgets.QPushButton(self.groupBox)
        self.flow_train.setEnabled(False)
        self.flow_train.setCheckable(True)
        self.flow_train.setObjectName("flow_train")
        self.verticalLayout_5.addWidget(self.flow_train)
        self.flowSelector = QtWidgets.QComboBox(self.groupBox)
        self.flowSelector.setObjectName("flowSelector")
        self.verticalLayout_5.addWidget(self.flowSelector)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.widget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.featureextractor_train = QtWidgets.QPushButton(self.groupBox_2)
        self.featureextractor_train.setEnabled(False)
        self.featureextractor_train.setCheckable(True)
        self.featureextractor_train.setObjectName("featureextractor_train")
        self.verticalLayout_6.addWidget(self.featureextractor_train)
        self.featureextractor_infer = QtWidgets.QPushButton(self.groupBox_2)
        self.featureextractor_infer.setEnabled(False)
        self.featureextractor_infer.setCheckable(True)
        self.featureextractor_infer.setObjectName("featureextractor_infer")
        self.verticalLayout_6.addWidget(self.featureextractor_infer)
        self.feSelector = QtWidgets.QComboBox(self.groupBox_2)
        self.feSelector.setObjectName("feSelector")
        self.verticalLayout_6.addWidget(self.feSelector)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(self.widget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.sequence_train = QtWidgets.QPushButton(self.groupBox_3)
        self.sequence_train.setEnabled(False)
        self.sequence_train.setCheckable(True)
        self.sequence_train.setObjectName("sequence_train")
        self.verticalLayout_7.addWidget(self.sequence_train)
        self.sequence_infer = QtWidgets.QPushButton(self.groupBox_3)
        self.sequence_infer.setEnabled(False)
        self.sequence_infer.setCheckable(True)
        self.sequence_infer.setObjectName("sequence_infer")
        self.verticalLayout_7.addWidget(self.sequence_infer)
        self.sequenceSelector = QtWidgets.QComboBox(self.groupBox_3)
        self.sequenceSelector.setObjectName("sequenceSelector")
        self.verticalLayout_7.addWidget(self.sequenceSelector)
        self.verticalLayout.addWidget(self.groupBox_3)
        self.labelBox = QtWidgets.QGroupBox(self.widget)
        self.labelBox.setObjectName("labelBox")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.labelBox)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.importPredictions = QtWidgets.QPushButton(self.labelBox)
        self.importPredictions.setEnabled(False)
        self.importPredictions.setObjectName("importPredictions")
        self.gridLayout_4.addWidget(self.importPredictions, 2, 0, 1, 1)
        self.finalize_labels = QtWidgets.QPushButton(self.labelBox)
        self.finalize_labels.setEnabled(False)
        self.finalize_labels.setObjectName("finalize_labels")
        self.gridLayout_4.addWidget(self.finalize_labels, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.labelBox)
        self.groupBox_4 = QtWidgets.QGroupBox(self.widget)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.predictionsCombo = QtWidgets.QComboBox(self.groupBox_4)
        self.predictionsCombo.setObjectName("predictionsCombo")
        self.verticalLayout_3.addWidget(self.predictionsCombo)
        self.exportPredictions = QtWidgets.QPushButton(self.groupBox_4)
        self.exportPredictions.setEnabled(False)
        self.exportPredictions.setObjectName("exportPredictions")
        self.verticalLayout_3.addWidget(self.exportPredictions)
        self.verticalLayout.addWidget(self.groupBox_4)
        self.horizontalLayout.addWidget(self.widget)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.videoPlayer = VideoPlayer(self.centralwidget)
        self.videoPlayer.setObjectName("videoPlayer")
        self.verticalLayout_2.addWidget(self.videoPlayer)
        self.labels = LabelImg(self.centralwidget)
        self.labels.setObjectName("labels")
        self.label_4 = QtWidgets.QLabel(self.labels)
        self.label_4.setGeometry(QtCore.QRect(10, 270, 55, 16))
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.labels)
        self.predictions = LabelImg(self.centralwidget)
        self.predictions.setObjectName("predictions")
        self.label_5 = QtWidgets.QLabel(self.predictions)
        self.label_5.setGeometry(QtCore.QRect(10, 270, 71, 20))
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.predictions)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1810, 22))
        self.menubar.setObjectName("menubar")
        self.menuDeepEthogram = QtWidgets.QMenu(self.menubar)
        self.menuDeepEthogram.setObjectName("menuDeepEthogram")
        self.menuBehaviors = QtWidgets.QMenu(self.menubar)
        self.menuBehaviors.setObjectName("menuBehaviors")
        self.menuVideo = QtWidgets.QMenu(self.menubar)
        self.menuVideo.setObjectName("menuVideo")
        self.menuImport = QtWidgets.QMenu(self.menubar)
        self.menuImport.setObjectName("menuImport")
        self.menuBatch = QtWidgets.QMenu(self.menubar)
        self.menuBatch.setObjectName("menuBatch")
        MainWindow.setMenuBar(self.menubar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actionNew_Project = QtWidgets.QAction(MainWindow)
        self.actionNew_Project.setObjectName("actionNew_Project")
        self.actionSave_Project = QtWidgets.QAction(MainWindow)
        self.actionSave_Project.setEnabled(False)
        self.actionSave_Project.setObjectName("actionSave_Project")
        self.actionAdd = QtWidgets.QAction(MainWindow)
        self.actionAdd.setEnabled(False)
        self.actionAdd.setObjectName("actionAdd")
        self.actionRemove = QtWidgets.QAction(MainWindow)
        self.actionRemove.setEnabled(False)
        self.actionRemove.setObjectName("actionRemove")
        self.actionStyle = QtWidgets.QAction(MainWindow)
        self.actionStyle.setObjectName("actionStyle")
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setEnabled(False)
        self.actionOpen.setObjectName("actionOpen")
        self.actionEdit_list = QtWidgets.QAction(MainWindow)
        self.actionEdit_list.setObjectName("actionEdit_list")
        self.actionNext = QtWidgets.QAction(MainWindow)
        self.actionNext.setObjectName("actionNext")
        self.actionPrevious = QtWidgets.QAction(MainWindow)
        self.actionPrevious.setObjectName("actionPrevious")
        self.actionOpen_Project = QtWidgets.QAction(MainWindow)
        self.actionOpen_Project.setObjectName("actionOpen_Project")
        self.importLabels = QtWidgets.QAction(MainWindow)
        self.importLabels.setObjectName("importLabels")
        self.actionAdd_videos = QtWidgets.QAction(MainWindow)
        self.actionAdd_videos.setObjectName("actionAdd_videos")
        self.classifierInference = QtWidgets.QAction(MainWindow)
        self.classifierInference.setCheckable(True)
        self.classifierInference.setObjectName("classifierInference")
        self.actionOvernight = QtWidgets.QAction(MainWindow)
        self.actionOvernight.setCheckable(True)
        self.actionOvernight.setObjectName("actionOvernight")
        self.actionAdd_multiple = QtWidgets.QAction(MainWindow)
        self.actionAdd_multiple.setObjectName("actionAdd_multiple")
        self.menuDeepEthogram.addAction(self.actionNew_Project)
        self.menuDeepEthogram.addAction(self.actionSave_Project)
        self.menuDeepEthogram.addAction(self.actionOpen_Project)
        self.menuBehaviors.addAction(self.actionAdd)
        self.menuBehaviors.addAction(self.actionRemove)
        self.menuVideo.addAction(self.actionOpen)
        self.menuVideo.addAction(self.actionAdd_multiple)
        self.menuImport.addAction(self.importLabels)
        self.menuBatch.addAction(self.classifierInference)
        self.menuBatch.addAction(self.actionOvernight)
        self.menubar.addAction(self.menuDeepEthogram.menuAction())
        self.menubar.addAction(self.menuBehaviors.menuAction())
        self.menubar.addAction(self.menuVideo.menuAction())
        self.menubar.addAction(self.menuImport.menuAction())
        self.menubar.addAction(self.menuBatch.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "MainWindow", None, -1))
        self.videoBox.setTitle(QtWidgets.QApplication.translate("MainWindow", "Video Info", None, -1))
        self.name_constant.setText(QtWidgets.QApplication.translate("MainWindow", "Name", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("MainWindow", "N frames", None, -1))
        self.label_3.setText(QtWidgets.QApplication.translate("MainWindow", "FPS", None, -1))
        self.label_6.setText(QtWidgets.QApplication.translate("MainWindow", "Duration", None, -1))
        self.label_7.setText(QtWidgets.QApplication.translate("MainWindow", "N labeled", None, -1))
        self.label_8.setText(QtWidgets.QApplication.translate("MainWindow", "N unlabeled", None, -1))
        self.groupBox.setTitle(QtWidgets.QApplication.translate("MainWindow", "FlowGenerator", None, -1))
        self.flow_train.setText(QtWidgets.QApplication.translate("MainWindow", "Train", None, -1))
        self.groupBox_2.setTitle(QtWidgets.QApplication.translate("MainWindow", "FeatureExtractor", None, -1))
        self.featureextractor_train.setText(QtWidgets.QApplication.translate("MainWindow", "Train", None, -1))
        self.featureextractor_infer.setText(QtWidgets.QApplication.translate("MainWindow", "Infer", None, -1))
        self.groupBox_3.setTitle(QtWidgets.QApplication.translate("MainWindow", "Sequence", None, -1))
        self.sequence_train.setText(QtWidgets.QApplication.translate("MainWindow", "Train", None, -1))
        self.sequence_infer.setText(QtWidgets.QApplication.translate("MainWindow", "Infer", None, -1))
        self.labelBox.setTitle(QtWidgets.QApplication.translate("MainWindow", "Labels", None, -1))
        self.importPredictions.setText(QtWidgets.QApplication.translate("MainWindow", "Import predictions as labels", None, -1))
        self.finalize_labels.setText(QtWidgets.QApplication.translate("MainWindow", "Finalize Labels", None, -1))
        self.groupBox_4.setTitle(QtWidgets.QApplication.translate("MainWindow", "Predictions", None, -1))
        self.exportPredictions.setText(QtWidgets.QApplication.translate("MainWindow", "Export predictions to CSV", None, -1))
        self.label_4.setText(QtWidgets.QApplication.translate("MainWindow", "Labels", None, -1))
        self.label_5.setText(QtWidgets.QApplication.translate("MainWindow", "Predictions", None, -1))
        self.menuDeepEthogram.setTitle(QtWidgets.QApplication.translate("MainWindow", "File", None, -1))
        self.menuBehaviors.setTitle(QtWidgets.QApplication.translate("MainWindow", "Behaviors", None, -1))
        self.menuVideo.setTitle(QtWidgets.QApplication.translate("MainWindow", "Video", None, -1))
        self.menuImport.setTitle(QtWidgets.QApplication.translate("MainWindow", "Import", None, -1))
        self.menuBatch.setTitle(QtWidgets.QApplication.translate("MainWindow", "Batch", None, -1))
        self.actionNew_Project.setText(QtWidgets.QApplication.translate("MainWindow", "New Project", None, -1))
        self.actionSave_Project.setText(QtWidgets.QApplication.translate("MainWindow", "Save Project (ctrl+s)", None, -1))
        self.actionAdd.setText(QtWidgets.QApplication.translate("MainWindow", "Add", None, -1))
        self.actionRemove.setText(QtWidgets.QApplication.translate("MainWindow", "Remove", None, -1))
        self.actionStyle.setText(QtWidgets.QApplication.translate("MainWindow", "Style", None, -1))
        self.actionOpen.setText(QtWidgets.QApplication.translate("MainWindow", "Add or open", None, -1))
        self.actionEdit_list.setText(QtWidgets.QApplication.translate("MainWindow", "Edit list", None, -1))
        self.actionNext.setText(QtWidgets.QApplication.translate("MainWindow", "Next", None, -1))
        self.actionPrevious.setText(QtWidgets.QApplication.translate("MainWindow", "Previous", None, -1))
        self.actionOpen_Project.setText(QtWidgets.QApplication.translate("MainWindow", "Open Project", None, -1))
        self.importLabels.setText(QtWidgets.QApplication.translate("MainWindow", "Labels", None, -1))
        self.actionAdd_videos.setText(QtWidgets.QApplication.translate("MainWindow", "Add videos", None, -1))
        self.classifierInference.setText(QtWidgets.QApplication.translate("MainWindow", "Feature extractor inference + sequence inference", None, -1))
        self.actionOvernight.setText(QtWidgets.QApplication.translate("MainWindow", "Overnight", None, -1))
        self.actionAdd_multiple.setText(QtWidgets.QApplication.translate("MainWindow", "Add multiple", None, -1))

from deepethogram.gui.custom_widgets import LabelImg, VideoPlayer
