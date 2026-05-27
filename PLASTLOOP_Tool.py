# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 10:29:51 2025

@author: Darius Steegborn
"""
#this is the main file for the chemical recycling simulation tool. In here, the GUI is generated and obtained data is visualised.

# import all necessary libraries
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QAction, QDialog, QDialogButtonBox, QListWidget,
    QListWidgetItem, QMessageBox, QSizePolicy, QInputDialog, QFileDialog,
    QMainWindow, QLabel, QWidget, QTabWidget, QFormLayout,
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QTableView,
    QComboBox, QLineEdit, QHeaderView, QGroupBox, QScrollArea
)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt, QAbstractTableModel, QVariant
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import json # to store and load data

# import the file, where background calculations are performed
from PLASTLOOP_background_calculations import (
    readin_data,
    regressModell,
    estimationOfResults,
    getLCAlist,
    calculateLCAForTrials,
    calculateLCAForPrediction,
    visualizationGetData
)


#in here, "helper"-classes are programmed to easily process and visualize data. 

#This class is only for displaying the DataFrame tables in the GUI.
class PandasModel(QAbstractTableModel):
    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else None

    def setDataFrame(self, df):
        # Qt model reset pattern: inform the view that the entire model content changes.
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    # Required model methods
    def rowCount(self, parent=None):
        # Number of rows = len(DataFrame)
        return 0 if self._df is None else len(self._df)

    def columnCount(self, parent=None):
        # Number of columns = DataFrame.shape[1]
        return 0 if self._df is None else self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        # Called by the view to display each cell. We only handle DisplayRole.
        if not index.isValid() or self._df is None:
            return QVariant()
        if role == Qt.DisplayRole:
            value = self._df.iat[index.row(), index.column()]
            return "" if value is None else str(value)
        return QVariant()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        # Provides column headers (horizontal) and index labels (vertical).
        if self._df is None or role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        else:
            # Optional: show DataFrame index as row header
            return str(self._df.index[section])

    def flags(self, index):
        # Read-only table
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable
    
#This class for creating a pop-up in which you can move items in a list from left to right
class TransferDialog(QDialog):
    def __init__(self, parent=None, left_title="Available Items", right_title="Selected Items", items=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Items")
        self.resize(1500, 800)

        # --- Left column ---
        self.leftLabel = QLabel(left_title)
        self.leftLabel.setStyleSheet("font-weight: bold;")
        self.leftList = QListWidget()
        if items:
            for text in items:
                self.leftList.addItem(QListWidgetItem(text))

        # --- Right column ---
        self.rightLabel = QLabel(right_title)
        self.rightLabel.setStyleSheet("font-weight: bold;")
        self.rightList = QListWidget()

        # --- Buttons in the middle ---
        self.toRightBtn = QPushButton("→")
        self.toLeftBtn = QPushButton("←")
        self.toRightBtn.clicked.connect(self.moveRight)
        self.toLeftBtn.clicked.connect(self.moveLeft)

        # --- Layouts for columns ---
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.leftLabel)
        leftLayout.addWidget(self.leftList)

        midLayout = QVBoxLayout()
        midLayout.addStretch()
        midLayout.addWidget(self.toRightBtn)
        midLayout.addWidget(self.toLeftBtn)
        midLayout.addStretch()

        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.rightLabel)
        rightLayout.addWidget(self.rightList)

        # --- OK / Cancel ---
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        # --- Main layout ---
        mainLayout = QVBoxLayout()
        listsLayout = QHBoxLayout()
        listsLayout.addLayout(leftLayout)
        listsLayout.addLayout(midLayout)
        listsLayout.addLayout(rightLayout)
        mainLayout.addLayout(listsLayout)
        mainLayout.addWidget(self.buttonBox)
        self.setLayout(mainLayout)

    # --- Move items ---
    def moveRight(self):
        # Move selected items from left list to right list.
        for item in self.leftList.selectedItems():
            row = self.leftList.row(item)
            self.leftList.takeItem(row)
            self.rightList.addItem(item.text())

    def moveLeft(self):
        # Move selected items from right list back to left list.
        for item in self.rightList.selectedItems():
            row = self.rightList.row(item)
            self.rightList.takeItem(row)
            self.leftList.addItem(item.text())

    # Helper to read the final selection (right side)
    def selected_items(self):
        return [self.rightList.item(i).text() for i in range(self.rightList.count())]


# this class is to create a pop up to match entries of two tables with each other. Used for Process-data and LCA-background-data linking
class MappingDialog(QtWidgets.QDialog):
    def __init__(self, labdata, LCA_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Linking of LCA Inventory to loaded data")
        self.resize(1400, 900)
        self.labdata = labdata
        self.LCA_list = LCA_list
        self.mapping = {}  # {lab_col: lca_item}

        # Left list: dataset columns
        self.left = QtWidgets.QListWidget()
        self.left.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        for c in self.labdata:   # just iterate directly
            self.left.addItem(str(c))
        
        # Right list: available LCA inventory categories
        self.right = QtWidgets.QListWidget()
        self.right.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        for x in self.LCA_list:
            self.right.addItem(str(x))

        # Buttons
        self.btn_link = QtWidgets.QPushButton("Match the selected elements")
        self.btn_unlink = QtWidgets.QPushButton("Remove a selected pair from below")
        self.btn_link.clicked.connect(self.add_pair)
        self.btn_unlink.clicked.connect(self.remove_selected_pair)

        mid = QtWidgets.QVBoxLayout()
        mid.addStretch()
        mid.addWidget(self.btn_link)
        mid.addWidget(self.btn_unlink)
        mid.addStretch()

        # Table showing created pairs (not a model/view table; a direct widget-based table)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Loaded data from csv", "LCA Inverntory Categories"])
        hdr = self.table.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # Layout
        left_box = QtWidgets.QVBoxLayout()
        left_box.addWidget(QtWidgets.QLabel("<b>Loaded data from .csv</b>"))
        left_box.addWidget(self.left)

        right_box = QtWidgets.QVBoxLayout()
        right_box.addWidget(QtWidgets.QLabel("<b>LCA Inventory data</b>"))
        right_box.addWidget(self.right)

        top = QtWidgets.QHBoxLayout()
        top.addLayout(left_box, 1)
        top.addLayout(mid)
        top.addLayout(right_box, 1)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(QtWidgets.QLabel("<b>Matched pairs</b>"))
        root.addWidget(self.table, 1)
        root.addWidget(btns)

    def add_pair(self):
        # User picks one item on each side and clicks "Match".
        l_sel = self.left.selectedItems()
        r_sel = self.right.selectedItems()
        if not l_sel or not r_sel:
            QtWidgets.QMessageBox.information(
                self,
                "Warning",
                "No pair was selected. Please select an element of each side"
            )
            return

        a = l_sel[0].text()  # dataset column
        b = r_sel[0].text()  # LCA inventory item

        # Remove existing mapping if present (left-side uniqueness).
        # If a was mapped before, return old LCA item to the right list.
        if a in self.mapping:
            old_b = self.mapping[a]
            self.right.addItem(old_b)

        self.mapping[a] = b
        self.refresh_table()

        # Remove from lists so user does not map the same items repeatedly.
        self.remove_item_by_text(self.left, a)
        self.remove_item_by_text(self.right, b)

    def remove_selected_pair(self):
        # Remove the currently selected mapping row; return entries to both lists.
        row = self.table.currentRow()
        if row < 0:
            return
        a = self.table.item(row, 0).text()
        b = self.table.item(row, 1).text()

        self.left.addItem(a)
        self.right.addItem(b)

        if a in self.mapping:
            del self.mapping[a]
        self.refresh_table()

    def remove_item_by_text(self, list_widget, text):
        # Utility function: remove first list item matching the text.
        for i in range(list_widget.count()):
            if list_widget.item(i).text() == text:
                list_widget.takeItem(i)
                return

    def refresh_table(self):
        # Rebuild mapping table from `self.mapping`.
        self.table.setRowCount(0)
        for a, b in self.mapping.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(a))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(b))

    def get_mapping(self):
        return dict(self.mapping)

# in here the actual GUI generation starts

#define the main GUI-Window
class MainWindow(QMainWindow): 
    def __init__(self):         
        super().__init__()      
        icon_path = resource_path(os.path.join("Grafiken", "ChemRecPiktogramm.ico"))
        self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("Pyrolysis evaluation tool")
        self.initUI() #erstelle das Userinterface, welches in einem eigenen Befehl definiert wird

    def initUI(self): 
        # put style of QGroupBoxes
        self.setStyleSheet("""
            QGroupBox {
                font-size: 14pt;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
            }
        """)
               
        # ---- Central + tabs container ----
        central = QWidget(self)
        self.setCentralWidget(central)
        rootLayout = QVBoxLayout(central)
    
        self.tabs = QTabWidget(self)
        rootLayout.addWidget(self.tabs)
    
        #generate the first tab for machine learning algorithms
        
        tab1 = QWidget(self)
        tab1Layout = QVBoxLayout(tab1)
        tab1topLayout = QVBoxLayout()
        tab1botLayout = QVBoxLayout()
        
        self.readInDataLabel = QLabel("Readin your data")
        tab1topLayout.addWidget(self.readInDataLabel)

        # Read-in controls
        self.buttonReadInData = QPushButton("Read in dataset", self)
        self.buttonReadInData.clicked.connect(self.getData)
        tab1topLayout.addWidget(self.buttonReadInData)
    
        # Table: shows full loaded dataset via PandasModel
        self.tableDataReadIn = QTableView(self)
        self.tableDataReadIn.setSortingEnabled(True)
        self.tableDataReadIn.setAlternatingRowColors(True)
        tab1topLayout.addWidget(self.tableDataReadIn)
        
        self.CalculateRegressionLabel = QLabel("Machine learning model model fitting")
        tab1topLayout.addWidget(self.CalculateRegressionLabel)
        
        # Row: model selection + output selection
        modelRow = QHBoxLayout()
        self.modelLabel = QLabel("Select ML model:")
        self.outputLabel = QLabel("Select output:")

        # Model family selection:
        # - Must match strings expected by backend regressModell dispatcher.
        self.modelSelection = QComboBox(self)
        self.modelSelection.addItems([
            "Linear Regression",
            "Linear Regression (Scheffé)",
            "Ridge Regression",
            "Partial Least Squares Regression",
            "Support Vector Regression",
            "Decision Tree Regression"
        ])

        # Output selection (populated after data load)
        self.comboColumns = QComboBox(self)
        self.comboColumns.setEnabled(False)  # disabled until data is loaded
        
        # make both combo boxes the same width for visual alignment
        same_width = self.modelSelection.sizeHint().width()
        self.modelSelection.setFixedWidth(same_width)
        self.comboColumns.setFixedWidth(same_width)

        modelRow.addWidget(self.modelLabel)
        modelRow.addWidget(self.modelSelection)
        modelRow.addWidget(self.outputLabel)
        modelRow.addWidget(self.comboColumns)
        modelRow.addStretch()
        tab1botLayout.addLayout(modelRow)
    
        # Model for the read-in table
        self.model = PandasModel()
        self.tableDataReadIn.setModel(self.model)
    
        # Calculate model button (enabled once dataset is loaded)
        self.buttonCalculateModel = QPushButton("Calculate regressions model", self)
        self.buttonCalculateModel.setEnabled(False)
        self.buttonCalculateModel.clicked.connect(self.calculateModel)
        tab1botLayout.addWidget(self.buttonCalculateModel)
    
        # --- Regression results area (tables + explanation) ---
        Regresslayout = QVBoxLayout()  # stack rows vertically
        
        # Row 1: coefficients / model summary table
        self.tableRegressmodel = QTableView(self)
        self.tableRegressmodel.setSortingEnabled(True)
        self.tableRegressmodel.setAlternatingRowColors(True)
        Regresslayout.addWidget(self.tableRegressmodel)
        
        # model2 is the DataFrame->table adapter for regression output
        self.model2 = PandasModel()
        self.tableRegressmodel.setModel(self.model2)
        
        # Row 2: metrics table + explanation label
        bottomRow = QHBoxLayout()
        
        self.tableRegressmodelValScores = QTableView(self)
        self.tableRegressmodelValScores.setSortingEnabled(True)
        self.tableRegressmodelValScores.setAlternatingRowColors(True)
        bottomRow.addWidget(self.tableRegressmodelValScores, 1)  # left
        
        # model4 is the DataFrame->table adapter for validation KPIs
        self.model4 = PandasModel()
        self.tableRegressmodelValScores.setModel(self.model4)
        
        # Right: explanatory text for user interpretation of metrics
        self.valScoresLabel = QLabel(
            "Explanation: The parameters are calculated using the 'Leave One Out' method (LOO). "
            "The average R-squared value indicates how well the model matches the actual data used for model generation (train-data based). "
            "The average Root Mean Square Error (RMSE) describes the deviation between the model's predicted values and the testing data points (test-data based).\n\n"
            "*The R-squared and RMSE are calculated n times within the “Leave One Out” method, where n is the number of experimental trials in the dataset. "
            "The shown values are their average."
        )
        self.valScoresLabel.setWordWrap(True)
        self.valScoresLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bottomRow.addWidget(self.valScoresLabel, 1)  # right
        
        Regresslayout.addLayout(bottomRow)
        
        tab1botLayout.addLayout(Regresslayout)
        tab1Layout.addLayout(tab1topLayout)
        tab1Layout.addLayout(tab1botLayout)
        
        self.tabs.addTab(tab1, "Data Reading and Modeling")
    
    
        # generate the second tab for the machine learning validation    
    
    
        tab_metric =QWidget(self)
        tab_metricLayout = QVBoxLayout(tab_metric)
        self.tabs.addTab(tab_metric, "Modeling validation")
        # disabled until we have read in a dataset 
        self.tabs.setTabEnabled(1, False)
        self.metricButton = QPushButton("Calculate Validation")
        self.metricButton.clicked.connect(self.calculateValidationMetrics)
        
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        
        self.scrollWidget = QWidget()
        self.scrollLayout = QVBoxLayout(self.scrollWidget)
        
        self.scrollArea.setWidget(self.scrollWidget)
        
        validationMetricLayout = QHBoxLayout()
        self.trainmetricFigure = Figure(figsize=(7, 5))
        self.trainmetricCanvas = FigureCanvas(self.trainmetricFigure)
        self.trainmetricAx = self.trainmetricFigure.add_subplot(111)
        
        self.testmetricFigure = Figure(figsize=(7, 5))
        self.testmetricCanvas = FigureCanvas(self.testmetricFigure)
        self.testmetricAx = self.testmetricFigure.add_subplot(111)
        
        validationMetricLayout.addWidget(self.trainmetricCanvas)
        validationMetricLayout.addWidget(self.testmetricCanvas)
        
        validationMetricLayout.setStretch(0, 1)
        validationMetricLayout.setStretch(1, 1)
        
        self.scrollLayout.addLayout(validationMetricLayout)
        
        self.diagGroup = QGroupBox("Model diagnostics — residuals and QQ plot")
        self.diagLayout = QVBoxLayout(self.diagGroup)
        
        self.scrollLayout.addWidget(self.diagGroup)
        
        tab_metricLayout.addWidget(self.metricButton)
        tab_metricLayout.addWidget(self.scrollArea)

        
        # generate the third tab to visualize and predict results based on machine learning algorithms
        
        tab2 = QWidget(self)
        tab2Layout = QVBoxLayout(tab2)
        
        # ---- Group: estimation of outputs ----------------------------------------------------------------
        groupEstimation = QGroupBox("Estimation of outputs based on model")
        tab2toplayout = QHBoxLayout()
        dynamicInputLayout = QVBoxLayout()

        # Dynamic inputs form:
        # - Built at runtime based on namesOfParameters loaded from CSV.
        self.dynamicInputs = QWidget(self)
        self.dynamicForm = QFormLayout(self.dynamicInputs)
        self.dynamicForm.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        dynamicInputLayout.addWidget(self.dynamicInputs)
    
        # Keep references to created line edits
        self.input_fields = {}  # name -> QLineEdit
        
        # Output table for predictions
        self.tableEstRestults = QTableView(self)
        self.tableEstRestults.setSortingEnabled(True)
        self.tableEstRestults.setAlternatingRowColors(True)
        self.model3 = PandasModel()
        self.tableEstRestults.setModel(self.model3)
        
        # Estimation button
        self.buttonEstimateResults = QPushButton(
            "Estimate outputs based on Regressionmodel", self
        )
        self.buttonEstimateResults.clicked.connect(self.estimateResult)
        self.buttonEstimateResults.setEnabled(False)  # enable when inputs are ready/valid
        
        dynamicInputLayout.addWidget(self.buttonEstimateResults)
        dynamicInputLayout.addStretch()
        tab2toplayout.addLayout(dynamicInputLayout)
        tab2toplayout.addWidget(self.tableEstRestults, stretch=1)
        groupEstimation.setLayout(tab2toplayout)
        
        # ---- Group: regression visualization -------------------------------------------------------------
        groupVisualization = QGroupBox("Regression Visualization")

        tab2bottomlayout = QHBoxLayout()
        visualizationparameterslayout = QVBoxLayout()

        # Parameter & output selection for visualization
        self.VisParameter1Label = QLabel("Select Parameter 1 for the visualization")
        self.VisParameter1 = QComboBox()
        self.VisParameter2Label = QLabel("Select Parameter 2 for the visualization")
        self.VisParameter2 = QComboBox()
        self.VisOutputLabel = QLabel("Select Output for the visualization")
        self.VisOutput = QComboBox()

        # Plot button
        self.buttonVisualize = QPushButton("Plot Diagram", self)
        self.buttonVisualize.setEnabled(False)
        self.buttonVisualize.clicked.connect(self.visualizePlot)

        # Layout: selection widgets
        visualizationparameterslayout.addWidget(self.VisParameter1Label)
        visualizationparameterslayout.addWidget(self.VisParameter1)
        visualizationparameterslayout.addWidget(self.VisParameter2Label)
        visualizationparameterslayout.addWidget(self.VisParameter2)
        visualizationparameterslayout.addWidget(self.VisOutputLabel)
        visualizationparameterslayout.addWidget(self.VisOutput)
        
        # Dynamic fixed parameters:
        # - If two parameters vary, all other parameters must be fixed by user input (vis_values).
        self.visDynamicWidget = QWidget(self)
        self.visDynamicForm = QFormLayout(self.visDynamicWidget)
        self.visDynamicForm.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.visInputFields = {}  # name -> QLineEdit (for visualization)
        visualizationparameterslayout.addWidget(QLabel("Additional parameters"))
        visualizationparameterslayout.addWidget(self.visDynamicWidget)
        visualizationparameterslayout.addWidget(self.buttonVisualize)
        visualizationparameterslayout.addStretch()
        tab2bottomlayout.addLayout(visualizationparameterslayout)
        
        # Matplotlib 3D plot area
        self.regressFig = Figure(figsize=(7, 5))
        self.regressAx = self.regressFig.add_subplot(111, projection='3d')
        self.regressCanvas = FigureCanvas(self.regressFig)
        tab2bottomlayout.addWidget(self.regressCanvas)
        
        groupVisualization.setLayout(tab2bottomlayout)
        tab2Layout.addWidget(groupVisualization)
        tab2Layout.addWidget(groupEstimation)
       
        self.tabs.addTab(tab2, "Regression visualization and output prediction")
        
        # disable tab 2 until we have read in a dataset
        self.tabs.setTabEnabled(2, False)
        
        # generate the third tab to visualize and predict results based on machine learning algorithms
        
        tab3 = QWidget(self)
        tab3Layout = QVBoxLayout(tab3)

        self.linkLCADataLabel = QLabel("Link your data to LCA data or reload already linked data")
        tab3Layout.addWidget(self.linkLCADataLabel)

        # Buttons for linking / reloading LCA mapping
        LCAbuttonsLayout = QGridLayout()
        self.linkLCAdataButton = QPushButton("New: Link loaded data to LCA data")
        self.chosen_parameters = self.linkLCAdataButton.clicked.connect(self.chose_LCAParameters)

        self.loadLCAdataButton = QPushButton("Use the already linked data from last time")
        self.loadLCAdataButton.clicked.connect(self.load_LCAdata)

        LCAbuttonsLayout.addWidget(self.loadLCAdataButton,0,0)
        LCAbuttonsLayout.addWidget(self.linkLCAdataButton,0,1)
        
        self.calculateLCALabel = QLabel("Calculate LCA for your data or your prediction")
        LCAbuttonsLayout.addWidget(self.calculateLCALabel, 1,0)

        # Buttons for performing LCA calculations
        self.calculateLCAtrialsButton = QPushButton("Calculate LCA results for conducted trials")
        self.chosen_parameters = self.calculateLCAtrialsButton.clicked.connect(self.calculateLCAresultsForTrials)

        self.calculateLCAPredictionButton = QPushButton("Calculate the LCA results for the prediction")
        self.calculateLCAPredictionButton.clicked.connect(self.calculateLCAresultsForPrediction)

        LCAbuttonsLayout.addWidget(self.calculateLCAtrialsButton, 2,0)
        LCAbuttonsLayout.addWidget(self.calculateLCAPredictionButton,2,1)
        
        tab3Layout.addLayout(LCAbuttonsLayout)

        # disabled until mapping exists
        self.calculateLCAPredictionButton.setEnabled(False)
        self.calculateLCAtrialsButton.setEnabled(False)
        
        # Table for LCA results
        self.tableLCAcalcs = QTableView(self)
        tab3Layout.addWidget(self.tableLCAcalcs)
        self.modelLCAtable = PandasModel()
        self.tableLCAcalcs.setModel(self.modelLCAtable)
        
        # Plot area for LCA results (2D stacked bars)
        self.lcaFig = Figure(figsize=(7, 5))
        self.lcaAx = self.lcaFig.add_subplot(111)
        self.lcaCanvas = FigureCanvas(self.lcaFig)
        tab3Layout.addWidget(self.lcaCanvas)
                
        self.tabs.addTab(tab3, "Life Cycle Assessment")

        # disable tab 3 until we have read in a dataset
        self.tabs.setTabEnabled(3, False)
        
    # in here, the dynamic behaviour of the fields (adapting to chaning inputs) are ensured 
    def buildDynamicInputsEstimation(self, names):
        # Clear previous inputs if any
        for i in reversed(range(self.dynamicForm.count())):
            item = self.dynamicForm.itemAt(i)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            self.dynamicForm.removeItem(item)
        self.input_fields.clear()
    
        # Optional: placeholder & normalization hints per name (case-insensitive)
        placeholders = {}

        for raw_name in names:
            name = str(raw_name)
            le = QLineEdit(self)
            # Pick a placeholder if we have a hint
            key = name.strip().lower()
            le.setPlaceholderText(placeholders.get(key, "enter a value"))
            # Keep track and add to the form
            self.input_fields[name] = le
            self.dynamicForm.addRow(QLabel(f"{name}:"), le)
            # Re-check completeness on each change
            le.textChanged.connect(self.checkInputs)
    
        # After (re)building, ensure button reflects current state
        self.checkInputs()
  
    
    def refreshVisInputs(self):
        # Clear previous rows
        for i in reversed(range(self.visDynamicForm.count())):
            item = self.visDynamicForm.itemAt(i)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            self.visDynamicForm.removeItem(item)
        self.visInputFields.clear()
    
        # If inputs are not loaded yet, keep plot disabled.
        if not getattr(self, "namesOfParameters", None):
            self.checkVisInputs()  # keep button state sane
            return
    
        # Determine which parameters are varying (chosen in the two combos)
        chosen = {self.VisParameter1.currentText(), self.VisParameter2.currentText()}

        # All remaining parameters must be provided as fixed values
        remaining = [p for p in self.namesOfParameters if p not in chosen]
    
        for name in remaining:
            le = QLineEdit(self)
            le.setPlaceholderText("enter a value")
            self.visInputFields[name] = le
            self.visDynamicForm.addRow(QLabel(f"{name}:"), le)
            # react to user typing
            le.textChanged.connect(self.checkVisInputs)
    
        # evaluate initial state after rebuild
        self.checkVisInputs()
    
    
    def checkVisInputs(self):
        """
        Enable 'Plot Diagram' only if all additional parameter fields are non-empty.
        If there are no additional fields (only 2 inputs total), enable the button.
        """
        # If visInputFields not ready, keep disabled
        if not hasattr(self, "visInputFields"):
            self.buttonVisualize.setEnabled(False)
            return
    
        # If there are no remaining fixed parameters, plotting is possible immediately.
        if not self.visInputFields:
            self.buttonVisualize.setEnabled(True)
            return
    
        all_filled = all(le.text().strip() != "" for le in self.visInputFields.values())
        self.buttonVisualize.setEnabled(all_filled)
    
    
    #Enable Estimate button only if all dynamic fields are non-empty
    def checkInputs(self):
        if not self.input_fields:
            self.buttonEstimateResults.setEnabled(False)
            return
        all_filled = all(w.text().strip() for w in self.input_fields.values())
        self.buttonEstimateResults.setEnabled(all_filled)
    
    
    #refresh the visualization parameters, when 
    
    def _rebuild_vis_combo(self, combo: QComboBox, options, keep_text=None):
        """Repopulate a combo with `options`, trying to keep selection."""
        combo.blockSignals(True)
        current = combo.currentText() if keep_text is None else keep_text
        combo.clear()
        combo.addItems([str(o) for o in options])
        if current in options:
            combo.setCurrentText(current)
        elif options:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def syncVisCombos(self, changed=None):
        """
        Keep VisParameter1 and VisParameter2 mutually exclusive.
        Rebuilds both combos and then refreshes the dynamic inputs.
        `changed` is either 'VisParameter1' or 'VisParameter2' (or None on init).
        """
        if not getattr(self, "namesOfParameters", None):
            return
    
        all_params = list(self.namesOfParameters)
        sel1 = self.VisParameter1.currentText()
        sel2 = self.VisParameter2.currentText()
    
        # available options exclude the other combo's current pick
        opts1 = [p for p in all_params if p != sel2]
        opts2 = [p for p in all_params if p != sel1]
    
        # Rebuild while trying to preserve the combo that didn't change
        self._rebuild_vis_combo(self.VisParameter1, opts1, keep_text=(None if changed == "VisParameter1" else sel1))
        self._rebuild_vis_combo(self.VisParameter2, opts2, keep_text=(None if changed == "VisParameter2" else sel2))
    
        # If they still collide (e.g., after data reload), force the second to a different value
        if self.VisParameter1.currentText() == self.VisParameter2.currentText() and len(all_params) > 1:
            for p in all_params:
                if p != self.VisParameter1.currentText():
                    self.VisParameter2.blockSignals(True)
                    self.VisParameter2.setCurrentText(p)
                    self.VisParameter2.blockSignals(False)
                    break
    
        # Update the extra (remaining) parameter inputs
        self.refreshVisInputs()
    
    
    #Load the data table, then unlock the tabs and create input-dependent GUI components.
    def getData(self):
        counter = 1
        while counter >= 1:        
            start_dir = resource_path("inputdata")
            self.filepath, _ = QFileDialog.getOpenFileName(self,"Select dataframe to load",start_dir,"CSV Files (*.csv)")
            if self.filepath[-4:] != ".csv":
                counter = counter +1
                if counter >= 4:
                    QMessageBox.warning(self, "Alter...","At least try....")
                else:
                    QMessageBox.warning(self,"Not Good.....","You didnt select a .csv file, try again")
            else:
                counter = 0
        
        # Ask user how many columns after the first should be treated as inputs.
        self.amountInputs, self.check = QInputDialog.getInt(
            self,
            "Number of input parameters",
            "Insert the number of input parameters:"
        )
        
        # Backend read-in: returns full DataFrame + list of input column names
        self.dataFull, self.namesOfParameters = readin_data(self.amountInputs, self.filepath)

        # Display loaded dataset in table view (rounded for readability)
        self.model.setDataFrame(self.dataFull.round(3))
        self.tableDataReadIn.resizeColumnsToContents()
        self.tableDataReadIn.resizeRowsToContents()
        
        # Populate output selection combos:
        # - outputs are assumed to be all columns AFTER:
        #     [first ID column] + [input columns]
        self.comboColumns.clear()
        if self.dataFull is not None:
            cols = self.dataFull.columns[len(self.namesOfParameters)+1:]  # skip the Inputparameters
            self.comboColumns.addItems([str(c) for c in cols])
            self.VisOutput.addItems([str(c) for c in cols])

            # Populate parameter selection for visualization
            self.VisParameter1.addItems(self.namesOfParameters)
            self.VisParameter2.addItems(self.namesOfParameters)

            # reset then add again (redundant but functional)
            self.VisParameter1.clear()
            self.VisParameter2.clear()
            self.VisParameter1.addItems(self.namesOfParameters)
            self.VisParameter2.addItems(self.namesOfParameters)
            
            # sensible distinct defaults if possible
            if len(self.namesOfParameters) >= 2:
                self.VisParameter1.setCurrentIndex(0)
                self.VisParameter2.setCurrentIndex(1)
            elif len(self.namesOfParameters) == 1:
                self.VisParameter1.setCurrentIndex(0)
                self.VisParameter2.setCurrentIndex(0)  # will be resolved by sync
            
            # connect using lambdas so we know which combo changed
            self.VisParameter1.currentTextChanged.connect(lambda _: self.syncVisCombos("VisParameter1"))
            self.VisParameter2.currentTextChanged.connect(lambda _: self.syncVisCombos("VisParameter2"))
            
            # initial sync + build of the extra fields
            self.syncVisCombos()
            
            self.refreshVisInputs()  # initial build

            # enable controls now that data is loaded
            self.comboColumns.setEnabled(True)
            self.buttonCalculateModel.setEnabled(True)
        else:
            self.comboColumns.setEnabled(False)
        
        # Build estimation inputs dynamically from parameter names
        if getattr(self, "namesOfParameters", None):
            self.buildDynamicInputsEstimation(self.namesOfParameters)
            
        # unlock tab 2,3 and tab 4 (validation, prediction + LCA)
        self.tabs.setTabEnabled(1, True)  # validation tab
        self.tabs.setTabEnabled(2, True)  # regression visualization + prediction
        self.tabs.setTabEnabled(3, True)  # LCA
    
    
    # calculate the machine learning model based on chosen modeling method 
    
    def calculateModel(self):
        self.coefficientsDataFrame, self.validationData = regressModell(
            self.dataFull,
            self.comboColumns.currentText(),
            self.namesOfParameters,
            self.filepath,
            self.modelSelection.currentText()
        )
        self.model2.setDataFrame(self.coefficientsDataFrame.round(5))
        self.tableRegressmodel.resizeColumnsToContents()
        self.tableRegressmodel.resizeRowsToContents()

        self.model4.setDataFrame(self.validationData.round(5))
        self.tableRegressmodelValScores.resizeColumnsToContents()
        self.tableRegressmodelValScores.resizeRowsToContents()
    
    def calculateValidationMetrics(self):
        self.trainmetrics = pd.DataFrame(index=range(self.modelSelection.count()),columns=["Model","averaged_RMSE_train","averaged_R2_train"])
        self.testmetrics  = pd.DataFrame(index=range(self.modelSelection.count()),columns=["Model","RMSE_test","R2_test"])
        
        self.predictors = {}           
        self.residuals_by_model = {}  
        self.qq_by_model = {}          
        
        # take rows with valid y
        df_diag = self.dataFull[self.dataFull[self.comboColumns.currentText()].notna()].copy()
        
        for i in range(self.modelSelection.count()):
            self.coefficientsDataFrame, self.validationData, self.predict_fn = regressModell(
                self.dataFull,
                self.comboColumns.currentText(),
                self.namesOfParameters,
                self.filepath,
                self.modelSelection.itemText(i),
                return_predict_fn=True
            )
            self.trainmetrics.loc[i] = [
                self.modelSelection.itemText(i),
                self.validationData.loc[0, "averaged_RMSE_train"],
                self.validationData.loc[0, "averaged_R2_train"],
            ]
            self.testmetrics.loc[i] = [
                self.modelSelection.itemText(i),
                self.validationData.loc[0, "RMSE_test"],
                self.validationData.loc[0, "R2_test"],
            ]
        
            # store predictor
            self.predictors[self.modelSelection.itemText(i)] = self.predict_fn
            
            try:
                        y_true = pd.to_numeric(df_diag[self.comboColumns.currentText()], errors="coerce").to_numpy(dtype=float)
                        X = df_diag[list(self.namesOfParameters)]
            
                        y_pred = self.predict_fn(X)
                        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
            
                        residuals = y_true - y_pred
                        residuals = residuals[np.isfinite(residuals)]
            
            except Exception:
                residuals = np.array([], dtype=float)
    
            self.residuals_by_model[self.modelSelection.itemText(i)] = residuals
            
            # calculate and store QQ-data
            qq_data = None
            if residuals.size >= 3:
                try:
                    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    
                    qq_data = {
                        "osm": osm,
                        "osr": osr,
                        "slope": slope,
                        "intercept": intercept,
                        "r": r
                    }
                except Exception:
                    qq_data = None
    
            self.qq_by_model[self.modelSelection.itemText(i)] = qq_data
        
                # Remove old twin axis (if it exists)
        if len(self.trainmetricFigure.axes) > 1:
            self.trainmetricFigure.delaxes(self.trainmetricFigure.axes[1])
        
        self.trainmetricAx.clear()
        
        if len(self.testmetricFigure.axes) > 1:
            self.testmetricFigure.delaxes(self.testmetricFigure.axes[1])
        
                
        self.testmetricAx.clear()
        
        models = self.trainmetrics["Model"].tolist()
        x = range(len(models))
        
        rmse_train = self.trainmetrics["averaged_RMSE_train"].astype(float).to_numpy()
        r2_train   = self.trainmetrics["averaged_R2_train"].astype(float).to_numpy()
        
        ax1 = self.trainmetricAx
        ax2 = ax1.twinx()
        
        l1 = ax1.plot(x, rmse_train, marker="o", color="black", linestyle="-", label="RMSE train")
        l2 = ax2.plot(x, r2_train, marker="s", color="black", linestyle="--", label="R² train")
        
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(models, rotation=45, ha="right")
        
        ax1.set_ylabel("RMSE", color="black")
        ax2.set_ylabel("R²", color="black")
        
        ax1.tick_params(axis="y", colors="black")
        ax2.tick_params(axis="y", colors="black")
        
        ax1.set_title("Train metrics")
        
        # combined legend
        lines = l1 + l2
        labels = [line.get_label() for line in lines]
        ax1.legend(
            lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.65),
            ncol=2,
            frameon=False
        )
        self.trainmetricFigure.subplots_adjust(left=0.10, right=0.90, top=0.88, bottom=0.38)
        
        
        self.trainmetricCanvas.draw()
        self.trainmetricCanvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.trainmetricCanvas.setMinimumHeight(800)
        self.scrollLayout.setSpacing(20)
        self.scrollLayout.setContentsMargins(10, 10, 10, 10)
        
        

        rmse_test = self.testmetrics["RMSE_test"].astype(float).to_numpy()
        r2_test   = self.testmetrics["R2_test"].astype(float).to_numpy()
        
        ax1 = self.testmetricAx
        ax2 = ax1.twinx()
        
        l1 = ax1.plot(x, rmse_test, marker="o", color="black", linestyle="-", label="RMSE test")
        l2 = ax2.plot(x, r2_test, marker="s", color="black", linestyle="--", label="R² test")
        
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(models, rotation=45, ha="right")
        
        ax1.set_ylabel("RMSE", color="black")
        ax2.set_ylabel("R²", color="black")
        
        ax1.tick_params(axis="y", colors="black")
        ax2.tick_params(axis="y", colors="black")
        
        ax1.set_title("Test metrics")
        lines = l1 + l2
        labels = [line.get_label() for line in lines]
        ax1.legend(
            lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.65),
            ncol=2,
            frameon=False
        )
        
        self.testmetricFigure.subplots_adjust(left=0.10, right=0.90, top=0.88, bottom=0.38)
        self.testmetricCanvas.draw()
        
        # --- clear old diagnostic widgets ---
        while self.diagLayout.count():
            item = self.diagLayout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        
        # --- create one diagnostics figure per model (residual histogram + QQ) ---
        for model_name in self.trainmetrics["Model"].tolist():
        
            residuals = self.residuals_by_model.get(model_name, np.array([], dtype=float))
            qq = self.qq_by_model.get(model_name, None)
        
            fig = Figure(figsize=(8, 3.5))
            ax_hist = fig.add_subplot(1, 2, 1)
            ax_qq   = fig.add_subplot(1, 2, 2)
        
            # Residual histogram (black)
            ax_hist.hist(residuals, bins=30, color="black")
            ax_hist.set_title(f"{model_name} — Residuals")
            ax_hist.set_xlabel("Residual")
            ax_hist.set_ylabel("Count")
        
            # QQ plot (black)
            if qq is not None and len(qq["osm"]) > 0:
                osm = np.asarray(qq["osm"], dtype=float)
                osr = np.asarray(qq["osr"], dtype=float)
                slope = float(qq["slope"])
                intercept = float(qq["intercept"])
        
                ax_qq.plot(osm, osr, "o", color="black")
                ax_qq.plot(osm, slope * osm + intercept, "-", color="black")
                ax_qq.set_title("QQ plot")
                ax_qq.set_xlabel("Theoretical quantiles")
                ax_qq.set_ylabel("Ordered residuals")
            else:
                ax_qq.set_title("QQ plot")
                ax_qq.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_qq.transAxes)
                ax_qq.set_xticks([])
                ax_qq.set_yticks([])
        
            fig.tight_layout()
        
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            canvas.setMinimumHeight(500)
            self.diagLayout.addWidget(canvas)
            self.diagLayout.setSpacing(20)
            self.diagLayout.setContentsMargins(10, 10, 10, 10)
     
        
    # read numeric values in the estimation form to predict model results 
    def _compute_metrics_for_model(self, model_name: str):
        if not getattr(self, "filepath", None) or not getattr(self, "namesOfParameters", None):
            raise RuntimeError("No dataset loaded yet")
    
        output_name = self.comboColumns.currentText()
        if not output_name:
            raise RuntimeError("No output selected")
    
        coeff_df, validation_df, predict_fn = regressModell(
            output_name,
            self.namesOfParameters,
            self.filepath,
            model_name,
            return_predict_fn=True,
        )
    
        # ValidationMetricsTab erwartet (output_name, validation_df, predict_fn)
        return output_name, validation_df, predict_fn
    
    # calculate the model results based on the parameters used for prediction    
    def estimateResult(self):
        # Build values list from the textfields in the GUI 
        self.values = []
        for name in self.namesOfParameters:
            txt = self.input_fields[name].text().strip()
            try:
                val = float(txt)
            except ValueError:
                # Graceful failure: do nothing if parsing fails.
                # A more explicit UX pattern would show a QMessageBox.
                return
    
            self.values.append(val)

        self.estimatedResults, self.estParamCheck = estimationOfResults(
            self.namesOfParameters,
            self.values,
            self.filepath
        )
        
        # Warn if any entered parameter is outside training range (extrapolation).
        if sum(self.estParamCheck)>0:
            QMessageBox.warning(
                self,
                "ACHTUNG",
                "Enterd parameters are outside of the data that trained the model! "
                "You are extrapolating which will give bad results!"
            )
        
        self.model3.setDataFrame(self.estimatedResults.round(3))
        self.tableEstRestults.resizeColumnsToContents()
        self.tableEstRestults.resizeRowsToContents()
        
    
    # plot a 3D surface of the model based on the chosen parameters
    def visualizePlot(self):
        # get the values from the textboxes first (fixed params)
        self.vis_values = {}
        for name, le in self.visInputFields.items():
            txt = le.text().strip()
            try:
                self.vis_values[name] = float(txt)
            except ValueError:
                QMessageBox.warning(self, "Invalid input", f"'{name}' must be a number.")
                return
        
        # The two varying parameters define the 2D slice through input space
        self.vizualizationParameters = [self.VisParameter1.currentText(), self.VisParameter2.currentText()]

        # Single output chosen for the surface plot
        self.vizualizationOutputs   = [self.VisOutput.currentText()]
        
        # Backend returns a DataFrame to plot
        self.dataForVizualization = visualizationGetData(
            self.dataFull,
            self.namesOfParameters,
            self.vizualizationParameters,
            self.vizualizationOutputs,
            self.vis_values
        )
        
        p1 = self.VisParameter1.currentText()
        p2 = self.VisParameter2.currentText()
        out = self.VisOutput.currentText()
        df = self.dataForVizualization

        # sanity checks
        for c in (p1, p2, out):
            if c not in df.columns:
                QMessageBox.warning(self, "Missing data", f"Column '{c}' not found in data.")
                return
        
        # ensure 3D axes exist
        try:
            is3d = getattr(self.regressAx, 'name', '') == '3d'
        except Exception:
            is3d = False
        if not is3d:
            self.regressFig.clear()
            self.regressAx = self.regressFig.add_subplot(111, projection='3d')
        
        ax = self.regressAx
        ax.clear()
        
        # try to build a rectangular grid first (best for plot_surface)
        try:
            Zp = df.pivot_table(index=p2, columns=p1, values=out)
            # If pivot produced a proper 2D grid (no ragged edges)
            if Zp.notna().any().any() and Zp.index.size > 1 and Zp.columns.size > 1:
                X, Y = np.meshgrid(Zp.columns.values, Zp.index.values)
                Z = Zp.values
                surf = ax.plot_surface(
                    X, Y, Z,
                    cmap="turbo",  # nice bright colormap
                    linewidth=0, antialiased=False, alpha=0.95
                )
            else:
                raise ValueError("Pivot not grid-like")
        except Exception:
            # fallback: scattered data → triangulated surface
            x = df[p1].to_numpy()
            y = df[p2].to_numpy()
            z = df[out].to_numpy()
            surf = ax.plot_trisurf(x, y, z, cmap="turbo", linewidth=0.2, antialiased=True)

        # Set axis limits based on computed surfaces.
        # NOTE: This assumes X/Y/Z exist even in fallback; potential issue if pivot failed.
        ax.set_xlim(df[p1].min(), df[p1].max())
        ax.set_ylim(df[p2].min(), df[p2].max())
        ax.set_zlim(df[out].min(), df[out].max())

        # overlay black dots for observed/measured points if available
        if "is_point" in df.columns:
            pts = df[df["is_point"].astype(bool)]
            if not pts.empty:
                ax.scatter(pts[p1], pts[p2], pts[out], color="k", s=30, depthshade=False)
        
        # labels & style
        ax.set_xlabel(p1)
        ax.set_xlabel(p1, labelpad=15)
        ax.set_ylabel(p2)
        ax.set_ylabel(p2, labelpad=15)
        ax.set_zlabel(out)

        # viewing geometry and aesthetics
        ax.view_init(elev=25, azim=-60)
        ax.set_box_aspect((1.3, 1.0, 0.6))
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_alpha(0.0)
        ax.grid(True)
        
        ax.set_proj_type("ortho")

        # Put z-axis visually on the LEFT by rotating the view
        ax.view_init(elev=22, azim=35)
        
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(out, labelpad=12)
        ax.zaxis.set_label_coords(-0.12, 0.5)
                
        ax.set_box_aspect((1.3, 1.0, 0.6))
        ax.grid(True, linestyle=':', linewidth=0.5, color='gray', alpha=0.7)
        
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # tight layout and draw
        self.regressFig.tight_layout()
        self.regressCanvas.draw_idle()
        
    # load possible LCA paramters, chose suitable for your case and match them with your process data. Matching is stored afterwards
        
    def chose_LCAParameters(self):
        self.LCA_list = getLCAlist()
        dlg = TransferDialog(self, items=self.LCA_list)
        if dlg.exec_():
            self.chosen = dlg.selected_items()
            
            dlg2 = MappingDialog(self.dataFull.columns.tolist(), self.chosen, self)
            if dlg2.exec_():
                self.linkedParameters = dlg2.get_mapping()
                
                json_path = resource_path(os.path.join("backgrounddata", "linked_parameters.json"))
                with open(json_path, "w") as f:
                    json.dump(self.linkedParameters, f, indent=4)
                
                self.calculateLCAPredictionButton.setEnabled(True)
                self.calculateLCAtrialsButton.setEnabled(True)
                        
    #load already matched LCA-background and process data
    def load_LCAdata(self):
        json_path = resource_path(os.path.join("backgrounddata", "linked_parameters.json"))
        try:
            with open(json_path, "r") as f:
                self.linkedParameters = json.load(f)
        except FileNotFoundError:
            QMessageBox.warning(
                self,
                "Error",
                "No linked LCA data file found yet.\n"
                "Please create a new link first."
            )
            return
        
        self.calculateLCAPredictionButton.setEnabled(True)
        self.calculateLCAtrialsButton.setEnabled(True)
        
    # calculate the LCA and display the results table as well as the graph 
        
    def calculateLCAresultsForTrials(self):
        dfLCAtrialsResult,self.dataFullwithLCA = calculateLCAForTrials(self.dataFull, self.linkedParameters)
        #rebuild inputDataframe on tab 1
        self.dataFull = self.dataFullwithLCA
        self.model.setDataFrame(self.dataFull.round(3))
        self.tableDataReadIn.resizeColumnsToContents()
        self.tableDataReadIn.resizeRowsToContents()
        
        #rebuild selection for LCA calculations
        self.comboColumns.clear()
        cols = self.dataFull.columns[len(self.namesOfParameters)+1:]  # skip the Inputparameters
        self.comboColumns.addItems([str(c) for c in cols])
        self.VisOutput.addItems([str(c) for c in cols])
        #Visulalising data in the last tab
        # If the first column is the contributor label, make it the index
        if "Contributor" in dfLCAtrialsResult.columns:
            dfLCAtrialsResult = dfLCAtrialsResult.set_index("Contributor")
    
        # Keep a clean, numeric frame for plotting; leave non-numerics alone
        numeric = dfLCAtrialsResult.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        self.LCAresultTable = dfLCAtrialsResult.copy()
        self.modelLCAtable.setDataFrame(self.LCAresultTable.round(5))
        self.tableLCAcalcs.resizeColumnsToContents()
        self.tableLCAcalcs.resizeRowsToContents()
        
        # 2) Build the stacked bars (positives up, negatives down) across columns
        self.lcaAx.clear()
    
        cols = numeric.columns.tolist()
        x = np.arange(len(cols))  # one stack per column
    
        # running bottoms for each column
        pos_bottom = pd.Series(0.0, index=cols)
        neg_bottom = pd.Series(0.0, index=cols)
    
        # optional: order contributors by total absolute contribution (largest first)
        order = numeric.abs().sum(axis=1).sort_values(ascending=False).index
    
        used_in_legend = set()
        for contributor in order:
            row = numeric.loc[contributor].fillna(0.0)
    
            pos = row.clip(lower=0)
            neg = row.clip(upper=0)
    
            # plot positives
            if (pos != 0).any():
                bars = self.lcaAx.bar(
                    x,
                    pos.values,
                    bottom=pos_bottom.values,
                    label=(contributor if contributor not in used_in_legend else None),
                )
                pos_bottom += pos
                used_in_legend.add(contributor)
    
            # plot negatives
            if (neg != 0).any():
                bars = self.lcaAx.bar(
                    x,
                    neg.values,
                    bottom=neg_bottom.values,
                    label=(contributor if contributor not in used_in_legend else None),
                )
                neg_bottom += neg
                used_in_legend.add(contributor)
    
        # formatting
        self.lcaAx.axhline(0, linewidth=1)
        self.lcaAx.set_xticks(x)
        self.lcaAx.set_xticklabels([c[:8] + "…" if len(c) > 9 else c for c in cols], rotation=30, ha="right")
        self.lcaAx.set_ylabel("GWP [kg CO₂ per kg feedstock]")
        self.lcaAx.set_title("Life Cycle Assessment results of loaded data")

        self.lcaAx.margins(x=0.005)
        self.lcaAx.set_xlim(-0.5, len(cols) - 0.5)

        # legend below plot, expanded
        self.lcaAx.legend(
            loc="lower center",
            bbox_to_anchor=(0.0, -0.35, 1.0, 0.15),
            mode="expand",
            ncol=min(8, len(self.lcaAx.get_legend_handles_labels()[1])),
            fontsize=9,
            frameon=False,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=0.8,
            borderaxespad=0.0,
        )

        self.lcaFig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.23)
        self.lcaCanvas.draw_idle()    

        
    def calculateLCAresultsForPrediction(self):
        dfLCAtrialsResult = calculateLCAForPrediction(
            self.namesOfParameters,
            self.values,
            self.estimatedResults,
            self.linkedParameters
        )

        if "Contributor" in dfLCAtrialsResult.columns:
            dfLCAtrialsResult = dfLCAtrialsResult.set_index("Contributor")
    
        numeric = dfLCAtrialsResult.apply(pd.to_numeric, errors="coerce")
        self.LCAresultTable = dfLCAtrialsResult.copy()
        self.modelLCAtable.setDataFrame(self.LCAresultTable.round(5))
        self.tableLCAcalcs.resizeColumnsToContents()
        self.tableLCAcalcs.resizeRowsToContents()
        
        self.tableLCAcalcs.resizeColumnsToContents()
        self.tableLCAcalcs.resizeRowsToContents()
    
        self.lcaAx.clear()
    
        cols = numeric.columns.tolist()
        x = np.arange(len(cols))
    
        pos_bottom = pd.Series(0.0, index=cols)
        neg_bottom = pd.Series(0.0, index=cols)
    
        order = numeric.abs().sum(axis=1).sort_values(ascending=False).index
    
        used_in_legend = set()
        for contributor in order:
            row = numeric.loc[contributor].fillna(0.0)
    
            pos = row.clip(lower=0)
            neg = row.clip(upper=0)
    
            if (pos != 0).any():
                bars = self.lcaAx.bar(
                    x,
                    pos.values,
                    bottom=pos_bottom.values,
                    label=(contributor if contributor not in used_in_legend else None),
                )
                pos_bottom += pos
                used_in_legend.add(contributor)
    
            if (neg != 0).any():
                bars = self.lcaAx.bar(
                    x,
                    neg.values,
                    bottom=neg_bottom.values,
                    label=(contributor if contributor not in used_in_legend else None),
                )
                neg_bottom += neg
                used_in_legend.add(contributor)
    
        self.lcaAx.axhline(0, linewidth=1)
        self.lcaAx.set_xticks(x)
        self.lcaAx.set_xticklabels(cols, ha="right")
        self.lcaAx.set_ylabel("GWP [kg CO₂ per kg feedstock]")
        self.lcaAx.set_title("Life Cycle Assessment results of prediction based on regression model")

        self.lcaAx.margins(x=0.005)
        self.lcaAx.set_xlim(-0.5, len(cols) - 0.5)

        # legend below, anchored to figure coordinates (not axes)
        handles, labels = self.lcaAx.get_legend_handles_labels()
        self.lcaAx.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.0, 0.02, 1.0, 0.14),
            bbox_transform=self.lcaFig.transFigure,
            mode="expand",
            ncol=min(8, len(labels)),
            fontsize=9,
            frameon=False,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=0.8,
        )
        
        # center the axes and leave margins for the legend
        self.lcaFig.subplots_adjust(left=0.35, right=0.65, top=0.90, bottom=0.26)
        
        self.lcaCanvas.draw_idle()

        
# this is a small "helper" to be able to call subfolders after convertig file into .exe
    
def resource_path(relative_path: str) -> str:
    """
    Gibt den absoluten Pfad zu einer Ressource zurück.
    Funktioniert sowohl im normalen Python-Skript als auch in der
    PyInstaller-EXE.
    """
    if hasattr(sys, "_MEIPASS"):
        # Pfad, den PyInstaller zur Verfügung stellt
        base_path = sys._MEIPASS
    else:
        # Pfad der aktuellen Datei
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

# initialise the code

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 14))
    
    icon_path = resource_path(os.path.join("Grafiken", "ChemRecPiktogramm.ico"))
    app.setWindowIcon(QIcon(icon_path))
    
    window = MainWindow()          # create the main GUI window
    window.showMaximized()         # show it maximized for better table/plot visibility
    sys.exit(app.exec_())          # start Qt event loop

if __name__ == "__main__":
    main()