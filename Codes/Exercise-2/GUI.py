from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication

import GUI_background
import sys # Only needed for access to command line arguments


#######################
# You need one (and only one) QApplication instance per application.
# Pass in sys.argv to allow command line arguments for your app.
# If you know you won't use command line arguments QApplication([]) works too.
# Subclass QMainWindow to customize your application's main window

app = QApplication(sys.argv)

window = GUI_background.MainWindow()
window.setWindowTitle("Image Manipulation Tool by Ibrahim Furkan Tezcan  |  v3.2-alpha  |  Release: 15.Oct.2023")
window.show()

app.exec()

