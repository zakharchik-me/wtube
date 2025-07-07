import sys
import os
import glob
from PyQt6.QtWidgets import QApplication
from app.window import MainWindow

def _cleanup_test_logs():
    for path in glob.glob('logs/test/annotations/*'):
        try:
            os.remove(path)
        except OSError as e:
            print(f"Could not remove {path}: {e}")

    for path in glob.glob('logs/test/images/*'):
        try:
            os.remove(path)
        except OSError as e:
            print(f"Could not remove {path}: {e}")

def main():
    _cleanup_test_logs()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
