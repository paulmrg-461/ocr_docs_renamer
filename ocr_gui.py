import sys
import os
from typing import Optional, Tuple, List

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QMessageBox,
)

# Importar funciones existentes del renombrador
from ocr_pdf import process_prefix_rename, process_classify_rename


class RenamerWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(int, int)  # renamed_count, skipped_count
    error = pyqtSignal(str)

    def __init__(
        self,
        mode: str,
        dir_path: str,
        out_dir: Optional[str],
        lang: str,
        dpi: int,
        min_score: float,
        allow_fallback: bool,
    ):
        super().__init__()
        self.mode = mode
        self.dir_path = dir_path
        self.out_dir = out_dir
        self.lang = lang
        self.dpi = dpi
        self.min_score = min_score
        self.allow_fallback = allow_fallback

    def run(self):
        try:
            if self.mode == "prefijos":
                renamed, skipped = process_prefix_rename(
                    dir_path=self.dir_path,
                    lang=self.lang,
                    dpi=self.dpi,
                    min_score=self.min_score,
                    out_dir=self.out_dir,
                    allow_fallback=self.allow_fallback,
                )
            else:
                renamed, skipped = process_classify_rename(
                    dir_path=self.dir_path,
                    lang=self.lang,
                    dpi=self.dpi,
                    out_root=self.out_dir,
                    min_score=self.min_score,
                    allow_fallback=self.allow_fallback,
                )
            # Emitir logs resumidos
            for src, dst in renamed:
                self.log.emit(f"[OK] Renombrado: {os.path.basename(src)} -> {os.path.basename(dst)}")
            for src, motivo in skipped:
                self.log.emit(f"[OMITIDO] {os.path.basename(src)} ({motivo})")
            self.finished.emit(len(renamed), len(skipped))
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR Docs Renamer - Interfaz")
        self.setMinimumSize(760, 560)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Directorio de entrada
        dir_layout = QHBoxLayout()
        layout.addLayout(dir_layout)
        dir_layout.addWidget(QLabel("Directorio de entrada:"))
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Selecciona la carpeta con PDFs a procesar")
        dir_layout.addWidget(self.dir_edit)
        self.dir_btn = QPushButton("Examinar...")
        dir_layout.addWidget(self.dir_btn)

        # Directorio de salida
        out_layout = QHBoxLayout()
        layout.addLayout(out_layout)
        out_layout.addWidget(QLabel("Directorio de salida:"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Opcional: carpeta destino (por defecto: la misma carpeta de entrada)")
        out_layout.addWidget(self.out_edit)
        self.out_btn = QPushButton("Examinar...")
        out_layout.addWidget(self.out_btn)

        # Parámetros
        params_layout = QHBoxLayout()
        layout.addLayout(params_layout)
        # Modo
        params_layout.addWidget(QLabel("Modo:"))
        self.mode_prefijos_btn = QCheckBox("Prefijos (TE/CE/MP/CMP/NPMP/AP/RREX/AC/DF/NC/FJ)")
        self.mode_prefijos_btn.setChecked(True)
        params_layout.addWidget(self.mode_prefijos_btn)
        # Idioma
        params_layout.addWidget(QLabel("Idioma:"))
        self.lang_edit = QLineEdit("es")
        self.lang_edit.setFixedWidth(100)
        params_layout.addWidget(self.lang_edit)
        # DPI
        params_layout.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(100, 600)
        self.dpi_spin.setValue(300)
        params_layout.addWidget(self.dpi_spin)
        # Min Score
        params_layout.addWidget(QLabel("Min Score:"))
        self.min_score_spin = QDoubleSpinBox()
        self.min_score_spin.setRange(0.0, 1.0)
        self.min_score_spin.setSingleStep(0.05)
        self.min_score_spin.setValue(0.80)
        params_layout.addWidget(self.min_score_spin)
        # Fallback
        self.fallback_chk = QCheckBox("Permitir fallback (radicado/expediente por dígitos largos)")
        params_layout.addWidget(self.fallback_chk)

        # Botones de acción
        action_layout = QHBoxLayout()
        layout.addLayout(action_layout)
        self.run_btn = QPushButton("Procesar")
        action_layout.addWidget(self.run_btn)
        self.stop_btn = QPushButton("Detener")
        self.stop_btn.setEnabled(False)
        action_layout.addWidget(self.stop_btn)

        # Área de logs
        layout.addWidget(QLabel("Logs:"))
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit, stretch=1)

        # Estado
        self.status_label = QLabel("Listo.")
        layout.addWidget(self.status_label)

        # Conexiones
        self.dir_btn.clicked.connect(self.choose_input_dir)
        self.out_btn.clicked.connect(self.choose_output_dir)
        self.run_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)

        self.worker: Optional[RenamerWorker] = None

    def append_log(self, text: str):
        self.log_edit.append(text)

    def choose_input_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Selecciona carpeta de entrada")
        if d:
            self.dir_edit.setText(d)

    def choose_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Selecciona carpeta de salida")
        if d:
            self.out_edit.setText(d)

    def start_processing(self):
        dir_path = self.dir_edit.text().strip()
        out_dir = self.out_edit.text().strip() or None
        lang = self.lang_edit.text().strip() or "latin"
        dpi = int(self.dpi_spin.value())
        min_score = float(self.min_score_spin.value())
        allow_fallback = self.fallback_chk.isChecked()

        if not dir_path or not os.path.isdir(dir_path):
            QMessageBox.warning(self, "Error", "Debes seleccionar un directorio de entrada válido.")
            return
        if out_dir and not os.path.isdir(out_dir):
            QMessageBox.warning(self, "Error", "El directorio de salida no existe.")
            return

        self.log_edit.clear()
        self.status_label.setText("Procesando...")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        mode = "prefijos" if self.mode_prefijos_btn.isChecked() else "clasificar"
        self.worker = RenamerWorker(
            mode=mode,
            dir_path=dir_path,
            out_dir=out_dir,
            lang=lang,
            dpi=dpi,
            min_score=min_score,
            allow_fallback=allow_fallback,
        )
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def stop_processing(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.status_label.setText("Detenido.")
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.append_log("[INFO] Proceso detenido por el usuario.")

    def on_finished(self, renamed_count: int, skipped_count: int):
        self.status_label.setText(f"Terminado. Renombrados: {renamed_count} | Omitidos: {skipped_count}")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_error(self, msg: str):
        self.append_log(f"[ERROR] {msg}")
        self.status_label.setText("Error.")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()