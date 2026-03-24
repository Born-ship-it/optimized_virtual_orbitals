Place all figures (.pdf, .png, .pgf) in this folder.

Recommended workflow:
  - Generate plots from Python (matplotlib) and save as .pdf (vector).
  - Name convention: fig_<chapter>_<description>.pdf
    e.g.  fig_results_energy_recovery_H2O.pdf
  - Reference in LaTeX as: \includegraphics{fig_results_energy_recovery_H2O}
    (no extension needed; graphicx will find the file)

Do NOT commit large binary figure files to Git if avoidable.
Use .gitignore to exclude them and store in cloud storage.
